import easy_tensor as tf
from tensorflow.python.platform import tf_logging as logging
import threading
from functools import wraps


# using sugartensor code (sg_producer_func)
# # https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def numpy_producer_func(func):
    r"""
    Decorates a function `func` as sg_producer_func
    This is only for large numpy array without making tf.constant
    :param func:
    :return:
    """

    @wraps(func)
    def wrapper(sources, range_dequeue, capacity, num_threads, preprocess_fn=None):
        """

        :param sources:
        :param range_dequeue:
        :param capacity:
        :param num_threads:
        :param shuffle:
        :param preprocess_fn:
        :return:
        """
        def enqueue_func(sess, op):
            idx = sess.run(range_dequeue)

            _sources = func(sources, preprocess_fn)
            feed_dict = {}
            for ph, _source in zip(placeholders, _sources):
                feed_dict[ph] = _source[idx]
            sess.run(op, feed_dict=feed_dict)

        placeholders = []
        shapes = []
        dtypes = []
        for source in sources:
            dtype = source.dtype
            shape = list(source.shape)[1:]
            placeholders.append(tf.placeholder(dtype=dtype, shape=shape))
            shapes.append(shape)
            dtypes.append(dtype)

        # create FIFO queue
        queue = tf.FIFOQueue(capacity, shapes=shapes, dtypes=dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op] * num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)

        return queue.dequeue()

    return wrapper


@numpy_producer_func
def gen_queue_producer(sources, preprocess_fn=None):
    """
    example
    ```
    # make a queue
    train_idx_queue = tf.train.range_input_producer(self.num_train) #
    sources_train = [self.train, self.train_label, self.train_lens] # list of numpy
    dequeue_train = gen_queue_producer(sources=sources_train, range_dequeue=train_idx_queue.dequeue(),
                                   capacity=128, num_threads=self.num_threads,
                                   preprocess_fn=preprocess_fn)

    shapes = [temp.get_shape() for temp in dequeue_train]

    # make a batch queue
    train_result = tf.train.shuffle_batch(dequeue_train,
                                          num_threads=self.num_threads,
                                          batch_size=batch_size,
                                          shapes=shapes,
                                          capacity=batch_size*(self.num_threads + 24),
                                          min_after_dequeue=batch_size*16,
                                          allow_smaller_final_batch=False,
                                          name='train_queue')
    ```
    :param sources:
    :param preprocess_fn:
    :return:
    """

    _sources = sources

    if preprocess_fn:
        _sources = preprocess_fn(_sources)

    return _sources


class _FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                               queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op)  # call enqueue function
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1
