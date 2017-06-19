import tensorflow as tf
from .easy_decoder import read_image


def queue_producer(tensors, capacity, shapes=None, threads=1, shuffle=False, min_after_dequeue=0,
                   enqueue_many=False):
    """
    todo : queue를 생성하는 함수
    :param tensors:
    :param capacity:
    :param shapes:
    :param threads:
    :param shuffle:
    :param min_after_dequeue:
    :param enqueue_many:
    :return:
    """
    if not isinstance(tensors, (tuple, list)):
        tensors = [tensors]
    dtypes = [img.dtype for img in tensors]
    shapes = shapes or [img.get_shape() for img in tensors]
    if shuffle:
        q = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes=dtypes, shapes=shapes)
        # waitempty = (min_after_dequeue == 0)
    else:
        q = tf.FIFOQueue(capacity, dtypes=dtypes, shapes=shapes)
        # waitempty = True
    if enqueue_many:
        enq = q.enqueue_many(tensors)
    else:
        enq = q.enqueue(tensors)

    qr = tf.train.QueueRunner(q, enqueue_ops=[enq]*threads)
    tf.train.add_queue_runner(qr)

    return q


def filename_read_producer(fname, capacity, decoder=read_image, shape=None, preprocess=None, threads=1,
                        shuffle=False, min_after_dequeue=0, channels=None):
    """
    example::
        todo : add some example

    :param fname:
    :param capacity:
    :param decoder:
    :param shape:
    :param preprocess:
    :param threads:
    :param shuffle:
    :param min_after_dequeue:
    :param channels:
    :return:
    """
    img = decoder(fname, channels=channels)

    if preprocess is not None:
        img = preprocess(img)

    q = queue_producer(img, capacity, shapes=[shape], threads=threads,
                       shuffle=shuffle, min_after_dequeue=min_after_dequeue)
    return q


def file_matching_producer(pattern, decoder=read_image, **kwargs):
    files = tf.matching_files(pattern)
    q = filename_read_producer(files, decoder, **kwargs)

    return q

# tensorflow default queue producer alias
slice_input_producer = tf.train.slice_input_producer
range_input_procuder = tf.train.range_input_producer
string_input_producer = tf.train.string_input_producer
input_producer = tf.train.input_producer
