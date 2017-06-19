import easy_tensor as tf
from easy_tensor.utility import patch
from six import wraps
from rnn import *


def _layer(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):
        name = kwargs.pop('name', None)
        scope = kwargs.pop('scope', None) or fn.__name__
        reuse = kwargs.pop('reuse', None)

        with tf.variable_scope(name, reuse=reuse, scope=scope):
            out = fn(*args, **kwargs)

        return out

    return wrapped


def _layer_sflow(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):
        n = kwargs.pop('name', None)
        sc = kwargs.pop('scope', None) or fn.__name__
        # with tf.variable_scope(sc, fn.__name__):
        with tf.variable_scope(n, sc):
            return tf.options.call_with_default_context(fn, *args, **kwargs)

    patch.method([tf.Tensor, tf.Variable], wrapped)

    return wrapped


def _kernel_shape(nd, k, indim, outdim):
    if isinstance(k, int):
        k = [k for _ in range(nd)]
    k = list(k)
    assert len(k) == nd
    k.extend([indim, outdim])
    return k


def _stride_shape(nd, s):
    """

    :param nd:
    :param s: int | list | tuple
    :return:
    """
    if isinstance(s, int):
        s = [s for _ in range(nd)]
    s = list(s)
    assert len(s) == nd
    s = [1] + s + [1]
    return s


# cnn


# pooling


