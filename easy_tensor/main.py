import types
from functools import wraps
from contextlib import contextmanager
import easy_tensor as tf
import math


# constant
floatx = tf.float32
intx = tf.int32
eps = 1e-8
pi = math.pi

# global step
_global_step = tf.Variable(0, name='global_step', trainable=False)


def get_global_step():
    global _global_step
    return _global_step


def session(graph=None, config=None):
    """
    https://www.tensorflow.org/how_tos/using_gpu/
    :param graph:
    :param config:
    :return:
    """
    if config is None:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # config = tf.ConfigProto(allow_soft_placement=False)
        # config.gpu_options.allocator_type = 'BFC'
        # config.gpu_options.allow_growth = False
    sess = tf.Session(config=config, graph=graph)
    # sess.graph = graph
    return sess  #.to_default()
