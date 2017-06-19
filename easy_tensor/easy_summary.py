from __future__ import absolute_import
import easy_tensor as tf
import re
from tensorflow.python.ops import gen_logging_ops


# reference: sugartensor
def _pretty_name(tensor):
    name = ''.join(tensor.name.split(':')[:-1])
    return re.sub(r'gpu_[0-9]+/', '', name)


def _scalar(name, tensor):
    if not tf.get_variable_scope().reuse:
        val = gen_logging_ops._scalar_summary(name, tensor)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, val)


def _histogram(name, tensor):
    if not tf.get_variable_scope().reuse:
        val = gen_logging_ops._histogram_summary(name, tensor)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, val)


def summary_loss(tensor, prefix='losses', name=''):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    _scalar(name, tf.reduce_mean(tensor))
    _histogram(name + '-h', tensor)


def summary_gradient(tensor, gradient, prefix=None, name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    # noinspection PyBroadException
    _scalar(name + '/grad', tf.reduce_mean(tf.abs(gradient)))
    _histogram(name + '/grad-h', tf.abs(gradient))


def summary_activation(tensor, prefix=None, name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    _scalar(name + '/ratio',
            tf.reduce_mean(tf.cast(tf.greater(tensor, 0), tf.floatx)))
    _scalar(name + '/max', tf.reduce_max(tensor))
    _scalar(name + '/min', tf.reduce_max(tensor))
    _histogram(name + '/ratio-h', tensor)


def summary_param(tensor, prefix=None, name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    _scalar(name + '/abs', tf.reduce_mean(tf.abs(tensor)))
    _histogram(name + '/abs-h', tf.abs(tensor))


def summary_image(tensor, prefix=None, name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    if not tf.get_variable_scope().reuse:
        tf.summary.image(name + '-im', tensor)


def summary_audio(tensor, sample_rate=16000, prefix=None, name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    if not tf.get_variable_scope().reuse:
        tf.summary.audio(name + '-au', tensor, sample_rate)