import logging
import os
import sys
import time


_logger = logging.getLogger('easy_tensor')
_logger.addHandler(logging.StreamHandler())


def _log_prefix():

    # Returns (filename, line number) for the stack frame.
    def _get_file_line():

        # pylint: disable=protected-access
        # noinspection PyProtectedMember
        f = sys._getframe()
        # pylint: enable=protected-access
        our_file = f.f_code.co_filename
        f = f.f_back
        while f:
            code = f.f_code
            if code.co_filename != our_file:
                return code.co_filename, f.f_lineno
            f = f.f_back
        return '<unknown>', 0

    # current time
    now = time.time()
    now_tuple = time.localtime(now)
    now_millisecond = int(1e3 * (now % 1.0))

    # current filename and line
    filename, line = _get_file_line()
    basename = os.path.basename(filename)

    s = '%02d%02d:%02d:%02d:%02d.%03d:%s:%d] ' % (
        now_tuple[1],  # month
        now_tuple[2],  # day
        now_tuple[3],  # hour
        now_tuple[4],  # min
        now_tuple[5],  # sec
        now_millisecond,
        basename,
        line)

    return s


def logging_verbosity(verbosity=0):
    _logger.setLevel(verbosity)


def logging_debug(msg, *args, **kwargs):
    _logger.debug('D ' + _log_prefix() + msg, *args, **kwargs)


def logging_info(msg, *args, **kwargs):
    _logger.info('I ' + _log_prefix() + msg, *args, **kwargs)


def logging_warn(msg, *args, **kwargs):
    _logger.warn('W ' + _log_prefix() + msg, *args, **kwargs)


def logging_error(msg, *args, **kwargs):
    _logger.error('E ' + _log_prefix() + msg, *args, **kwargs)


def logging_fatal(msg, *args, **kwargs):
    _logger.fatal('F ' + _log_prefix() + msg, *args, **kwargs)
