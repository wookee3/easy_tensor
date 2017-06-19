import easy_tensor as tf


class patch(object):

    @staticmethod
    def method(clz, fun, name=None):
        name = name or fun.__name__
        for c in clz:
            setattr(c, name, fun)

    @staticmethod
    def methods(clz, funs, names=None):
        if names is None:
            for f in funs:
                name = f.__name__
                for c in clz:
                    setattr(c, name, f)
        else:
            for f, name in zip(funs, names):
                for c in clz:
                    setattr(c, name, f)

    @staticmethod
    def getter(clz, fun, name=None):
        name = name or fun.__name__
        p = property(fun)
        for c in clz:
            setattr(c, name, p)

    @staticmethod
    def getters(clz, funs):
        for f in funs:
            p = property(f)
            name = f.__name__
            for c in clz:
                setattr(c, name, p)


def patchmethod(*cls, **kwargs):
    """
    클래스 멤버 패치 @patchmethod(Cls1, ..., [name='membername'])
    ex)
    class A(object):
        def __init__(self, data):
            self.data = data

    @patchmethod(A)
    def sample(self):
        ''' haha docstrings '''
        print self.data

    @patchmethod(A, name='membermethod)
    def sample(self):
        ''' haha docstrings '''
        print self.data

    a = A()
    a.sample()

    """

    def _patch(fun):
        m = kwargs.pop('name', None) or fun.__name__
        for c in cls:
            setattr(c, m, fun)
            # c.__dict__[m].__doc__ = fun.__doc__

    def wrap(fun):
        _patch(fun)
        return fun

    return wrap


def patchproperty(*cls, **kwargs):
    """
    class getter 함수 패치 decorator
    EX)
    class B(A):
        pass

    @patchproperty(B)
    def prop(self):
        return 'hello'

    :param cls:
    :param kwargs:
    """

    def _patch(fun):
        m = kwargs.pop('property', None) or fun.__name__
        p = property(fun)
        for c in cls:
            setattr(c, m, p)

    def wrap(fun):
        _patch(fun)
        return fun

    return wrap
