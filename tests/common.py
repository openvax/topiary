from functools import wraps


def eq_(x, y):
    if x != y:
        raise ValueError("Expected %s = %s" % (x, y))


def raises(e_expected):
    def outer_fn(fn):
        try:
            fn()
        except Exception as e:
            return type(e) == e_expected
        return False

    return wraps(outer_fn)
