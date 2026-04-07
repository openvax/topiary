from contextlib import contextmanager

import pytest


def eq_(x, y):
    assert x == y, "Expected %s == %s" % (x, y)


@contextmanager
def assert_raises(e_expected):
    with pytest.raises(e_expected):
        yield
