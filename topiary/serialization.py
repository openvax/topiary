"""Shared conversion of scientific scalars to ordinary Python values."""

from numbers import Integral, Real

import numpy as np


def normalize_python_types(value):
    """Copy nested data with Python equivalents of NumPy scalar types.

    Parameters
    ----------
    value : object
        A scalar or nested dict, list or tuple. Numeric and boolean values
        may come from NumPy, for example from a pandas row or custom creator.

    Returns
    -------
    object
        Booleans become ``bool``, integral numbers become ``int``, real
        numbers become ``float``, and strings become ``str``. Containers
        retain their shape with recursively normalized keys and values;
        the input is not mutated. ``None`` and empty containers are valid.

        Other objects are left unchanged, so serialization still rejects
        unsupported types instead of silently stringifying them. This is
        not coercion of strings or truthy values: ``"False"`` stays a string
        and ``0`` stays an integer. Booleans are checked before integers
        because Python treats ``bool`` as an integer subclass.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, dict):
        return {normalize_python_types(k): normalize_python_types(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_python_types(v) for v in value]
    if isinstance(value, tuple):
        return tuple(normalize_python_types(v) for v in value)
    return value
