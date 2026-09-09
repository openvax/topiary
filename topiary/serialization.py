"""Shared conversion of scientific scalars to ordinary Python values."""

import dataclasses

import numpy as np


def normalize_python_types(value, *, dataclasses_as_dict=False):
    """Copy nested data, simplifying scalar types without changing values.

    Parameters
    ----------
    value : object
        A scalar or nested dict, list or tuple, for example from a pandas
        row or custom creator. ``None`` and empty containers are valid.
    dataclasses_as_dict : bool, optional
        Also represent dataclass instances as field dictionaries, recursively.
        Used by fragment serialization. By default dataclass objects are left
        intact, like other custom objects. No arbitrary objects are deep-copied:
        their copy hooks can change stored values before an encoder sees them.

    Returns
    -------
    object
        NumPy boolean, signed/unsigned integer and floating
        scalars use NumPy's lossless ``item`` conversion. Extended-precision
        floats without an equivalent Python type remain NumPy scalars.
        Built-in str/int/float subclasses use their stored value, not an
        overridden display/conversion method; a string-backed enum containing
        ``"variant:snv"`` therefore becomes that string, not its member name.

        Dates, durations, decimals, fractions, complex numbers, bytes,
        arrays and other objects are left unchanged for their encoders.
        In particular, durations must not become unitless integers merely
        because NumPy classifies them as integral. This function does not
        promise that every output is JSON-serializable: unsupported objects
        still need an explicit encoder.

        Dicts, lists and tuples are copied with recursively normalized keys
        and values; the input is not mutated. ``"False"`` remains a string,
        ``0`` remains an integer, and ``False`` remains a boolean.
    """
    if dataclasses_as_dict and dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, str):
        # Also covers np.str_; its item() can discard trailing NUL characters.
        return str.__str__(value)
    if isinstance(value, np.generic):
        # Whitelist primitive dtype kinds. Temporal kinds (m/M) carry units;
        # item() can discard those units, so they must not enter this path.
        return np.generic.item(value) if value.dtype.kind in "biuf" else value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    if isinstance(value, dict):
        return {
            normalize_python_types(k, dataclasses_as_dict=dataclasses_as_dict):
            normalize_python_types(v, dataclasses_as_dict=dataclasses_as_dict)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [normalize_python_types(v, dataclasses_as_dict=dataclasses_as_dict) for v in value]
    if isinstance(value, tuple):
        return tuple(normalize_python_types(v, dataclasses_as_dict=dataclasses_as_dict) for v in value)
    return value
