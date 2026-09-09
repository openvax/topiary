"""Native scalar normalization without guessing meanings for other objects."""

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum, IntEnum
from fractions import Fraction

import numpy as np
import pytest

from topiary import ProteinFragment, normalize_python_types, write_fragments


class Origin(str, Enum):
    SNV = "variant:snv"


class Count(IntEnum):
    ONE = 1


class DisplayString(str):
    def __str__(self):
        return "display, not data"


class DisplayInteger(int):
    def __int__(self):
        return 999


class DisplayFloat(float):
    def __float__(self):
        return 999.0


@pytest.mark.parametrize("value,expected", [
    (Origin.SNV, "variant:snv"),
    (DisplayString("variant:snv"), "variant:snv"),
    (Count.ONE, 1),
    (DisplayInteger(2), 2),
    (DisplayFloat(0.5), 0.5),
])
def test_builtin_subclasses_preserve_stored_values(value, expected):
    normalized = normalize_python_types(value)
    assert normalized == expected
    assert type(normalized) is type(expected)
    # Python's JSON encoder uses the underlying value, not display hooks.
    assert json.dumps(normalized) == json.dumps(value)


@pytest.mark.parametrize("factory", [str, np.str_, DisplayString])
@pytest.mark.parametrize("text", ["", "False", "\0", "a\0", "a\0b", "é🙂", "\ud800"])
def test_string_contents_are_preserved_exactly(factory, text):
    normalized = normalize_python_types(factory(text))
    assert type(normalized) is str
    assert normalized == text
    assert json.loads(json.dumps(normalized)) == text


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("boundary", ["min", "max"])
def test_numpy_integer_boundaries_are_exact(dtype, boundary):
    expected = getattr(np.iinfo(dtype), boundary)
    normalized = normalize_python_types(dtype(expected))
    assert type(normalized) is int
    assert normalized == expected
    assert json.loads(json.dumps(normalized)) == expected


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("value", [0.0, -0.0, 0.5, float("inf"), -float("inf"), float("nan")])
def test_numpy_float_values_and_signed_zero_are_preserved(dtype, value):
    original = dtype(value)
    normalized = normalize_python_types(original)
    assert type(normalized) is float
    if np.isnan(original):
        assert np.isnan(normalized)
    else:
        assert normalized == original
        assert np.signbit(normalized) == np.signbit(original)


@pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize("unit", ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"])
@pytest.mark.parametrize("value", [1, "NaT"])
def test_numpy_temporal_scalars_keep_their_units(dtype, unit, value):
    original = dtype(value, unit)
    assert normalize_python_types(original) is original


@pytest.mark.parametrize("value", [
    Decimal("0.12345678901234567890123456789"), Fraction(1, 3),
    date(2026, 1, 1), datetime(2026, 1, 1), timedelta(seconds=1),
    b"bytes", np.bytes_(b"bytes"), np.void(b"bytes"),
])
def test_rich_scalars_are_left_for_their_encoders(value):
    assert normalize_python_types(value) is value


def test_extended_precision_is_not_narrowed():
    value = np.longdouble(1) + np.finfo(np.longdouble).eps
    normalized = normalize_python_types(value)
    assert type(normalized) is type(value)
    assert normalized == value


def test_dataclasses_are_expanded_only_for_serialization():
    @dataclass
    class Settings:
        enabled: object
        label: object

    settings = Settings(np.bool_(True), np.str_("label\0"))
    assert normalize_python_types(settings) is settings
    normalized = normalize_python_types({"nested": [(settings,)]}, dataclasses_as_dict=True)
    assert normalized == {"nested": [({"enabled": True, "label": "label\0"},)]}
    assert normalized["nested"][0][0]["enabled"] is True
    assert type(settings.enabled) is np.bool_
    assert settings.label == "label\0"


def test_serialization_does_not_invoke_custom_copy_hooks():
    class CustomValue:
        def __deepcopy__(self, memo):
            raise AssertionError("copying is not serialization")

    original = CustomValue()
    fragment = ProteinFragment(fragment_id="custom", annotations={"x": original})
    assert fragment.to_dict()["annotations"]["x"] is original
    seen = []

    def encoder(value):
        assert value is original
        seen.append(value)
        return "explicit representation"

    assert json.loads(fragment.to_json(default=encoder))["annotations"]["x"] == "explicit representation"
    assert seen == [original]


@pytest.mark.parametrize("value", [None, True, False, 0, 1, 0.5, "False", "0", "", {}, [], ()])
def test_native_types_keep_their_meaning(value):
    normalized = normalize_python_types(value)
    assert normalized == value
    assert type(normalized) is type(value)


def test_nested_containers_and_keys_are_copied_and_normalized():
    original = {np.str_("settings"): [{np.int64(2): (np.bool_(False), np.float32(0.5))}]}
    normalized = normalize_python_types(original)
    assert normalized == {"settings": [{2: (False, 0.5)}]}
    assert type(next(iter(normalized))) is str
    assert type(next(iter(normalized["settings"][0]))) is int
    flag, fraction = normalized["settings"][0][2]
    assert flag is False
    assert type(fraction) is float
    assert type(original["settings"][0][2][0]) is np.bool_
    normalized["settings"].append("new")
    assert len(original["settings"]) == 1


@pytest.mark.parametrize("value", [object(), np.complex64(1j), np.array([True])])
def test_unsupported_objects_are_not_stringified(value, tmp_path):
    assert normalize_python_types(value) is value
    fragment = ProteinFragment(fragment_id="unsupported", sequence="M", annotations={"x": value})
    with pytest.raises(TypeError, match="not JSON serializable"):
        fragment.to_json()
    with pytest.raises(TypeError, match="not JSON serializable"):
        write_fragments([fragment], tmp_path / "unsupported.tsv")


def test_custom_json_encoder_is_still_supported():
    fragment = ProteinFragment(fragment_id="custom", sequence="M", annotations={"x": 1j})
    encoded = fragment.to_json(default=lambda value: [value.real, value.imag])
    assert json.loads(encoded)["annotations"]["x"] == [0.0, 1.0]
