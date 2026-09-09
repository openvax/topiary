"""Native scalar normalization without guessing meanings for other objects."""

import json

import numpy as np
import pytest

from topiary import ProteinFragment, normalize_python_types, write_fragments


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
