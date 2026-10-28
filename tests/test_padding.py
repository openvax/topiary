import pytest

from topiary import check_padding_around_mutation


def test_default_padding():
    # expect padding to be one less than the largest epitope length
    assert check_padding_around_mutation(None, [8, 9, 10]) == 9


def test_zero_padding_uses_default():
    assert check_padding_around_mutation(0, [8, 9, 10]) == 9


def test_explicit_padding():
    assert check_padding_around_mutation(12, [8, 9, 10]) == 12


def test_boolean_padding_keeps_python_integer_compatibility():
    assert check_padding_around_mutation(True, []) is True


def test_empty_epitope_lengths_require_no_padding():
    assert check_padding_around_mutation(None, []) == 0


def test_invalid_padding():
    # padding is insufficient for the epitope lengths given
    with pytest.raises(ValueError, match="cannot be less than 8"):
        check_padding_around_mutation(2, [9])


@pytest.mark.parametrize("padding", [0.0, "", 1.5, "3", object()])
def test_non_integer_padding_is_rejected(padding):
    with pytest.raises(TypeError, match="must be an integer or None"):
        check_padding_around_mutation(padding, [9])
