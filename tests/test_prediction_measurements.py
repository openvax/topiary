"""The public reduction shares the DSL's conflict and missing-value policy."""

import numpy as np
import pandas as pd
import pytest

from topiary import prediction_field_values


@pytest.mark.parametrize("values,expected", [
    ([0.1 + 0.2, 0.3], 0.3), ([50., None], 50.),
    ([None, None], np.nan), ([np.inf, np.inf], np.inf),
    ([-np.inf, -np.inf], -np.inf),
])
def test_consistent_grouped_measurements_have_one_order_independent_value(values, expected):
    frame = pd.DataFrame(dict(observation=["run"] * 2, value=values), index=[3, 3])
    for ordered in (frame, frame.iloc[::-1]):
        answer = prediction_field_values(ordered, "value", group_keys=["observation"])
        assert answer.index.tolist() == ["run"]
        assert np.isnan(answer.iloc[0]) if np.isnan(expected) else answer.iloc[0] == expected


@pytest.mark.parametrize("values", [[50., 60.], [np.inf, 50.], [-np.inf, np.inf]])
def test_distinct_measurements_including_infinities_cannot_be_reduced(values):
    frame = pd.DataFrame(dict(observation=["run"] * 2, value=values))
    for ordered in (frame, frame.iloc[::-1]):
        with pytest.raises(ValueError, match="Conflicting prediction measurements"):
            prediction_field_values(ordered, "value", group_keys=["observation"])


def test_empty_measurements_keep_the_requested_group_index():
    frame = pd.DataFrame(columns=["observation", "value"])
    answer = prediction_field_values(frame, "value", group_keys=["observation"])
    assert answer.empty and answer.index.name == "observation"
