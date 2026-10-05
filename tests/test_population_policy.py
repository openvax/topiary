"""Population operations compose across Python, parsed DSL and saved policies."""

import numpy as np
import pandas as pd
import pytest

from topiary import (
    Affinity, Column, EvalContext, ordinal_rank, apply_sort,
)
from .test_candidate_tables import source
from .test_twin_conformance import POPULATION_TWINS


@pytest.mark.parametrize('run', POPULATION_TWINS)
def test_occurrence_population_not_prediction_rows(run):
    frame = source(n_rna_alt=[4., 16.]).df
    repeated = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    ctx = EvalContext(repeated)
    expression = Column('n_rna_alt').log2() / Column('n_rna_alt').log2().population_max()
    assert run(expression, ctx).tolist() == [.5, 1.]
    assert run(Affinity.value.dense_rank(), ctx).tolist() == [1., 2.]


@pytest.mark.parametrize('run', POPULATION_TWINS)
@pytest.mark.parametrize('bottom,expected', [(False, [1., np.nan]), (True, [1., 2.])])
def test_explicit_missing_ranks(run, bottom, expected):
    ctx = EvalContext(source(values=(50., np.nan)).df)
    np.testing.assert_equal(run(Affinity.value.dense_rank(False, bottom), ctx).to_numpy(), expected)
    assert run(Column('optional').fillna(1), ctx).tolist() == [1., 1.]


@pytest.mark.parametrize('run', POPULATION_TWINS)
def test_lexicographic_ties_and_text(run):
    frame = source(values=(50., 50.), gene=['Z', 'A']).df
    expression = ordinal_rank(Affinity.value, Column('gene'))
    assert run(expression, EvalContext(frame)).tolist() == [2., 1.]
    assert run(expression, EvalContext(frame.iloc[::-1])).tolist() == [1., 2.]
    frame['gene'] = ['A', 'A']
    assert run(expression, EvalContext(frame)).tolist() == [1., 1.]


@pytest.mark.parametrize('run', POPULATION_TWINS)
@pytest.mark.parametrize('expression', [Column('n_rna_alt').population_max(),
                                       Column('n_rna_alt').dense_rank(),
                                       ordinal_rank(Column('n_rna_alt'))])
def test_multiple_named_samples_require_separate_partitions(run, expression):
    frame = source(sample_name=['one', 'two']).df
    with pytest.raises(ValueError, match='one sample'):
        run(expression, EvalContext(frame))


@pytest.mark.parametrize('run', POPULATION_TWINS)
def test_fill_preserves_conflict_checks(run):
    frame = source().df
    frame = pd.concat([frame, frame.iloc[:1].assign(n_rna_alt=999)], ignore_index=True)
    with pytest.raises(ValueError, match='Conflicting'):
        run(Column('n_rna_alt').fillna(1), EvalContext(frame))
    with pytest.raises(ValueError, match='Conflicting'):
        run(ordinal_rank(Column('n_rna_alt')), EvalContext(frame))




def test_population_max_all_missing_and_empty():
    ctx = EvalContext(source(n_rna_alt=[None, None]).df)
    assert Column('n_rna_alt').population_max().eval(ctx).isna().all()
    assert Column('n_rna_alt').dense_rank().eval(ctx).isna().all()
    ctx = EvalContext(source().df.iloc[:0])
    for node in (Column('n_rna_alt').population_max(), Column('n_rna_alt').dense_rank(),
                 ordinal_rank(Column('gene'))):
        assert node.eval(ctx).empty


@pytest.mark.parametrize('run', POPULATION_TWINS)
def test_optional_column_still_validates_other_references(run):
    frame = source().df
    node = Column('optional').fillna(1) + Column('n_rna_alt')
    assert run(node, EvalContext(frame)).tolist() == [6., 16.]
    ranked = apply_sort(frame, repr(node), sort_direction='desc')
    assert ranked.peptide.iloc[0] == 'GILGFVFTL'
    with pytest.raises(ValueError, match='required'):
        apply_sort(frame, Column('optional').fillna(1) + Column('required'))
