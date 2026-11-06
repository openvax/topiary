"""Shared mhctools #631 corpus and composed cache/occurrence regression."""

import json
from pathlib import Path

import pandas as pd
import pytest
from mhctools import Prediction

from topiary import CachedPredictor, from_predictions
from topiary.alleles import AlleleError, canonical_allele_name

CORPUS = json.loads((Path(__file__).parent / 'data' /
                     'canonical_allele_identities.json').read_text())


@pytest.mark.parametrize('raw,canonical', CORPUS['accepted'])
def test_shared_canonical_identities(raw, canonical):
    # Identity preservation and presentation eligibility are separate policies.
    if raw == 'HLA-A*24:09N':
        with pytest.raises(AlleleError, match='null allele'):
            canonical_allele_name(raw)
    else:
        assert canonical_allele_name(raw) == canonical
        assert canonical_allele_name(canonical) == canonical


@pytest.mark.parametrize('raw', CORPUS['rejected'])
def test_shared_rejected_identities(raw):
    with pytest.raises(AlleleError):
        canonical_allele_name(raw)


def test_beta_pair_and_resolution_survive_cache_serialization_and_occurrences(tmp_path):
    identities = [
        'HLA-DRB1*01:01', 'HLA-DRA*01:01/DRB1*01:01',
        'HLA-DQB1*06:02', 'HLA-DQA1*01:02/DQB1*06:02',
        'HLA-DPA1*01:03/DPB1*04:01', 'H2-AA*b/AB*b',
        'BoLA-T2b', 'DLA-88*001:01', 'HLA-A*02:01:01:01',
    ]
    peptide = 'SIINFEKLL'
    predictions = [Prediction(
        kind='pMHC_affinity', score=i / 20, value=100 + i,
        peptide=peptide, allele=allele, predictor_name='identity-fixture',
        predictor_version='1') for i, allele in enumerate(identities)]
    frame = from_predictions(predictions)
    cache = CachedPredictor(frame)
    path = tmp_path / 'predictions.tsv'
    cache.save(path)
    restored = CachedPredictor.from_topiary_output(path)
    output = restored.predict_dataframe([peptide])
    assert set(output.allele) == set(identities)
    assert len(output) == len(identities)
    expected = {a: 100 + i for i, a in enumerate(identities)}
    assert dict(zip(output.allele, output.value)) == expected
    # Two proteins containing the same peptide must retain both occurrences for
    # every identity, including beta-only and explicit-pair keys together.
    scanned = restored.predict_proteins_dataframe({'first': peptide, 'second': peptide})
    selected = scanned[scanned.peptide == peptide]
    assert set(selected.allele) == set(identities)
    counts = selected.groupby('allele').size().to_dict()
    assert counts == {a: 2 for a in identities}
    pd.testing.assert_series_equal(
        selected.groupby('allele').value.first().sort_index(),
        pd.Series(expected, name='value').rename_axis('allele').sort_index(),
        check_dtype=False)
