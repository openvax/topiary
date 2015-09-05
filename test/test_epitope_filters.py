from nose.tools import eq_
from topiary.filters import apply_epitope_filters

class MockEpitopePrediction(object):
    def __init__(self, value, percentile_rank, novel_epitope):
        self.value = value
        self.percentile_rank = percentile_rank
        self.novel_epitope = novel_epitope

epitope_prediction_strong_mutant = MockEpitopePrediction(
    value=50.0, percentile_rank=0.4, novel_epitope=True)

epitope_prediction_strong_wildtype = MockEpitopePrediction(
    value=50.0, percentile_rank=0.4, novel_epitope=False)

epitope_prediction_weak_mutant = MockEpitopePrediction(
    value=20000.0, percentile_rank=30.0, novel_epitope=True)


epitope_prediction_weak_wildtype = MockEpitopePrediction(
    value=20000, percentile_rank=30.0, novel_epitope=False)

epitope_predictions = [
    epitope_prediction_strong_mutant,
    epitope_prediction_strong_wildtype,
    epitope_prediction_weak_mutant,
    epitope_prediction_weak_wildtype
]

def test_apply_epitope_filters_nothing():
    filtered = apply_epitope_filters(
        epitope_predictions,
        ic50_cutoff=None,
        percentile_cutoff=None,
        only_novel_epitopes=False)
    eq_(len(filtered), len(epitope_predictions))

def test_apply_epitope_filters_ic50_some():
    filtered = apply_epitope_filters(
        epitope_predictions,
        ic50_cutoff=500.0,
        percentile_cutoff=None,
        only_novel_epitopes=False)
    eq_(len(filtered), len(epitope_predictions) / 2)


def test_apply_epitope_filters_ic50_all():
    filtered = apply_epitope_filters(
        epitope_predictions,
        ic50_cutoff=1.0,
        percentile_cutoff=None,
        only_novel_epitopes=False)
    eq_(len(filtered), 0)

def test_apply_epitope_filters_percentile_some():
    filtered = apply_epitope_filters(
        epitope_predictions,
        ic50_cutoff=None,
        percentile_cutoff=2.0,
        only_novel_epitopes=False)
    eq_(len(filtered), len(epitope_predictions) / 2)

def test_apply_epitope_filters_percentile_all():
    filtered = apply_epitope_filters(
        epitope_predictions,
        ic50_cutoff=None,
        percentile_cutoff=0.001,
        only_novel_epitopes=False)
    eq_(len(filtered), 0)

def test_apply_epitope_filters_mutant():
    filtered = apply_epitope_filters(
        epitope_predictions,
        ic50_cutoff=None,
        percentile_cutoff=None,
        only_novel_epitopes=True)
    eq_(len(filtered), len(epitope_predictions) / 2)
