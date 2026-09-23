"""Explicitly reported partial prediction batches, with strict defaults."""

import warnings

import pandas as pd

class PartialPredictionWarning(UserWarning):
    """One or more model/input pairs were skipped and sent to a report handler."""


def predict_with_cache_miss_report(predict_batch, named_inputs, *, on_miss=None, context=None):
    """Predict a batch, optionally isolating and reporting cache coverage gaps.

    Parameters
    ----------
    predict_batch : callable
        Accepts a mapping of names to complete protein/peptide sequences and
        returns a DataFrame. Must support independent subsets of its inputs.
        Failed batches may be retried; full sequences preserve flank context.
    named_inputs : mapping
        Input identity to sequence. An empty mapping is passed through once.
    on_miss : callable, optional
        Explicit opt-in to partial results. Called with a dict for each input
        whose singleton batch raises ``CachedPredictorCoverageError``. Retain
        these records alongside the returned table. If absent, no retry or
        skipping occurs. Handler failures propagate instead of losing reports.
    context : dict, optional
        Report provenance, such as model key, method/version and prediction
        stage. Each failure additionally records ``source_sequence_name``,
        ``error_type``, ``message`` and ``scope='model_input'``.

    Returns
    -------
    pandas.DataFrame
        Successful batch rows in input partition order. No fabricated rows or
        scores. A missed input loses all its rows for this model, including any
        covered windows; other inputs and models may still succeed. All-missing
        input returns an empty frame. A partial result emits
        ``PartialPredictionWarning``. The handler, not DataFrame attrs, is the
        authoritative report across subsequent joins, filtering and exports.

    Notes
    -----
    Only cache coverage errors are isolated. Bare ``KeyError``, validation,
    setup, fallback and other prediction errors still raise. Successful batches
    stay batched; only failing partitions are bisected, avoiding one model call
    per protein when there are few misses. This does not relax cache matching
    by allele, kind, genotype, version or flanks.
    """
    # CachedPredictor uses predictor.py's normalization, so resolve the error
    # type at call time rather than creating an import cycle during setup.
    from .cached import CachedPredictorCoverageError

    if on_miss is not None and not callable(on_miss):
        raise TypeError("on_miss must be callable or None")
    if on_miss is None or not named_inputs:
        return predict_batch(named_inputs)
    frames, failures = [], []

    def predict(items):
        try:
            frame = predict_batch(dict(items))
        except CachedPredictorCoverageError as error:
            if len(items) > 1:
                middle = len(items) // 2
                predict(items[:middle])
                predict(items[middle:])
            else:
                failure = dict(context or {}, source_sequence_name=items[0][0],
                               error_type=type(error).__name__, message=str(error), scope="model_input")
                on_miss(failure)
                failures.append(failure)
        else:
            frames.append(frame)

    predict(list(named_inputs.items()))
    if failures:
        warnings.warn(
            f"Partial predictions: skipped {len(failures)} model/input pair(s) with cache coverage gaps; "
            "see the cache miss report. Each skipped input loses all rows for that model.",
            PartialPredictionWarning, stacklevel=2)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
