"""Shared input policy for protein scans, independent of antigen category."""

import warnings

import pandas as pd

from .prediction_batch import predict_with_cache_miss_report


class ProteinInputWarning(UserWarning):
    """Protein scan inputs containing X were excluded by policy or a producer."""


def filter_protein_sequences(named_sequences, *, allow_x=False):
    """Select complete scan inputs under Topiary's unknown-residue policy.

    Parameters
    ----------
    named_sequences : mapping
        Source names to protein sequences. Empty mappings return an empty dict.
    allow_x : bool, default False
        Pass through proteins containing ``X`` (including lowercase ``x``).
        Otherwise exclude those entire proteins, including their clean windows,
        and emit one :class:`ProteinInputWarning` with a count and example names.

    Returns
    -------
    dict
        Selected inputs in their original order. Names, sequences and the input
        mapping are not modified. Other residues are left to the producer.
        This selects inputs, not predictions: it neither probes model support
        nor relaxes cache coverage for the inputs that remain.

    Notes
    -----
    Excluding whole inputs preserves coordinates and flank context; trimming
    around X or substituting residues would describe a different model input.
    The policy applies to the supplied scan sequences, which may already be
    extracted regions or variant fragments rather than full-length proteins.
    """
    selected, excluded = {}, []
    for name, sequence in named_sequences.items():
        if not allow_x and "X" in sequence.upper():
            excluded.append(name)
        else:
            selected[name] = sequence
    if excluded:
        warnings.warn(
            f"Skipped {len(excluded)} protein scan input(s) containing X, including all their windows; "
            f"example source names: {excluded[:3]!r}. "
            "Use allow_x_in_proteins=True / --allow-x-in-proteins to opt in.",
            ProteinInputWarning, stacklevel=2,
        )
    return selected


def predict_proteins_with_x_recovery(predict_batch, named_sequences, *, on_cache_miss=None, context=None):
    """Preserve supported protein inputs and isolate recognized X rejections.

    Parameters
    ----------
    predict_batch : callable
        Map source names to complete sequences and return a DataFrame. Must
        support independent subsets; failed batches may be retried. Original
        proteins are never trimmed, substituted or split into smaller windows.
    named_sequences : mapping
        Inputs already selected under the caller's policy. Empty input is passed
        through once. This function does not exclude X from successful batches.
    on_cache_miss : callable, optional
        Explicit partial-cache reporting handler, as in
        :func:`predict_with_cache_miss_report`. Cache misses remain strict
        without a handler.
    context : dict, optional
        Model/stage provenance for cache reports and the rejection warning.

    Returns
    -------
    pandas.DataFrame
        Successful rows in partition order. If a singleton containing X raises
        a recognized residue error, all its rows are omitted for this producer.
        One :class:`ProteinInputWarning` reports the count and example names.
        All-rejected input returns an empty frame. Other models can still score
        the same protein. Successful batches are not retried or changed.

    Notes
    -----
    Legacy mhctools and MHCflurry expose bare ValueError for residue rejection,
    so recovery recognizes only their specific validation messages. Unknown
    validation errors, configuration/runtime errors, cache misses and invalid
    result tables are not reclassified merely because a protein contains X.
    """
    rejected = {}

    def predict(inputs):
        try:
            return predict_batch(inputs)
        except ValueError as error:
            # mhctools/base_predictor.py:_check_peptide_inputs and
            # mhcflurry/class1_affinity_predictor.py:predict_to_dataframe /
            # predict_multiple use these messages, not a residue-specific type.
            # Verified with mhctools 4.9.3 / MHCflurry 2.3.16. Typed contract:
            # https://github.com/openvax/mhctools/issues/659
            # Do not match "unsupported" generally or catch ValueError subclasses
            # (e.g. allele errors) that already carry a different contract.
            message = str(error)
            count, separator, remainder = message.partition(" peptides have nonstandard amino acids: ")
            residue_error = (
                message.startswith("Invalid character 'X' in peptide '") and message.endswith("'")
                or bool(separator) and count.isdecimal() and int(count) > 0 and bool(remainder)
            )
            if type(error) is not ValueError or not residue_error or not any(
                "X" in sequence.upper() for sequence in inputs.values()
            ):
                raise
            items = list(inputs.items())
            if len(items) == 1:
                rejected[items[0][0]] = None
                return pd.DataFrame()
            middle = len(items) // 2
            frames = [predict(dict(items[:middle])), predict(dict(items[middle:]))]
            populated = [frame for frame in frames if not frame.empty]
            return pd.concat(populated, ignore_index=True) if populated else pd.DataFrame()

    result = predict_with_cache_miss_report(
        predict, named_sequences, on_miss=on_cache_miss, context=context)
    if rejected:
        provenance = context or {}
        warnings.warn(
            f"Skipped {len(rejected)} protein scan input(s) containing X after producer residue rejection "
            f"({provenance.get('prediction_method_name', 'model')}, {provenance.get('stage', 'protein')}); "
            f"example source names: {list(rejected)[:3]!r}. Each loses all rows for this producer.",
            ProteinInputWarning, stacklevel=2,
        )
    return result
