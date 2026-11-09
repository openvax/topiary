"""Shared input policy for protein scans, independent of antigen category."""

import warnings

import pandas as pd


class ProteinInputWarning(UserWarning):
    """Protein scan inputs containing X were excluded by the input policy."""


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
            'Use protein_x_policy="windows" / --protein-x-policy windows to score X-free windows.',
            ProteinInputWarning,
            stacklevel=2,
        )
    return selected


def protein_scan_flank_lengths(model):
    """Read a producer's declared protein context, without guessing from rows.

    Parameters
    ----------
    model : predictor
        A live producer or CachedPredictor. Peptide-only producers may declare
        ``uses_flanking_sequences=False``; others declare
        ``protein_flank_lengths()``. Caches combine stored and fallback limits.

    Returns
    -------
    tuple of int or None
        Maximum N/C context lengths, or None when context is unknown. Unknown
        is not zero; observed short flanks never establish a scanner limit.
    """
    from .cached import CachedPredictor, prediction_flank_limits

    if isinstance(model, CachedPredictor):
        return model.protein_window_flank_limits()
    resolver = getattr(model, "protein_flank_lengths", None)
    limits = (
        resolver()
        if callable(resolver)
        else (
            (0, 0) if getattr(model, "uses_flanking_sequences", None) is False else None
        )
    )
    if limits is None:
        return None
    if len(limits) != 2:
        raise ValueError("protein_flank_lengths must return two nonnegative integers")
    parsed = prediction_flank_limits(
        dict(zip(("prediction_n_flank_limit", "prediction_c_flank_limit"), limits))
    )
    return None if None in parsed else parsed


def x_free_protein_windows(named_sequences, peptide_lengths, *, flank_lengths=(0, 0)):
    """Select original windows whose peptide and required context contain no X.

    Parameters
    ----------
    named_sequences : mapping of str to str
        Original proteins. Names, sequences and offsets are never rewritten.
    peptide_lengths : iterable of int
        Positive requested lengths. Short proteins contribute no windows.
    flank_lengths : pair of int, default (0, 0)
        Declared maximum N/C inference context. Context is shortened only at a
        true input boundary, never at an internal X. Matching is case insensitive.

    Returns
    -------
    list of tuple
        ``(source_name, offset, length)`` triples in source/length/offset order.
        Empty inputs or all-excluded windows return an empty list. No residues
        are substituted or concatenated; other amino acids remain unchanged.
    """
    from .cached import prediction_flank_limits
    from .protein_predictions import _protein_windows

    if len(flank_lengths) != 2:
        raise ValueError("flank_lengths must contain two nonnegative integers")
    n_limit, c_limit = prediction_flank_limits(
        dict(
            zip(("prediction_n_flank_limit", "prediction_c_flank_limit"), flank_lengths)
        )
    )
    if n_limit is None or c_limit is None:
        raise ValueError("X-free selection requires both flank limits")
    return [
        (name, offset, length)
        for name, offset, length in _protein_windows(named_sequences, peptide_lengths)
        if "X"
        not in named_sequences[name][
            max(0, offset - n_limit) : offset + length + c_limit
        ].upper()
    ]


def predict_protein_windows(model, named_sequences, windows):
    """Score selected original occurrences without introducing new termini.

    Parameters
    ----------
    model : predictor
        A CachedPredictor, native mhctools predictor, or a DataFrame producer
        supported by ``predict_peptide_occurrences``. Live models must declare
        their protein context and kinds. Native inputs are deduplicated and
        predicted in one aligned batch, then expanded to original occurrences.
    named_sequences : mapping of str to str
        Original complete scan inputs.
    windows : iterable of tuple
        ``(source_name, offset, length)`` triples at configured peptide lengths.
        Empty selections return an empty prediction table without inference.

    Returns
    -------
    pandas.DataFrame
        Validated predictions retaining original coordinates and required
        context. Missing/malformed live predictions and cache misses remain
        errors. Cache-only scans retain their native sparse-kind semantics.
    """
    from mhctools.base_predictor import BasePredictor
    from .cached import CachedPredictor
    from .peptide_occurrences import predict_peptide_occurrences
    from .predictor import from_predictions
    from .protein_predictions import _protein_windows, validate_protein_predictions

    lengths = model.default_peptide_lengths
    windows = _protein_windows(named_sequences, lengths, windows)
    if not windows:
        return from_predictions([])
    if isinstance(model, CachedPredictor):
        return model.predict_proteins_dataframe(named_sequences, windows=windows)
    limits = protein_scan_flank_lengths(model)
    if limits is None:
        raise ValueError("Selected protein windows require declared model flank limits")
    n_limit, c_limit = limits
    contexts = {}
    for name, offset, length in windows:
        sequence = named_sequences[name]
        key = (
            sequence[offset : offset + length],
            sequence[max(0, offset - n_limit) : offset],
            sequence[offset + length : offset + length + c_limit],
        )
        contexts.setdefault(key, []).append((name, offset))
    if isinstance(model, BasePredictor):
        inputs = list(contexts)
        if model.uses_flanking_sequences:
            results = model.predict_with_flanks(
                [p for p, _, _ in inputs],
                n_flanks=[n for _, n, _ in inputs],
                c_flanks=[c for _, _, c in inputs],
            )
        else:
            results = model.predict([p for p, _, _ in inputs])
        if len(results) != len(inputs):
            raise ValueError(
                "Selected protein prediction returned an incorrect result count"
            )
        frames = []
        for key, result in zip(inputs, results):
            frame = result.to_dataframe()
            if frame.empty or not frame.peptide.eq(key[0]).all():
                raise ValueError(
                    "Selected protein prediction returned an incorrect peptide"
                )
            for name, offset in contexts[key]:
                frames.append(
                    frame.assign(
                        source_sequence_name=name,
                        offset=offset,
                        n_flank=key[1],
                        c_flank=key[2],
                    )
                )
        raw = pd.concat(frames, ignore_index=True)
    else:
        records = []
        for (peptide, n_flank, c_flank), locations in contexts.items():
            for name, offset in locations:
                records.append(
                    dict(
                        prediction_id=str(len(records)),
                        source_sequence_name=name,
                        peptide_offset=offset,
                        peptide=peptide,
                        n_flank=n_flank,
                        c_flank=c_flank,
                    )
                )
        raw = predict_peptide_occurrences(
            pd.DataFrame(records),
            model,
            use_flanks=getattr(model, "uses_flanking_sequences", True),
        ).drop(columns="prediction_id")
    return validate_protein_predictions(
        raw, named_sequences, model, peptide_lengths=lengths, windows=windows
    )


def predict_x_free_proteins(model, named_sequences):
    """Scan original X-free windows, reusing identical inputs across all proteins.

    Parameters
    ----------
    model : predictor
        Protein producer or cache with selected peptide lengths. Unknown
        scanner context excludes X-containing inputs with ProteinInputWarning;
        clean inputs still use the native protein path.
    named_sequences : mapping of str to str
        Inputs kept under the explicit ``windows`` policy.

    Returns
    -------
    pandas.DataFrame
        Selected original occurrences, batched together when X is present and
        context is known. Clean-only inputs retain the native scan path.
        No backend call is made for an empty selection. Unrelated model errors
        and missing coverage for retained windows remain strict.
    """
    from .predictor import from_predictions

    if not named_sequences:
        return from_predictions([])
    unknown = {name: seq for name, seq in named_sequences.items() if "X" in seq.upper()}
    if not unknown:
        return model.predict_proteins_dataframe(named_sequences)
    limits = protein_scan_flank_lengths(model)
    if limits is None:
        warnings.warn(
            f"Skipped {len(unknown)} protein scan input(s) containing X because "
            f"{type(model).__name__} has unknown protein flank limits; "
            f"example source names: {list(unknown)[:3]!r}.",
            ProteinInputWarning,
            stacklevel=2,
        )
        clean = {
            name: seq for name, seq in named_sequences.items() if name not in unknown
        }
        return (
            model.predict_proteins_dataframe(clean) if clean else from_predictions([])
        )
    windows = x_free_protein_windows(
        named_sequences, model.default_peptide_lengths, flank_lengths=limits
    )
    return predict_protein_windows(model, named_sequences, windows)
