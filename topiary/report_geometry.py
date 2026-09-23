"""Reported mutation coordinates and their scope when rescanning peptides."""

import math
import re

from .ranking import is_stated
from .serialization import normalize_python_types


def mutation_intervals_from_positions(positions, peptide_length):
    """Convert reported one-based residue positions to half-open intervals.

    Parameters
    ----------
    positions : str, number or None
        One position, comma-separated positions, or inclusive ranges such as
        ``"2,7-9"``. This is the coordinate convention of pVACseq's
        ``Mutation Position`` / ``Pos`` fields. Missing or unsupported values
        remain unknown; a missing position does not establish absence of a
        mutation. Booleans and fractional positions are unsupported.
    peptide_length : int
        Length of the reported peptide. Reported positions outside this window
        are omitted, as in historical pVACseq flanking-only peptide reports.

    Returns
    -------
    list of tuple of int or None
        Sorted, merged zero-based half-open intervals, ``[]`` when all stated
        positions are outside the peptide, or ``None`` when unknown. Disjoint
        positions remain disjoint. A historical single position remains a
        single residue, not an inferred indel span or frameshift tail.

    Raises
    ------
    ValueError
        ``peptide_length`` is not a non-negative integer.

    Notes
    -----
    These intervals retain the report's definition of a changed position;
    they are not evidence of somatic origin or reference-proteome absence.
    """
    peptide_length = normalize_python_types(peptide_length)
    if isinstance(peptide_length, bool) or not isinstance(peptide_length, int) or peptide_length < 0:
        raise ValueError("peptide_length must be a non-negative integer")
    if not is_stated(positions):
        return None
    positions = normalize_python_types(positions)
    if isinstance(positions, bool):
        return None
    # pandas commonly reads a column containing missing positions as floats.
    if not isinstance(positions, str):
        try:
            numeric = float(positions)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric) or not numeric.is_integer():
            return None
        positions = str(int(numeric))
    intervals = []
    for part in positions.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if match is None:
            return None
        start = int(match[1])
        end = int(match[2] or start)
        if end < start:
            return None
        low, high = max(0, start - 1), min(peptide_length, end)
        if low < high:
            intervals.append((low, high))
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
        else:
            merged.append((low, high))
    return merged


def map_peptide_intervals(sequence, peptide, intervals):
    """Map reported peptide-relative targets into a containing sequence.

    Parameters
    ----------
    sequence : str
        Selected fragment sequence, either the peptide or a longer context.
    peptide : str or None
        Peptide whose coordinates were reported. It must occur exactly once
        in ``sequence``; a reported protein offset is not a context offset.
    intervals : iterable of (int, int) or None
        Zero-based half-open target intervals in the peptide. A zero-width
        interval represents a deletion/junction. ``None`` means unknown;
        ``[]`` states only that the reported peptide has no target.

    Returns
    -------
    list of tuple of int or None
        Intervals shifted into the unique peptide occurrence, or ``None`` for
        missing/ambiguous placement. An empty interval list remains ``[]``
        only when the selected sequence is the reported peptide: absence of
        targets in one peptide cannot establish absence in a larger context.

    Raises
    ------
    ValueError
        A stated interval has fractional/boolean bounds, is reversed, or lies
        outside the reported peptide. Invalid coordinates are never rounded.
    """
    if intervals is None or not is_stated(peptide):
        return None
    if not isinstance(peptide, str) or not isinstance(sequence, str):
        raise ValueError("sequence and peptide must be strings")
    validated = []
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("Mutation intervals must contain start/end pairs")
        bounds = []
        for bound in interval:
            bound = normalize_python_types(bound)
            if isinstance(bound, bool):
                raise ValueError("Mutation interval bounds must be whole numbers, not booleans")
            try:
                numeric = float(bound)
            except (TypeError, ValueError) as error:
                raise ValueError("Mutation interval bounds must be whole numbers") from error
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError("Mutation interval bounds must be whole numbers")
            bounds.append(int(numeric))
        low, high = bounds
        if not 0 <= low <= high <= len(peptide):
            raise ValueError(f"Mutation interval {(low, high)} is outside peptide {peptide!r}")
        validated.append((low, high))
    if not validated:
        return [] if sequence == peptide else None
    offset = sequence.find(peptide)
    if offset < 0 or sequence.find(peptide, offset + 1) >= 0:
        return None
    return [(low + offset, high + offset) for low, high in validated]
