"""
Peptide amino acid properties for ranking and analysis.

Computes properties directly from the peptide sequence column using
vectorized pandas string operations. All properties are pure functions
of the amino acid sequence — no external dependencies.

Usage::

    from topiary.properties import add_peptide_properties

    df = predictor.predict_from_named_sequences(seqs)
    df = add_peptide_properties(df)                             # all properties
    df = add_peptide_properties(df, groups=["core"])            # named group
    df = add_peptide_properties(df, include=["charge", "cysteine_count"])

Then use in ranking via :class:`~topiary.ranking.Column`::

    from topiary.ranking import Column, Affinity
    score = 0.5 * Affinity.score - 0.2 * Column("cysteine_count")
"""



# ---------------------------------------------------------------------------
# Amino acid property tables
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity scale
_KD_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Net charge at pH 7.4 (simple model: K,R = +1; D,E = -1; H ≈ +0.1)
_CHARGE = {
    "K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0,
}

# Average amino acid molecular weights (Da)
_MASS = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

# Instability index dipeptide weights (Guruprasad et al. 1990)
# Subset of the 400 dipeptide DIWV values; missing pairs default to 0.
_DIWV = {
    "WW": 1.0, "WC": 1.0, "WM": 24.68, "CW": -14.03,
    "CH": 33.60, "CK": -7.49, "CC": 1.0, "CF": -6.54,
    "WG": -9.37, "WL": -7.49, "DG": 1.0, "DK": -7.49,
    "DD": 1.0, "DE": 1.0, "DF": -6.54, "DW": 1.0,
    "DY": 1.0, "DA": 2.36, "DN": 2.05, "DR": -6.54,
    "DS": 0.46, "EG": -6.54, "EK": -7.49, "ED": 2.27,
    "EE": 33.60, "EF": -6.54, "EW": -14.03, "EY": -6.54,
    "EA": 11.0, "EN": 1.0, "ER": 2.36, "ES": 0.46,
    "FW": 1.0, "FL": 1.0, "FK": -14.03, "FD": 13.34,
    "FE": -6.54, "FF": 1.0, "FG": 1.0, "FY": 33.60,
    "FA": 1.0, "GG": 13.34, "GK": -7.49, "GD": -7.49,
    "GE": -6.54, "GF": -6.54, "GW": 13.34, "GY": -7.49,
    "GA": -7.49, "GN": -7.49, "GR": 1.0, "GS": 1.0,
    "HG": 1.0, "HK": 24.68, "HD": 1.0, "HE": 1.0,
    "HF": -6.54, "HW": -1.88, "HY": 44.94, "HA": 1.0,
    "HN": 24.68, "HR": 1.0, "HS": 2.36, "IG": 1.0,
    "IK": -7.49, "ID": 1.0, "IE": 44.94, "IF": 1.0,
    "IW": 1.0, "IY": 1.0, "IA": 1.0, "IN": 1.0,
    "IR": 1.0, "IS": 1.0, "KG": -7.49, "KK": 1.0,
    "KD": 1.0, "KE": 1.0, "KF": 1.0, "KW": 1.0,
    "KY": 1.0, "KA": 1.0, "KN": 1.0, "KR": 33.60,
    "KS": 1.0, "LG": 14.45, "LK": -7.49, "LD": 1.0,
    "LE": -6.54, "LF": 1.0, "LW": 24.68, "LY": 1.0,
    "LA": 1.0, "LN": 1.0, "LR": -6.54, "LS": 1.0,
    "MG": 1.0, "MK": 1.0, "MD": 1.0, "ME": 1.0,
    "MF": 1.0, "MW": 1.0, "MY": 24.68, "MA": 13.34,
    "MN": 1.0, "MR": -6.54, "MS": 44.94, "NG": -14.03,
    "NK": 1.0, "ND": 1.0, "NE": 1.0, "NF": -14.03,
    "NW": -9.37, "NY": 1.0, "NA": 1.0, "NN": 1.0,
    "NR": 1.0, "NS": 1.0, "PG": 1.0, "PK": 1.0,
    "PD": -6.54, "PE": 18.38, "PF": 20.26, "PW": -1.88,
    "PY": 1.0, "PA": 20.26, "PN": -6.54, "PR": -6.54,
    "PS": 20.26, "QG": 1.0, "QK": 1.0, "QD": 20.26,
    "QE": 33.60, "QF": -6.54, "QW": 1.0, "QY": -6.54,
    "QA": 1.0, "QN": 1.0, "QR": 1.0, "QS": 44.94,
    "RG": -7.49, "RK": 1.0, "RD": 1.0, "RE": 1.0,
    "RF": 1.0, "RW": 58.28, "RY": -6.54, "RA": 1.0,
    "RN": 13.34, "RR": 58.28, "RS": 44.94, "SG": 1.0,
    "SK": 1.0, "SD": 1.0, "SE": 1.0, "SF": 1.0,
    "SW": 1.0, "SY": 1.0, "SA": 1.0, "SN": 1.0,
    "SR": 20.26, "SS": 1.0, "TG": -7.49, "TK": 1.0,
    "TD": 1.0, "TE": 1.0, "TF": 13.34, "TW": -14.03,
    "TY": 1.0, "TA": 1.0, "TN": -14.03, "TR": 1.0,
    "TS": 1.0, "VG": -7.49, "VK": -7.49, "VD": -14.03,
    "VE": 1.0, "VF": 1.0, "VW": 1.0, "VY": -7.49,
    "VA": 1.0, "VN": 1.0, "VR": 1.0, "VS": 1.0,
    "WK": 1.0, "WD": 1.0, "WE": 1.0, "WF": 1.0,
    "WY": 1.0, "WA": -14.03, "WN": 13.34, "WR": 1.0,
    "WS": 1.0, "YG": -7.49, "YK": 1.0, "YD": 24.68,
    "YE": -6.54, "YF": 13.34, "YW": -9.37, "YY": 13.34,
    "YA": 24.68, "YN": 1.0, "YR": -15.91, "YS": 1.0,
}

# TCR-facing positions for MHC-I peptides (0-indexed)
_TCR_POSITIONS = {
    8: [3, 4, 5, 6],       # 8-mer: p4-p7
    9: [3, 4, 5, 7],       # 9-mer: p4,p5,p6,p8
    10: [3, 4, 5, 6, 8],   # 10-mer: p4-p7,p9
    11: [3, 4, 5, 6, 7, 9],  # 11-mer: p4-p8,p10
}

_DIFFICULT_NTERM = {"Q", "E", "C"}
_DIFFICULT_CTERM = {"P", "C"}


# ---------------------------------------------------------------------------
# Per-peptide computation functions (operate on a single string)
# ---------------------------------------------------------------------------


def _sum_lookup(peptide, table):
    return sum(table.get(c, 0.0) for c in peptide)


def _mean_lookup(peptide, table):
    if not peptide:
        return 0.0
    return _sum_lookup(peptide, table) / len(peptide)


def _max_window_mean(peptide, table, window=7):
    """Max mean of a sliding window across the peptide."""
    if len(peptide) <= window:
        return _mean_lookup(peptide, table)
    values = [table.get(c, 0.0) for c in peptide]
    best = -float("inf")
    window_sum = sum(values[:window])
    best = window_sum / window
    for i in range(1, len(values) - window + 1):
        window_sum += values[i + window - 1] - values[i - 1]
        best = max(best, window_sum / window)
    return best


def _instability_index(peptide):
    """Guruprasad instability index (>40 = unstable)."""
    if len(peptide) < 2:
        return 0.0
    total = sum(_DIWV.get(peptide[i:i+2], 0.0) for i in range(len(peptide) - 1))
    return (10.0 / len(peptide)) * total


def _tcr_residues(peptide):
    """Extract TCR-facing residues for MHC-I peptides."""
    positions = _TCR_POSITIONS.get(len(peptide))
    if positions is None:
        return None
    return "".join(peptide[i] for i in positions if i < len(peptide))


# ---------------------------------------------------------------------------
# Property registry
# ---------------------------------------------------------------------------

# Each property: (column_name, compute_fn(peptides: pd.Series) -> pd.Series)

def _compute_charge(peptides):
    return peptides.apply(lambda p: _sum_lookup(p, _CHARGE))


def _compute_hydrophobicity(peptides):
    return peptides.apply(lambda p: _mean_lookup(p, _KD_HYDRO))


def _compute_aromaticity(peptides):
    return peptides.str.count("[FWY]")


def _compute_molecular_weight(peptides):
    # Sum of residue masses minus (n-1) water molecules lost in peptide bonds
    return peptides.apply(
        lambda p: _sum_lookup(p, _MASS) - 18.015 * (len(p) - 1) if p else 0.0
    )


def _compute_cysteine_count(peptides):
    return peptides.str.count("C")


def _compute_instability_index(peptides):
    return peptides.apply(_instability_index)


def _compute_max_7mer_hydrophobicity(peptides):
    return peptides.apply(lambda p: _max_window_mean(p, _KD_HYDRO, 7))


def _compute_cterm_7mer_hydrophobicity(peptides):
    return peptides.apply(lambda p: _mean_lookup(p[-7:], _KD_HYDRO))


def _compute_difficult_nterm(peptides):
    return peptides.str[0].isin(_DIFFICULT_NTERM)


def _compute_difficult_cterm(peptides):
    return peptides.str[-1].isin(_DIFFICULT_CTERM)


def _compute_asp_pro_bonds(peptides):
    return peptides.str.count("DP")


def _compute_tcr_charge(peptides):
    return peptides.apply(
        lambda p: _sum_lookup(r, _CHARGE) if (r := _tcr_residues(p)) else float("nan")
    )


def _compute_tcr_aromaticity(peptides):
    return peptides.apply(
        lambda p: sum(1 for c in r if c in "FWY") if (r := _tcr_residues(p)) else float("nan")
    )


def _compute_tcr_hydrophobicity(peptides):
    return peptides.apply(
        lambda p: _mean_lookup(r, _KD_HYDRO) if (r := _tcr_residues(p)) else float("nan")
    )


# Property name -> (compute function, groups it belongs to)
_PROPERTIES = {
    "charge":                    (_compute_charge, {"core", "manufacturability", "immunogenicity"}),
    "hydrophobicity":            (_compute_hydrophobicity, {"core", "manufacturability", "immunogenicity"}),
    "aromaticity":               (_compute_aromaticity, {"core", "manufacturability", "immunogenicity"}),
    "molecular_weight":          (_compute_molecular_weight, {"core", "manufacturability", "immunogenicity"}),
    "cysteine_count":            (_compute_cysteine_count, {"manufacturability"}),
    "instability_index":         (_compute_instability_index, {"manufacturability"}),
    "max_7mer_hydrophobicity":   (_compute_max_7mer_hydrophobicity, {"manufacturability"}),
    "cterm_7mer_hydrophobicity": (_compute_cterm_7mer_hydrophobicity, {"manufacturability"}),
    "difficult_nterm":           (_compute_difficult_nterm, {"manufacturability"}),
    "difficult_cterm":           (_compute_difficult_cterm, {"manufacturability"}),
    "asp_pro_bonds":             (_compute_asp_pro_bonds, {"manufacturability"}),
    "tcr_charge":                (_compute_tcr_charge, {"immunogenicity"}),
    "tcr_aromaticity":           (_compute_tcr_aromaticity, {"immunogenicity"}),
    "tcr_hydrophobicity":        (_compute_tcr_hydrophobicity, {"immunogenicity"}),
}


def available_properties():
    """Return dict of property_name -> set of groups it belongs to."""
    return {name: groups for name, (_, groups) in _PROPERTIES.items()}


def add_peptide_properties(
    df,
    groups=None,
    include=None,
    peptide_column="peptide",
    prefix="",
):
    """Add amino acid property columns to a predictions DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column with peptide sequences.

    groups : list of str, optional
        Named property groups: ``"core"``, ``"manufacturability"``,
        ``"immunogenicity"``. If None and include is None, computes all.

    include : list of str, optional
        Specific property names to compute. Overrides groups.

    peptide_column : str
        Column containing peptide sequences (default ``"peptide"``).

    prefix : str
        Prefix for output column names (e.g. ``"wt_"`` for WT peptides).

    Returns
    -------
    pd.DataFrame
        Copy with new property columns added.
    """
    if peptide_column not in df.columns:
        raise ValueError(
            f"Column {peptide_column!r} not found. "
            f"Available: {sorted(df.columns)}"
        )

    if include is not None:
        names = include
        bad = [n for n in names if n not in _PROPERTIES]
        if bad:
            raise ValueError(
                f"Unknown properties: {bad}. "
                f"Available: {sorted(_PROPERTIES.keys())}"
            )
    elif groups is not None:
        group_set = set(groups)
        bad = group_set - {"core", "manufacturability", "immunogenicity"}
        if bad:
            raise ValueError(
                f"Unknown groups: {sorted(bad)}. "
                f"Available: core, manufacturability, immunogenicity"
            )
        names = [
            name for name, (_, prop_groups) in _PROPERTIES.items()
            if prop_groups & group_set
        ]
    else:
        names = list(_PROPERTIES.keys())

    df = df.copy()
    peptides = df[peptide_column]
    for name in names:
        compute_fn, _ = _PROPERTIES[name]
        df[prefix + name] = compute_fn(peptides)

    return df
