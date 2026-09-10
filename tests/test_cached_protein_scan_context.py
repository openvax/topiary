"""A cached prediction rebound to a new occurrence must describe that
occurrence — all of it, or none of it.

:meth:`CachedPredictor.predict_proteins_dataframe` answers "where does
this peptide sit in the sequence I just handed you".  The cache rows it
answers from were built somewhere else, and every column saying *where a
peptide was found* therefore describes the wrong place until it is
rewritten.  Rewriting only some of them is the bug these tests pin: the
row named the requested protein while the coordinate still pointed into
the protein the cache came from, and the two coordinates then collided
into two columns of the same name (issue #296).

The flanking residues are the same mistake one column over, with the
opposite fix: they are a prediction *input*, so they cannot be rewritten
to match a new occurrence — the row simply does not apply there.
"""

import pandas as pd
import pytest

from topiary import (
    PROTEIN_SCAN_COLUMNS,
    Affinity,
    CachedPredictor,
    TopiaryPredictor,
    apply_filter,
    evaluate_scores,
)

ALLELE = "HLA-A*02:01"
PEPTIDE = "SIINFEKLA"


def _row(peptide, *, allele=ALLELE, score=0.5, affinity=150.0, **overrides):
    row = {
        "peptide": peptide,
        "allele": allele,
        "peptide_length": len(peptide),
        "kind": "pMHC_affinity",
        "score": score,
        "affinity": affinity,
        "percentile_rank": 2.0,
        "value": affinity,
        "prediction_method_name": "netmhcpan",
        "predictor_version": "4.2",
    }
    row.update(overrides)
    return row


def _covering_cache(sequences, length=9, **row_overrides):
    """A cache holding every ``length``-mer window of *sequences*.

    A sliding-window scan misses on any window the cache lacks, so a test
    about *which* row comes back has to cover them all or it ends up
    testing the miss path instead.
    """
    rows = {}
    for sequence in sequences:
        for offset in range(len(sequence) - length + 1):
            peptide = sequence[offset:offset + length]
            rows[peptide] = _row(peptide, **row_overrides)
    return CachedPredictor.from_dataframe(pd.DataFrame(list(rows.values())))


# ---------------------------------------------------------------------------
# The coordinate: exactly one, and it describes the requested occurrence
# ---------------------------------------------------------------------------


def test_protein_scan_emits_one_coordinate_not_two():
    # The cache row remembers offset 77 in some other protein.  The scan
    # is asked about 'x', where the peptide sits at 0.
    cache = _covering_cache(
        [PEPTIDE], source_sequence_name="original", peptide_offset=77,
    )
    out = cache.predict_proteins_dataframe({"x": PEPTIDE})

    assert "peptide_offset" not in out.columns, (
        "the producer must not emit topiary's spelling alongside "
        "mhctools' 'offset' — the adapter renames one onto the other"
    )
    assert out["offset"].tolist() == [0]
    assert out["source_sequence_name"].tolist() == ["x"]


def test_predictor_result_has_unique_columns_and_the_new_coordinate():
    cache = _covering_cache(
        [PEPTIDE], source_sequence_name="original", peptide_offset=77,
    )
    df = TopiaryPredictor(models=[cache]).predict_from_named_sequences(
        {"x": PEPTIDE}
    )

    assert df.columns.is_unique
    assert df["peptide_offset"].tolist() == [0]
    # A duplicate column made frame[key] a DataFrame, and every consumer
    # reading a dtype off it failed far from the cause.
    assert isinstance(df["peptide_offset"], pd.Series)


def test_stale_cache_offset_does_not_survive_a_save_load_round_trip(tmp_path):
    path = tmp_path / "netmhcpan42.tsv"
    _covering_cache(
        [PEPTIDE], source_sequence_name="original", peptide_offset=77,
    ).save(path)

    cache = CachedPredictor.from_topiary_output(path)
    df = TopiaryPredictor(models=[cache]).predict_from_named_sequences(
        {"x": PEPTIDE}
    )

    assert df.columns.is_unique
    assert df["peptide_offset"].tolist() == [0]


def test_repeated_occurrences_keep_their_own_nonzero_offsets():
    # The peptide occurs twice, at 2 and at 13, and neither is 0.
    protein = "GG" + PEPTIDE + "GG" + PEPTIDE + "GG"
    assert protein.index(PEPTIDE) == 2
    cache = _covering_cache([protein], source_sequence_name="original",
                            peptide_offset=77)

    out = cache.predict_proteins_dataframe({"prot": protein})
    hits = out[out["peptide"] == PEPTIDE]

    assert sorted(hits["offset"].tolist()) == [2, 13]
    assert set(hits["source_sequence_name"]) == {"prot"}


def test_distinct_source_names_are_preserved_across_proteins():
    left = "GG" + PEPTIDE
    right = "GGGG" + PEPTIDE
    cache = _covering_cache([left, right], source_sequence_name="original",
                            peptide_offset=77)

    out = cache.predict_proteins_dataframe({"left": left, "right": right})
    hits = out[out["peptide"] == PEPTIDE]

    assert set(
        zip(hits["source_sequence_name"], hits["offset"])
    ) == {("left", 2), ("right", 4)}


def test_named_peptides_reset_the_stale_offset_to_zero():
    # The caller supplied the peptide whole under the name 'x', so it
    # starts at 0 of 'x' — not partway through itself.
    cache = _covering_cache(
        [PEPTIDE], source_sequence_name="original", peptide_offset=77,
    )
    df = TopiaryPredictor(models=[cache]).predict_from_named_peptides(
        {"x": PEPTIDE}
    )

    assert df.columns.is_unique
    assert df["peptide_offset"].tolist() == [0]
    assert df["source_sequence_name"].tolist() == ["x"]


def test_declared_scan_columns_are_what_a_scan_actually_emits():
    """The published shape and the emitted shape are one answer.

    A constant naming the output is only useful if it is the output; a
    test that reads either one alone cannot see them drift apart.
    """
    cache = _covering_cache([PEPTIDE])
    out = cache.predict_proteins_dataframe({"x": PEPTIDE})
    empty = cache.predict_proteins_dataframe({"tiny": "MA"})

    assert list(out.columns) == list(PROTEIN_SCAN_COLUMNS)
    assert list(empty.columns) == list(PROTEIN_SCAN_COLUMNS)
    assert len(set(PROTEIN_SCAN_COLUMNS)) == len(PROTEIN_SCAN_COLUMNS)


def test_empty_protein_scan_result_has_unique_columns():
    cache = _covering_cache([PEPTIDE])
    # Nothing to scan: every sequence is shorter than the window.
    out = cache.predict_proteins_dataframe({"tiny": "MA"})

    assert out.empty
    assert out.columns.is_unique
    assert "peptide_offset" not in out.columns


# ---------------------------------------------------------------------------
# Downstream: the frame reaches the DSL intact
# ---------------------------------------------------------------------------


def test_scan_result_survives_apply_filter_and_evaluate_scores():
    protein = "GG" + PEPTIDE + "GG"
    cache = _covering_cache([protein], source_sequence_name="original",
                            peptide_offset=77, affinity=42.0)
    df = TopiaryPredictor(models=[cache]).predict_from_named_sequences(
        {"prot": protein}
    )

    kept = apply_filter(df, Affinity.value <= 500)
    assert len(kept) == len(df)
    assert kept["peptide_offset"].tolist() == df["peptide_offset"].tolist()

    scores = evaluate_scores(df, Affinity.value)
    assert isinstance(scores, pd.Series)
    assert set(scores.dropna().tolist()) == {42.0}


def test_predictor_filter_by_runs_on_a_cached_protein_scan():
    protein = "GG" + PEPTIDE + "GG"
    cache = _covering_cache([protein], source_sequence_name="original",
                            peptide_offset=77, affinity=42.0)

    df = TopiaryPredictor(
        models=[cache], filter_by=Affinity.value <= 100,
    ).predict_from_named_sequences({"prot": protein})

    assert not df.empty
    assert df.columns.is_unique
    assert (df["peptide_offset"] >= 0).all()


# ---------------------------------------------------------------------------
# Flanks: a prediction input, so it selects occurrences rather than
# being rewritten to fit one
# ---------------------------------------------------------------------------


def test_flankless_cache_rows_apply_at_any_occurrence():
    # A predictor that does not read flanks produces context-independent
    # scores, so reusing them across proteins is exactly what the cache
    # is for.
    protein = "GG" + PEPTIDE + "GG"
    cache = _covering_cache([protein])

    out = cache.predict_proteins_dataframe({"prot": protein})

    assert set(out["peptide"]) >= {PEPTIDE}
    assert (out["source_sequence_name"] == "prot").all()


def _flanked_cache(*proteins, n_flank, c_flank):
    """Cache covering every window of *proteins*, PEPTIDE's row flanked.

    Every protein a test scans has to be covered, or the scan misses on
    some unrelated window and raises before flank matching is reached —
    which would pass the test for the wrong reason.
    """
    rows = {}
    for protein in proteins:
        for offset in range(len(protein) - 9 + 1):
            peptide = protein[offset:offset + 9]
            rows[peptide] = _row(peptide)
    rows[PEPTIDE] = _row(PEPTIDE, n_flank=n_flank, c_flank=c_flank)
    return CachedPredictor.from_dataframe(pd.DataFrame(list(rows.values())))


def test_flanked_row_applies_where_the_real_neighbours_match():
    protein = "MA" + PEPTIDE + "GG"
    cache = _flanked_cache(protein, n_flank="MA", c_flank="GG")

    out = cache.predict_proteins_dataframe({"prot": protein})
    hit = out[out["peptide"] == PEPTIDE]

    assert hit["offset"].tolist() == [2]
    assert hit["n_flank"].tolist() == ["MA"]
    assert hit["c_flank"].tolist() == ["GG"]


def test_flanked_row_is_not_reused_at_a_differently_flanked_occurrence():
    # Cached in the context 'MA...GG'; asked about an occurrence whose
    # real neighbours are 'WW...CC'.  The cached score is a prediction
    # about a different protein context, so it must not come back
    # wearing this occurrence's coordinates.
    cached_protein = "MA" + PEPTIDE + "GG"
    other = "WW" + PEPTIDE + "CC"
    cache = _flanked_cache(
        cached_protein, other, n_flank="MA", c_flank="GG",
    )

    with pytest.raises(KeyError, match="different flanking context"):
        cache.predict_proteins_dataframe({"other": other})


def test_flank_mismatch_error_names_the_occurrence_and_the_stored_context():
    cached_protein = "MA" + PEPTIDE + "GG"
    other = "WW" + PEPTIDE + "CC"
    cache = _flanked_cache(
        cached_protein, other, n_flank="MA", c_flank="GG",
    )

    with pytest.raises(KeyError) as excinfo:
        cache.predict_proteins_dataframe({"other": other})

    message = str(excinfo.value)
    assert PEPTIDE in message
    assert "'other'" in message
    assert "offset 2" in message
    assert "'MA'" in message and "'GG'" in message


def test_a_shorter_stored_flank_matches_the_tail_of_the_real_context():
    # Predictors store a bounded number of flanking residues, so a
    # one-residue stored flank is the tail of what really precedes the
    # peptide, not a claim that nothing else does.
    protein = "WMA" + PEPTIDE + "GGT"
    cache = _flanked_cache(protein, n_flank="A", c_flank="G")

    out = cache.predict_proteins_dataframe({"prot": protein})

    assert PEPTIDE in set(out["peptide"])


def test_a_stored_flank_longer_than_the_sequence_provides_does_not_apply():
    protein = "A" + PEPTIDE + "G"
    cache = _flanked_cache(protein, n_flank="MMMA", c_flank="G")

    with pytest.raises(KeyError, match="different flanking context"):
        cache.predict_proteins_dataframe({"prot": protein})


# ---------------------------------------------------------------------------
# The adapter refuses a frame that states the coordinate twice
# ---------------------------------------------------------------------------


def test_normalizing_a_two_coordinate_frame_reports_the_contract():
    from topiary.predictor import _normalize_prediction_frame

    df = pd.DataFrame([{
        "peptide": PEPTIDE, "allele": ALLELE, "kind": "pMHC_affinity",
        "value": 50.0, "prediction_method_name": "netmhcpan",
        "offset": 0, "peptide_offset": 77,
    }])

    with pytest.raises(ValueError, match="coordinate twice"):
        _normalize_prediction_frame(df)


# ---------------------------------------------------------------------------
# Coverage is per kind: one kind cannot vouch for another (#302)
# ---------------------------------------------------------------------------


def _mixed_flank_cache(protein, *, flanked_kind, n_flank, c_flank):
    """Cache where one kind reads flanks and the rest do not.

    One cache holds exactly one ``(method, version)`` pair, so this is
    not two different predictors' caches merged — that is refused at
    construction. It is one model whose rows reached the cache by two
    routes, which :meth:`CachedPredictor.concat` joins without complaint
    because the version invariant holds: a protein scan records the
    flanking context it scored in, and a peptide-level run has none to
    record.
    """
    rows = [
        _row(protein[offset:offset + 9])
        for offset in range(len(protein) - 9 + 1)
    ]
    rows.append(_row(
        PEPTIDE, kind=flanked_kind, score=0.9, affinity=None,
        n_flank=n_flank, c_flank=c_flank,
    ))
    return CachedPredictor.from_dataframe(pd.DataFrame(rows))


def test_a_kind_that_applies_does_not_vouch_for_one_that_does_not():
    # The affinity row has no flank context and applies here; the
    # presentation row was predicted somewhere else.  Reporting only the
    # affinity would read as "no presentation prediction", which is not
    # what the cache says.
    protein = "MA" + PEPTIDE + "GG"
    cache = _mixed_flank_cache(
        protein, flanked_kind="pMHC_presentation",
        n_flank="WW", c_flank="CC",
    )

    with pytest.raises(KeyError, match="different flanking context"):
        cache.predict_proteins_dataframe({"prot": protein})


def test_the_uncovered_kind_is_named_in_the_error():
    protein = "MA" + PEPTIDE + "GG"
    cache = _mixed_flank_cache(
        protein, flanked_kind="pMHC_presentation",
        n_flank="WW", c_flank="CC",
    )

    with pytest.raises(KeyError) as excinfo:
        cache.predict_proteins_dataframe({"prot": protein})

    message = str(excinfo.value)
    assert "pMHC_presentation" in message
    assert PEPTIDE in message
    assert "'WW'" in message and "'CC'" in message


def test_a_kind_the_cache_never_held_stays_quiet():
    # A cache that simply has no presentation row for a peptide is not
    # the same as one whose only presentation row does not apply here.
    # Only the second is worth an error.
    protein = "MASIINFEKLAGGQ"
    rows = [
        _row(protein[offset:offset + 9])
        for offset in range(len(protein) - 9 + 1)
    ]
    rows.append(_row(protein[0:9], kind="pMHC_presentation", score=0.9))
    cache = CachedPredictor.from_dataframe(pd.DataFrame(rows))

    out = cache.predict_proteins_dataframe({"prot": protein})

    kinds = out.groupby("peptide")["kind"].apply(lambda s: set(s)).to_dict()
    assert kinds[protein[0:9]] == {"pMHC_affinity", "pMHC_presentation"}
    assert kinds[PEPTIDE] == {"pMHC_affinity"}


def test_a_flanked_kind_that_applies_is_still_returned_alongside_others():
    protein = "MA" + PEPTIDE + "GG"
    cache = _mixed_flank_cache(
        protein, flanked_kind="pMHC_presentation",
        n_flank="MA", c_flank="GG",
    )

    out = cache.predict_proteins_dataframe({"prot": protein})
    hits = out[out["peptide"] == PEPTIDE]

    assert set(hits["kind"]) == {"pMHC_affinity", "pMHC_presentation"}
    assert set(hits["offset"]) == {2}


def test_concat_of_a_scan_cache_and_a_peptide_cache_reaches_the_same_check():
    """The mix arrives through a public door, not only a built frame.

    A scan-built cache and a peptide-built cache from the same model at
    the same version satisfy the version invariant, so ``concat`` joins
    them — and the result holds a flanked and a flankless row for one
    ``(peptide, allele)``. Building the frame by hand would test the
    check without showing that a caller can get here.
    """
    scanned = CachedPredictor.from_dataframe(pd.DataFrame([
        _row(PEPTIDE, kind="pMHC_presentation", score=0.9, affinity=None,
             n_flank="WW", c_flank="CC"),
    ]))
    from_peptides = CachedPredictor.from_dataframe(pd.DataFrame([
        _row(PEPTIDE),
    ]))

    merged = CachedPredictor.concat([scanned, from_peptides])
    rows = merged.predict_peptides_dataframe([PEPTIDE])

    assert set(rows["kind"]) == {"pMHC_affinity", "pMHC_presentation"}
    assert set(rows["n_flank"]) == {"WW", ""}

    # Scanning a protein whose real flanks are MA / GG must not let the
    # applicable affinity row vouch for the presentation row.
    protein = "MA" + PEPTIDE + "GG"
    covering = CachedPredictor.concat([
        merged,
        CachedPredictor.from_dataframe(pd.DataFrame([
            _row(protein[offset:offset + 9])
            for offset in range(len(protein) - 9 + 1)
            if protein[offset:offset + 9] != PEPTIDE
        ])),
    ])

    with pytest.raises(KeyError, match="pMHC_presentation"):
        covering.predict_proteins_dataframe({"prot": protein})
