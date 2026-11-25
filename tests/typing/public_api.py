"""Static consumer checks across expanded package-wide public APIs (#532).

Expected failures use precise ignores so ``warn_unused_ignores`` detects any
regression that turns a validated boundary back into ``Any``.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import assert_type

import pandas as pd

from topiary import (
    ProteinFragment,
    TopiaryResult,
    attach_dna_evidence,
    attach_rna_evidence,
    attach_sequence_source,
    available_evidence_columns,
    cleavage_interval_coverage,
    derive_mhc_class,
    describe_read_evidence,
    detect_lens_version,
    from_cleavage_results,
    iter_fragments,
    make_fragment_id,
    read_fragments,
    read_lens,
    require_sample_name,
    select_cleavage_sites,
    source_column,
    source_columns,
    write_fragments,
)


def expanded_public_api_workflow(path: Path, df: pd.DataFrame) -> None:
    # Protein fragments and TSV IO
    frag_id = make_fragment_id("variant:snv", "SIINFEKL", variant="chr1:100:A>G")
    assert_type(frag_id, str)
    frag = ProteinFragment(
        fragment_id=frag_id,
        source_type="variant:snv",
        sequence="SIINFEKL",
        sample_name="tumor_1",
    )
    assert_type(frag, ProteinFragment)
    assert_type(require_sample_name(frag), str)
    write_fragments([frag], path)
    assert_type(read_fragments(path), list[ProteinFragment])
    assert_type(iter_fragments(path), Iterator[ProteinFragment])

    # LENS reports
    assert_type(detect_lens_version(["peptide", "allele"]), str | None)
    assert_type(read_lens(path), TopiaryResult)
    assert_type(
        read_lens(path, binding_metrics={("netmhcpan", "aff_nm"): ("affinity", "value")}),
        TopiaryResult,
    )

    # Evidence helpers
    assert_type(attach_rna_evidence(df), pd.DataFrame)
    assert_type(attach_dna_evidence(df), pd.DataFrame)
    assert_type(attach_sequence_source(df, "rna_assembly"), pd.DataFrame)
    assert_type(source_column("lens", "priority_score"), str)
    assert_type(source_columns(df), tuple)
    assert_type(source_columns(df, source="lens"), tuple)
    assert_type(available_evidence_columns(df), tuple)
    assert_type(describe_read_evidence(df), dict)

    # Allele class derivation & cleavage tables
    assert_type(derive_mhc_class(pd.Series(["HLA-A*02:01"])), pd.Series)
    cleavage_df = from_cleavage_results([])
    assert_type(cleavage_df, pd.DataFrame)
    assert_type(select_cleavage_sites(cleavage_df, 0, 9), pd.DataFrame)
    assert_type(cleavage_interval_coverage(cleavage_df, 0, 9), pd.DataFrame)

    # Negative checks: validated boundaries must reject invalid argument types.
    ProteinFragment(fragment_id=123)  # type: ignore[arg-type]
    source_columns(df, source=123)  # type: ignore[arg-type]
    source_column(123, "priority_score")  # type: ignore[arg-type]
    attach_rna_evidence("not a DataFrame")  # type: ignore[arg-type]
    read_lens(path, tag=42)  # type: ignore[arg-type]
    write_fragments([frag], path, sep=9)  # type: ignore[arg-type]
    select_cleavage_sites(cleavage_df, "0", 9)  # type: ignore[arg-type]
