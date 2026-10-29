"""Static consumer workflow, checked by lint.sh; no prediction runs here.

Expected failures use precise ignores so warn_unused_ignores detects any
regression that turns a validated boundary back into Any.
"""

from pathlib import Path
from typing import assert_type

import pandas as pd

from topiary import (
    CachedPredictor, Metadata, TopiaryResult, combine_predictions,
    read_csv, read_tsv, read_isovar_hypotheses,
    read_isovar_patient_background_pairs, stack_results, to_tsv,
)


def table_workflow(path: Path) -> None:
    result = read_tsv(path)
    assert_type(result, TopiaryResult)
    assert_type(read_csv(str(path)), TopiaryResult)
    assert_type(result.df, pd.DataFrame)
    assert_type(result.long_df, pd.DataFrame)
    assert_type(result.wide_df, pd.DataFrame)
    assert_type(result.metadata, Metadata)
    assert_type(result.metadata.topiary_version, str | None)
    assert_type(result.metadata.sources, list[str])
    assert_type(result.head().to_wide().to_long(), TopiaryResult)
    assert_type(result.filter_by("affinity.value < 500"), TopiaryResult)
    assert_type(result.sort_by(["affinity.value"]), TopiaryResult)
    assert_type(stack_results([result]), TopiaryResult)
    assert_type(combine_predictions([result, result.df]), TopiaryResult)
    assert_type(result.combine_predictions((result, result.df)), TopiaryResult)
    assert_type(read_isovar_hypotheses(path), TopiaryResult)
    assert_type(read_isovar_patient_background_pairs({}), TopiaryResult)

    cache = CachedPredictor.from_topiary_output(path)
    assert_type(cache, CachedPredictor)
    assert_type(cache.alleles, list[str])
    assert_type(cache.default_peptide_lengths, list[int])
    cache.alleles = ["HLA-A*02:01"]
    cache.default_peptide_lengths = None
    frame = cache.predict_contextual_peptides_dataframe(
        ["SIINFEKLA"], n_flanks=[""], c_flanks=["GG"])
    assert_type(frame, pd.DataFrame)
    assert_type(cache.predict_peptides_dataframe(["SIINFEKLA"]), pd.DataFrame)
    assert_type(cache.predict_proteins_dataframe({"p": "SIINFEKLAGG"}), pd.DataFrame)
    assert_type(cache.to_dataframe(), pd.DataFrame)
    to_tsv(TopiaryResult(frame, extra={"nested": {"counts": [1, None]}}), path)
    cache.save(path)

    read_tsv(123)  # type: ignore[arg-type]
    TopiaryResult(["not a DataFrame"])  # type: ignore[arg-type]
    result.df = []  # type: ignore[assignment]
    result.head("five")  # type: ignore[arg-type]
    Metadata(sources="not a list")  # type: ignore[arg-type]
    cache.predict_proteins_dataframe(["not a mapping"])  # type: ignore[arg-type]
    cache.predict_contextual_peptides_dataframe(
        ["SIINFEKLA"], n_flanks=[None], c_flanks=[""])  # type: ignore[list-item]
    cache.save(123)  # type: ignore[arg-type]
