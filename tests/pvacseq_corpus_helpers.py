"""Independent source-column oracle for the pinned, real pVACseq corpus."""

import json
from pathlib import Path

from .sid_data import sid_data_root

import mhcgnomes
import numpy as np
import pandas as pd


ROOT = sid_data_root("pvacseq/osteosarc")
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
REPORTS = MANIFEST["reports"]


def selected_columns(aggregated):
    """Source-to-public-field expectations, independent of the loader's maps."""
    if aggregated:
        text = {"Gene": "gene", "Best Transcript": "transcript",
                "ID": "variant", "Best Peptide": "peptide", "Allele": "allele",
                "Pos": "mutation_position"}
        numbers = {"RNA Depth": "pvacseq_tumor_rna_depth",
                   "RNA VAF": "pvacseq_tumor_rna_vaf", "DNA VAF": "pvacseq_tumor_dna_vaf",
                   "RNA Expr": "gene_expression", "IC50 MT": "value", "IC50 WT": "wt_value",
                   "%ile MT": "percentile_rank", "%ile WT": "wt_percentile_rank"}
    else:
        text = {"Gene Name": "gene", "Transcript": "transcript",
                "MT Epitope Seq": "peptide", "WT Epitope Seq": "wt_peptide",
                "HLA Allele": "allele", "Mutation Position": "mutation_position"}
        numbers = {"Tumor RNA Depth": "pvacseq_tumor_rna_depth",
                   "Tumor RNA VAF": "pvacseq_tumor_rna_vaf",
                   "Tumor DNA Depth": "pvacseq_tumor_dna_depth",
                   "Tumor DNA VAF": "pvacseq_tumor_dna_vaf", "Gene Expression": "gene_expression",
                   "Median MT IC50 Score": "value", "Median WT IC50 Score": "wt_value",
                   "Median MT Percentile": "percentile_rank",
                   "Median WT Percentile": "wt_percentile_rank"}
    specs = [(src, dst, "pvacseq", "pMHC_affinity", numeric)
             for numeric, mapping in ((False, text), (True, numbers))
             for src, dst in mapping.items()]
    if not aggregated:
        for algorithm, method, kind, metrics in (
            ("MHCflurry", "mhcflurry", "pMHC_affinity", ("IC50 Score", "Percentile")),
            ("NetMHCpan", "netmhcpan", "pMHC_affinity", ("IC50 Score", "Percentile")),
            ("NetMHCIIpan", "netmhciipan", "pMHC_affinity", ("IC50 Score", "Percentile")),
            ("MHCflurryEL Processing", "mhcflurry", "antigen_processing", ("Score", "Percentile")),
            ("MHCflurryEL Presentation", "mhcflurry", "pMHC_presentation", ("Score", "Percentile")),
            ("NetMHCpanEL", "netmhcpan", "pMHC_presentation", ("Score", "Percentile")),
            ("NetMHCIIpanEL", "netmhciipan", "pMHC_presentation", ("Score", "Percentile")),
            ("BigMHC_IM", "bigmhc_im", "immunogenicity", ("Score", "Percentile")),
        ):
            for sequence, scope in (("MT", ""), ("WT", "wt_")):
                for metric in metrics:
                    target = {"IC50 Score": "value", "Score": "score",
                              "Percentile": "percentile_rank"}[metric]
                    specs.append((f"{algorithm} {sequence} {metric}", scope + target,
                                  method, kind, True))
    return specs


def assert_selected_columns(raw, frame, aggregated):
    checked = 0
    for source, target, method, kind, numeric in selected_columns(aggregated):
        if source not in raw:
            continue
        mask = pd.Series(True, index=raw.index)
        if kind != "pMHC_affinity":
            prefix = source.rsplit(" MT ", 1)[0].rsplit(" WT ", 1)[0]
            companions = [c for c in raw if c.startswith(prefix + " ")]
            mask = raw[companions].notna().any(axis=1)
        expected = raw.loc[mask, source].reset_index(drop=True)
        actual = frame.loc[frame.prediction_method_name.eq(method) & frame.kind.eq(kind),
                           target].reset_index(drop=True)
        assert len(actual) == len(expected), (source, method, kind)
        assert actual.isna().equals(expected.isna()), source
        if target == "allele":
            expected = expected.map(lambda x: mhcgnomes.parse(x).to_string())
        valid = expected.notna()
        if numeric:
            np.testing.assert_allclose(actual[valid].astype(float), expected[valid].astype(float),
                                       rtol=1e-12, atol=1e-12, err_msg=source)
        else:
            assert actual[valid].astype(str).tolist() == expected[valid].astype(str).tolist(), source
        checked += 1
    assert checked >= 14
