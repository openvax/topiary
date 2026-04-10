"""
Commandline arguments for expression data and legacy RNA expression filtering.
"""

import logging
import warnings

from ..rna import load_cufflinks_fpkm_dict, load_transcript_fpkm_dict_from_gtf
from ..rna.expression_loader import load_expression_from_spec

logger = logging.getLogger(__name__)


def add_expression_args(arg_parser):
    """Add --gene-expression, --transcript-expression, --variant-expression."""
    expr_group = arg_parser.add_argument_group(
        title="Expression Data",
        description=(
            "Load expression quantification to annotate predictions. "
            "Values become columns accessible in --ranking and --rank-by. "
            "Format: [name:]file[:id_col[:val_col]]. "
            "Common formats (Salmon .sf, Kallisto abundance.tsv, RSEM, "
            "StringTie .gtf, Cufflinks .fpkm_tracking) are auto-detected."
        ),
    )
    expr_group.add_argument(
        "--gene-expression",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Gene-level expression file, joined on gene_id. "
            "Example: --gene-expression salmon_quant.sf "
            "or --gene-expression gene_tpm:quant.sf:Name:TPM"
        ),
    )
    expr_group.add_argument(
        "--transcript-expression",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Transcript-level expression file, joined on transcript_id. "
            "Example: --transcript-expression kallisto/abundance.tsv"
        ),
    )
    expr_group.add_argument(
        "--variant-expression",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Per-variant data file (e.g. isovar output), joined on variant. "
            "Example: --variant-expression isovar_output.tsv "
            "Loads all numeric columns by default."
        ),
    )
    return expr_group


def expression_data_from_args(args):
    """Parse expression CLI args into loaded DataFrames.

    Returns
    -------
    dict with keys 'gene', 'transcript', 'variant', each mapping to a
    list of (name_prefix, id_col, DataFrame) tuples. Empty lists if no
    args provided.
    """
    result = {"gene": [], "transcript": [], "variant": []}

    for spec in getattr(args, "gene_expression", []) or []:
        name, id_col, df = load_expression_from_spec(spec, default_name="gene")
        result["gene"].append((name, id_col, df))

    for spec in getattr(args, "transcript_expression", []) or []:
        name, id_col, df = load_expression_from_spec(spec, default_name="transcript")
        result["transcript"].append((name, id_col, df))

    for spec in getattr(args, "variant_expression", []) or []:
        name, id_col, df = load_expression_from_spec(spec, default_name="variant")
        result["variant"].append((name, id_col, df))

    return result


# --- Legacy RNA flags (deprecated) ---


def add_rna_args(arg_parser):
    rna_group = arg_parser.add_argument_group(
        title="RNA-Seq (deprecated)",
        description=(
            "Legacy expression filtering. Use --gene-expression / "
            "--transcript-expression instead."
        ),
    )

    rna_group.add_argument(
        "--rna-transcript-fpkm-tracking-file",
        help="(Deprecated) Cufflinks tracking file for transcript FPKM. "
             "Use --transcript-expression instead.",
    )

    rna_group.add_argument(
        "--rna-transcript-fpkm-gtf-file",
        help="(Deprecated) GTF file with transcript FPKM. "
             "Use --transcript-expression instead.",
    )

    rna_group.add_argument(
        "--rna-min-transcript-expression",
        help="(Deprecated) Minimum transcript FPKM. "
             "Use --ranking 'transcript_tpm >= N' instead.",
        default=0.0,
        type=float,
    )

    rna_group.add_argument(
        "--rna-gene-fpkm-tracking-file",
        help="(Deprecated) Cufflinks tracking file for gene FPKM. "
             "Use --gene-expression instead.",
        required=False,
    )

    rna_group.add_argument(
        "--rna-min-gene-expression",
        help="(Deprecated) Minimum gene FPKM. "
             "Use --ranking 'gene_tpm >= N' instead.",
        default=0.0,
        type=float,
    )

    return rna_group


def rna_gene_expression_dict_from_args(args):
    if args.rna_gene_fpkm_tracking_file:
        warnings.warn(
            "--rna-gene-fpkm-tracking-file is deprecated. "
            "Use --gene-expression instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return load_cufflinks_fpkm_dict(args.rna_gene_fpkm_tracking_file)
    return None


def rna_transcript_expression_dict_from_args(args):
    if args.rna_transcript_fpkm_tracking_file:
        warnings.warn(
            "--rna-transcript-fpkm-tracking-file is deprecated. "
            "Use --transcript-expression instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return load_cufflinks_fpkm_dict(args.rna_transcript_fpkm_tracking_file)
    elif args.rna_transcript_fpkm_gtf_file:
        warnings.warn(
            "--rna-transcript-fpkm-gtf-file is deprecated. "
            "Use --transcript-expression instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return load_transcript_fpkm_dict_from_gtf(args.rna_transcript_fpkm_gtf_file)
    return None
