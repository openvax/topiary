"""
Commandline arguments for expression data and legacy RNA expression filtering.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def add_expression_args(arg_parser):
    """Add --gene-expression, --transcript-expression, --variant-expression."""
    expr_group = arg_parser.add_argument_group(
        title="Expression Data",
        description=(
            "Load expression quantification to annotate predictions. "
            "Values become columns accessible in --filter-by and --sort-by. "
            "Format: [name:]file[:id_col[:val_col]]. "
            "The name is a column prefix, not a column name: a value "
            "column becomes {name}_{column} in lower case (unless it "
            "already starts with the name), so a single --gene-expression "
            "with the default name 'gene' and a TPM column gives 'gene_tpm'. "
            "Repeating an option numbers the default names (gene_1, gene_2). "
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
            "or --gene-expression gene:quant.sf:Name:TPM "
            "(either one adds the column 'gene_tpm')"
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
    result: dict[str, list[tuple[str, str, Any]]] = {
        "gene": [], "transcript": [], "variant": [],
    }

    for level, attr, default_name in [
        ("gene", "gene_expression", "gene"),
        ("transcript", "transcript_expression", "transcript"),
        ("variant", "variant_expression", "variant"),
    ]:
        specs = getattr(args, attr, []) or []
        if specs:
            from ..rna.expression_loader import load_expression_from_spec
        for i, spec in enumerate(specs):
            suffix = f"_{i + 1}" if len(specs) > 1 else ""
            name, id_col, df = load_expression_from_spec(
                spec, default_name=f"{default_name}{suffix}",
            )
            result[level].append((name, id_col, df))

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
             "Use --transcript-expression with --filter-by 'transcript_tpm >= N'.",
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
             "Use --gene-expression with --filter-by 'gene_tpm >= N'.",
        default=0.0,
        type=float,
    )

    return rna_group


def rna_gene_expression_dict_from_args(args):
    if args.rna_gene_fpkm_tracking_file:
        from ..rna import load_cufflinks_fpkm_dict

        logging.warning(
            "--rna-gene-fpkm-tracking-file is deprecated; use --gene-expression instead.")
        return load_cufflinks_fpkm_dict(args.rna_gene_fpkm_tracking_file)
    if getattr(args, "rna_min_gene_expression", 0.0):
        logging.warning(
            "--rna-min-gene-expression is deprecated and only applies to "
            "--rna-gene-fpkm-tracking-file (it has no effect with --gene-expression); "
            "use --filter-by 'gene_tpm >= N' or 'gene_fpkm >= N' instead."
        )
    return None


def rna_transcript_expression_dict_from_args(args):
    if args.rna_transcript_fpkm_tracking_file:
        from ..rna import load_cufflinks_fpkm_dict

        logging.warning(
            "--rna-transcript-fpkm-tracking-file is deprecated; use --transcript-expression instead.")
        return load_cufflinks_fpkm_dict(args.rna_transcript_fpkm_tracking_file)
    elif args.rna_transcript_fpkm_gtf_file:
        from ..rna import load_transcript_fpkm_dict_from_gtf

        logging.warning(
            "--rna-transcript-fpkm-gtf-file is deprecated; use --transcript-expression instead.")
        return load_transcript_fpkm_dict_from_gtf(args.rna_transcript_fpkm_gtf_file)
    if getattr(args, "rna_min_transcript_expression", 0.0):
        logging.warning(
            "--rna-min-transcript-expression is deprecated and only applies to "
            "--rna-transcript-fpkm-tracking-file / --rna-transcript-fpkm-gtf-file "
            "(it has no effect with --transcript-expression); "
            "use --filter-by 'transcript_tpm >= N' or 'transcript_fpkm >= N' instead."
        )
    return None
