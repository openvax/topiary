"""
Commandline arguments for expression data.
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
