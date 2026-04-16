# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Common commandline arguments used by scripts
"""

from argparse import ArgumentParser

import pandas as pd
from mhctools.cli import add_mhc_args, predictors_from_args
from varcode.cli import add_variant_args, variant_collection_from_args

from .cached_predictor import (
    add_cached_predictor_args,
    cached_predictor_from_args,
    cached_predictor_in_use,
)
from .filtering import add_filter_args
from .rna import (
    add_expression_args,
    add_rna_args,
    expression_data_from_args,
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)
from .sequence import add_sequence_args
from .errors import add_error_args
from .outputs import add_output_args
from .protein_changes import add_protein_change_args
from ..inputs import (
    exclude_by,
    read_fasta,
    read_peptide_csv,
    read_peptide_fasta,
    read_sequence_csv,
    slice_regions,
)
from ..sources import (
    cta_sequences,
    ensembl_proteome,
    non_cta_sequences,
    sequences_from_gene_names,
    sequences_from_gene_ids,
    sequences_from_transcript_ids,
    tissue_expressed_sequences,
)
from ..predictor import TopiaryPredictor
from ..ranking import (
    Affinity,
    BoolOp,
    KindAccessor,
    Presentation,
    _kind_matches,
    _resolve_qualified_kind,
    parse,
)


def _add_input_args(arg_parser):
    input_group = arg_parser.add_argument_group(
        title="Direct Inputs",
        description="Peptide/sequence inputs (bypass variant pipeline)",
    )
    input_group.add_argument(
        "--peptide-csv",
        default=None,
        help="CSV with 'peptide' column (optional: 'name'). "
             "Each peptide is scored as-is (no sliding window).",
    )
    input_group.add_argument(
        "--sequence-csv",
        default=None,
        help="CSV with 'sequence' column (optional: 'name'). "
             "Full-length proteins scanned with a sliding window.",
    )
    input_group.add_argument(
        "--fasta",
        default=None,
        help="FASTA of full-length protein sequences scanned with a "
             "sliding window. (For pre-defined peptide lists, use "
             "--peptide-fasta instead.)",
    )
    input_group.add_argument(
        "--peptide-fasta",
        default=None,
        help="FASTA where each entry is a single peptide scored as-is "
             "(no sliding window). (For full-length proteins, use "
             "--fasta instead.)",
    )
    input_group.add_argument(
        "--regions",
        default=None,
        nargs="+",
        help="Restrict prediction to regions of named protein sequences. "
             "Only applies to sequence inputs (--fasta, --sequence-csv, "
             "--gene-names, etc.), not peptide inputs. Sequences not "
             "listed are scanned in full. "
             "Format: NAME:START-END (0-indexed, half-open: START is "
             "included, END is excluded). "
             "E.g. --regions spike:319-541 nucleocapsid:0-50",
    )
    input_group.add_argument(
        "--exclude-fasta",
        default=None,
        nargs="*",
        help="FASTA file(s) of reference sequences to exclude against. "
             "Predicted peptides matching the reference are removed.",
    )
    input_group.add_argument(
        "--exclude-mode",
        choices=["substring", "exact"],
        default="substring",
        help="How to match against the exclusion set. "
             "'substring' (default): exclude if any reference k-mer "
             "appears inside the predicted peptide (e.g. an 8-mer from "
             "heart tissue inside a 9-mer CTA peptide → excluded). "
             "'exact': exclude only if the entire predicted peptide "
             "matches a reference k-mer at the same length.",
    )
    input_group.add_argument(
        "--exclude-ensembl",
        default=False,
        action="store_true",
        help="Exclude peptides found in the Ensembl proteome "
             "(default: human GRCh38).",
    )
    input_group.add_argument(
        "--exclude-non-cta",
        default=False,
        action="store_true",
        help="Exclude peptides from non-CTA (cancer-testis antigen) proteins. "
             "Human only. Requires pirlygenes.",
    )
    input_group.add_argument(
        "--exclude-tissues",
        default=None,
        nargs="*",
        help="Exclude peptides from genes expressed in these tissues "
             "(e.g. heart_muscle lung brain). Requires pirlygenes.",
    )

    # -- built-in sequence sources --
    input_group.add_argument(
        "--ensembl-proteome",
        default=False,
        action="store_true",
        help="Predict across all Ensembl protein sequences.",
    )
    input_group.add_argument(
        "--gene-names",
        default=None,
        nargs="*",
        help="Gene names to predict (e.g. BRAF TP53 EGFR). "
             "Uses longest protein-coding transcript.",
    )
    input_group.add_argument(
        "--gene-ids",
        default=None,
        nargs="*",
        help="Ensembl gene IDs to predict.",
    )
    input_group.add_argument(
        "--transcript-ids",
        default=None,
        nargs="*",
        help="Ensembl transcript IDs to predict.",
    )
    input_group.add_argument(
        "--cta",
        default=False,
        action="store_true",
        help="Predict across CTA (cancer-testis antigen) proteins. "
             "Human only. Requires pirlygenes.",
    )
    input_group.add_argument(
        "--ensembl-release",
        default=None,
        type=int,
        help="Ensembl release number (default: latest installed).",
    )

    return input_group


def create_arg_parser(
    rna=True,
    expression=True,
    mhc=True,
    variants=True,
    protein_changes=True,
    filters=True,
    sequence_options=True,
    error_options=True,
    output=True,
    direct_inputs=True,
):
    arg_parser = ArgumentParser()
    if expression:
        add_expression_args(arg_parser)
    if rna:
        add_rna_args(arg_parser)
    if mhc:
        add_mhc_args(arg_parser)
        add_cached_predictor_args(arg_parser)
        # mhctools' add_mhc_args registers --mhc-predictor / --mhc-alleles
        # as required.  Relax them so users can run exclusively from a
        # --mhc-cache-file / --mhc-cache-directory; predict_epitopes_from_args
        # validates that at least one MHC source is supplied.
        #
        # Accessing _actions here is intentional — it's argparse's de-facto
        # public API for mutating registered arguments post-registration.
        # If mhctools ever changes its add_mhc_args action layout this loop
        # will still work (it only looks at option_strings), and worst-case
        # would fall back to mhctools' original required=True behavior.
        for action in arg_parser._actions:
            if action.option_strings and any(
                opt in ("--mhc-predictor", "--mhc-alleles")
                for opt in action.option_strings
            ):
                action.required = False
    if variants:
        add_variant_args(arg_parser)
    if protein_changes:
        add_protein_change_args(arg_parser)
    if filters:
        add_filter_args(arg_parser)
    if sequence_options:
        add_sequence_args(arg_parser)
    if error_options:
        add_error_args(arg_parser)
    if output:
        add_output_args(arg_parser)
    if direct_inputs:
        _add_input_args(arg_parser)
    return arg_parser


# keeping global instance for backwards compatibility with existing code
arg_parser = create_arg_parser()


def _build_filter_and_sort(args):
    """Build a (filter_by, sort_by, sort_direction) triple from CLI args.

    Returns ``(None, [], direction)`` when nothing is configured.
    """
    import operator as _op

    ranking_text = getattr(args, "filter_by", None)
    has_presentation = getattr(args, "presentation_cutoff", None) is not None
    has_sort_by = getattr(args, "sort_by", None) is not None
    sort_direction = getattr(args, "sort_direction", "auto")
    filter_logic = getattr(args, "filter_logic", "any")
    combine_op = _op.and_ if filter_logic == "all" else _op.or_

    filter_clauses = []
    if ranking_text:
        filter_clauses.append(parse(ranking_text))
    else:
        if args.ic50_cutoff is not None:
            filter_clauses.append(Affinity.value <= args.ic50_cutoff)
        if args.percentile_cutoff is not None:
            filter_clauses.append(Affinity.rank <= args.percentile_cutoff)
        if has_presentation:
            filter_clauses.append(Presentation.rank <= args.presentation_cutoff)

    if not filter_clauses:
        filter_by = None
    elif len(filter_clauses) == 1:
        filter_by = filter_clauses[0]
    else:
        filter_by = BoolOp(combine_op, filter_clauses)
        # Flatten same-op chains for a cleaner string form
        filter_by = _flatten_boolop(filter_by)

    sort_by = []
    if has_sort_by:
        for item in _split_top_level_commas(args.sort_by.strip()):
            sort_by.append(_parse_sort_expr(item))

    return filter_by, sort_by, sort_direction


def _flatten_boolop(node):
    """Flatten BoolOp(op, [BoolOp(op, [...]), ...]) into a single BoolOp."""
    if not isinstance(node, BoolOp):
        return node
    flat = []
    for c in node.children:
        if isinstance(c, BoolOp) and c.op is node.op:
            flat.extend(c.children)
        else:
            flat.append(c)
    return BoolOp(node.op, flat)


def _split_top_level_commas(text):
    """Split a comma-separated sort string on top-level commas only."""
    if not text:
        return []

    parts = []
    current = []
    paren_depth = 0
    bracket_depth = 0

    for ch in text:
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == "," and paren_depth == 0 and bracket_depth == 0:
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
            continue
        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_sort_expr(text):
    """Parse one CLI --sort-by item.

    Bare kind names use the default sort field for that kind:
    affinity -> raw value (IC50), others -> normalized score.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty sort key in --sort-by")

    if _looks_like_plain_kind_name(text):
        kind, method = _resolve_qualified_kind(text)
        accessor = KindAccessor(kind, method=method)
        if _kind_matches(kind, Affinity.kind):
            return accessor.value
        return accessor.score

    return parse(text)


def _looks_like_plain_kind_name(text):
    """Return True when *text* is a bare kind token like ba or mhcflurry_el."""
    if any(ch.isspace() for ch in text):
        return False
    if any(ch in text for ch in ".()[]+-*/"):
        return False
    try:
        _resolve_qualified_kind(text)
        return True
    except ValueError:
        return False


def _parse_regions(region_strings):
    """Parse --regions args like 'spike:319-541' into {name: [(start, end)]}."""
    if not region_strings:
        return None
    regions = {}
    for s in region_strings:
        if ":" not in s:
            raise ValueError(
                "Region must be NAME:START-END, got %r" % s
            )
        name, interval = s.rsplit(":", 1)
        if "-" not in interval:
            raise ValueError(
                "Region interval must be START-END, got %r" % interval
            )
        start_str, end_str = interval.split("-", 1)
        if not start_str or not end_str:
            raise ValueError(
                "Region coordinates cannot be empty, got %r" % s
            )
        try:
            start, end = int(start_str), int(end_str)
        except ValueError:
            raise ValueError(
                "Region coordinates must be integers, got %r" % s
            )
        if start >= end:
            raise ValueError(
                "Region start must be less than end, got %r" % s
            )
        regions.setdefault(name, []).append((start, end))
    return regions


def _get_direct_input(args):
    """Check for direct peptide/sequence inputs. Returns (dict, is_peptides) or (None, None)."""
    release = getattr(args, "ensembl_release", None)

    # File-based sources
    peptide_csv = getattr(args, "peptide_csv", None)
    sequence_csv = getattr(args, "sequence_csv", None)
    fasta = getattr(args, "fasta", None)
    peptide_fasta = getattr(args, "peptide_fasta", None)

    # Built-in sources
    use_ensembl = getattr(args, "ensembl_proteome", False)
    gene_names = getattr(args, "gene_names", None)
    gene_id_list = getattr(args, "gene_ids", None)
    transcript_id_list = getattr(args, "transcript_ids", None)
    use_cta = getattr(args, "cta", False)

    file_sources = [s for s in [peptide_csv, sequence_csv, fasta, peptide_fasta] if s]
    builtin_sources = [s for s in [use_ensembl, gene_names, gene_id_list,
                                   transcript_id_list, use_cta] if s]

    if len(file_sources) + len(builtin_sources) > 1:
        raise ValueError(
            "Only one sequence source may be specified (file or built-in)"
        )

    is_peptides = False
    sequences = None

    if peptide_csv:
        sequences = read_peptide_csv(peptide_csv)
        is_peptides = True
    elif peptide_fasta:
        sequences = read_peptide_fasta(peptide_fasta)
        is_peptides = True
    elif sequence_csv:
        sequences = read_sequence_csv(sequence_csv)
    elif fasta:
        sequences = read_fasta(fasta)
    elif use_ensembl:
        sequences = ensembl_proteome(release=release)
    elif gene_names:
        sequences = sequences_from_gene_names(gene_names, release=release)
    elif gene_id_list:
        sequences = sequences_from_gene_ids(gene_id_list, release=release)
    elif transcript_id_list:
        sequences = sequences_from_transcript_ids(transcript_id_list, release=release)
    elif use_cta:
        sequences = cta_sequences(release=release)

    if sequences is None:
        return None, None

    # Apply region slicing (only meaningful for protein sequences, not peptides)
    region_strings = getattr(args, "regions", None)
    if region_strings and is_peptides:
        raise ValueError(
            "--regions cannot be used with peptide inputs "
            "(--peptide-csv, --peptide-fasta)"
        )
    if region_strings:
        regions = _parse_regions(region_strings)
        sequences = slice_regions(sequences, regions)

    return sequences, is_peptides


def _validate_input_modes(args):
    """Reject incompatible combinations of direct and variant-style inputs."""
    direct_input_flags = [
        getattr(args, "peptide_csv", None),
        getattr(args, "sequence_csv", None),
        getattr(args, "fasta", None),
        getattr(args, "peptide_fasta", None),
        getattr(args, "ensembl_proteome", False),
        getattr(args, "gene_names", None),
        getattr(args, "gene_ids", None),
        getattr(args, "transcript_ids", None),
        getattr(args, "cta", False),
    ]
    if not any(direct_input_flags):
        return

    incompatible_flags = []
    if getattr(args, "vcf", None):
        incompatible_flags.append("--vcf")
    if getattr(args, "maf", None):
        incompatible_flags.append("--maf")
    if getattr(args, "variant", None):
        incompatible_flags.append("--variant")
    if getattr(args, "json_variants", None):
        incompatible_flags.append("--json-variants")
    if getattr(args, "protein_change", None):
        incompatible_flags.append("--protein-change")
    if getattr(args, "rna_gene_fpkm_tracking_file", None):
        incompatible_flags.append("--rna-gene-fpkm-tracking-file")
    if getattr(args, "rna_transcript_fpkm_tracking_file", None):
        incompatible_flags.append("--rna-transcript-fpkm-tracking-file")
    if getattr(args, "rna_transcript_fpkm_gtf_file", None):
        incompatible_flags.append("--rna-transcript-fpkm-gtf-file")
    if getattr(args, "gene_expression", None):
        incompatible_flags.append("--gene-expression")
    if getattr(args, "transcript_expression", None):
        incompatible_flags.append("--transcript-expression")
    if getattr(args, "variant_expression", None):
        incompatible_flags.append("--variant-expression")

    if incompatible_flags:
        raise ValueError(
            "Direct sequence inputs can't be combined with variant/RNA inputs: %s"
            % ", ".join(incompatible_flags)
        )


def predict_epitopes_from_args(args):
    """
    Returns an epitope collection from the given commandline arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed commandline arguments for Topiary
    """
    if cached_predictor_in_use(args):
        models = cached_predictor_from_args(args)
    else:
        if not getattr(args, "mhc_predictor", None):
            raise ValueError(
                "Must supply either --mhc-predictor (live predictor) or "
                "--mhc-cache-file / --mhc-cache-directory (cached "
                "predictions).  Both are missing."
            )
        models = predictors_from_args(args)

    filter_by, sort_by, sort_direction = _build_filter_and_sort(args)

    predictor = TopiaryPredictor(
        models=models,
        padding_around_mutation=args.padding_around_mutation,
        filter_by=filter_by,
        sort_by=sort_by,
        sort_direction=sort_direction,
        min_transcript_expression=args.rna_min_transcript_expression,
        min_gene_expression=args.rna_min_gene_expression,
        only_novel_epitopes=args.only_novel_epitopes,
        raise_on_error=not args.skip_variant_errors,
    )

    _validate_input_modes(args)

    # Check for direct peptide/sequence inputs first
    direct_input, is_peptides = _get_direct_input(args)
    if direct_input is not None:
        if is_peptides:
            df = predictor.predict_from_named_peptides(direct_input)
        else:
            df = predictor.predict_from_named_sequences(direct_input)
        return _apply_exclusion(df, args)

    # Check that at least some variant input is present
    has_variant_input = any([
        getattr(args, "vcf", None),
        getattr(args, "maf", None),
        getattr(args, "variant", None),
        getattr(args, "json_variants", None),
        getattr(args, "protein_change", None),
    ])
    if not has_variant_input:
        raise ValueError(
            "No input specified. Use one of: "
            "--peptide-csv, --sequence-csv, --fasta, --peptide-fasta, "
            "--gene-names, --gene-ids, --transcript-ids, --cta, "
            "--ensembl-proteome, --vcf, --maf, --variant, --json-variants"
        )

    # Use variant pipeline
    variants = variant_collection_from_args(args)
    gene_expression_dict = rna_gene_expression_dict_from_args(args)
    transcript_expression_dict = rna_transcript_expression_dict_from_args(args)
    expr_data = expression_data_from_args(args)

    df = predictor.predict_from_variants(
        variants=variants,
        transcript_expression_dict=transcript_expression_dict,
        gene_expression_dict=gene_expression_dict,
        expression_data=expr_data,
    )
    return _apply_exclusion(df, args)


def _apply_exclusion(df, args):
    """Apply all exclusion sources to the predictions."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    release = getattr(args, "ensembl_release", None)
    mode = getattr(args, "exclude_mode", "substring")

    # Collect all reference sequences into one dict
    ref_sequences = {}

    exclude_paths = getattr(args, "exclude_fasta", None)
    if exclude_paths:
        for path in exclude_paths:
            ref_sequences.update(read_fasta(path))

    if getattr(args, "exclude_ensembl", False):
        ref_sequences.update(ensembl_proteome(release=release))

    if getattr(args, "exclude_non_cta", False):
        ref_sequences.update(non_cta_sequences(release=release))

    exclude_tissues = getattr(args, "exclude_tissues", None)
    if exclude_tissues:
        ref_sequences.update(
            tissue_expressed_sequences(exclude_tissues, release=release)
        )

    if ref_sequences:
        df = exclude_by(df, ref_sequences, mode=mode)
    return df
