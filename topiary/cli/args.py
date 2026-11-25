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

from importlib import import_module
import logging
from pathlib import Path
import re
from typing import Any

from mhctools.cli import add_mhc_args, predictors_from_args
from varcode.cli import add_variant_args, variant_collection_from_args

from .. import __version__
from .cached_predictor import (
    add_cached_predictor_args,
    cached_predictor_from_args,
    cached_predictor_in_use,
)
from .filtering import add_filter_args
from .help import HelpArgumentParser
from .rna import (
    add_expression_args,
    expression_data_from_args,
)
from .sequence import add_sequence_args
from .errors import add_error_args
from .outputs import add_output_args, validate_output_column_args
from .references import (
    exclusion_options,
    has_builtin_exclusions,
    require_human,
    require_release,
    source_options,
    validate_source_options,
)
from .protein_changes import (
    add_protein_change_args,
    genome_from_args,
    protein_change_effects_from_args,
)

LAZY_RUNTIME_IMPORTS = {
    "_class_of_allele": ("..alleles", "_class_of_allele"),
    "exclude_by": ("..inputs", "exclude_by"),
    "read_fasta": ("..inputs", "read_fasta"),
    "read_peptide_csv": ("..inputs", "read_peptide_csv"),
    "read_peptide_fasta": ("..inputs", "read_peptide_fasta"),
    "read_sequence_csv": ("..inputs", "read_sequence_csv"),
    "slice_regions": ("..inputs", "slice_regions"),
    "cta_sequences": ("..sources", "cta_sequences"),
    "ensembl_proteome": ("..sources", "ensembl_proteome"),
    "non_cta_sequences": ("..sources", "non_cta_sequences"),
    "sequences_from_gene_names": ("..sources", "sequences_from_gene_names"),
    "sequences_from_gene_ids": ("..sources", "sequences_from_gene_ids"),
    "sequences_from_transcript_ids": ("..sources", "sequences_from_transcript_ids"),
    "tissue_expressed_sequences": ("..sources", "tissue_expressed_sequences"),
    "TopiaryPredictor": ("..predictor", "TopiaryPredictor"),
    "Affinity": ("..ranking", "Affinity"),
    "BoolOp": ("..ranking", "BoolOp"),
    "KindAccessor": ("..ranking", "KindAccessor"),
    "Presentation": ("..ranking", "Presentation"),
    "kind_matches": ("..ranking", "kind_matches"),
    "parse": ("..ranking", "parse"),
    "resolve_qualified_kind": ("..ranking", "resolve_qualified_kind"),
}


def resolve_runtime_symbol(name):
    """Return a lazily imported runtime symbol, honoring module attribute overrides.

    Parameters
    ----------
    name : str
        Symbol name defined in :data:`LAZY_RUNTIME_IMPORTS`.

    Returns
    -------
    object
        Resolved attribute from the corresponding Topiary submodule (or any
        override patched onto ``topiary.cli.args``).
    """
    if name in globals():
        return globals()[name]
    if name in LAZY_RUNTIME_IMPORTS:
        module_name, attr_name = LAZY_RUNTIME_IMPORTS[name]
        module = import_module(module_name, package=__package__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name):
    """PEP 562: defer heavy runtime imports until execution or explicit access."""
    return resolve_runtime_symbol(name)


def __dir__():
    """Include lazily imported runtime symbols in interactive discovery."""
    return sorted(set(globals()) | set(LAZY_RUNTIME_IMPORTS))


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
             "(default: the source species/release; human GRCh38 for unannotated inputs).",
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
             "(e.g. heart_muscle lung brain). Human only. Requires pirlygenes.",
    )
    input_group.add_argument(
        "--exclude-species",
        help="Species for built-in exclusions, independent of antigen origin and MHC. "
             "Explicitly selecting it starts a separate reference context; pair it "
             "with --exclude-ensembl-release (required for non-human species).",
    )
    input_group.add_argument(
        "--exclude-ensembl-release", type=int,
        help="Ensembl release for built-in exclusions. Defaults to the source release "
             "unless --exclude-species selects an independent reference context.",
    )

    # -- built-in sequence sources --
    gene_selection_help = (
        "Prefers complete protein-coding translations, then standard amino acids, then length."
    )
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
        help="Gene names to predict (e.g. BRAF TP53 EGFR). " + gene_selection_help,
    )
    input_group.add_argument(
        "--gene-ids",
        default=None,
        nargs="*",
        help="Ensembl gene IDs to predict. " + gene_selection_help,
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
             "Human only. Requires pirlygenes. " + gene_selection_help,
    )
    input_group.add_argument(
        "--species",
        help="Species of the antigen sequence source, independent of MHC alleles "
             "(default: human for direct inputs, actual annotation for variants). "
             "Non-human built-in sources require --ensembl-release. For variants, "
             "an explicit value must match their annotations.",
    )
    input_group.add_argument(
        "--ensembl-release",
        default=None,
        type=int,
        help=(
            "Source Ensembl release number. For variant inputs, an explicit value "
            "must match their annotations. Without it, direct human inputs use the release the "
            "installed pyensembl exposes as ensembl_grch38, whose data must "
            "already be installed; pass one to pin an analysis."
        ),
    )

    return input_group


class CommandLineError(ValueError):
    """The command line itself is malformed: flags missing or in conflict.

    argparse cannot express these checks, but they are the same kind of
    mistake it reports, so ``main`` reports them the same way: with the
    usage block and exit status 2. Everything else it catches (a missing
    file, an allele a predictor rejects, an unwritable output) exits 1
    without the usage block (#310). A ``ValueError`` subclass, so code
    calling these validators directly sees no change.
    """


MHC_SOURCE_ERROR = (
    "Must supply either --mhc-predictor (live predictor) or "
    "--mhc-cache-file / --mhc-cache-directory (cached predictions). "
    "Both are missing."
)

INPUT_SOURCE_ERROR = (
    "No input specified. Use one of: "
    "--peptide-csv, --sequence-csv, --fasta, --peptide-fasta, "
    "--gene-names, --gene-ids, --transcript-ids, --cta, "
    "--ensembl-proteome, --vcf, --maf, --variant, --json-variants, "
    "--protein-change"
)


def create_arg_parser(
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
    arg_parser = HelpArgumentParser(
        prog="topiary",
        description=(
            "Predict, filter, and rank MHC-presented peptides from "
            "sequence or variant inputs."
        ),
        epilog=(
            "Examples:\n"
            "  topiary --mhc-predictor random --mhc-alleles A0201 "
            "--peptide-csv peptides.csv --output-csv predictions.csv\n"
            "  topiary --mhc-cache-file predictions.tsv "
            "--mhc-cache-format topiary_output --peptide-csv peptides.csv\n\n"
            "Run 'topiary --help' to see all options."
        ),
    )
    arg_parser.add_argument(
        "--version",
        action="version",
        version=f"topiary {__version__}",
        help="Print the installed Topiary version and exit",
    )
    if expression:
        add_expression_args(arg_parser)
    if mhc:
        # --mhc-predictor is optional so a run can be answered entirely from
        # --mhc-cache-file / --mhc-cache-directory; predict_epitopes_from_args
        # checks that at least one MHC source was given.
        add_mhc_args(arg_parser, predictor_required=False)
        add_cached_predictor_args(arg_parser)
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
        filter_clauses.append(resolve_runtime_symbol("parse")(ranking_text))
    else:
        if args.ic50_cutoff is not None:
            filter_clauses.append(
                resolve_runtime_symbol("Affinity").value <= args.ic50_cutoff
            )
        if args.percentile_cutoff is not None:
            filter_clauses.append(
                resolve_runtime_symbol("Affinity").rank <= args.percentile_cutoff
            )
        if has_presentation:
            filter_clauses.append(
                resolve_runtime_symbol("Presentation").rank <= args.presentation_cutoff
            )

    if not filter_clauses:
        filter_by = None
    elif len(filter_clauses) == 1:
        filter_by = filter_clauses[0]
    else:
        filter_by = resolve_runtime_symbol("BoolOp")(combine_op, filter_clauses)
        # Flatten same-op chains for a cleaner string form
        filter_by = _flatten_boolop(filter_by)

    sort_by = []
    if has_sort_by:
        for item in _split_top_level_commas(args.sort_by.strip()):
            sort_by.append(_parse_sort_expr(item))

    return filter_by, sort_by, sort_direction


def _flatten_boolop(node):
    """Flatten BoolOp(op, [BoolOp(op, [...]), ...]) into a single BoolOp."""
    bool_op_cls = resolve_runtime_symbol("BoolOp")
    if not isinstance(node, bool_op_cls):
        return node
    flat = []
    for c in node.children:
        if isinstance(c, bool_op_cls) and c.op is node.op:
            flat.extend(c.children)
        else:
            flat.append(c)
    return bool_op_cls(node.op, flat)


def _split_top_level_commas(text):
    """Split a comma-separated sort string on top-level commas only."""
    if not text:
        return []

    parts: list[str] = []
    current: list[str] = []
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
        kind, method = resolve_runtime_symbol("resolve_qualified_kind")(text)
        accessor = resolve_runtime_symbol("KindAccessor")(kind, method=method)
        if resolve_runtime_symbol("kind_matches")(
            kind, resolve_runtime_symbol("Affinity").kind
        ):
            return accessor.value
        return accessor.score

    return resolve_runtime_symbol("parse")(text)


def _looks_like_plain_kind_name(text):
    """Return True when *text* is a bare kind token like ba or mhcflurry_el."""
    if any(ch.isspace() for ch in text):
        return False
    if any(ch in text for ch in ".()[]+-*/"):
        return False
    try:
        resolve_runtime_symbol("resolve_qualified_kind")(text)
        return True
    except ValueError:
        return False


def _parse_regions(region_strings):
    """Parse --regions args like 'spike:319-541' into {name: [(start, end)]}."""
    if not region_strings:
        return None
    regions: dict[str, list[tuple[int, int]]] = {}
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
    options = source_options(args)

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
    if use_cta:
        require_human(options, "--cta")
    elif builtin_sources:
        require_release(options, "--ensembl-release")

    is_peptides = False
    sequences = None

    if peptide_csv:
        sequences = resolve_runtime_symbol("read_peptide_csv")(peptide_csv)
        is_peptides = True
    elif peptide_fasta:
        sequences = resolve_runtime_symbol("read_peptide_fasta")(peptide_fasta)
        is_peptides = True
    elif sequence_csv:
        sequences = resolve_runtime_symbol("read_sequence_csv")(sequence_csv)
    elif fasta:
        sequences = resolve_runtime_symbol("read_fasta")(fasta)
    elif use_ensembl:
        sequences = resolve_runtime_symbol("ensembl_proteome")(**options)
    elif gene_names:
        sequences = resolve_runtime_symbol("sequences_from_gene_names")(
            gene_names, **options
        )
    elif gene_id_list:
        sequences = resolve_runtime_symbol("sequences_from_gene_ids")(
            gene_id_list, **options
        )
    elif transcript_id_list:
        sequences = resolve_runtime_symbol("sequences_from_transcript_ids")(
            transcript_id_list, **options
        )
    elif use_cta:
        sequences = resolve_runtime_symbol("cta_sequences")(
            release=options["release"]
        )

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
        sequences = resolve_runtime_symbol("slice_regions")(sequences, regions)

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
    if getattr(args, "genome", None):
        incompatible_flags.append("--genome (use --species / --ensembl-release for direct inputs)")
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
    if getattr(args, "gene_expression", None):
        incompatible_flags.append("--gene-expression")
    if getattr(args, "transcript_expression", None):
        incompatible_flags.append("--transcript-expression")
    if getattr(args, "variant_expression", None):
        incompatible_flags.append("--variant-expression")

    if incompatible_flags:
        raise CommandLineError(
            "Direct sequence inputs can't be combined with variant/RNA inputs: %s"
            % ", ".join(incompatible_flags)
        )


def _has_prediction_input(args):
    return any([
        getattr(args, "peptide_csv", None),
        getattr(args, "sequence_csv", None),
        getattr(args, "fasta", None),
        getattr(args, "peptide_fasta", None),
        getattr(args, "gene_names", None),
        getattr(args, "gene_ids", None),
        getattr(args, "transcript_ids", None),
        getattr(args, "cta", False),
        getattr(args, "ensembl_proteome", False),
        getattr(args, "vcf", None),
        getattr(args, "maf", None),
        getattr(args, "variant", None),
        getattr(args, "json_variants", None),
        getattr(args, "protein_change", None),
    ])


def _validate_required_prediction_args(args):
    if cached_predictor_in_use(args) and getattr(args, "mhc_predictor", None):
        # The Cached Predictions group already documents these as
        # mutually exclusive ("the cache supplies predictions"), but
        # passing both ran the cache and ignored --mhc-predictor without
        # a word -- so a command naming mhcflurry got NetMHCpan rows from
        # the file (#321). Say so instead of picking one silently.
        raise CommandLineError(
            "--mhc-predictor and --mhc-cache-file / --mhc-cache-directory "
            "are mutually exclusive: a cache supplies predictions instead "
            "of running a predictor. Drop --mhc-predictor to score from "
            "the cache, or drop the cache flags to run the predictor."
        )
    missing = []
    if not cached_predictor_in_use(args) and not getattr(args, "mhc_predictor", None):
        missing.append(MHC_SOURCE_ERROR)
    if not _has_prediction_input(args):
        missing.append(INPUT_SOURCE_ERROR)

    if len(missing) > 1:
        raise CommandLineError(
            "No prediction request specified.\n\n" + "\n\n".join(missing)
        )
    if missing:
        raise CommandLineError(missing[0])


_CLASS1_ONLY_MHCFLURRY_PREDICTORS = frozenset({"mhcflurry", "mhcflurry-affinity"})


def _iter_cli_strings(value):
    """Yield flattened string items from argparse string, list, or nested-list values."""
    if not value:
        return
    if isinstance(value, str):
        yield value
        return
    for item in value:
        yield from _iter_cli_strings(item)


def _raw_cli_alleles(args):
    """Return user-supplied allele strings before mhctools normalizes them."""
    raw_tokens = []
    for item in _iter_cli_strings(getattr(args, "mhc_alleles", None)):
        for token in re.split(r"[\s,]+", item.strip()):
            if token:
                raw_tokens.append(token)
    alleles_file = getattr(args, "mhc_alleles_file", None)
    if alleles_file:
        for line in Path(alleles_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            for token in re.split(r"[\s,]+", line):
                if token:
                    raw_tokens.append(token)
    return raw_tokens


def _cli_predictor_names(args):
    """Return normalized predictor names requested via ``--mhc-predictor``."""
    names = []
    for item in _iter_cli_strings(getattr(args, "mhc_predictor", None)):
        for part in item.split(","):
            part = part.strip().lower()
            if part:
                names.append(part)
    return names


def _validate_predictor_alleles(args):
    """Reject class II alleles for class-I-only MHCflurry predictors early (#574).

    Echoes each allele exactly as the user typed it rather than leaking
    mhctools' internal class II pairing spelling, and fails before loading
    MHCflurry model weights.
    """
    predictors = [
        name for name in _cli_predictor_names(args)
        if name in _CLASS1_ONLY_MHCFLURRY_PREDICTORS
    ]
    if not predictors:
        return
    raw_alleles = _raw_cli_alleles(args)
    class_of_allele = resolve_runtime_symbol("_class_of_allele")
    class2 = [allele for allele in raw_alleles if class_of_allele(allele) == "II"]
    if class2:
        predictor_label = predictors[0]
        noun = "allele" if len(class2) == 1 else "alleles"
        raise ValueError(
            f"{predictor_label} predicts class I only; unsupported class II "
            f"{noun}: {', '.join(class2)}"
        )


def predict_epitopes_from_args(args):
    """
    Returns an epitope collection from the given commandline arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed commandline arguments for Topiary
    """
    _validate_required_prediction_args(args)
    cache_misses: list[Any] = []
    report_cache_misses = bool(getattr(args, "cache_miss_report", None))
    reference_options = None

    def finish(df):
        df = _apply_exclusion(df, args, reference_options=reference_options)
        if report_cache_misses:
            df.attrs["topiary_cache_misses"] = cache_misses
        return df

    # Read inputs and parse expressions before building predictors, which
    # can take seconds: a missing file or a bad --filter-by fails at once (#310).
    filter_by, sort_by, sort_direction = _build_filter_and_sort(args)
    _validate_input_modes(args)
    direct_input, is_peptides = _get_direct_input(args)
    protein_change_effects = None
    variants = None
    expr_data = None
    if direct_input is None:
        _require_variant_input(args)
        if getattr(args, "protein_change", None):
            protein_change_effects = protein_change_effects_from_args(args)
        if any(getattr(args, name, None) for name in (
            "vcf", "maf", "variant", "json_variants",
        )):
            variant_args = args
            if (getattr(args, "genome", None) is None
                    and getattr(args, "ensembl_release", None) is not None):
                from copy import copy
                from ..sources import ensembl_genome
                variant_args = copy(args)
                variant_args.genome = ensembl_genome(**source_options(args))
            variants = variant_collection_from_args(variant_args)
        expr_data = expression_data_from_args(args)

    # Validate output column flags and predictor/allele compatibility before
    # loading heavy models or running predictions (#574).
    validate_output_column_args(
        args,
        is_variant=(direct_input is None),
        expression_data=expr_data,
        allow_extra_columns=cached_predictor_in_use(args),
    )

    # Read actual loaded annotations: --genome can be ignored by MAF, and VCF /
    # JSON can supply their own references. None of this consults MHC alleles.
    genomes: list[Any] = []
    if (has_builtin_exclusions(args)
            or getattr(args, "species", None) is not None
            or getattr(args, "ensembl_release", None) is not None):
        if variants is not None:
            genomes.extend(variant.genome for variant in variants)
        if getattr(args, "protein_change", None):
            genomes.append(genome_from_args(args))
        elif direct_input is None and not genomes and getattr(args, "genome", None):
            genomes.append(genome_from_args(args))
    validate_source_options(args, genomes)
    reference_options = exclusion_options(args, genomes)

    if cached_predictor_in_use(args):
        models = cached_predictor_from_args(args)
    else:
        _validate_predictor_alleles(args)
        models = predictors_from_args(args)

    skip_errors = bool(
        args.skip_variant_errors or getattr(args, "skip_errors", False)
    )
    predictor = resolve_runtime_symbol("TopiaryPredictor")(
        models=models,
        padding_around_mutation=args.padding_around_mutation,
        filter_by=filter_by,
        sort_by=sort_by,
        sort_direction=sort_direction,
        only_novel_epitopes=args.only_novel_epitopes,
        raise_on_error=not skip_errors,
        protein_x_policy=getattr(args, "protein_x_policy", "skip"),
        predict_wt=getattr(args, "predict_wt", False),
        cache_miss_handler=cache_misses.append if report_cache_misses else None,
    )

    if direct_input is not None:
        if not direct_input:
            logging.warning(
                "No %s found in the input; there is nothing to predict.",
                "peptides" if is_peptides else "protein sequences",
            )
        if is_peptides:
            df = predictor.predict_from_named_peptides(direct_input)
        else:
            df = predictor.predict_from_named_sequences(direct_input)
        return finish(df)

    shared = dict(expression_data=expr_data)
    if variants is not None:
        df = predictor.predict_from_variants(
            variants=variants,
            additional_effects=protein_change_effects,
            **shared,
        )
    else:
        # Named protein changes are already mutation effects; they skip
        # variant_collection_from_args entirely.
        df = predictor.predict_from_mutation_effects(
            effects=protein_change_effects,
            **shared,
        )
    return finish(df)


def _require_variant_input(args):
    """Without a direct peptide/sequence input, some variant source is required."""
    if not any(getattr(args, name, None) for name in (
            "vcf", "maf", "variant", "json_variants", "protein_change")):
        raise CommandLineError(INPUT_SOURCE_ERROR)


def _apply_exclusion(df, args, reference_options=None):
    """Apply all exclusion sources to the predictions."""
    import pandas as pd

    options = reference_options or exclusion_options(args)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    mode = getattr(args, "exclude_mode", "substring")

    # Collect all reference sequences into one dict
    ref_sequences: dict[tuple[int, str], str] = {}

    def add_reference(sequences):
        # Header identities are local to a source. Two FASTAs (e.g. human and
        # mouse backgrounds) may reuse a header without losing either protein.
        source_index = len(ref_sequences)
        ref_sequences.update({(source_index, name): seq for name, seq in sequences.items()})

    exclude_paths = getattr(args, "exclude_fasta", None)
    if exclude_paths:
        read_fasta = resolve_runtime_symbol("read_fasta")
        for path in exclude_paths:
            add_reference(read_fasta(path))

    if getattr(args, "exclude_ensembl", False):
        add_reference(resolve_runtime_symbol("ensembl_proteome")(**options))

    if getattr(args, "exclude_non_cta", False):
        add_reference(
            resolve_runtime_symbol("non_cta_sequences")(release=options["release"])
        )

    exclude_tissues = getattr(args, "exclude_tissues", None)
    if exclude_tissues:
        add_reference(
            resolve_runtime_symbol("tissue_expressed_sequences")(
                exclude_tissues, release=options["release"]
            )
        )

    if ref_sequences:
        df = resolve_runtime_symbol("exclude_by")(df, ref_sequences, mode=mode)
    return df
