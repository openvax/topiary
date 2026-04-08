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
from mhctools.cli import add_mhc_args, mhc_binding_predictor_from_args
from varcode.cli import add_variant_args, variant_collection_from_args

from .filtering import add_filter_args
from .rna import (
    add_rna_args,
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)
from .sequence import add_sequence_args
from .errors import add_error_args
from .outputs import add_output_args
from .protein_changes import add_protein_change_args
from ..inputs import read_fasta, read_peptide_csv, read_sequence_csv
from ..predictor import TopiaryPredictor
from ..ranking import (
    EpitopeFilter,
    RankingStrategy,
    affinity_filter,
    parse_ranking,
    presentation_filter,
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
             "Peptides are predicted across all specified alleles.",
    )
    input_group.add_argument(
        "--sequence-csv",
        default=None,
        help="CSV with 'sequence' column (optional: 'name'). "
             "Sequences are scanned with sliding window.",
    )
    input_group.add_argument(
        "--fasta",
        default=None,
        help="FASTA file of protein sequences to scan.",
    )
    return input_group


def create_arg_parser(
    rna=True,
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
    if rna:
        add_rna_args(arg_parser)
    if mhc:
        add_mhc_args(arg_parser)
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


def _build_ranking_strategy(args):
    """Build a RankingStrategy from CLI args, or return None."""
    # --ranking takes precedence over individual filter args
    ranking_text = getattr(args, "ranking", None)
    if ranking_text:
        result = parse_ranking(ranking_text)
        if isinstance(result, EpitopeFilter):
            return RankingStrategy(filters=[result])
        return result

    has_presentation = getattr(args, "presentation_cutoff", None) is not None
    has_rank_by = getattr(args, "rank_by", None) is not None
    filter_logic = getattr(args, "filter_logic", "any")

    if not (has_presentation or has_rank_by):
        return None

    filters = []
    if args.ic50_cutoff or args.percentile_cutoff:
        filters.append(affinity_filter(
            ic50_cutoff=args.ic50_cutoff,
            percentile_cutoff=args.percentile_cutoff,
        ))
    if has_presentation:
        filters.append(presentation_filter(max_rank=args.presentation_cutoff))

    sort_by = []
    if has_rank_by:
        from ..ranking import KindAccessor, _resolve_kind
        kind_names = [s.strip() for s in args.rank_by.split(",")]
        sort_by = [KindAccessor(_resolve_kind(k)).score for k in kind_names]

    return RankingStrategy(
        filters=filters,
        require_all=(filter_logic == "all"),
        sort_by=sort_by,
    )


def _get_direct_input(args):
    """Check for direct peptide/sequence inputs. Returns (dict, mode) or (None, None)."""
    peptide_csv = getattr(args, "peptide_csv", None)
    sequence_csv = getattr(args, "sequence_csv", None)
    fasta = getattr(args, "fasta", None)

    sources = [s for s in [peptide_csv, sequence_csv, fasta] if s is not None]
    if len(sources) > 1:
        raise ValueError(
            "Only one of --peptide-csv, --sequence-csv, --fasta may be specified"
        )
    if peptide_csv:
        return read_peptide_csv(peptide_csv), "peptides"
    if sequence_csv:
        return read_sequence_csv(sequence_csv), "sequences"
    if fasta:
        return read_fasta(fasta), "sequences"
    return None, None


def predict_epitopes_from_args(args):
    """
    Returns an epitope collection from the given commandline arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed commandline arguments for Topiary
    """
    mhc_model = mhc_binding_predictor_from_args(args)
    ranking_strategy = _build_ranking_strategy(args)

    predictor = TopiaryPredictor(
        mhc_model=mhc_model,
        padding_around_mutation=args.padding_around_mutation,
        ic50_cutoff=args.ic50_cutoff,
        percentile_cutoff=args.percentile_cutoff,
        ranking_strategy=ranking_strategy,
        min_transcript_expression=args.rna_min_transcript_expression,
        min_gene_expression=args.rna_min_gene_expression,
        only_novel_epitopes=args.only_novel_epitopes,
        raise_on_error=not args.skip_variant_errors,
    )

    # Check for direct peptide/sequence inputs first
    direct_input, mode = _get_direct_input(args)
    if direct_input is not None:
        return predictor.predict_from_named_sequences(direct_input)

    # Otherwise, use variant pipeline
    variants = variant_collection_from_args(args)
    gene_expression_dict = rna_gene_expression_dict_from_args(args)
    transcript_expression_dict = rna_transcript_expression_dict_from_args(args)

    return predictor.predict_from_variants(
        variants=variants,
        transcript_expression_dict=transcript_expression_dict,
        gene_expression_dict=gene_expression_dict,
    )
