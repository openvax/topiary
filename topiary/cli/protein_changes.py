# Copyright (c) 2018. Mount Sinai School of Medicine
#
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

import logging
from pyensembl import ensembl_grch38
from varcode import EffectCollection
from varcode.effects import Substitution
from varcode.reference import infer_genome
import re

def add_protein_change_args(arg_parser):
    protein_change_group = arg_parser.add_argument_group(
        title="Protein Changes",
        description="Input protein changes without associated genomic variants")

    protein_change_group.add_argument(
        "--protein-change",
        default=[],
        nargs=2,
        action="append",
        help="Protein modification without genomic variant (e.g. EGFR T790M)")

    return arg_parser

def genome_from_args(args):
    if args.genome:
        return infer_genome(args.genome)
    else:
        # no genome specified, assume it can be inferred from the file(s)
        # we're loading
        return ensembl_grch38

def transcript_sort_key(transcript):
    """
    Key function used to sort transcripts. Taking the negative of
    protein sequence length and nucleotide sequence length so that
    the transcripts with longest sequences come first in the list. This couldn't
    be accomplished with `reverse=True` since we're also sorting by
    transcript name (which places TP53-001 before TP53-002).
    """
    return (
        -len(transcript.protein_sequence),
        -len(transcript.sequence),
        transcript.name
    )

def best_transcript(transcripts):
    """
    Given a set of coding transcripts, choose the one with the longest
    protein sequence and in cases of ties use the following tie-breaking
    criteria:
        - transcript sequence (including UTRs)
        - transcript name (so TP53-001 should come before TP53-202)
    """
    assert len(transcripts) > 0
    sorted_list = sorted(transcripts, key=transcript_sort_key)
    return sorted_list[0]

def protein_change_effects_from_args(args):
    genome = genome_from_args(args)
    valid_gene_names = set(genome.gene_names())
    substitution_regex = re.compile("([A-Z]+)([0-9]+)([A-Z]+)")
    effects = []
    for gene_name, protein_change_string in args.protein_change:
        match_obj = substitution_regex.match(protein_change_string)
        if match_obj is None:
            logging.warn(
                "Unable to parse protein modification: '%s'" % protein_change_string)
            continue

        ref, base1_pos, alt = match_obj.groups()

        base1_pos = int(base1_pos)

        if gene_name not in valid_gene_names:
            logging.warn("Invalid gene name '%s' in protein modification: '%s'" % (
                gene_name, protein_change_string))
            continue

        candidate_transcripts = []
        for candidate_gene in genome.genes_by_name(gene_name):
            for candidate_transcript in candidate_gene.transcripts:
                if not candidate_transcript.is_protein_coding:
                    continue
                protein_sequence = candidate_transcript.protein_sequence
                if protein_sequence is None:
                    continue
                if len(protein_sequence) < (base1_pos + len(ref) - 1):
                    # protein sequence too short for this modification
                    # e.g. EGFR T790M can't happen in an EGFR transcript
                    # with only 789 amino acids
                    continue

                seq_at_pos = protein_sequence[base1_pos - 1: base1_pos + len(ref) - 1]
                if seq_at_pos != ref:
                    # if this transcript doesn't have the same reference amino
                    # acids as the change then skip it and use a different
                    # transcript
                    continue
                candidate_transcripts.append(candidate_transcript)
        if len(candidate_transcripts) > 0:
            transcript = best_transcript(candidate_transcripts)
            effects.append(Substitution(
                variant=None,
                transcript=transcript,
                aa_ref=ref,
                aa_alt=alt,
                aa_mutation_start_offset=base1_pos - 1))
    return EffectCollection(effects)
