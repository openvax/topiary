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
from collections.abc import Iterable
from numbers import Integral

from .amino_acids import AMINO_ACIDS

#: Residues an MHC binding predictor can score. Everything else — ``X`` for
#: an unknown residue, ``U``, ``*`` — is refused by the tools Topiary drives.
SCORABLE_AMINO_ACIDS = frozenset(AMINO_ACIDS)


def scorable_peptide(peptide):
    """Whether a predictor can score this peptide at all.

    Parameters
    ----------
    peptide : str
        Candidate peptide sequence.

    Returns
    -------
    bool
        True when the peptide is nonempty and every residue is one of the
        twenty standard amino acids. The MHC predictors Topiary drives
        refuse anything else rather than scoring it: mhctools raises
        "Invalid character 'X' in peptide", MHCflurry "peptides have
        nonstandard amino acids". An unknown residue carries no binding
        evidence, so there is no score to substitute.
    """
    return bool(peptide) and SCORABLE_AMINO_ACIDS.issuperset(peptide)


def scannable_segments(sequences):
    """Split named proteins into the runs a predictor can scan.

    A protein from Ensembl may carry ``X`` where the residue is unknown —
    a few CTA proteins do — and every window covering one is refused by
    every predictor. Scanning such a protein whole used to end the entire
    run, so one unknown residue produced no output for any protein (#573).
    Scanning its scorable runs instead keeps every window that can be
    scored and drops exactly those that cannot.

    Parameters
    ----------
    sequences : mapping of str to str
        Named protein sequences, as passed to a model's protein scanner.

    Returns
    -------
    dict of str to tuple
        ``{scan_name: (source_name, offset, segment)}`` in input order.
        A protein a predictor can scan whole keeps its own name and offset
        0, so the ordinary case passes through unchanged. A split protein
        contributes one entry per run of scorable residues, each under a
        name that collides with no other key, and ``offset`` locates the
        run in the source protein so reported coordinates can be restored.
        A protein with no scorable run contributes nothing. Flanking
        context stops at the unscorable residue, which is where the
        sequence stops saying what the neighbouring residue is.

    Notes
    -----
    Logs one summary warning when any protein is split, naming how many
    proteins and residues were affected and one example.
    """
    segments = {}
    skipped = {}
    for name, sequence in sequences.items():
        runs = list(_residue_runs(sequence))
        unscorable = len(sequence) - sum(len(run) for _, run in runs)
        if not unscorable:
            segments[name] = (name, 0, sequence)
            continue
        skipped[name] = unscorable
        for offset, run in runs:
            scan_name = f"{name}@{offset}"
            duplicate = 2
            while scan_name in segments or scan_name in sequences:
                scan_name = f"{name}@{offset}#{duplicate}"
                duplicate += 1
            segments[scan_name] = (name, offset, run)
    if skipped:
        example = next(iter(skipped))
        logging.warning(
            "%d of %d protein sequences contain residues no MHC predictor "
            "can score (%d residues in all); windows covering them are not "
            "scored. For example %s has %d such residue(s).",
            len(skipped), len(sequences), sum(skipped.values()),
            example, skipped[example],
        )
    return segments


def _residue_runs(sequence):
    """Yield ``(offset, run)`` for each maximal scorable run of residues."""
    offset = 0
    run = []
    for index, residue in enumerate(sequence):
        if residue in SCORABLE_AMINO_ACIDS:
            if not run:
                offset = index
            run.append(residue)
        elif run:
            yield offset, "".join(run)
            run = []
    if run:
        yield offset, "".join(run)


def peptide_occurrence_flanks(source_sequence, peptide, peptide_offset):
    """Return the residues flanking one exact peptide occurrence.

    Parameters
    ----------
    source_sequence : str
        Protein sequence containing the occurrence.
    peptide : str
        Nonempty peptide sequence.
    peptide_offset : int
        Zero-based offset of the peptide in ``source_sequence``.

    Returns
    -------
    tuple of str
        The complete N- and C-terminal source context. An empty string states
        that the peptide reaches that molecular terminus.

    Raises
    ------
    TypeError
        If either sequence is not a string or the offset is not an integer.
    ValueError
        If a sequence is empty, the offset is negative, or the exact peptide
        does not occur at the stated offset. Matching is case-sensitive; this
        function never searches for or chooses among repeated occurrences.
    """
    if not isinstance(source_sequence, str) or not isinstance(peptide, str):
        raise TypeError("source_sequence and peptide must be strings")
    if not source_sequence or not peptide:
        raise ValueError("source_sequence and peptide must be nonempty")
    if isinstance(peptide_offset, bool) or not isinstance(peptide_offset, Integral):
        raise TypeError("peptide_offset must be an integer")
    peptide_offset = int(peptide_offset)
    if peptide_offset < 0:
        raise ValueError("peptide_offset must be nonnegative")
    peptide_end = peptide_offset + len(peptide)
    if source_sequence[peptide_offset:peptide_end] != peptide:
        raise ValueError("peptide does not occur in source_sequence at peptide_offset")
    return source_sequence[:peptide_offset], source_sequence[peptide_end:]


def protein_subsequences_around_mutations(effects, padding_around_mutation):
    """
    From each effect get a mutant protein sequence and pull out a subsequence
    around the mutation (based on the given padding). Returns a dictionary
    of subsequences and a dictionary of subsequence start offsets.
    """
    protein_subsequences = {}
    protein_subsequence_start_offsets = {}
    for effect in effects:
        protein_sequence = effect.mutant_protein_sequence
        # some effects will lack a mutant protein sequence since
        # they are either silent or unpredictable
        if protein_sequence:
            mutation_start = effect.aa_mutation_start_offset
            mutation_end = effect.aa_mutation_end_offset
            seq_start_offset = max(0, mutation_start - padding_around_mutation)
            # some pseudogenes have stop codons in the reference sequence,
            # if we try to use them for epitope prediction we should trim
            # the sequence to not include the stop character '*'
            first_stop_codon_index = protein_sequence.find("*")
            if first_stop_codon_index < 0:
                first_stop_codon_index = len(protein_sequence)

            seq_end_offset = min(
                first_stop_codon_index, mutation_end + padding_around_mutation
            )
            subsequence = protein_sequence[seq_start_offset:seq_end_offset]
            protein_subsequences[effect] = subsequence
            protein_subsequence_start_offsets[effect] = seq_start_offset
    return protein_subsequences, protein_subsequence_start_offsets


def check_padding_around_mutation(
    given_padding: int | None,
    epitope_lengths: Iterable[int],
) -> int:
    """Resolve the protein padding needed around a mutation.

    Parameters
    ----------
    given_padding : int or None
        Explicit number of residues to retain on each side of the mutation.
        ``None`` or integer zero selects the minimum required by
        ``epitope_lengths``.
    epitope_lengths : iterable of int
        Peptide lengths that will be scanned. An empty iterable represents a
        whole-peptide model and requires no automatic padding.

    Returns
    -------
    int
        The explicit padding, or one less than the longest epitope length when
        automatic padding is selected.

    Raises
    ------
    TypeError
        If ``given_padding`` is neither an integer nor ``None``.
    ValueError
        If explicit padding cannot contain an epitope of every requested
        length around the mutation.
    """
    min_required_padding = max((length - 1 for length in epitope_lengths), default=0)
    if given_padding is not None and not isinstance(given_padding, int):
        raise TypeError(
            "Padding around mutation must be an integer or None, "
            f"got {given_padding!r} ({type(given_padding).__name__})"
        )
    if not given_padding:
        return min_required_padding
    if given_padding < min_required_padding:
        raise ValueError(
            "Padding around mutation %d cannot be less than %d "
            "for epitope lengths %s"
            % (given_padding, min_required_padding, epitope_lengths)
        )
    return given_padding


def contains_mutant_residues(
    peptide_start_in_protein,
    peptide_length,
    mutation_start_in_protein,
    mutation_end_in_protein,
):
    peptide_end_in_protein = peptide_start_in_protein + peptide_length - 1
    return (
        peptide_start_in_protein < mutation_end_in_protein
        and peptide_end_in_protein >= mutation_start_in_protein
    )


def peptide_mutation_interval(
    peptide_start_in_protein,
    peptide_length,
    mutation_start_in_protein,
    mutation_end_in_protein,
):
    """
    Half-open interval of mutated residues in the peptide, determined
    from the mutation interval in the original protein sequence.

    Parameters
    ----------
    peptide_start_in_protein : int
        Position of the first peptide residue within the protein
        (starting from 0)

    peptide_length : int

    mutation_start_in_protein : int
        Position of the first mutated residue starting from 0. In the case of a
        deletion, the position where the first residue had been.

    mutation_end_in_protein : int
        Position of the last mutated residue in the mutant protein. In the case
        of a deletion, this is equal to the mutation_start_in_protein.
    )
    """
    if peptide_start_in_protein > mutation_end_in_protein:
        raise ValueError("Peptide starts after mutation")
    elif peptide_start_in_protein + peptide_length < mutation_start_in_protein:
        raise ValueError("Peptide ends before mutation")

    # need a half-open start/end interval
    peptide_mutation_start_offset = min(
        peptide_length, max(0, mutation_start_in_protein - peptide_start_in_protein)
    )
    peptide_mutation_end_offset = min(
        peptide_length, max(0, mutation_end_in_protein - peptide_start_in_protein)
    )
    return (peptide_mutation_start_offset, peptide_mutation_end_offset)
