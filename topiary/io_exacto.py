"""Native Exacto tables, retaining observed RNA versus predicted protein evidence."""

from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
import csv
import gzip
from pathlib import Path

import pandas as pd
from varcode.effects.codon_tables import STANDARD

from .candidates import _identity
from .protein_fragment import ProteinFragment, make_fragment_id
from .result import TopiaryResult


EXACTO_SCHEMA_COMMIT = "307c08670d5e706734bddf393bcebc84db497f9f"
EXACTO_SCHEMAS = {
    "transcript_read_support": ("transcript_model_id", "read_name"),
    "peptide_variants": (
        "mutant_peptide_id", "peptide_id", "mutant_peptide_sequence", "k",
        "primary_structure_index_start", "primary_structure_index_end",
        "rna_variant_call_ids", "dna_variant_call_ids",
    ),
    "translations": ("peptide_id", "peptide_sequence", "rna_id", "rna_sequence", "orf_start", "orf_end"),
    "primary_structures": (
        "peptide_id", "primary_structure_index", "type", "amino_acid", "amino_acid_index", "codon_index",
        "nucleotide", "transcript_model_id", "reference_transcript_ids", "transcript_structure_index",
        "read_start", "read_end", "net_variant_nucleotides_count", "frameshift_state",
        "rna_variant_call_ids", "dna_variant_call_ids", "codon_rna_variant_call_ids",
        "codon_dna_variant_call_ids", "frameshift_rna_variant_call_ids", "frameshift_dna_variant_call_ids",
        "amino_acid_change",
    ),
}
_COLUMNS = (
    "rna_observations", "sample_name", "reference_name", "peptide", "allele", "kind", "value", "prediction_method_name",
    "sequence", "protein_sequence", "protein_hypothesis_sequence", "coding_sequence",
    "nucleotide_sequence", "orf_id", "orf_start", "orf_end", "reading_frame", "orf_completeness",
    "starts_at_start_codon", "ends_with_stop_codon", "peptide_start", "peptide_end", "n_flank", "c_flank",
    "source_type", "sequence_source", "tumor_specificity", "target_intervals", "mutation_intervals_in_peptide",
    "exacto_record_id", "exacto_native_rows", "exacto_rna_variant_call_ids", "exacto_dna_variant_call_ids",
    "exacto_transcript_model_id", "exacto_reference_transcript_ids", "exacto_rna_id", "exacto_primary_rows",
)


def _table(path, schema=None):
    if hasattr(path, "read"):
        context = nullcontext(path)
        name = getattr(path, "name", "stream")
    else:
        name = str(path)
        context = gzip.open(path, "rt", newline="") if name.endswith(".gz") else open(path, newline="")
    with context as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate Exacto columns")
        matches = [s for s, required in EXACTO_SCHEMAS.items() if set(required) <= set(columns)]
        if schema is not None and schema not in EXACTO_SCHEMAS:
            raise ValueError(f"Unsupported Exacto schema {schema!r}; choose {sorted(EXACTO_SCHEMAS)}")
        if schema is None:
            if len(matches) != 1:
                raise ValueError("Unrecognized or ambiguous Exacto header; expected peptide_variants, translations or primary_structures")
            schema = matches[0]
        elif schema not in matches:
            raise ValueError(f"Missing required columns for Exacto {schema}")
        rows = []
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Ragged Exacto row at line {reader.line_num}")
            rows.append(row)
    return rows, dict(schema=schema, file=str(name), columns=columns, records=deepcopy(rows))


def _integer(row, field):
    value = row[field]
    if not str(value).isdigit():
        raise ValueError(f"Exacto {field} must be a nonnegative integer: {value!r}")
    return int(value)


def _ids(value):
    return sorted(set(filter(None, value.split(","))))


def _translated(coding):
    return "".join("*" if coding[i:i + 3] in STANDARD.stop_codons
                   else STANDARD.forward_table.get(coding[i:i + 3], "X")
                   for i in range(0, len(coding) - 2, 3))


def _protein(sequence, coding):
    if not sequence or "*" in sequence[:-1] or not sequence.removesuffix("*"):
        raise ValueError("Exacto protein requires residues with at most one terminal stop")
    if set(coding) - set("ACGTRYSWKMBDHVN") or (sequence.endswith("*") and len(coding) % 3):
        raise ValueError("Invalid Exacto coding sequence or trailing bases after stop")
    if _translated(coding) != sequence:
        raise ValueError("Exacto translated sequence disagrees with coding nucleotides")
    start, stop = coding.startswith("ATG"), sequence.endswith("*")
    protein = sequence.removesuffix("*")
    return dict(sequence=protein, protein_hypothesis_sequence=protein,
                protein_sequence=protein if start and stop else None,
                coding_sequence=coding, starts_at_start_codon=start, ends_with_stop_codon=stop,
                orf_completeness="start_to_stop" if start and stop else "partial_end" if start
                else "partial_start" if stop else "partial_both")


def _intervals(indices):
    intervals = []
    for index in sorted(set(indices)):
        if intervals and intervals[-1][1] == index:
            intervals[-1][1] = index + 1
        else:
            intervals.append([index, index + 1])
    return intervals


def _primary(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if not row["peptide_id"]:
            raise ValueError("Empty Exacto peptide_id")
        groups[row["peptide_id"]].append((index, row))
    proteins = {}
    for identity, native in groups.items():
        ordered = sorted(native, key=lambda pair: _integer(pair[1], "primary_structure_index"))
        if [_integer(r, "primary_structure_index") for _, r in ordered] != list(range(len(ordered))):
            raise ValueError("Exacto primary indices must be unique and contiguous from zero")
        bases = []
        for _, row in ordered:
            if row["type"] not in {"base", "event"}:
                raise ValueError(f"Unsupported Exacto primary event type {row['type']!r}")
            if row["type"] == "event":
                if row["amino_acid"] or row["nucleotide"]:
                    raise ValueError("Sequence-bearing Exacto event records are unsupported")
                continue
            if len(row["nucleotide"]) != 1 or _integer(row, "read_start") != _integer(row, "read_end"):
                raise ValueError("Exacto primary base must name one nucleotide/read position")
            index = len(bases)
            if _integer(row, "codon_index") != index % 3 or _integer(row, "amino_acid_index") != index // 3:
                raise ValueError("Invalid Exacto codon/amino-acid indices")
            if bases and _integer(row, "read_start") != _integer(bases[-1], "read_start") + 1:
                raise ValueError("Gapped or overlapping Exacto coding sequence is unsupported")
            bases.append(row)
        if not bases:
            raise ValueError("Exacto primary structure has no coding bases")
        for field in ("transcript_model_id", "reference_transcript_ids"):
            if len({r[field] for _, r in ordered}) != 1:
                raise ValueError(f"Exacto {field} differs within a primary structure")
        coding = "".join(r["nucleotide"] for r in bases).upper().replace("U", "T")
        sequence, mutated, spans = [], [], {}
        for index in range(0, len(bases), 3):
            codon = bases[index:index + 3]
            if len(codon) != 3:
                if any(r["amino_acid"] for r in codon):
                    raise ValueError("Incomplete Exacto codon cannot claim an amino acid")
                continue
            amino = codon[0]["amino_acid"]
            if len(amino) != 1 or any(r["amino_acid"] != amino for r in codon):
                raise ValueError("Exacto codon rows disagree about the amino acid")
            changes = {r["amino_acid_change"] for r in codon}
            if len(changes) != 1 or not changes <= {"reference", "mutant"}:
                raise ValueError("Unsupported or inconsistent Exacto amino_acid_change")
            if changes == {"mutant"} and amino != "*":
                mutated.append(index // 3)
            for r in codon:
                spans[_integer(r, "primary_structure_index")] = (index // 3, _integer(r, "codon_index"))
            sequence.append(amino)
        row = dict(_protein("".join(sequence), coding), orf_id="exacto:" + identity,
                   # Read coordinates in structures are native; no reference transcript bounds are invented.
                   exacto_primary_rows=[index for index, _ in ordered],
                   exacto_native_rows=[index for index, _ in ordered],
                   exacto_transcript_model_id=bases[0]["transcript_model_id"],
                   exacto_reference_transcript_ids=_ids(bases[0]["reference_transcript_ids"]),
                   target_intervals=_intervals(mutated), exacto_record_id=identity)
        for assay in ("rna", "dna"):
            fields = [f"{prefix}{assay}_variant_call_ids" for prefix in ("", "codon_", "frameshift_")]
            row[f"exacto_{assay}_variant_call_ids"] = sorted({v for _, r in ordered for field in fields for v in _ids(r[field])})
        proteins[identity] = (row, spans)
    return proteins


def read_exacto(path, *, sample_name, schema=None, primary_structures=None, reference_name=None, tag=None,
                transcript_read_support=None, library_id=None, read_set_id=None):
    """Read native Exacto peptide/translation TSVs without running prediction.

    Parameters
    ----------
    path : path or text stream
        Native tab-separated table, optionally gzip-compressed. Supports the
        ``peptide_variants``, ``translations`` and ``primary_structures`` column
        schemas emitted by Exacto 0.4.6 at ``EXACTO_SCHEMA_COMMIT``. Extra columns
        and every native record are retained in metadata. Empty valid tables
        return an empty normalized result; unknown headers/event types raise.
    sample_name : str
        Required sample identity: these native files do not label the sample.
    schema : str, optional
        One of ``EXACTO_SCHEMAS``; default detects an unambiguous header. These
        names describe tested column schemas, not a producer-embedded version.
    primary_structures : path or text stream, optional
        Companion primary-structures TSV for a peptide-variants file. Recover
        exact peptide occurrences, translated context and producer-annotated
        mutant intervals; validate the peptide against that occurrence. Without
        it, the reported peptide remains usable and context/geometry stay unknown.
    transcript_read_support : path or text stream, optional
        Native transcript-model/read-name table. Requires explicit library and
        read-set IDs. Distinct read names become a transcript-level count with
        retained membership; missing models remain unknown. Counts are never
        promoted to variant support or ORF abundance.
    library_id, read_set_id : str, optional
        Evidence namespace for the companion read-support table. The read-set
        ID must identify a shared read-name namespace before counts can be unioned.
    reference_name : str, optional
        Explicit reference label, retained without inferring assembly or locus
        equivalence from Exacto's local numeric variant/model/peptide IDs.
    tag : str, optional
        Provenance/run label; defaults to the input filename. Required for
        streams with read support, to scope local transcript-model identities.

    Returns
    -------
    TopiaryResult
        Normalized source observations. Native Exacto has no HLA assignments or
        pMHC measurements: these remain null, and ORF-only rows have no peptide.
        Combining with LENS/pVACseq therefore ranks their reported candidates;
        this reader never creates peptide-HLA combinations. Protein sequences
        are predictions from RNA, not protein-expression measurements. Full
        products require a start codon and terminal stop; partial translations
        remain ``protein_hypothesis_sequence``. No tumor/normal specificity or
        comparator sequence is invented. Native DNA/RNA links and annotations
        are preserved in columns and ``extra['exacto']``. Translation ORF bounds
        are normalized from zero-based inclusive to half-open; the original
        values remain in metadata. All sequence choices stay explicit.
    """
    if not isinstance(sample_name, str) or not sample_name.strip():
        raise ValueError("Exacto sample_name must be a nonempty string")
    native, provenance = _table(path, schema)
    schema = provenance["schema"]
    if schema == "transcript_read_support":
        raise ValueError("Transcript read support is a companion table, not a sequence input")
    support_provenance, support = None, defaultdict(set)
    if transcript_read_support is not None:
        if schema == "translations" or (schema == "peptide_variants" and primary_structures is None):
            raise ValueError("Transcript read support requires primary-structure transcript model identities")
        if hasattr(path, "read") and not tag:
            raise ValueError("Stream read support requires a tag identifying the Exacto run")
        if any(not isinstance(v, str) or not v.strip() for v in (library_id, read_set_id)):
            raise ValueError("Transcript read support requires library_id and read_set_id")
        support_rows, support_provenance = _table(transcript_read_support, "transcript_read_support")
        for record in support_rows:
            if not record["transcript_model_id"] or not record["read_name"]:
                raise ValueError("Exacto read support requires transcript_model_id and read_name")
            support[record["transcript_model_id"]].add(record["read_name"])
    primary_provenance, proteins = None, {}
    if primary_structures is not None:
        if schema != "peptide_variants":
            raise ValueError("primary_structures companion applies only to peptide_variants")
        primary, primary_provenance = _table(primary_structures, "primary_structures")
        proteins = _primary(primary)
    rows = []
    if schema == "primary_structures":
        rows = [row for row, _ in _primary(native).values()]
    else:
        for index, native_row in enumerate(native):
            if not native_row["peptide_id"]:
                raise ValueError("Empty Exacto peptide_id")
            identity = native_row["peptide_id"]
            row = dict(orf_id="exacto:" + identity, exacto_native_rows=[index])
            if schema == "translations":
                rna = native_row["rna_sequence"].upper().replace("U", "T")
                start, end = _integer(native_row, "orf_start"), _integer(native_row, "orf_end")
                if start > end or end >= len(rna):
                    raise ValueError("Exacto ORF bounds exceed RNA sequence")
                row.update(_protein(native_row["peptide_sequence"], rna[start:end + 1]),
                           orf_id="exacto:translation:" + _identity([native_row["rna_id"], identity]),
                           nucleotide_sequence=rna, orf_start=start, orf_end=end + 1,
                           reading_frame=start % 3, exacto_rna_id=native_row["rna_id"], exacto_record_id=identity)
            else:
                peptide = native_row["mutant_peptide_sequence"]
                if not peptide or "*" in peptide or len(peptide) != _integer(native_row, "k"):
                    raise ValueError("Exacto peptide sequence disagrees with k")
                start, end = (_integer(native_row, f"primary_structure_index_{side}") for side in ("start", "end"))
                if start > end:
                    raise ValueError("Exacto primary interval is reversed")
                row.update(peptide=peptide, sequence=peptide, exacto_record_id=native_row["mutant_peptide_id"])
                for assay in ("rna", "dna"):
                    row[f"exacto_{assay}_variant_call_ids"] = _ids(native_row[f"{assay}_variant_call_ids"])
                if primary_structures is not None:
                    if identity not in proteins:
                        raise ValueError(f"Missing Exacto primary structure for peptide_id {identity}")
                    parent, spans = proteins[identity]
                    if start not in spans or end not in spans:
                        raise ValueError("Exacto peptide interval does not name coding bases")
                    if spans[start][1] != 0 or spans[end][1] != 2:
                        raise ValueError("Exacto peptide interval must span complete codons")
                    peptide_start, peptide_end = spans[start][0], spans[end][0] + 1
                    if parent["sequence"][peptide_start:peptide_end] != peptide:
                        raise ValueError("Exacto peptide disagrees with its primary structure")
                    row.update({k: deepcopy(v) for k, v in parent.items() if k not in
                                {"exacto_record_id", "exacto_native_rows", "exacto_rna_variant_call_ids", "exacto_dna_variant_call_ids"}})
                    row.update(peptide_start=peptide_start, peptide_end=peptide_end,
                               n_flank=parent["sequence"][:peptide_start], c_flank=parent["sequence"][peptide_end:],
                               mutation_intervals_in_peptide=[[max(a, peptide_start) - peptide_start,
                                                               min(b, peptide_end) - peptide_start]
                                                              for a, b in parent["target_intervals"]
                                                              if a < peptide_end and b > peptide_start])
            rows.append(row)
    for row in rows:
        members = support.get(row.get("exacto_transcript_model_id"))
        row["rna_observations"] = [] if members is None else [dict(
            sample_name=sample_name, entity_type="transcript",
            entity_id=f"exacto:{tag or provenance['file']}:{row['exacto_transcript_model_id']}",
            quantity="count", unit="reads", value=len(members), evidence_unit_ids=sorted(members),
            library_id=library_id, read_set_id=read_set_id, method="exacto_transcript_read_support")]
        row.update(sample_name=sample_name, reference_name=reference_name, source_type="rna:translation",
                   sequence_source="predicted_from_observed_rna", tumor_specificity=None)
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    return TopiaryResult(frame, sources=[tag or Path(provenance["file"]).name], form="long",
                         extra={"exacto": dict(provenance, tested_producer_commit=EXACTO_SCHEMA_COMMIT,
                                               primary_structures=primary_provenance, transcript_read_support=support_provenance)})


def read_exacto_fragments(path, **kwargs):
    """Convert supported native Exacto rows into optional prediction fragments.

    Parameters
    ----------
    path : path or text stream
        Native table accepted by ``read_exacto``.
    **kwargs
        Passed unchanged to ``read_exacto``, including required ``sample_name``.

    Returns
    -------
    list of ProteinFragment
        Distinct source records with their translated sequence, exact recovered
        novelty intervals (unknown without primary structures) and native evidence
        provenance. No predictor runs. This is a separate, explicit precursor to
        scanning new windows with ``TopiaryPredictor.predict_from_fragments``;
        it does not rescore or modify the original reported-candidate universe.
        Empty input returns an empty list. Missing comparator/specificity stays
        unknown; target intervals indicate sequence change, not tumor specificity.
    """
    result = read_exacto(path, **kwargs)
    fragments = []
    for row in result.df.to_dict("records"):
        # Each native occurrence remains distinct even for repeated sequences.
        identity = _identity([result.sources, row["orf_id"], row["exacto_record_id"], row["exacto_native_rows"]])
        intervals = row["target_intervals"] if isinstance(row["target_intervals"], list) else None
        provenance = result.extra["exacto"]
        evidence = {k: deepcopy(v) for k, v in provenance.items() if k not in {"records", "primary_structures", "transcript_read_support"}}
        evidence["records"] = [deepcopy(provenance["records"][i]) for i in row["exacto_native_rows"]]
        primary = provenance.get("primary_structures")
        if primary is not None:
            evidence["primary_structures"] = {
                "schema": primary["schema"], "file": primary["file"], "columns": primary["columns"],
                "records": [deepcopy(primary["records"][i]) for i in row["exacto_primary_rows"]],
            }
        support = provenance.get("transcript_read_support")
        if support is not None:
            evidence["transcript_read_support"] = {
                "schema": support["schema"], "file": support["file"], "columns": support["columns"],
                "records": [deepcopy(r) for r in support["records"]
                            if r["transcript_model_id"] == row["exacto_transcript_model_id"]],
            }
        fragments.append(ProteinFragment(
            fragment_id=make_fragment_id("exacto", row["sequence"], qualifiers=[identity]),
            sample_name=row["sample_name"], source_type=row["source_type"], sequence=row["sequence"],
            target_intervals=None if intervals is None else [tuple(pair) for pair in intervals],
            annotations={"exacto": evidence, "exacto_observation": row,
                         "sequence_source": row["sequence_source"], "tumor_specificity": None},
        ))
    return fragments
