"""Retain every SV nomination while ranking explicit protein evidence."""

from collections import defaultdict
from copy import deepcopy
import csv
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import pandas as pd

from .ranking import Column, apply_sort


SV_PROTEIN_EVIDENCE_TIERS = {
    10: "annotated_frame_reaches_alteration",
    20: "complete_ORF_annotated_start",
    30: "complete_ORF_five_prime_UTR_start",
    40: "complete_ORF_splice_supported_intronic_start",
    50: "complete_ORF_exploratory_or_ambiguous_start",
    60: "partial_ORF_with_full_interval_RNA",
    70: "ORF_without_full_interval_RNA",
    75: "ORF_RNA_with_ambiguous_event_linkage",
    80: "junction_RNA_without_recovered_protein",
    90: "DNA_and_same_sample_gene_expression",
    100: "DNA_and_other_sample_gene_expression",
    110: "DNA_without_qualifying_RNA",
    120: "nomination_without_reconstructed_protein",
}


def build_sv_interest_report(catalogue, orf_exports=(), *, comparisons=(), event_aliases=None):
    """Build an exhaustive, source-scoped SV protein-evidence report.

    Parameters
    ----------
    catalogue : dict
        ``osteosarc.load_sv_interest()`` or an equivalent mapping with
        ``targets``. Every target is retained, including unresolved entries.
    orf_exports : iterable of dict
        Isovar SV ORF v1/v2 exports. Duplicate reconstructions are reconciled
        by source-scoped fragment membership, never by adding their counts.
    comparisons : iterable of dict
        Optional Isovar ``sv_rna_prediction_comparison.v1`` results. Their
        annotated-frame hypotheses remain separate from exploratory ATGs.
    event_aliases : mapping, optional
        Explicit original export event ID to catalogue target ID mapping.
        Original IDs remain in the source observations. Unknown targets,
        conflicting sequences or inconsistent fragment membership raise
        ``ValueError``. No fuzzy event matching is performed.

    Returns
    -------
    dict
        Candidate summaries, protein hypotheses and unchanged catalogue with
        transparent ordinal evidence tiers. Lower tier numbers rank first;
        they are not probabilities. Full-ORF template ranks are calculated
        only within one sample/source and never used to compare abundance
        across libraries. Gene TPM remains separately sample-labelled.
        Protein abundance and observed translation remain unmeasured.

    Notes
    -----
    Start origin, RNA interval support and initiation are independent axes.
    Conflicting start assessments retain an ambiguous prior. Processed-read
    orientation never penalizes a candidate. Sequence completeness does not
    establish mutant specificity, mature transcript identity or presentation.
    Aliases and synonymous alternatives are not independent discoveries.
    """
    targets = catalogue["targets"]
    aliases = event_aliases or {}
    groups = {}

    def target_id(event):
        key = aliases.get(event, event)
        if key not in targets:
            raise ValueError("RNA event is absent from the interest catalogue: %s" % event)
        return key

    def observe(event, sample, source, candidate, kind, provenance):
        target = target_id(event)
        expected_reference = targets[target].get("assembly", catalogue.get("assembly"))
        observed_reference = provenance.get("reference_name")
        if expected_reference and observed_reference and expected_reference != observed_reference:
            raise ValueError("RNA and catalogue reference assemblies disagree")
        if not sample or not source:
            raise ValueError("RNA hypotheses require sample and source identity")
        sequence = candidate["amino_acids"]
        nucleotide = candidate.get("nucleotide_sequence")
        complete = candidate.get("ends_with_stop_codon", candidate.get("complete_candidate", False))
        if not sequence or type(complete) is not bool:
            raise ValueError("Protein hypotheses require a sequence and boolean completeness")
        identifier = candidate.get("candidate_id", candidate.get("hypothesis_id"))
        if not identifier:
            raise ValueError("Protein hypotheses require stable identity")
        key = target, identifier, sample, source, kind
        row = groups.setdefault(key, dict(
            event_id=target, hypothesis_id=identifier, sample_id=sample, source=source, kind=kind,
            amino_acids=sequence, nucleotide_sequence=nucleotide, complete=complete,
            fragment_ids=set(), source_observations=[], protein_abundance=None,
            translation_observed=False, initiation_observed=False))
        if (row["amino_acids"], row["nucleotide_sequence"], row["complete"]) != (sequence, nucleotide, complete):
            raise ValueError("Conflicting sequence for one hypothesis identity")
        support = candidate.get("rna_support")
        if support is not None:
            ids = support["fragment_ids"]
            if len(set(ids)) != support["fragments"] or len(ids) != len(set(ids)):
                raise ValueError("Fragment count disagrees with source-scoped membership")
            row["fragment_ids"].update(ids)
        observation = dict(original_event_id=event, provenance=deepcopy(provenance),
                           candidate=deepcopy(candidate))
        if observation not in row["source_observations"]:
            row["source_observations"].append(observation)

    for export in orf_exports:
        if export.get("schema") not in ("isovar.sv_rna_orfs.v1", "isovar.sv_rna_orfs.v2"):
            raise ValueError("Expected an Isovar SV ORF v1/v2 export")
        target_id(export["event_id"])  # Validate even empty exports.
        provenance = {k: deepcopy(v) for k, v in export.items() if k != "candidates"}
        for candidate in export["candidates"]:
            observe(export["event_id"], export["sample_id"], export["source"], candidate,
                    "exploratory_orf", provenance)
    for comparison in comparisons:
        if comparison.get("schema") != "isovar.sv_rna_prediction_comparison.v1":
            raise ValueError("Expected an Isovar RNA protein comparison")
        target_id(comparison["event_id"])
        provenance = {k: deepcopy(v) for k, v in comparison.items() if k != "hypotheses"}
        for candidate in comparison["hypotheses"]:
            if candidate["kind"] == "annotated_frame":
                observe(comparison["event_id"], comparison["sample_id"], comparison["rna_source"],
                        candidate, "annotated_frame", provenance)

    proteins = []
    for key, row in sorted(groups.items()):
        row["row_id"] = sha256(json.dumps(key).encode()).hexdigest()
        row["fragment_ids"] = sorted(row["fragment_ids"])
        row["full_orf_templates"] = len(row["fragment_ids"]) if row["kind"] == "exploratory_orf" else None
        row["independent_molecules"] = None
        summaries = [o["candidate"].get("start_evidence_summary", {}) for o in row["source_observations"]]
        priorities = {s.get("priority") for s in summaries}
        row["start_priority"] = next(iter(priorities)) if len(priorities) == 1 else None
        relations = set()
        for observation in row["source_observations"]:
            candidate = observation["candidate"]
            if row["kind"] == "annotated_frame":
                # A surrounding path can carry the SV while this translated
                # interval departs at a different, ordinary splice. Only the
                # translation's own departure evidence can establish linkage.
                relations.update(relation for p in candidate["paths"].values()
                                 for relation in p.get("candidate", {}).get("departure_relations", []))
            else:
                relations.update(j["relation"] for o in candidate.get("occurrences", []) for j in o["junctions"])
        row["event_linkage_relations"] = sorted(relations)
        linked = bool(relations & {"breakpoint_junction", "event_compatible_junction"})
        if row["kind"] == "annotated_frame":
            priority = 10
        elif not row["full_orf_templates"]:
            priority = 70
        elif not row["complete"]:
            priority = 60
        else:
            priority = {1: 20, 2: 30, 3: 40}.get(row["start_priority"], 50)
        if not linked and (row["kind"] == "annotated_frame" or row["full_orf_templates"]):
            priority = 75
        row.update(evidence_priority=priority, evidence_tier=SV_PROTEIN_EVIDENCE_TIERS[priority],
                   rna_support_rank_within_source=None,
                   adjacency_group_id=targets[row["event_id"]].get("adjacency_group_id", row["event_id"]))
        proteins.append(row)
    # The existing DSL controls evidence ordering. Abundance proxies have a
    # separate, explicit within-product denominator and never cross sources.
    if proteins:
        frame = pd.DataFrame(proteins)
        ranks = frame.groupby(["sample_id", "source"], dropna=False)["full_orf_templates"].rank(
            method="dense", ascending=False)
        frame["rna_support_rank_within_source"] = ranks
        ordered = apply_sort(frame, [Column("evidence_priority")], sort_direction="asc", group_keys=["row_id"])
        proteins = ordered.to_dict("records")
        for row in proteins:
            for field in ("full_orf_templates", "start_priority", "rna_support_rank_within_source"):
                value = row[field]
                row[field] = None if pd.isna(value) else int(value)
    by_event = defaultdict(list)
    for row in proteins:
        by_event[row["event_id"]].append(row)
    candidates = []
    for event, target in sorted(targets.items()):
        expression = target.get("gene_expression", [])
        dna_samples = target.get("dna_samples", [])
        positive_expression = [e for e in expression if e.get("tpm") is not None and e["tpm"] > 0]
        junction = any((e.get("split_path_templates") or 0) > 0 for e in target.get("rna_evidence", []))
        priority = (min(p["evidence_priority"] for p in by_event[event]) if by_event[event] else
                    80 if junction else
                    90 if dna_samples and any(e["sample_id"] in dna_samples for e in positive_expression) else
                    100 if dna_samples and positive_expression else 110 if dna_samples else 120)
        candidates.append(dict(
            event_id=event, genes=deepcopy(target.get("genes", [])), sv_type=target.get("sv_type"),
            adjacency_group_id=target.get("adjacency_group_id", event),
            evidence_priority=priority, evidence_tier=SV_PROTEIN_EVIDENCE_TIERS[priority],
            dna_samples=deepcopy(dna_samples), gene_expression=deepcopy(expression),
            expression_matches_a_DNA_sample=any(e["sample_id"] in dna_samples for e in positive_expression),
            rna_evidence=deepcopy(target.get("rna_evidence", [])),
            protein_hypothesis_rows=[p["row_id"] for p in by_event[event]], protein_abundance=None))
    if candidates:
        candidates = apply_sort(pd.DataFrame(candidates), [Column("evidence_priority")],
                                sort_direction="asc", group_keys=["event_id"]).to_dict("records")
    return dict(schema="topiary.sv_interest_report.v1", catalogue=deepcopy(catalogue),
                candidates=candidates, protein_hypotheses=proteins,
                policy=dict(evidence_tiers=SV_PROTEIN_EVIDENCE_TIERS,
                            ranking="ordinal evidence priority; not a translation probability",
                            abundance="protein unmeasured; template ranks only within sample/source",
                            orientation="descriptive only; no read-orientation penalty",
                            missing="unknown, never absence; all nominations retained",
                            aliases="same geometry is not necessarily the same inserted allele; never add counts",
                            novelty="candidate sequence does not establish mutant specificity or presentation"))


def write_sv_interest_report(report, prefix):
    """Write the full report as JSON, two TSVs, candidate protein FASTA and HTML.

    Parameters
    ----------
    report : dict
        Output of :func:`build_sv_interest_report`.
    prefix : str or pathlib.Path
        Output path without a format suffix; parent directories are created.

    Returns
    -------
    dict
        Paths by format. Unknown numbers become blank in TSV and null in
        JSON. Nested evidence is JSON-encoded in TSV. FASTA contains each
        distinct amino-acid sequence once, labelled as a hypothesis; it is
        not a vaccine selection. Empty reports still produce valid files.
    """
    prefix = Path(prefix).expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {kind: Path(str(prefix) + suffix) for kind, suffix in (
        ("json", ".json"), ("candidates", ".candidates.tsv"), ("proteins", ".proteins.tsv"),
        ("fasta", ".protein.fasta"), ("html", ".html"))}
    paths["json"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for kind, rows in (("candidates", report["candidates"]), ("proteins", report["protein_hypotheses"])):
        fields = list(rows[0]) if rows else ["event_id"]
        with paths[kind].open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                                 for k, v in row.items()})
    with paths["fasta"].open("w") as handle:
        for sequence in sorted({r["amino_acids"] for r in report["protein_hypotheses"]}):
            handle.write(">hypothesis_" + sha256(sequence.encode()).hexdigest() + "\n" + sequence + "\n")
    rows = []
    for row in report["candidates"]:
        expression = "; ".join("%s %s: %s TPM" % (e["sample_id"], e.get("gene", "gene"), e.get("tpm"))
                               for e in row["gene_expression"])
        rna = "; ".join("%s: %s" % (e["source_id"], e.get("split_path_templates")) for e in row["rna_evidence"])
        values = [row["event_id"], ", ".join(row["genes"]), row["evidence_tier"],
                  ", ".join(row["dna_samples"]), expression, rna, len(row["protein_hypothesis_rows"])]
        rows.append("<tr>" + "".join("<td>" + escape(str(v)) + "</td>" for v in values) + "</tr>")
    paths["html"].write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>SV candidate evidence</title>'
        '<style>body{font:15px system-ui;margin:2rem;color:#172b35}table{border-collapse:collapse;width:100%}'
        'th,td{text-align:left;vertical-align:top;padding:.6rem;border-bottom:1px solid #ddd}'
        'th{position:sticky;top:0;background:#eef4f5}input{padding:.5rem;width:28rem}</style>'
        '<h1>SV candidate evidence</h1><p>Every nomination is retained. Evidence tiers are ordinal, '
        'not probabilities. Gene TPM and source-specific read templates are RNA evidence; protein '
        'abundance is unmeasured. None means unassessed, not zero. Aliases and ORF alternatives are '
        'not independent events. Full sequences and provenance are in the accompanying JSON and TSV.</p>'
        '<label>Filter candidates <input id="search" type="search"></label><table><thead><tr>'
        '<th>Event</th><th>Genes</th><th>Protein evidence</th><th>DNA samples</th>'
        '<th>Gene expression</th><th>Junction templates by product</th><th>Protein rows</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table><script>'
        'document.getElementById("search").addEventListener("input",function(){'
        'for(const r of document.querySelectorAll("tbody tr"))'
        'r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase());});</script></html>')
    return paths
