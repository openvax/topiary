"""Regenerate the separately versioned translation and synthetic score fixtures.

Run from the checkout with ``PYTHONPATH=. python
tests/data/osteosarc_shared/regenerate.py NEW_DIRECTORY``. Inspect differences;
do not overwrite the reviewed pins automatically. Source acquisition is in
``python -m scripts.osteosarc_test_data``.
"""

import argparse
import json
from pathlib import Path
import tempfile

import isovar
from osteosarc import Variant, Variants, digest, extract_reads, Cache

from scripts.osteosarc_rna_overlay import POLICY
from scripts.osteosarc_variant_audit import reference_genome
from topiary import describe_isovar_result, fragment_from_isovar_result, osteosarc_fixture_paths


def regenerate(destination):
    root = Path(__file__).resolve().parent
    source = root / "vaccine-rna-v1"
    manifest = json.loads((source / "manifest.json").read_text())
    paths = osteosarc_fixture_paths(manifest, directory=source)
    case = next(c for c in manifest["cases"] if c["variant"]["gene"] == "NTF3")
    record = case["variant"]
    reference = root.parent / "osteosarc_all_variants"
    options = dict(protein_context_peptide_length=25, variant_sequence_assembly=True)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        genome = reference_genome(reference, temporary / "reference")
        variants = Variants([Variant(
            id=record["variant_id"], gene=record["gene"], assembly=record["assembly"],
            alleles=((record["chrom"], record["pos"], record["ref"], record["alt"]),), status="ready")])
        subset = extract_reads(paths[case["bam"]], variants.regions(padding=100),
                               index=paths[case["bam"] + ".bai"], cache=Cache(temporary / "cache", offline=True))
        with subset.open() as bam:
            result, = isovar.run_isovar(
                variants.to_varcode(genome=genome, assembly="GRCh38"), bam,
                read_collector=isovar.ReadCollector(**POLICY),
                protein_sequence_creator=isovar.ProteinSequenceCreator(**options))
        fragment = fragment_from_isovar_result(result)
    if fragment is None:
        raise ValueError("No reconstructed protein; investigate instead of inventing a fixture")
    translated = dict(
        schema_version=1, data_version="translation-v1", parent_dataset_sha256=digest(source / "manifest.json"),
        case_id=case["case_id"], species=manifest["species"], taxon_id=manifest["taxon_id"],
        isovar_version=isovar.__version__, read_collector=POLICY, protein_sequence_creator=options,
        reference_manifest_sha256=digest(reference / "reference/manifest.json"),
        description=describe_isovar_result(result), fragment=fragment.to_dict())
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "translation-v1.json"
    path.write_text(json.dumps(translated, indent=2, sort_keys=True) + "\n")
    rows = [dict(peptide=fragment.sequence[i:i + 9], peptide_length=9, allele="HLA-A*02:01",
                 kind="pMHC_affinity", prediction_method_name="synthetic_fixture_affinity",
                 predictor_version="fixture-v1", affinity=float((i + 1) * 10), value=float((i + 1) * 10),
                 n_flank="", c_flank="", allele_set="") for i in range(len(fragment.sequence) - 8)]
    prediction = dict(schema_version=1, data_version="prediction-contract-v1",
                      interpretation="Synthetic numeric scores for cache, coverage and ranking regressions; not biological binding predictions.",
                      parent_translation_sha256=digest(path), rows=rows)
    (destination / "prediction-contract-v1.json").write_text(json.dumps(prediction, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    regenerate(parser.parse_args().destination)
