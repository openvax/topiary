"""Replay the original selected RNA records and independently verify deletion labels.

Run from a checkout with current Isovar/Varcode: PYTHONPATH=. python this_script.py NEW_OUTPUT_DIR.
Historical fixtures and receipts are never overwritten.
"""
import sys
import json
import hashlib
from collections import Counter
from pathlib import Path
from importlib.metadata import version
import isovar
import pysam
from tests.sid_data import sid_data_root, sid_read
from tests.osteosarc_all_helpers import reference_models, validate_rna_protein
from scripts.osteosarc_variant_audit import reference_genome, audit_variant
from scripts.osteosarc_rna_overlay import POLICY, digest
from topiary import describe_isovar_result
root = sid_data_root('osteosarc_all_variants')
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
old = json.loads((root / 'expected.json').read_text())
records = json.loads((root / 'checked-inventory.json').read_text())['variants']
bam_path = sid_read('osteosarc_all_variants/source/t2-all-variant-regions.bam')
receipt = json.loads((root / 'source/bam.receipt.json').read_text())
selection = json.loads((root.parent / 'manifest.json').read_text())['read_selections']['osteosarc_all_variants/source/t2-all-variant-regions.bam']
with pysam.AlignmentFile(bam_path) as bam:
    records_by_sam = Counter((read.to_string() for read in bam))
assert sum(records_by_sam.values()) == selection['selected_records']
record_digest = hashlib.sha256(json.dumps(sorted(records_by_sam.items()), separators=(',', ':')).encode()).hexdigest()
assert record_digest == selection['sam_records_sha256']
genome = reference_genome(root, out / 'reference-index')
models = reference_models(root / 'reference')
collector = isovar.ReadCollector(**POLICY)
creator = isovar.ProteinSequenceCreator(protein_context_peptide_length=25, variant_sequence_assembly=True)
old_by_id = {x['input']['variant_id']: x for x in old}
changes = []
observations = []
for entry in records:
    if entry['input_status'] != 'ready':
        continue
    variant = audit_variant(entry, genome)
    with pysam.AlignmentFile(bam_path) as bam:
        result, = isovar.run_isovar([variant], bam, read_collector=collector, protein_sequence_creator=creator)
    observed = describe_isovar_result(result)
    validate_rna_protein(result, models)
    expected = dict(old_by_id[entry['variant_id']]['rna'])
    expected['transcript_ids'] = sorted(expected['transcript_ids'])
    diff = {key: {'before': expected.get(key), 'after': observed.get(key)} for key in expected.keys() | observed.keys() if expected.get(key) != observed.get(key)}
    if diff:
        assert set(diff) == {'predicted_effect'}, (entry['variant_id'], diff)
        changes.append({'variant_id': entry['variant_id'], **diff['predicted_effect'], 'range': result.predicted_effect.format_short_description(deletion_notation='range')})
    observations.append({'variant_id': entry['variant_id'], 'rna': observed})
    print(entry['variant_id'], diff, flush=True)
(out / 'observations.json').write_text(json.dumps(observations, indent=2) + '\n')
summary = {'dependency_versions': {p: version(p) for p in ['isovar', 'varcode', 'pyensembl', 'pysam']}, 'entries_checked': len(observations), 'historical_run_sha256': digest(root / 'run.json'), 'historical_expected_sha256': digest(root / 'expected.json'), 'exported_bam_sha256': digest(bam_path), 'historical_bam_sha256': receipt['sha256'], 'sam_records_sha256': record_digest, 'selected_records': sum(records_by_sam.values()), 'bundle': 'openvax-v1', 'reference_sha256': digest(root / 'reference/manifest.json'), 'inventory_sha256': digest(root / 'checked-inventory.json'), 'observations_sha256': digest(out / 'observations.json'), 'collection_policy': POLICY, 'protein_context_peptide_length': 25, 'variant_sequence_assembly': True, 'changes': changes, 'checks': ['all other RNA/outcome fields identical', 'independent raw-reference RNA translation and edit-interval oracle']}
assert len(changes) == 4, changes
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
