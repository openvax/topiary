"""Regenerate synthetic exports using the published producer and its tagged tests."""

from pathlib import Path
import argparse
import json
import tempfile
import sys
import types
import isovar
import pysam

parser = argparse.ArgumentParser(description='Regenerate the Isovar 1.37.0 export fixtures')
parser.add_argument('producer_checkout', type=Path, help='Isovar checkout at tag v1.37.0')
args = parser.parse_args()
assert isovar.__version__ == '1.37.0'
# Load fixtures from the checked-out, unmodified v1.37.0 producer tests;
# Isovar itself was already imported from the installed published wheel.
package = types.ModuleType('tests')
package.__path__ = [str(args.producer_checkout.resolve() / 'tests')]
sys.modules['tests'] = package
from tests.test_protein_hypotheses import synthetic_result, READS
from tests.test_sv_rna_orf_export import reconstruction
from tests.test_sv_rna_comparison import inputs

out = Path(__file__).resolve().parents[1] / 'tests/data/isovar_exports'
settings = dict(max_protein_sequences_per_variant=0)
plain = isovar.export_protein_hypotheses([synthetic_result(settings=settings)],
    sample_id='tumor-1', source='synthetic-rna.bam')

header = dict(SQ=[dict(SN='1', LN=1000)],
              RG=[dict(ID='rg1', SM='tumor-1', LB='library'), dict(ID='rg2', SM='tumor-1')])
temporary = tempfile.TemporaryDirectory(prefix='topiary-isovar-fixtures-')
path = Path(temporary.name) / 'labels.bam'
with pysam.AlignmentFile(path, 'wb', header=header) as bam:
    for name, read in READS.items():
        group, query, mate = read.source_alignments[0][0]
        segment = pysam.AlignedSegment(bam.header)
        segment.query_name = query
        segment.query_sequence = 'CCATGGCTCAAGGCTAA'
        segment.query_qualities = pysam.qualitystring_to_array('I' * len(segment.query_sequence))
        segment.reference_id = 0
        segment.reference_start = 99
        segment.cigarstring = '17M'
        segment.mapping_quality = 60
        segment.flag = mate | (1 if mate else 0)
        segment.set_tag('RG', group)
        segment.set_tag('CB', 'cell-' + ('1' if query in ('a','b') else '2'))
        if query != 'b':
            segment.set_tag('UB', 'umi-' + query)
        bam.write(segment)
pysam.index(str(path))
with pysam.AlignmentFile(path) as bam:
    labelled = isovar.export_protein_hypotheses([synthetic_result(settings=settings)],
        sample_id='tumor-1', source='synthetic-rna.bam', cell_umi_alignment_file=bam,
        alignment_header=bam.header)

reconstructed = reconstruction()
identities = [o['identity'] for o in reconstructed['observations'].values()]
scope = dict(source='source', sample_id='sample', library='library')
reconstructed['cell_umi_evidence'] = dict(reads=[dict(identity=i, library_scope_known=True,
    scope=scope, cell_barcode='cell', label=['sample','source','library','cell','umi'],
    status='resolved_label') for i in identities])
orf = isovar.export_sv_rna_orfs(reconstructed)
comparison_input = inputs()
comparison = isovar.compare_sv_rna_predictions(*comparison_input)
(out / 'orfs-input.json').write_text(json.dumps(reconstructed, indent=2) + '\n')
(out / 'comparison-input.json').write_text(json.dumps(comparison_input, indent=2) + '\n')
for name, value in [('protein-v2', plain), ('protein-v2-labelled', labelled),
                    ('orfs-v4', orf), ('comparison-v3', comparison)]:
    (out / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    print(name, value['schema'])
print('labelled', labelled['events'][0]['protein_hypotheses'][0]['rna_support'])
print('isovar', isovar.__file__)

temporary.cleanup()
