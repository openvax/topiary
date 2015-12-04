from __future__ import print_function, division, absolute_import

from topiary.rna import load_transcript_fpkm_dict_from_gtf

from nose.tools import eq_

from .data import data_path


def test_load_stringtie_gtf_transcripts():
    transcript_fpkms = load_transcript_fpkm_dict_from_gtf(
        data_path("B16-StringTie-chr1-subset.gtf"))
    transcript_ids = set(transcript_fpkms.keys())
    expected_fpkms_dict = {
        "ENSMUST00000192505": 0.125126,
        "ENSMUST00000191939": 0.680062,
        "ENSMUST00000182774": 0.054028,
    }
    expected_transcript_ids = set(expected_fpkms_dict.keys())
    eq_(expected_transcript_ids, transcript_ids)
    for transcript_id, fpkm in expected_fpkms_dict.items():
        eq_(fpkm, transcript_fpkms[transcript_id])
