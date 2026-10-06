"""Both GTF readers preserve known abundances with the pandas parser backend."""

import pytest

from .common import eq_
from .data import data_path
from .test_twin_conformance import GTF_FPKM_TWINS


@pytest.mark.parametrize("name,reader", GTF_FPKM_TWINS, ids=[name for name, _ in GTF_FPKM_TWINS])
def test_load_stringtie_gtf_transcripts(name, reader):
    transcript_fpkms = reader(
        data_path("B16-StringTie-chr1-subset.gtf")
    )
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
