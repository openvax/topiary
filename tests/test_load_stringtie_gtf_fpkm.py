from topiary.rna import load_transcript_fpkm_dict_from_gtf

from .common import eq_
from .data import data_path


def test_load_stringtie_gtf_transcripts():
    transcript_fpkms = load_transcript_fpkm_dict_from_gtf(
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


def test_load_expression_reads_a_gtf_as_pandas():
    """The other GTF door, which had no test and was broken.

    ``topiary.rna.gtf`` and ``expression_loader._load_gtf`` both read
    GTFs, and only the first was covered. gtfparse 2.x returns polars by
    default; ``_load_gtf`` treats the result as pandas, so every GTF
    read through ``load_expression`` (and through the CLI's
    ``--transcript-expression``) raised "expected 17 values when
    selecting columns by boolean mask" from polars, while this file's
    other test kept passing.
    """
    import pandas as pd

    from topiary.rna.expression_loader import load_expression

    df = load_expression(data_path("B16-StringTie-chr1-subset.gtf"))

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # Transcript rows only, carrying the abundance column.
    assert "reference_id" in df.columns
    assert "TPM" in df.columns or "FPKM" in df.columns
