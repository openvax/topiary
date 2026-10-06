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

"""
test_cufflinks : Test that we can correctly load Cufflinks tracking files which
contain the estimated expression levels of genes and isoforms (computed from
RNA-Seq reads).
"""


from __future__ import print_function, division, absolute_import

from contextlib import nullcontext

import pandas as pd
import pytest

from topiary.rna import load_cufflinks_dataframe

from .common import eq_
from .data import data_path
from .test_twin_conformance import CUFFLINKS_FPKM_TWINS


@pytest.mark.parametrize("mode", ["default", "copy_on_write"])
@pytest.mark.parametrize("custom_columns", [False, True])
@pytest.mark.parametrize("drop_hidata", [False, True])
@pytest.mark.parametrize("replacement", [None, 0., 12.5, 100.])
def test_hidata_replacement_agrees_across_public_readers(
    tmp_path, mode, custom_columns, drop_hidata, replacement,
):
    frame = pd.DataFrame(dict(
        tracking_id=["ENSG000001", "ENSG000002", "ENSG000003", "ENSG000004", "ENSG000005"],
        FPKM=[7., 0., 2., 5., .25],
        FPKM_status=["FAIL", "HIDATA", "OK", "HIDATA", "LOWDATA"],
        locus=["chr1:1-10", "chr1:11-20", "chr1:21-30", "chr1:31-40", "chr1:41-50"],
        gene_short_name=["FAILED", "HIGH1", "OK1", "HIGH2", "LOW1"],
    ))
    columns = dict(id_column="tracking_id", fpkm_column="FPKM", status_column="FPKM_status",
                   locus_column="locus", gene_names_column="gene_short_name")
    if custom_columns:
        frame = frame.rename(columns={column: "custom_" + column for column in frame})
        columns = {argument: "custom_" + column for argument, column in columns.items()}
    path = tmp_path / "expression.tsv"
    frame.to_csv(path, sep="\t", index=False)
    expected = {"ENSG000003": 2., "ENSG000005": .25}
    if not drop_hidata:
        expected.update(ENSG000002=0. if replacement is None else replacement,
                        ENSG000004=5. if replacement is None else replacement)
    # pandas 3 always uses CoW; setting its deprecated option would itself warn.
    context = (pd.option_context("mode.copy_on_write", True)
               if mode == "copy_on_write" and int(pd.__version__.split(".")[0]) < 3
               else nullcontext())
    with context:
        for name, read_values in CUFFLINKS_FPKM_TWINS:
            actual = read_values(path, sep="\t", drop_hidata=drop_hidata,
                                 replace_hidata_fpkm_value=replacement, **columns)
            assert actual == expected, name


@pytest.mark.parametrize("fpkm_column", ["FPKM", "custom_fpkm"])
def test_fractional_hidata_replacement_accepts_integer_valued_input(tmp_path, fpkm_column):
    path = tmp_path / "integer-expression.tsv"
    path.write_text(f"tracking_id\t{fpkm_column}\tFPKM_status\tlocus\tgene_short_name\n"
                    "ENSG000001\t0\tHIDATA\tchr1:1-10\tGENE1\n"
                    "ENSG000002\t2\tOK\tchr1:11-20\tGENE2\n")
    for name, read_values in CUFFLINKS_FPKM_TWINS:
        assert read_values(path, sep="\t", fpkm_column=fpkm_column, drop_hidata=False,
                           replace_hidata_fpkm_value=12.5) == {"ENSG000001": 12.5, "ENSG000002": 2.}, name


def test_load_cufflinks_genes():
    genes_df = load_cufflinks_dataframe(
        data_path("genes.fpkm_tracking"),
        drop_lowdata=True,
        drop_hidata=True,
        drop_failed=True,
        drop_novel=False,
    )
    gene_ids = set(genes_df.id)
    expected_gene_ids = {
        "ENSG00000240361",
        "ENSG00000268020",
        "ENSG00000186092",
        "ENSG00000269308",
        "CUFF.1",
        "CUFF.2",
        "CUFF.3",
        "CUFF.4",
        "CUFF.5",
    }
    eq_(gene_ids, expected_gene_ids)


def test_load_cufflinks_genes_drop_novel():
    genes_df = load_cufflinks_dataframe(
        data_path("genes.fpkm_tracking"),
        drop_lowdata=True,
        drop_hidata=True,
        drop_failed=True,
        drop_novel=True,
    )
    gene_ids = set(genes_df.id)
    expected_gene_ids = {
        "ENSG00000240361",
        "ENSG00000268020",
        "ENSG00000186092",
        "ENSG00000269308",
    }
    eq_(gene_ids, expected_gene_ids)


def test_load_cufflinks_isoforms():
    transcripts_df = load_cufflinks_dataframe(
        data_path("isoforms.fpkm_tracking"),
        drop_lowdata=True,
        drop_hidata=True,
        drop_failed=True,
        drop_novel=False,
    )
    transcript_ids = set(transcripts_df.id)
    expected_transcript_ids = {
        "ENST00000492842",
        "ENST00000594647",
        "ENST00000335137",
        "ENST00000417324",
        "ENST00000461467",
        "ENST00000518655",
        "CUFF.7604.1",
    }
    eq_(transcript_ids, expected_transcript_ids)


def test_load_cufflinks_isoforms_drop_novel():
    transcripts_df = load_cufflinks_dataframe(
        data_path("isoforms.fpkm_tracking"),
        drop_lowdata=True,
        drop_hidata=True,
        drop_failed=True,
        drop_novel=True,
    )
    transcript_ids = set(transcripts_df.id)
    expected_transcript_ids = {
        "ENST00000492842",
        "ENST00000594647",
        "ENST00000335137",
        "ENST00000417324",
        "ENST00000461467",
        "ENST00000518655",
    }
    eq_(transcript_ids, expected_transcript_ids)
