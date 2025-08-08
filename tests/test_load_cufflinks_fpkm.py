# Copyright (c) 2015. Mount Sinai School of Medicine
#
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

from topiary.rna import load_cufflinks_dataframe

from nose.tools import eq_

from .data import data_path

def test_load_cufflinks_genes():
    genes_df = load_cufflinks_dataframe(
        data_path("genes.fpkm_tracking"),
        drop_lowdata=True,
        drop_hidata=True,
        drop_failed=True,
        drop_novel=False)
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
        "CUFF.5"
    }
    eq_(gene_ids, expected_gene_ids)

def test_load_cufflinks_genes_drop_novel():
    genes_df = load_cufflinks_dataframe(
        data_path("genes.fpkm_tracking"),
        drop_lowdata=True,
        drop_hidata=True,
        drop_failed=True,
        drop_novel=True)
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
        drop_novel=False)
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
        drop_novel=True)
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
