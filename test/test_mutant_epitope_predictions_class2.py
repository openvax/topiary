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


from __future__ import print_function, division, absolute_import

from mhctools import NetMHCIIpan
from nose.tools import eq_
from pyensembl import ensembl_grch37
from topiary import predict_epitopes_from_variants
from varcode import Variant, VariantCollection

# TODO: find out about these variants,
# what do we expect from them? Are they SNVs?
variants = VariantCollection([
    Variant(
      contig=10,
      start=100018900,
      ref='C',
      alt='T',
      ensembl=ensembl_grch37),
    Variant(
      contig=11,
      start=32861682,
      ref='G',
      alt='A',
      ensembl=ensembl_grch37)])

alleles = [
  "HLA-DPA1*01:05/DPB1*100:01",
  "DRB10102"
]

epitope_lengths = [15, 16]

mhc_model = NetMHCIIpan(
    alleles=alleles,
    epitope_lengths=epitope_lengths)

def test_netmhcii_pan_epitopes():
    epitope_predictions = predict_epitopes_from_variants(
        mhc_model=mhc_model,
        variants=variants,
        transcript_expression_dict=None,
        only_novel_epitopes=True)

    # expect (15 + 16 mutant peptides) * (2 alleles) * 2 variants =
    # 124 total epitope predictions
    eq_(len(epitope_predictions), 124)
    unique_alleles = {
      epitope_prediction.allele
      for epitope_prediction in epitope_predictions
    }
    assert len(unique_alleles) == 2, \
      "Expected 2 unique alleles, got %s" % (unique_alleles,)
    unique_lengths = {
      epitope_prediction.peptide_length
      for epitope_prediction in epitope_predictions
    }
    assert unique_lengths == {15, 16}, \
      "Expected epitopes of length 15 and 16 but got lengths %s" % (unique_lengths,)
