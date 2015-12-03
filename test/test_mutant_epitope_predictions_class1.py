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

from mhctools import NetMHCpan
from nose.tools import eq_, raises
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
    'A02:01',
    'a0204',
    'B*07:02',
    'HLA-B14:02',
    'HLA-C*07:02',
    'hla-c07:01'
]

epitope_lengths = [9]

mhc_model = NetMHCpan(
    alleles=alleles,
    epitope_lengths=epitope_lengths)


def test_epitope_prediction_without_padding():
    output_without_padding = predict_epitopes_from_variants(
        variants=variants,
        mhc_model=mhc_model,
        transcript_expression_dict=None,
        only_novel_epitopes=True)
    # one prediction for each variant * number of alleles
    strong_binders = [
        epitope_prediction
        for epitope_prediction in output_without_padding
        if epitope_prediction.value <= 500.0
    ]
    eq_(len(strong_binders), 4)


@raises(ValueError)
def test_epitope_prediction_with_invalid_padding():
    predict_epitopes_from_variants(
        variants=variants,
        mhc_model=mhc_model,
        transcript_expression_dict=None,
        padding_around_mutation=7)


@raises(ValueError)
def test_epitope_prediction_with_invalid_zero_padding():
    predict_epitopes_from_variants(
        variants=variants,
        mhc_model=mhc_model,
        transcript_expression_dict=None,
        padding_around_mutation=7)


def test_epitope_prediction_with_valid_padding():
    output_with_padding = predict_epitopes_from_variants(
        variants=variants,
        mhc_model=mhc_model,
        transcript_expression_dict=None,
        padding_around_mutation=8,
        only_novel_epitopes=True)
    # 6 alleles * 2 mutations * 9 distinct windows = 108
    eq_(len(output_with_padding), 108)
