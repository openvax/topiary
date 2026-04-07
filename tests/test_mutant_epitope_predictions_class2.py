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

import pytest
from pyensembl import ensembl_grch37
from topiary import TopiaryPredictor
from varcode import Variant, VariantCollection

from .common import eq_

try:
    from mhctools import NetMHCIIpan

    mhc_model = NetMHCIIpan(
        alleles=["HLA-DPA1*01:05/DPB1*100:01", "DRB10102"],
        default_peptide_lengths=[15, 16],
    )
    HAS_NETMHC = True
except Exception:
    mhc_model = None
    HAS_NETMHC = False

pytestmark = pytest.mark.skipif(not HAS_NETMHC, reason="NetMHCIIpan not installed")

# TODO: find out about these variants,
# what do we expect from them? Are they SNVs?
variants = VariantCollection(
    [
        Variant(contig=10, start=100018900, ref="C", alt="T", ensembl=ensembl_grch37),
        Variant(contig=11, start=32861682, ref="G", alt="A", ensembl=ensembl_grch37),
    ]
)


def test_netmhcii_pan_epitopes():
    epitope_predictions = TopiaryPredictor(
        mhc_model=mhc_model, only_novel_epitopes=True
    ).predict_from_variants(variants=variants)

    # expect (15 + 16 mutant peptides) * (2 alleles) * 2 variants =
    # 124 total epitope predictions
    eq_(len(epitope_predictions), 124)
    unique_alleles = set(epitope_predictions.allele)
    assert len(unique_alleles) == 2, "Expected 2 unique alleles, got %s" % (
        unique_alleles,
    )
    unique_lengths = set(epitope_predictions.peptide_length)
    assert unique_lengths == {
        15,
        16,
    }, "Expected epitopes of length 15 and 16 but got lengths %s" % (unique_lengths,)
