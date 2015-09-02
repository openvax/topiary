# Copyright (c) 2015. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper functions and shared datasets for tests
"""


from __future__ import print_function, division, absolute_import
import os

from varcode import Variant, VariantCollection
from pyensembl import ensembl_grch38

def data_path(name):
    """
    Return the absolute path to a file in the varcode/test/data directory.
    The name specified should be relative to varcode/test/data.
    """
    return os.path.join(os.path.dirname(__file__), "data", name)

# BRAF variant coordinates from COSMIC entry:
# http://cancer.sanger.ac.uk/cosmic/mutation/overview?id=476
braf_V600E_variant = Variant(7, 140753336, "A", "T", ensembl_grch38)

# TP53 variant coordinates from COSMIC entry:
# http://cancer.sanger.ac.uk/cosmic/mutation/overview?id=10656
tp53_R248W_variant = Variant(17, 7674221, "G", "A", ensembl_grch38)

cancer_test_variants = VariantCollection([
    braf_V600E_variant,
    tp53_R248W_variant
])

cancer_test_variant_gene_ids = {
    gene_id
    for v in cancer_test_variants
    for gene_id in v.gene_ids
}

cancer_test_variant_transcript_ids = {
    transcript_id
    for v in cancer_test_variants
    for transcript_id in v.transcript_ids
}
