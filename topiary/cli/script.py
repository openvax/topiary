# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to generate epitope predictions from somatic cancer variants
and (optionally) tumor RNA-seq data.

Example usage:
    topiary \
        --mhc-predictor netmhcpan
        --mhc-alleles-file HLA.txt
        --vcf somatic.vcf
        --rna-gene-fpkm-file genes.fpkm_tracking
        --rna-transcript-fpkm-file isoforms.fpkm_tracking
        --filter-ic50 500
        --filter-percentile 2
        --output results.csv
"""

import sys

import argcomplete

from ..cached import CachedPredictorCoverageError
from .args import arg_parser, predict_epitopes_from_args

from .outputs import write_outputs


def parse_args(args_list=None):
    if args_list is None:
        args_list = sys.argv[1:]
    argcomplete.autocomplete(arg_parser)
    return arg_parser.parse_args(args_list)


def main(args_list=None):
    """
    Script entry-point to predict neo-epitopes from genomic variants using
    Topiary.
    """
    args = parse_args(args_list)
    try:
        df = predict_epitopes_from_args(args)
    except (OSError, ValueError, CachedPredictorCoverageError) as e:
        # CachedPredictorCoverageError alongside OSError/ValueError: it's
        # the one CachedPredictor failure a CLI user needs to see cleanly
        # rather than as a traceback -- missed peptides with no fallback
        # configured, and a coverage gap a flank or genotype mismatch
        # leaves in a protein scan (#296, #302, #304). Catching this
        # specific subclass rather than bare KeyError means an unrelated
        # dict-lookup bug elsewhere in the same call graph still surfaces
        # as a traceback instead of being silently reported as a clean
        # CLI error. str(CachedPredictorCoverageError(...)) would print
        # with an extra layer of quoting (it subclasses KeyError, whose
        # str() reprs its args), so unwrap that one specifically.
        #
        # The isinstance guard matters: OSError's args are
        # (errno, strerror), so args[0] on a missing input file is the
        # bare integer 2 while str(e) is the readable
        # "[Errno 2] No such file or directory: '...'". Unwrapping
        # unconditionally printed "topiary: error: 2".
        if isinstance(e, CachedPredictorCoverageError) and e.args:
            message = str(e.args[0])
        else:
            message = str(e)
        arg_parser.error(message)
    write_outputs(df, args)
    print("Total count: %d" % len(df))
    return 0
