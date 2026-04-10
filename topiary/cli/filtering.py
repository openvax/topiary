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
Commandline arguments related to epitope filtering
"""

def add_filter_args(arg_parser):
    filter_group = arg_parser.add_argument_group(
        title="Filtering Options",
        description="Criteria for removing epitopes from results",
    )

    filter_group.add_argument(
        "--ic50-cutoff",
        help="Drop epitopes with predicted IC50 nM affinity above this value",
        default=None,
        type=float,
    )

    filter_group.add_argument(
        "--percentile-cutoff",
        help="Drop epitopes with predicted IC50 percentile rank above this value",
        default=None,
        type=float,
    )

    filter_group.add_argument(
        "--only-novel-epitopes",
        help="".join(
            [
                "Drop epitopes which do not contain mutated residues or occur ",
                "in the self-ligandome.",
            ]
        ),
        default=False,
        action="store_true",
    )

    filter_group.add_argument(
        "--wildtype-ligandome-directory",
        help="".join(
            [
                "Directory of 'self' ligand peptide sets, in files named ",
                "by allele (e.g. 'A0201'). Any predicted mutant epitope which ",
                "is in the files associated with the given alleles is treated as ",
                "wildtype (non-mutated).",
            ]
        ),
    )

    filter_group.add_argument(
        "--presentation-cutoff",
        help=(
            "Drop epitopes with presentation (EL) percentile rank above "
            "this value. Only applies to predictors that produce "
            "pMHC_presentation scores (e.g. NetMHCpan 4.1+)."
        ),
        default=None,
        type=float,
    )

    filter_group.add_argument(
        "--filter-logic",
        help="How to combine multiple filter criteria: 'any' (OR) or 'all' (AND)",
        choices=["any", "all"],
        default="any",
    )

    filter_group.add_argument(
        "--rank-by",
        help=(
            "Ranking expression or comma-separated prediction kinds. "
            "Simple: 'pMHC_presentation,pMHC_affinity' ranks by presentation "
            "score, falling back to affinity. "
            "Expression: '0.5 * affinity.descending_cdf(500, 200) + "
            "0.5 * presentation.score.ascending_cdf(0.5, 0.3)'. "
            "Transforms: ascending_cdf, descending_cdf, logistic, clip, "
            "hinge, log, log2, log10, log1p, exp, sqrt. "
            "Aggregations: mean, geomean, minimum, maximum, median."
        ),
        default=None,
        type=str,
    )

    filter_group.add_argument(
        "--filter-by",
        help=(
            "Filter expression. Examples: "
            "'affinity <= 500', "
            "'affinity <= 500 | el.rank <= 2', "
            "'ba <= 500 & gene_tpm >= 5'. "
            "Overrides --ic50-cutoff, --percentile-cutoff, --presentation-cutoff."
        ),
        default=None,
        type=str,
    )

    return filter_group
