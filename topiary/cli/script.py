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

r"""
Script to generate epitope predictions from somatic cancer variants
and (optionally) tumor RNA-seq data.

Example usage:
    topiary \
        --mhc-predictor netmhcpan \
        --mhc-alleles-file HLA.txt \
        --vcf somatic.vcf \
        --gene-expression genes.fpkm_tracking \
        --transcript-expression isoforms.fpkm_tracking \
        --ic50-cutoff 500 \
        --percentile-cutoff 2 \
        --output-csv results.csv
"""

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

import argcomplete

from ..cached import CachedPredictorCoverageError, PredictorSetupError
from ..ranking import stated_values
from ..serialization import normalize_python_types
from .args import arg_parser, predict_epitopes_from_args

from .outputs import write_outputs


def parse_args(args_list=None):
    if args_list is None:
        args_list = sys.argv[1:]
    argcomplete.autocomplete(arg_parser)
    return arg_parser.parse_args(args_list)


def _cache_report_path(args):
    """Keep the new sidecar from overwriting any input or prediction output."""
    value = getattr(args, "cache_miss_report", None)
    if value is None:
        return None
    if not value or value == "-":
        raise ValueError("--cache-miss-report requires a JSON file path, not stdout")
    path = Path(value).expanduser().resolve()
    for name, values in vars(args).items():
        if not (name in ("fasta", "vcf", "maf", "json_variants", "regions", "gene_expression",
                         "transcript_expression", "variant_expression", "output_html")
                or name.endswith(("_file", "_files", "_csv", "_fasta", "_path", "_directory"))):
            continue
        for item in values if isinstance(values, (list, tuple)) else [values]:
            if not isinstance(item, str) or not item or item == "-":
                continue
            # Existing files cover inputs (including aliases/hardlinks); explicit
            # output paths must also be protected before those files exist.
            other = Path(item).expanduser()
            if other.is_file() or name in ("output_csv", "output_html", "mhc_cache_directory"):
                other = other.resolve()
                if (path == other or (path.exists() and other.exists() and path.samefile(other))
                        or name == "mhc_cache_directory" and other in path.parents):
                    raise ValueError("--cache-miss-report must differ from inputs, cache directories and outputs")
    return path


def main(args_list=None):
    """
    Script entry-point to predict neo-epitopes from genomic variants using
    Topiary.
    """
    args = parse_args(args_list)
    cache_misses = []
    try:
        report_path = _cache_report_path(args)
        # Predictor and input-reader progress belongs with diagnostics; stdout
        # is reserved for the result table, including parseable CSV pipelines.
        with redirect_stdout(sys.stderr):
            df = predict_epitopes_from_args(args)
        if report_path is not None:
            cache_misses = df.attrs["topiary_cache_misses"]
            report_path.write_text(json.dumps(normalize_python_types(dict(
                schema="topiary.cache_miss_report.v1", complete=not cache_misses,
                scope="skipped_model_input_pairs", prediction_rows=len(df), failures=cache_misses,
            )), indent=2) + "\n")
    except (
        OSError, ValueError, PredictorSetupError,
        CachedPredictorCoverageError,
    ) as e:
        # Each of these names something the caller can act on: a
        # missing input file, a bad argument, predictor setup still to
        # finish, or a cache that cannot answer for this peptide.
        #
        # All four are narrow deliberately. Bare KeyError would swallow
        # an unrelated dict-lookup bug, and bare RuntimeError is worse:
        # NotImplementedError and RecursionError subclass it, so an
        # unimplemented abstract method would print as a clean user
        # error. Those must keep reaching the user as tracebacks.
        #
        # str(e) suits all four: CachedPredictorCoverageError defines
        # __str__, so its message arrives without KeyError's repr
        # quoting and no per-type formatting is needed here. Removing
        # that __str__ would bring the quoting back (#296, #302, #304).
        message = str(e) or type(e).__name__
        arg_parser.error(message)
    try:
        write_outputs(df, args)
        sys.stdout.flush()
    except BrokenPipeError:
        # A consumer such as `head` may stop before all rows are written.
        # Redirect the descriptor so Python's final flush cannot fail again.
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())
        return 3 if cache_misses else 0

    counts = []
    for column, label in (("peptide", "unique peptide"), ("allele", "named allele")):
        count = df.loc[stated_values(df[column]), column].nunique() if column in df else 0
        counts.append(f"{count} {label}{'' if count == 1 else 's'}")
    print(
        f"{len(df)} prediction row{'' if len(df) == 1 else 's'} "
        f"({', '.join(counts)})", file=sys.stderr,
    )
    if df.empty:
        print("No prediction rows to display or save.", file=sys.stderr)
    if cache_misses:
        print(f"Partial result: {len(cache_misses)} model/input pair(s) skipped; "
              f"see {report_path}. Exit status 3.", file=sys.stderr)
    return 3 if cache_misses else 0
