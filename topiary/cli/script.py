# PYTHON_ARGCOMPLETE_OK
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
import shutil
import sys

import argcomplete

from .args import (
    CommandLineError,
    _has_prediction_input,
    arg_parser,
    predict_epitopes_from_args,
)
from .cached_predictor import cached_predictor_in_use
from .outputs import print_default_columns, write_outputs


def parse_args(args_list=None):
    if args_list is None:
        args_list = sys.argv[1:]
    argcomplete.autocomplete(arg_parser)
    return arg_parser.parse_args(args_list)


def _exit_with_error(message):
    """Report a failure topiary detected: one line on stderr, exit 1.

    argparse's own errors keep its usage block and exit status 2, which
    mean the command line itself is malformed. Everything found after
    parsing (a missing input, an unparseable allele, a predictor not set
    up, an output that can't be written) exits 1 without the usage block,
    so the message stays visible and scripts can tell the two apart (#310).
    """
    arg_parser.exit(1, f"{arg_parser.prog}: error: {message}\n")


def actionable_error_types():
    """Return the tuple of actionable exception classes for CLI error reporting.

    Each type names something the caller can act on: a missing input file, a
    bad argument, a missing optional dependency, an allele ``mhcgnomes``
    cannot parse, predictor setup still to finish, or a cache that cannot
    answer for a peptide.

    The list is narrow deliberately: bare ``KeyError`` would swallow an
    unrelated dict-lookup bug, and ``RuntimeError`` would swallow
    ``NotImplementedError`` or ``RecursionError``. Optional heavy exception
    classes are looked up from already-loaded modules so parser construction
    and tab completion never trigger heavy imports.
    """
    types = [OSError, ValueError, ImportError]
    mhcgnomes_errors = sys.modules.get("mhcgnomes.errors")
    if mhcgnomes_errors is not None:
        parse_error = getattr(mhcgnomes_errors, "ParseError", None)
        if isinstance(parse_error, type) and issubclass(parse_error, Exception):
            types.append(parse_error)
    cached_module = sys.modules.get("topiary.cached")
    if cached_module is not None:
        for name in ("PredictorSetupError", "CachedPredictorCoverageError"):
            err_type = getattr(cached_module, name, None)
            if isinstance(err_type, type) and issubclass(err_type, Exception):
                types.append(err_type)
    return tuple(types)


def format_actionable_error(error):
    """Format one user-facing error line for an actionable CLI failure.

    ``str`` suits every actionable type: ``CachedPredictorCoverageError``
    defines ``__str__``, so its message arrives without ``KeyError``'s repr
    quoting (#296, #302, #304). An ``mhcgnomes`` allele parse error says
    which kind of input it was about, since ``mhcgnomes``'s own text only
    names the string. Topiary's own :class:`topiary.AlleleError` is a
    ``ValueError`` whose message already says that, so it passes through
    unchanged.
    """
    message = str(error) or type(error).__name__
    mhcgnomes_errors = sys.modules.get("mhcgnomes.errors")
    parse_error = (
        getattr(mhcgnomes_errors, "ParseError", None)
        if mhcgnomes_errors is not None
        else None
    )
    if isinstance(parse_error, type) and isinstance(error, parse_error):
        return f"could not parse MHC allele: {message}"
    return message


def _cache_report_path(args):
    """Keep the new sidecar from overwriting any input or prediction output."""
    value = getattr(args, "cache_miss_report", None)
    if value is None:
        return None
    if not value or value == "-":
        raise CommandLineError("--cache-miss-report requires a JSON file path, not stdout")
    path = Path(value).expanduser().resolve()
    for name, values in vars(args).items():
        if not (name in ("fasta", "vcf", "maf", "json_variants", "regions", "gene_expression",
                         "transcript_expression", "variant_expression", "output_html",
                         "output_provenance")
                or name.endswith(("_file", "_files", "_csv", "_fasta", "_path", "_directory"))):
            continue
        for item in values if isinstance(values, (list, tuple)) else [values]:
            if not isinstance(item, str) or not item or item == "-":
                continue
            # Existing files cover inputs (including aliases/hardlinks); explicit
            # output paths must also be protected before those files exist.
            other = Path(item).expanduser()
            if other.is_file() or name in (
                "output_csv", "output_html", "output_provenance", "mhc_cache_directory",
            ):
                other = other.resolve()
                if (path == other or (path.exists() and other.exists() and path.samefile(other))
                        or name == "mhc_cache_directory" and other in path.parents):
                    raise CommandLineError("--cache-miss-report must differ from inputs, cache directories and outputs")
    return path


def _check_paths_before_predicting(args):
    """Fail in the first second on paths that would fail after the run.

    Predicting can take minutes, and an unwritable output used to surface
    only when the results were about to be saved (#310).
    """
    for flag, value in (
        ("--output-csv", args.output_csv),
        ("--output-html", args.output_html),
        ("--output-provenance", getattr(args, "output_provenance", None)),
    ):
        if not value or value == "-":
            continue
        path = Path(value).expanduser()
        if path.is_dir():
            raise ValueError(f"{flag} {value} is a directory")
        if not path.parent.is_dir():
            raise ValueError(f"{flag} {value}: directory {path.parent} does not exist")
        if not os.access(path if path.exists() else path.parent, os.W_OK):
            raise ValueError(f"{flag} {value} is not writable")
    # A model directory, an executable, or a program name on PATH: mhctools
    # accepts all three, and fails only after loading everything else.
    program = getattr(args, "mhc_predictor_path", None)
    if program and not (Path(program).expanduser().exists() or shutil.which(program)):
        raise ValueError(
            f"--mhc-predictor-path {program} is not an existing file or directory, "
            "or a program on PATH")


def main(args_list=None):
    """
    Script entry-point to predict neo-epitopes from genomic variants using
    Topiary.
    """
    args = parse_args(args_list)
    if (
        getattr(args, "print_columns", False)
        and not _has_prediction_input(args)
        and not cached_predictor_in_use(args)
        and not getattr(args, "mhc_predictor", None)
    ):
        try:
            print_default_columns(args)
        except actionable_error_types() as e:
            _exit_with_error(format_actionable_error(e))
        return 0
    cache_misses = []
    try:
        report_path = _cache_report_path(args)
        _check_paths_before_predicting(args)
        # Predictor and input-reader progress belongs with diagnostics; stdout
        # is reserved for the result table, including parseable CSV pipelines.
        with redirect_stdout(sys.stderr):
            df = predict_epitopes_from_args(args)
        if report_path is not None:
            from ..serialization import normalize_python_types

            cache_misses = df.attrs["topiary_cache_misses"]
            report_path.write_text(json.dumps(normalize_python_types(dict(
                schema="topiary.cache_miss_report.v1", complete=not cache_misses,
                scope="skipped_model_input_pairs", prediction_rows=len(df), failures=cache_misses,
            )), indent=2) + "\n")
    except CommandLineError as e:
        # A ValueError subclass, so it must precede the general clause.
        arg_parser.error(str(e))
    except actionable_error_types() as e:
        _exit_with_error(format_actionable_error(e))
    try:
        write_outputs(df, args)
        sys.stdout.flush()
    except BrokenPipeError:
        # A consumer such as `head` may stop before all rows are written.
        # Redirect the descriptor so Python's final flush cannot fail again.
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())
        return 3 if cache_misses else 0
    except actionable_error_types() as e:
        # BrokenPipeError is an OSError, so it must stay the first clause.
        _exit_with_error(format_actionable_error(e))

    from ..ranking import stated_values

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
