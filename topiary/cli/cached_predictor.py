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


"""Commandline arguments for :class:`~topiary.CachedPredictor`.

Adds a group of ``--mhc-cache-*`` flags so users can run topiary from
pre-computed prediction files without invoking a live MHC predictor:

    topiary --mhc-cache-file predictions.csv --mhc-cache-format mhcflurry \\
        --peptide-csv input.csv --output-csv results.csv

Supported formats mirror :class:`CachedPredictor`'s ``from_*`` loaders.
"""

from ..cached import CachedPredictor


_CACHE_FORMATS = (
    "topiary_output",
    "mhcflurry",
    "tsv",
    "netmhcpan_stdout",
    "netmhc_stdout",
    "netmhcpan_cons_stdout",
    "netmhciipan_stdout",
    "netmhcstabpan_stdout",
)


def add_cached_predictor_args(arg_parser):
    group = arg_parser.add_argument_group(
        title="Cached Predictions",
        description=(
            "Run topiary from a pre-computed prediction file instead of "
            "invoking a live MHC predictor.  Mutually exclusive with "
            "--mhc-predictor (the cache supplies predictions)."
        ),
    )
    group.add_argument(
        "--mhc-cache-file",
        default=None,
        help=(
            "Path to a pre-computed prediction file.  Requires "
            "--mhc-cache-format.  Alternative to --mhc-cache-directory."
        ),
    )
    group.add_argument(
        "--mhc-cache-directory",
        default=None,
        help=(
            "Load every matching file in a directory as shards and merge "
            "them.  Use --mhc-cache-directory-pattern to restrict the "
            "glob (defaults to '*').  Uses the topiary_output format for "
            "every matched file — all shards must share the same "
            "(predictor_name, predictor_version)."
        ),
    )
    group.add_argument(
        "--mhc-cache-directory-pattern",
        default="*",
        help="Glob pattern for --mhc-cache-directory.  Default: *.",
    )
    group.add_argument(
        "--mhc-cache-format",
        choices=_CACHE_FORMATS,
        default=None,
        help=(
            "Format of --mhc-cache-file.  Required when --mhc-cache-file "
            "is set."
        ),
    )
    group.add_argument(
        "--mhc-cache-predictor-name",
        default=None,
        help=(
            "Override the prediction_method_name column value.  Required "
            "for 'tsv' format when the file doesn't carry this column."
        ),
    )
    group.add_argument(
        "--mhc-cache-predictor-version",
        default=None,
        help=(
            "Override the predictor_version column value.  Required for "
            "'tsv' format when the file doesn't carry this column; "
            "optional for the NetMHC-family formats (auto-parsed from "
            "the stdout preamble when omitted) and for 'mhcflurry' "
            "(auto-composed from the local install via "
            "topiary.mhcflurry_composite_version() when omitted)."
        ),
    )
    group.add_argument(
        "--mhc-cache-tsv-column",
        action="append",
        default=[],
        metavar="CANONICAL=FILE_COLUMN",
        help=(
            "Column-name mapping for 'tsv' format.  Repeatable.  Example: "
            "--mhc-cache-tsv-column affinity=IC50 "
            "--mhc-cache-tsv-column percentile_rank=Rank"
        ),
    )
    group.add_argument(
        "--mhc-cache-tsv-sep",
        default="\t",
        help="Column separator for 'tsv' format.  Default is tab.",
    )
    group.add_argument(
        "--mhc-cache-netmhcpan-mode",
        default="binding_affinity",
        choices=("binding_affinity", "elution_score"),
        help=(
            "Mode for 'netmhcpan_stdout' format on NetMHCpan 4+ output.  "
            "Default: binding_affinity."
        ),
    )
    group.add_argument(
        "--mhc-cache-netmhc-version",
        default="4",
        choices=("3", "4", "4.1"),
        help=(
            "NetMHC classic version for 'netmhc_stdout' format.  "
            "Default: 4."
        ),
    )
    group.add_argument(
        "--mhc-cache-netmhciipan-version",
        default="4.3",
        choices=("legacy", "4", "4.3"),
        help=(
            "NetMHCIIpan version for 'netmhciipan_stdout' format.  "
            "Default: 4.3."
        ),
    )
    group.add_argument(
        "--mhc-cache-netmhciipan-mode",
        default="elution_score",
        choices=("binding_affinity", "elution_score"),
        help=(
            "Mode for 'netmhciipan_stdout' format on NetMHCIIpan 4+ "
            "output.  Default: elution_score."
        ),
    )


def cached_predictor_in_use(args) -> bool:
    """True when the caller has asked for cached predictions (file or dir)."""
    return bool(
        getattr(args, "mhc_cache_file", None)
        or getattr(args, "mhc_cache_directory", None)
    )


def cached_predictor_from_args(args) -> CachedPredictor:
    """Construct a CachedPredictor from the parsed CLI args.

    Assumes :func:`cached_predictor_in_use` returned True.  Raises
    ``ValueError`` with an actionable message on any missing required
    flag for the chosen format.
    """
    cache_dir = getattr(args, "mhc_cache_directory", None)
    cache_file = getattr(args, "mhc_cache_file", None)

    if cache_file and cache_dir:
        raise ValueError(
            "--mhc-cache-file and --mhc-cache-directory are mutually "
            "exclusive.  Pick one."
        )

    if cache_dir is not None:
        pattern = getattr(args, "mhc_cache_directory_pattern", None) or "*"
        return CachedPredictor.from_directory(cache_dir, pattern=pattern)

    fmt = getattr(args, "mhc_cache_format", None)
    if not fmt:
        raise ValueError(
            "--mhc-cache-file requires --mhc-cache-format to be set "
            f"(choose one of {list(_CACHE_FORMATS)})."
        )

    if fmt == "topiary_output":
        return CachedPredictor.from_topiary_output(cache_file)

    if fmt == "mhcflurry":
        return CachedPredictor.from_mhcflurry(
            cache_file,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt == "tsv":
        columns = _parse_tsv_columns(args.mhc_cache_tsv_column)
        return CachedPredictor.from_tsv(
            cache_file,
            columns=columns,
            sep=args.mhc_cache_tsv_sep,
            prediction_method_name=args.mhc_cache_predictor_name,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt == "netmhcpan_stdout":
        return CachedPredictor.from_netmhcpan_stdout(
            cache_file,
            mode=args.mhc_cache_netmhcpan_mode,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt == "netmhc_stdout":
        return CachedPredictor.from_netmhc_stdout(
            cache_file,
            version=args.mhc_cache_netmhc_version,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt == "netmhcpan_cons_stdout":
        return CachedPredictor.from_netmhcpan_cons_stdout(
            cache_file,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt == "netmhciipan_stdout":
        return CachedPredictor.from_netmhciipan_stdout(
            cache_file,
            version=args.mhc_cache_netmhciipan_version,
            mode=args.mhc_cache_netmhciipan_mode,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt == "netmhcstabpan_stdout":
        return CachedPredictor.from_netmhcstabpan_stdout(
            cache_file,
            predictor_version=args.mhc_cache_predictor_version,
        )

    raise ValueError(f"Unknown --mhc-cache-format: {fmt!r}")


def _parse_tsv_columns(column_args):
    """Parse list of ``canonical=file_col`` strings into a mapping dict."""
    out = {}
    for entry in column_args:
        if "=" not in entry:
            raise ValueError(
                f"--mhc-cache-tsv-column expects KEY=VALUE; got {entry!r}."
            )
        k, v = entry.split("=", 1)
        out[k.strip()] = v.strip()
    return out
