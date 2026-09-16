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

from mhctools.cli import mhc_alleles_from_args
from mhctools.allele_normalization import normalize_allele_name_or_raw

from ..cached import CachedPredictor
from ..ranking import is_stated, stated_values


_CACHE_FORMATS = (
    "topiary_output",
    "mhcflurry",
    "tsv",
    "netmhcpan",
    "netmhc",
    "netmhccons",
    "netmhciipan",
    "netmhcstabpan",
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
            "Path to a pre-computed prediction file. Format is detected "
            "when possible; generic TSVs require --mhc-cache-format tsv. "
            "Alternative to --mhc-cache-directory."
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
            "Format of --mhc-cache-file.  Optional: topiary sniffs the "
            "format from the file's content when omitted — NetMHC-family "
            "stdout captures carry a 'version X.Y' preamble, mhcflurry "
            "CSVs have 'mhcflurry_*' columns, topiary's own output has "
            "'prediction_method_name' + 'predictor_version'.  Only the "
            "'tsv' format requires an explicit --mhc-cache-format "
            "(generic tables can't be auto-detected). TSV files must carry "
            "a 'kind' column on every row (e.g. pMHC_affinity), either "
            "directly or mapped with --mhc-cache-tsv-column kind=FILE_COLUMN."
        ),
    )
    group.add_argument(
        "--mhc-cache-predictor-name",
        default=None,
        help=(
            "Fill missing prediction_method_name values in 'tsv', "
            "'topiary_output', or directory shards. Existing stated values "
            "must agree. Other formats already identify their predictor."
        ),
    )
    group.add_argument(
        "--mhc-cache-predictor-version",
        default=None,
        help=(
            "Supply predictor version provenance. For 'tsv', 'topiary_output', "
            "and directory shards, fills missing cells; existing stated "
            "values must agree. Required when those files lack a version; "
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
            "--mhc-cache-tsv-column percentile_rank=Rank. A 'kind' column "
            "is required; map a differently named one with "
            "--mhc-cache-tsv-column kind=FILE_COLUMN."
        ),
    )
    group.add_argument(
        "--mhc-cache-tsv-sep",
        default="\t",
        help="Column separator for 'tsv' format.  Default is tab.",
    )
    group.add_argument(
        "--mhc-cache-netmhc-version",
        default="4",
        choices=("3", "4", "4.1"),
        help=(
            "NetMHC classic version for 'netmhc' format.  "
            "Default: 4."
        ),
    )
    group.add_argument(
        "--mhc-cache-netmhciipan-version",
        default="4.3",
        choices=("legacy", "4", "4.3"),
        help=(
            "NetMHCIIpan version for 'netmhciipan' format.  "
            "Default: 4.3."
        ),
    )
    group.add_argument(
        "--mhc-cache-netmhciipan-mode",
        default="elution_score",
        choices=("binding_affinity", "elution_score"),
        help=(
            "Mode for 'netmhciipan' format on NetMHCIIpan 4+ "
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
    flag for the chosen format, or when the cache cannot answer for the
    genotype or peptide lengths the command asked for.
    """
    cache = _restrict_to_requested_alleles(_build_cached_predictor(args), args)
    # Match mhctools' live-predictor precedence for the legacy length flag.
    requested = (
        getattr(args, "mhc_peptide_lengths", None)
        or getattr(args, "mhc_epitope_lengths", None)
    )
    if requested:
        try:
            cache.default_peptide_lengths = requested
        except ValueError as error:
            raise ValueError(
                f"--mhc-peptide-lengths / --mhc-epitope-lengths: {error}"
            ) from error
    return cache


def _requested_alleles(args):
    """The alleles the command asked for, or ``[]`` if it named none.

    Parsed by mhctools' own ``mhc_alleles_from_args`` so a spelling
    normalizes here exactly as it does on the live-predictor path -- a
    second copy of that rule would let ``HLA-A*02:01`` match a cache on
    one path and miss it on the other.
    """
    if not getattr(args, "mhc_alleles", "") and not getattr(
        args, "mhc_alleles_file", None
    ):
        return []
    return list(mhc_alleles_from_args(args))


def _restrict_to_requested_alleles(cache, args):
    """Hold the cache to the genotype the command asked for.

    The cache used to decide the genotype by itself: asking for
    ``HLA-A*02:01`` against a cache holding only ``HLA-B*07:02`` returned
    B*07:02 rows, exit 0, no warning (#321). For the use this mode exists
    for -- re-scoring a cohort per patient from one shared table -- that
    silently answers every patient with whatever the table happens to
    hold.

    An uncovered allele is refused rather than dropped, matching how a
    missed peptide behaves. A covered subset is filtered to what was
    asked for, so an unrequested allele never comes back.

    Rows with no allele are always kept: an allele-free kind
    (antigen_processing, the half-lives) is not a prediction about any
    allele, so filtering it out with the unrequested ones would delete
    evidence the request never excluded.
    """
    requested = _requested_alleles(args)
    if not requested:
        return cache

    available = {str(a): normalize_allele_name_or_raw(str(a))
                 for a in cache.alleles if is_stated(a)}
    wanted = set(requested)
    covered_raw = {raw for raw, norm in available.items() if norm in wanted}
    missing = sorted(wanted - {available[raw] for raw in covered_raw})
    if missing:
        raise ValueError(
            f"This cache has no predictions for {missing}; it covers "
            f"{sorted(set(available.values()))}. Re-predict for the "
            f"requested allele(s), or ask only for alleles the cache "
            f"holds -- answering with the alleles it happens to contain "
            f"would report predictions the command did not request."
        )

    frame = cache.to_dataframe()
    keep = frame["allele"].isin(covered_raw) | ~stated_values(frame["allele"])
    if bool(keep.all()):
        return cache
    return CachedPredictor(frame[keep], fallback=cache.fallback)


def _build_cached_predictor(args) -> CachedPredictor:
    """Load the cache named by the CLI args, without applying the
    request-coverage checks :func:`cached_predictor_from_args` adds."""
    cache_dir = getattr(args, "mhc_cache_directory", None)
    cache_file = getattr(args, "mhc_cache_file", None)

    if cache_file and cache_dir:
        raise ValueError(
            "--mhc-cache-file and --mhc-cache-directory are mutually "
            "exclusive.  Pick one."
        )

    if cache_dir is not None:
        pattern = getattr(args, "mhc_cache_directory_pattern", None) or "*"
        return CachedPredictor.from_directory(
            cache_dir, pattern=pattern,
            prediction_method_name=args.mhc_cache_predictor_name,
            predictor_version=args.mhc_cache_predictor_version,
        )

    fmt = getattr(args, "mhc_cache_format", None)
    if not fmt:
        fmt = _sniff_format(cache_file)
        if fmt is None:
            raise ValueError(
                f"Could not auto-detect --mhc-cache-format for "
                f"{cache_file!r}.  Pass --mhc-cache-format explicitly "
                f"(one of {list(_CACHE_FORMATS)})."
            )

    if fmt == "topiary_output":
        return CachedPredictor.from_topiary_output(
            cache_file,
            prediction_method_name=args.mhc_cache_predictor_name,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt != "tsv" and args.mhc_cache_predictor_name is not None:
        raise ValueError(
            "--mhc-cache-predictor-name is only supported for topiary_output, "
            f"tsv, and directory shards; {fmt!r} identifies its own predictor."
        )

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

    if fmt == "netmhcpan":
        return CachedPredictor.from_netmhcpan_stdout(
            cache_file,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt == "netmhc":
        return CachedPredictor.from_netmhc_stdout(
            cache_file,
            version=args.mhc_cache_netmhc_version,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt == "netmhccons":
        return CachedPredictor.from_netmhcpan_cons_stdout(
            cache_file,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt == "netmhciipan":
        return CachedPredictor.from_netmhciipan_stdout(
            cache_file,
            version=args.mhc_cache_netmhciipan_version,
            mode=args.mhc_cache_netmhciipan_mode,
            predictor_version=args.mhc_cache_predictor_version,
        )

    if fmt == "netmhcstabpan":
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


def _sniff_format(path):
    """Guess the cache format from file content.

    Returns one of the strings in ``_CACHE_FORMATS`` or ``None`` when
    detection is ambiguous (e.g. a generic TSV without identifying
    columns).  ``"tsv"`` is never returned — generic TSVs need an
    explicit ``--mhc-cache-format tsv``.

    Detection order:
    1. Parquet magic bytes → ``topiary_output``.
    2. NetMHC-family ``...version ...`` preamble lines → respective
       tool name.
    3. ``mhcflurry_*`` columns in the first text line → ``mhcflurry``.
    4. ``prediction_method_name`` + ``predictor_version`` columns in
       the first text line → ``topiary_output``.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"PAR1":
        return "topiary_output"

    with open(path, "r", errors="replace") as f:
        head = f.read(10000)

    # NetMHC-family preamble detection.  Order matters: more specific
    # tool names come first so e.g. "NetMHCstabpan" doesn't get
    # mis-tagged as "NetMHC".
    import re
    for needle, fmt in (
        (r"\bNetMHCstabpan\b", "netmhcstabpan"),
        (r"\bNetMHCIIpan\b", "netmhciipan"),
        (r"\bNetMHCcons\b", "netmhccons"),
        (r"\bNetMHCpan\b", "netmhcpan"),
        (r"\bNetMHC\b", "netmhc"),
    ):
        if re.search(needle, head):
            return fmt

    first_line = head.split("\n", 1)[0]
    if "mhcflurry_affinity" in first_line or "mhcflurry_presentation" in first_line:
        return "mhcflurry"
    if (
        "prediction_method_name" in first_line
        and "predictor_version" in first_line
    ):
        return "topiary_output"
    return None
