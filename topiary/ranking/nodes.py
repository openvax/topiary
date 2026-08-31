"""
Filtering and ranking of epitope predictions across prediction kinds.

Single-tree DSL: every node is a :class:`DSLNode` whose ``.eval(ctx)``
returns a :class:`pandas.Series` indexed by the unique peptide-allele
group tuples of the DataFrame.  Booleans and numbers live in the same
tree — comparisons (``<=``, ``>=``, ...) return boolean-valued nodes
that still participate in arithmetic (pandas idiom).

Examples::

    from topiary import Affinity, Presentation

    # Comparison (boolean node)
    Affinity.value <= 500

    # Compound boolean
    (Affinity.value <= 500) | (Presentation.rank <= 2.0)
    (Affinity.value <= 500) & (Presentation.rank <= 2.0)

    # Composite numeric ranking
    0.5 * Affinity.value.norm(mean=500, std=200) \
      + 0.5 * Presentation.score.norm(mean=0.5, std=0.3)

    # Boolean-as-number composition (allowed and encouraged)
    (Affinity <= 500) * Affinity.score \
      + (Affinity > 500) * 0.5 * Affinity.score

Applying a tree to a DataFrame::

    from topiary.ranking import apply_filter, apply_sort
    df = apply_filter(df, (Affinity.value <= 500) | (Presentation.rank <= 2.0))
    df = apply_sort(df, [Presentation.score, Affinity.score])
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType
from difflib import get_close_matches
from typing import Optional

import numpy as np
from packaging.version import InvalidVersion, Version
import pandas as pd
from mhctools import MHC_DEPENDENCE_VALUES, Kind


# =============================================================================
# mhctools Kind compatibility
# =============================================================================


def _kind_name(kind):
    """Return the canonical mhctools kind name."""
    return getattr(kind, "name", str(kind))


def _kind_value(kind):
    """Return the DataFrame ``kind`` value for a kind constant."""
    return getattr(kind, "value", kind)


def _kind_short_name(kind):
    """Return the DSL short name for a kind."""
    return _kind_name(kind).lower().replace("pmhc_", "")


def _kind_matches(left, right):
    """Check whether two kind constants refer to the same prediction kind."""
    return _kind_value(left) == _kind_value(right)


def _iter_known_kinds(kind_source=Kind):
    """Enumerate mhctools kind constants across enum and string-class APIs."""
    try:
        candidates = list(kind_source)
    except TypeError:
        candidates = [
            value
            for name, value in vars(kind_source).items()
            if not name.startswith("_") and isinstance(value, str)
        ]

    seen = set()
    kinds = []
    for kind in candidates:
        name = _kind_name(kind)
        if name in seen:
            continue
        seen.add(name)
        kinds.append(kind)
    return kinds


def _build_kind_aliases(kind_source=Kind):
    """Build parser aliases for the currently installed mhctools Kind API."""
    aliases = {}
    for kind in _iter_known_kinds(kind_source):
        name = _kind_name(kind).lower()
        aliases[name] = kind
        aliases[_kind_short_name(kind)] = kind
    aliases["el"] = kind_source.pMHC_presentation
    aliases["ba"] = kind_source.pMHC_affinity
    aliases["aff"] = kind_source.pMHC_affinity
    aliases["ic50"] = kind_source.pMHC_affinity
    aliases["processing"] = kind_source.antigen_processing
    return aliases


# =============================================================================
# Group key detection and EvalContext
# =============================================================================


def _missing_column_error(col_names, available, label="Column"):
    """ValueError for referenced column(s) that aren't in the DataFrame.

    Takes one name or several; naming all of them at once saves the
    caller a round-trip per typo.
    """
    names = (
        list(col_names) if isinstance(col_names, (list, tuple, set))
        else [col_names]
    )
    # Suggest only real string labels — a frame is allowed non-string
    # column labels, and str()ing them would suggest a '7' that isn't a
    # usable key.  The displayed list is stringified so it can be sorted.
    string_labels = sorted(c for c in available if isinstance(c, str))

    def suggest(name):
        if not isinstance(name, str):
            return []
        return get_close_matches(name, string_labels, n=3, cutoff=0.6)

    if len(names) == 1:
        msg = f"{label} {names[0]!r} not found in DataFrame."
        close = suggest(names[0])
        if close:
            return ValueError(msg + f" Did you mean: {close}?")
        return ValueError(
            msg + f" Available columns: {sorted(str(c) for c in available)}"
        )

    msg = f"{label}s {names!r} not found in DataFrame."
    hints = [f"{n!r} -> {close}" for n in names if (close := suggest(n))]
    if hints:
        return ValueError(msg + f" Did you mean: {'; '.join(hints)}?")
    return ValueError(
        msg + f" Available columns: {sorted(str(c) for c in available)}"
    )


_SAMPLE_GROUP_KEY = "sample_name"
#: Column holding the allele set a genotype-level prediction was scored
#: against, comma-joined and sorted. Populated only for kinds whose
#: ``mhc_dependence`` is ``haplotype``; blank elsewhere.
ALLELE_SET_COLUMN = "allele_set"
#: Group-key columns that describe *which alleles* a row is about.
#: Everything else in the group key identifies the peptide.
_ALLELE_DIMENSION_KEYS = ("allele", ALLELE_SET_COLUMN)
_ALLELE_SET_SEPARATOR = ","
_GROUP_KEYS = ["source_sequence_name", "peptide", "peptide_offset", "allele"]
_GROUP_KEYS_VARIANT = ["variant", "peptide", "peptide_offset", "allele"]


_GROUP_KEYS_FRAGMENT = ["fragment_id", "peptide", "peptide_offset", "allele"]


def _is_blank(value) -> bool:
    """True for null or whitespace-only — "this cell carries no value"."""
    return not is_stated(value)


def _has_real_values(values) -> bool:
    """True when *values* holds at least one non-null, non-blank entry.

    ``mhctools`` stamps ``""`` on columns a run carries no value for —
    ``sample_name`` on a single-sample run, ``allele`` on an
    allele-independent prediction — so a column's presence says nothing
    about whether it carries identity.  One definition of "blank" for
    every caller that has to make that distinction.
    """
    # The columns this is asked about are near-constant, so the distinct
    # set is tiny: one hashing pass, then a test over a couple of values
    # rather than every row.
    return any(not _is_blank(v) for v in pd.unique(values.to_numpy()))


def split_allele_set(value):
    """Parse an ``allele_set`` cell into a list of allele names.

    Splitting on the separator is not optional: allele names prefix one
    another (``HLA-A*02:01`` is a prefix of ``HLA-A*02:010``, and
    mhcgnomes parses both as real, distinct alleles), so a substring
    test against the joined string reports false membership.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [
        part.strip() for part in text.split(_ALLELE_SET_SEPARATOR)
        if part.strip()
    ]


def format_allele_set(alleles):
    """Render alleles as a canonical ``allele_set`` cell.

    Sorted so two predictions over the same set compare and hash equal
    regardless of the order the caller listed them in.
    """
    names = sorted({str(a).strip() for a in alleles if str(a).strip()})
    return _ALLELE_SET_SEPARATOR.join(names)


def _peptide_keys(group_keys):
    """The group keys that identify the peptide rather than the alleles."""
    return [k for k in group_keys if k not in _ALLELE_DIMENSION_KEYS]


def _with_optional_allele_set_key(df, group_keys):
    """Add ``allele_set`` to the group key when the frame populates it.

    A genotype-level row names one allele (the predictor's deconvolved
    best presenter) but is not *about* that allele, so it must not share
    a group with that allele's own predictions.  Keying on the set
    separates them while leaving ``allele`` readable.  Frames with no
    genotype-level rows keep their narrower key, the same way a blank
    ``sample_name`` is left out.
    """
    group_keys = list(group_keys)
    if (
        ALLELE_SET_COLUMN in df.columns
        and ALLELE_SET_COLUMN not in group_keys
        and "allele" in group_keys
        and _has_real_values(df[ALLELE_SET_COLUMN])
    ):
        group_keys.insert(group_keys.index("allele") + 1, ALLELE_SET_COLUMN)
    return group_keys


def _with_optional_sample_key(df, group_keys):
    group_keys = list(group_keys)
    if (
        _SAMPLE_GROUP_KEY in df.columns
        and _SAMPLE_GROUP_KEY not in group_keys
        and _has_real_values(df[_SAMPLE_GROUP_KEY])
    ):
        group_keys.insert(0, _SAMPLE_GROUP_KEY)
    return group_keys


def _pick_group_keys(df):
    # fragment_id is the most specific identity (from predict_from_fragments);
    # variant is for the legacy varcode pipeline; source_sequence_name is the
    # generic fallback.
    if "fragment_id" in df.columns:
        keys = _GROUP_KEYS_FRAGMENT
    elif "variant" in df.columns:
        keys = _GROUP_KEYS_VARIANT
    else:
        keys = _GROUP_KEYS
    _check_inferred_group_keys(df, keys)
    return _with_optional_sample_key(df, _with_optional_allele_set_key(df, keys))


def _check_inferred_group_keys(df, keys):
    """Explain a frame that inference can't key, instead of a KeyError.

    Inference works from a fixed set of identity columns.  A frame
    missing one of them used to reach ``df[self.group_keys]`` and raise
    a bare ``KeyError``; say what's missing and point at the way out.
    """
    if len(df.columns) == 0:
        # Nothing to infer from at all (e.g. ``pd.DataFrame()``); there is
        # no identity to get wrong, so stay quiet.
        return
    missing = [k for k in keys if k not in df.columns]
    if not missing:
        return
    raise ValueError(
        f"Cannot infer group keys: this DataFrame is missing "
        f"{missing!r}, expected alongside "
        f"{[k for k in keys if k in df.columns]!r}. "
        f"Pass group_keys=[...] to name the group identity explicitly."
    )


def _normalize_group_keys(df, group_keys):
    """Validate explicit *group_keys* against *df* and return them as a list.

    Fails loudly on a bare string, an empty sequence, duplicate entries,
    or names that aren't columns, so a caller with a stable provenance
    identity finds out here rather than through a KeyError deep inside a
    node.
    """
    if isinstance(group_keys, str):
        raise ValueError(
            f"group_keys must be a sequence of column names, not the string "
            f"{group_keys!r}; pass [{group_keys!r}] for a single key."
        )
    keys = list(group_keys)
    if not keys:
        raise ValueError(
            "group_keys must be a non-empty sequence of column names; "
            "pass group_keys=None to infer them from the DataFrame."
        )
    # Compare by repr, not ==: a key may be array-like, whose __eq__
    # returns an array and would raise before the real error is reached.
    counts = Counter(repr(k) for k in keys)
    duplicates = sorted(k for k, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"group_keys has duplicate entries: {duplicates}")
    for key in keys:
        try:
            hash(key)
        except TypeError:
            raise ValueError(
                f"group_keys entries must be column names, got a "
                f"{type(key).__name__}: {key!r}"
            ) from None
        if key not in df.columns:
            raise _missing_column_error(
                key, df.columns, label="group_keys column",
            )
    return keys


class _PeptideAlleleLookup:
    """Resolve the alleles declared for one peptide.

    A mapping may be keyed by the full peptide-key tuple, or — when its
    keys are not tuples — by the ``peptide`` value alone, which is the
    spelling a caller reaches for first. Whichever it is, a key that
    matches no peptide in the frame is an error rather than a silent
    no-op: a typo'd peptide would otherwise declare nothing and look
    exactly like a peptide deliberately left undeclared.
    """

    __slots__ = ("_source", "_peptide_keys", "_by_peptide_column", "_seen")

    def __init__(self, source, peptide_keys):
        self._source = source
        self._peptide_keys = list(peptide_keys)
        self._seen = set()
        self._by_peptide_column = False
        if isinstance(source, Mapping) and source:
            tupled = all(isinstance(k, tuple) for k in source)
            if not tupled:
                if "peptide" not in self._peptide_keys:
                    raise ValueError(
                        "alleles mapping is keyed by peptide, but 'peptide' "
                        "is not one of the group keys "
                        f"({self._peptide_keys}). Key it by the group-key "
                        "tuple instead."
                    )
                self._by_peptide_column = True

    def for_peptide(self, row):
        """Alleles declared for the peptide described by *row*, or ``()``."""
        if callable(self._source):
            declared = self._source(dict(row))
        else:
            key = (
                row["peptide"] if self._by_peptide_column
                else tuple(row[k] for k in self._peptide_keys)
            )
            if key not in self._source:
                return ()
            self._seen.add(key)
            declared = self._source[key]
        if declared is None:
            return ()
        if isinstance(declared, str):
            raise ValueError(
                f"alleles for a peptide must be a sequence of allele "
                f"names, not the string {declared!r}; pass "
                f"[{declared!r}] for a single allele."
            )
        declared = list(declared)
        if not declared:
            # Empty is meaningful per peptide — "declare nothing for this
            # one" — where an empty frame-wide sequence is a mistake, so
            # it must not go through the frame-level check.
            return ()
        return _normalize_alleles(declared) or ()

    def check_all_used(self):
        """Raise if a mapping entry named a peptide the frame lacks."""
        if callable(self._source):
            return
        unused = sorted(
            (repr(k) for k in self._source if k not in self._seen),
            key=str,
        )
        if unused:
            raise ValueError(
                f"alleles mapping names {len(unused)} peptide(s) not in the "
                f"frame: {unused[:5]}. A key that matches nothing declares "
                f"nothing, which is indistinguishable from a peptide left "
                f"undeclared on purpose."
            )


def _normalize_alleles(alleles):
    """Validate a declared allele set and return it as a list (or None).

    A mapping or callable is per-peptide and is kept as-is; only the
    flat "same set for every peptide" form is materialized here.
    """
    if alleles is None:
        return None
    if callable(alleles) or isinstance(alleles, Mapping):
        return alleles
    if isinstance(alleles, str):
        raise ValueError(
            f"alleles must be a sequence of allele names, not the string "
            f"{alleles!r}; pass [{alleles!r}] for a single allele."
        )
    declared = list(alleles)
    if not declared:
        raise ValueError(
            "alleles must be a non-empty sequence of allele names; pass "
            "alleles=None to use only the alleles present in the DataFrame."
        )
    blank = [a for a in declared if not is_stated(a)]
    if blank:
        raise ValueError(
            "alleles must all name an allele; got a blank entry. An "
            "allele-free prediction is expressed by leaving the row's "
            "allele empty, not by declaring a blank allele."
        )
    return declared


#: Order used to pick one model when a kind has several and the caller
#: hasn't said which.  This is a **tie-break convention, not a quality
#: ranking** — topiary is not asserting that one predictor is better
#: than another.  It orders general-purpose predictors ahead of ones
#: whose output for a kind is secondary to their main job (NetMHCstabPan
#: predicts stability; its affinity comes along with it), and mode
#: variants after the model they vary.  Anything unlisted sorts
#: alphabetically after these, so the answer is always deterministic.
#:
#: Callers who care which model answers should qualify the reference —
#: ``Affinity["netmhcpan"]`` — or pass their own *preference*.
CANONICAL_METHOD_PREFERENCE = (
    "mhcflurry",
    "netmhcpan",
    "netmhcpan_ba",
    "netmhcpan_el",
    "netmhcstabpan",
)


def resolve_default_methods(df, preference=None):
    """Pick one ``prediction_method_name`` per kind that has several.

    Unqualified references raise when a kind was produced by more than
    one model, which is the right default — silently choosing a model is
    not something topiary should do behind a caller's back.  This is the
    explicit way to say "pick the canonical one", so consumers stop
    writing their own preference table and disagreeing with each other
    about what canonical means.

    Kinds with a single model are omitted: the result only speaks where
    there is a real choice.  Returns a mapping suitable for
    ``default_methods=``.

    *preference* overrides :data:`CANONICAL_METHOD_PREFERENCE`.
    """
    if df is None or df.empty:
        return {}
    needed = {"kind", "prediction_method_name"}
    if not needed.issubset(df.columns):
        return {}
    order = list(CANONICAL_METHOD_PREFERENCE if preference is None else preference)
    rank = {name.lower(): i for i, name in enumerate(order)}

    resolved = {}
    for kind_value, group in df.groupby("kind", sort=False, dropna=True):
        methods = sorted({
            str(m) for m in group["prediction_method_name"].dropna().unique()
        })
        if len(methods) < 2:
            continue
        resolved[str(kind_value)] = min(
            methods, key=lambda m: (rank.get(m.lower(), len(order)), m.lower()),
        )
    return resolved


def resolve_default_versions(df, prefer="newest"):
    """Pick one ``predictor_version`` per (kind, model) that has several.

    The version-level counterpart of :func:`resolve_default_methods`, and
    for the same reason: raising on ambiguity is the right default, but a
    consumer needs one supported way to say "use the newest" rather than
    each inventing its own preference table and disagreeing about what
    newest means.

    Ordering is PEP 440 where the versions parse as PEP 440, which is the
    only ordering under which ``4.10`` beats ``4.9``; anything unparseable
    sorts before everything that parses, and ties break on the string, so
    the answer is always deterministic. Pass ``prefer="oldest"`` for a
    pipeline pinned to an older validated model.

    Pairs with a single version are omitted: the result only speaks where
    there is a real choice. Returns a mapping suitable for
    ``default_versions=``.

    :func:`describe_default_versions` returns the candidates this chose
    between, for telling a user what was ambiguous and what won.
    """
    if prefer not in ("newest", "oldest"):
        raise ValueError(
            f"prefer must be 'newest' or 'oldest', got {prefer!r}"
        )
    if df is None or df.empty:
        return {}
    needed = {"kind", "prediction_method_name", "predictor_version"}
    if not needed.issubset(df.columns):
        return {}

    return {
        key: (candidates[-1] if prefer == "newest" else candidates[0])
        for key, candidates in describe_default_versions(df).items()
    }


def describe_default_versions(df):
    """The versions :func:`resolve_default_versions` is choosing between.

    ``{(kind, model): [version, ...]}`` for every pair a model produced at
    more than one version, ordered oldest to newest by the same PEP 440
    rule the resolver uses — so the winner is the last entry for
    ``prefer="newest"`` and the first for ``prefer="oldest"``.

    Picking a version silently is right for scoring but not something to
    do quietly to a user. Telling them "netMHCpan reports 4.1b and 4.2,
    scoring with 4.2" needs what the winner won *against*, and deriving
    that from the frame means re-implementing "was a version named at
    all" — the rule whose subtlety is the whole reason a caller should
    not be writing it. A blank, ``None``, ``NaN`` or the literal string
    ``"nan"`` is not a version, here as everywhere else.

    Keys match :func:`resolve_default_versions` exactly, so the two zip.
    """
    if df is None or df.empty:
        return {}
    needed = {"kind", "prediction_method_name", "predictor_version"}
    if not needed.issubset(df.columns):
        return {}

    described = {}
    grouped = df.dropna(subset=["prediction_method_name"]).groupby(
        ["kind", "prediction_method_name"], sort=False, dropna=True,
    )
    for (kind_value, method), group in grouped:
        known = group["predictor_version"][
            _known_versions(group["predictor_version"])
        ]
        versions = {str(v).strip() for v in known.unique()}
        if len(versions) < 2:
            continue
        described[(str(kind_value), str(method))] = sorted(
            versions, key=_version_sort_key,
        )
    return described


#: Text that means "no value was stated here".
#:
#: These are the spellings a missing value takes once anything calls
#: ``str()`` on it — which happens on ``astype(str)``, on a CSV round
#: trip, and on a cache export — plus the blank forms. A column that has
#: been through any of those carries ``"nan"`` where it once carried
#: ``NaN``, and a check that only tests for blankness lets it through as
#: a real value.
#: The text a missing value becomes under ``str()``.
#:
#: The empty string is deliberately **not** here: ``str(None)`` is
#: ``"None"``, never ``""``. A blank cell is a stated-but-empty value,
#: which some frames use as a group of its own (an allele-free
#: prediction row), so collapsing it into null would merge groups a
#: caller meant to keep apart.
NULL_TEXT = frozenset({"nan", "none", "<na>", "nat", "null"})

NOT_STATED = NULL_TEXT | {""}

#: Deprecated alias for :data:`NOT_STATED`. Versions were never special.
NOT_STATED_VERSIONS = NOT_STATED

_CONTAINER_TYPES = (pd.Series, pd.Index, np.ndarray, list, tuple, set)


def is_stated(value) -> bool:
    """Whether *value* is a value at all, rather than a missing one.

    **The one definition of "did the source say anything here?"** —
    used for allele names, predictor versions, kinds, method names and
    fragment fields alike. Nothing in topiary should write this test
    again; a second copy is how the two answers drift apart.

    Not stated: ``None``, ``NaN``, ``pd.NA``, ``pd.NaT``, the empty and
    whitespace-only strings, and every spelling in :data:`NOT_STATED` —
    which includes ``"nan"`` and ``"None"``, because those are what a
    missing value *becomes* the moment it is stringified.

    The obvious version of this test, ``if str(v).strip()``, excludes
    only the blank spellings: ``str(None)`` is ``"None"`` and
    ``str(float("nan"))`` is ``"nan"``, both truthy. So the naive rule
    admits most of the ways a value goes missing, and the mistake is
    invisible until a frame round-trips through text.

    Parameters
    ----------
    value : scalar
        One cell. Pass a Series to :func:`stated_values` instead; a
        container here raises rather than being silently truthy.

    Returns
    -------
    bool
        True when the value says something.

    Raises
    ------
    TypeError
        If *value* is a container. Answering "yes" for a column of
        missing values is the outcome this function exists to prevent.

    See Also
    --------
    stated_values : the same rule over a Series.
    is_named_version : this rule, named for the ``predictor_version`` case.

    Examples
    --------
    >>> is_stated("HLA-A*02:01")
    True
    >>> [is_stated(v) for v in (None, float("nan"), "", " ", "nan", "None")]
    [False, False, False, False, False, False]
    """
    if isinstance(value, _CONTAINER_TYPES):
        raise TypeError(
            f"is_stated takes one value, got {type(value).__name__}. "
            f"Use stated_values() for a Series."
        )
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() not in NOT_STATED


def stated_values(values) -> pd.Series:
    """:func:`is_stated` over a Series, as a boolean mask.

    The vectorized form of the same rule, built from the same
    :data:`NOT_STATED` set rather than restating the test, so the two
    cannot drift. ``tests/test_public_helpers.py`` asserts they agree
    across every spelling in the set.

    Parameters
    ----------
    values : pandas.Series

    Returns
    -------
    pandas.Series
        Boolean mask, True where the row states a value.
    """
    text = values.astype(str).str.strip().str.lower()
    return values.notna() & ~text.isin(NOT_STATED)


def is_named_version(value) -> bool:
    """Whether *value* names a predictor version.

    :func:`is_stated` under the name the ``predictor_version`` case
    reaches for. A missing version is the absence of a version claim,
    not a version called ``"nan"`` — treating it as one produces a
    phantom second version, and then an ambiguity error naming a version
    the caller cannot possibly pass.

    Examples
    --------
    >>> is_named_version("4.1b")
    True
    >>> [is_named_version(v) for v in (None, float("nan"), "", "nan")]
    [False, False, False, False]
    """
    return is_stated(value)


def known_versions(values) -> pd.Series:
    """:func:`stated_values` over a ``predictor_version`` column."""
    return stated_values(values)


def _known_versions(values) -> pd.Series:
    """Deprecated internal alias for :func:`known_versions`."""
    return stated_values(values)


def _version_sort_key(version):
    """PEP 440 order where possible, deterministic where not.

    ``packaging`` is imported at module scope rather than here so a
    missing install fails loudly. Catching ImportError alongside
    InvalidVersion would silently demote *every* version to string
    order, which is the ordering this function exists to avoid.
    """
    try:
        return (1, Version(str(version)), "")
    except InvalidVersion:
        # Not PEP 440 (a build tag, a date, a git hash). Sorting these
        # before every parseable version keeps "newest" meaning a real
        # release rather than whichever string happened to sort last.
        return (0, _UNPARSEABLE_VERSION, str(version))


class _UnparseableVersion:
    """Sorts equal to itself, so the string tiebreak decides."""

    __slots__ = ()

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, _UnparseableVersion)

    def __hash__(self):
        return hash(_UnparseableVersion)


_UNPARSEABLE_VERSION = _UnparseableVersion()


def _normalize_default_versions(mapping):
    """Canonicalize ``default_versions`` keys to ``(kind, model)`` pairs."""
    aliases = _build_kind_aliases()
    out = {}
    for key, version in mapping.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
        ):
            raise TypeError(
                f"default_versions keys must be (kind, model) pairs, "
                f"got {key!r}. A version is only meaningful within a "
                f"model, so the model has to be named."
            )
        kind_key, method = key
        if not isinstance(version, str):
            raise TypeError(
                f"default_versions[{key!r}] must be a version string, "
                f"got {type(version).__name__}"
            )
        if not isinstance(method, str):
            raise TypeError(
                f"default_versions[{key!r}] model must be a string, "
                f"got {type(method).__name__}"
            )
        kind_value = _canonical_kind_key(kind_key, aliases, "default_versions")
        out[(kind_value, method)] = version
    return out


def validate_default_versions(df, default_versions):
    """Raise if *default_versions* names a (kind, model) or version *df* lacks.

    Same rationale as :func:`validate_default_methods`: a default is only
    consulted when a reference is actually ambiguous, so an entry naming
    a model or version that never ran sits inert until the day it starts
    deciding.
    """
    normalized = _normalize_default_versions(default_versions)
    if df is None or df.empty:
        return
    needed = {"kind", "prediction_method_name", "predictor_version"}
    if not needed.issubset(df.columns):
        return
    for (kind_value, method), version in normalized.items():
        rows = df[
            (df["kind"] == kind_value)
            & (df["prediction_method_name"] == method)
        ]
        if rows.empty:
            continue
        available = sorted({
            str(v).strip() for v in rows["predictor_version"][
                _known_versions(rows["predictor_version"])
            ].unique()
        })
        if version not in available:
            raise ValueError(
                f"No {kind_value} predictions from {method} at "
                f"predictor_version {version!r}. Available: {available}"
            )


def validate_default_methods(df, default_methods):
    """Raise if *default_methods* names a kind or model *df* doesn't have.

    ``EvalContext`` only consults a default when a kind is actually
    ambiguous, so an entry naming a model that never ran is otherwise
    silently inert — and stays inert until the day two models do produce
    that kind, when it starts deciding.  Checking up front turns a typo
    in a config file into an error at the point it was written.
    """
    if not default_methods:
        return
    normalized = _normalize_default_methods(default_methods)
    if df is None or df.empty or "kind" not in df.columns:
        return
    if "prediction_method_name" not in df.columns:
        return
    for kind_value, method in normalized.items():
        rows = df[df["kind"] == kind_value]
        if rows.empty:
            continue
        available = sorted({
            str(m) for m in rows["prediction_method_name"].dropna().unique()
        })
        lowered = method.lower()
        if not any(lowered in name.lower() for name in available):
            raise _method_not_found_error(kind_value, method, available)


def _normalize_default_methods(mapping):
    """Canonicalize ``default_methods`` keys to DataFrame ``kind`` values.

    Accepts canonical names (``"pMHC_affinity"``), DSL short names
    (``"affinity"``, ``"ba"``, ``"el"``, ...), and mhctools ``Kind``
    constants.  Values are method strings passed through unchanged.
    """
    aliases = _build_kind_aliases()
    out = {}
    for key, method in mapping.items():
        if not isinstance(method, str):
            raise TypeError(
                f"default_methods[{key!r}] must be a method name string, "
                f"got {type(method).__name__}"
            )
        out[_canonical_kind_key(key, aliases, "default_methods")] = method
    return out


def _canonical_kind_key(key, aliases, label):
    """A kind spelling → its canonical ``kind`` value, or a helpful raise."""
    kind = aliases.get(_kind_name(key).lower())
    if kind is None:
        # Surface the canonical kind values and every DSL short alias so
        # a user who typed 'banana' sees that 'ba' / 'affinity' /
        # 'pMHC_affinity' all map to the same kind. The alias dict is
        # lower-cased for case-insensitive lookup; skip lower-case
        # duplicates of canonicals to keep the list readable.
        canonical = {_kind_value(k) for k in aliases.values()}
        canonical_lower = {c.lower() for c in canonical}
        shorts = {a for a in aliases.keys() if a not in canonical_lower}
        accepted = sorted(shorts | canonical)
        raise ValueError(
            f"{label} key {key!r} is not a known kind. "
            f"Accepted spellings: {accepted}"
        )
    return _kind_value(kind)


class EvalContext:
    """Context for vectorized DSL evaluation.

    Wraps a prediction DataFrame and exposes the unique group-key
    index.  Every :class:`DSLNode` ``.eval(ctx)`` returns a
    ``pd.Series`` indexed by :attr:`group_index` (one value per
    peptide-allele group).

    Parameters
    ----------
    df : pandas.DataFrame
        Prediction rows, long-form.
    alleles : sequence, mapping, or callable, optional
        Alleles to evaluate peptides against — a patient's genotype,
        typically.  Group keys otherwise come only from the rows, so a
        peptide whose evidence is allele-free has no per-allele group to
        read; declaring the set adds one group per peptide per allele
        for :func:`peptide_view` to broadcast into.  Added groups carry
        no rows, so allele-scoped fields read NaN there.

        Three forms, the same shape
        :func:`~topiary.from_predictions` takes for ``allele_set``:

        - a **sequence** — the same alleles for every peptide;
        - a **mapping** — per peptide, keyed by the ``peptide`` value or
          by the full peptide-key tuple;
        - a **callable** — receives a dict of the peptide keys, returns
          that peptide's alleles.

        Per-peptide sets matter when peptides were not each reported
        against the whole genotype, which is the normal case for a
        reader that emits one row per (peptide, allele) passing its own
        threshold.  Declaring the union there invents groups for
        pairings that were never scored, and an expression reading only
        peptide-level evidence gives every one of them a real number.

        A peptide the mapping or callable declares nothing for keeps
        only the groups its own rows name.  A mapping key matching no
        peptide in the frame raises, since a key that declares nothing
        is indistinguishable from a peptide left undeclared on purpose.
    group_keys : list of str, optional
        Override the auto-detected peptide-allele group keys.  Use this
        when the frame carries a stable provenance identity (e.g. a
        variant or prediction ID) that inferred sequence-oriented keys
        would collapse: two rows with the same peptide, context and
        offset but different origins stay in separate groups.  The names
        are validated against ``df`` up front.
        :func:`~topiary.ranking.apply_filter`,
        :func:`~topiary.ranking.apply_sort` and
        :func:`~topiary.ranking.evaluate_scores` forward the same kwarg,
        so filtering, sorting and scoring can share one grouping.
    default_methods : dict, optional
        Per-kind default ``prediction_method_name`` for resolving
        unqualified Field references when multiple methods produce
        the same kind.  Keys may be canonical kind names
        (``"pMHC_affinity"``), short names (``"affinity"``, ``"ba"``,
        ``"el"``, ...), or mhctools ``Kind`` constants.  Without
        this kwarg, ambiguous references raise ``ValueError`` — the
        safety behavior is preserved by default.

        Example::

            ctx = EvalContext(
                df,
                default_methods={
                    "pMHC_affinity": "mhcflurry",
                    "pMHC_stability": "netmhcstabpan",
                },
            )
    default_versions : dict, optional
        Per-(kind, model) default ``predictor_version``, for resolving
        unqualified references when one model produced the same kind at
        several versions.  Keyed on the pair because a version is only
        meaningful within a model: ``4.2`` says nothing on its own.
        Keys accept the same kind spellings as *default_methods*::

            EvalContext(
                df,
                default_versions={("pMHC_affinity", "netmhcpan"): "4.2"},
            )

        :func:`resolve_default_versions` builds one for the common
        "newest wins" case.  Without this kwarg an ambiguous version
        raises, as an ambiguous method does.
    filter_context : bool, optional
        When true, directional ``Comparison`` nodes
        (``<``, ``<=``, ``>``, ``>=``) with unqualified same-kind refs
        auto-aggregate across methods (nanmin for ``<``/``<=``, nanmax
        for ``>``/``>=``) instead of raising on ambiguity.
        :func:`apply_filter` sets this to ``True`` automatically;
        :func:`apply_sort` leaves it ``False`` so sort stays strict.
    """

    __slots__ = (
        "df", "group_keys", "default_methods", "default_versions",
        "filter_context", "kind_support", "alleles",
        "_group_index", "_key_frame", "_group_tuples_cache",
        "_group_codes_cache", "_method_override",
    )

    def __init__(
        self, df, group_keys=None, default_methods=None, filter_context=False,
        kind_support=None, alleles=None, default_versions=None,
    ):
        self.df = df
        if group_keys is None:
            self.group_keys = _pick_group_keys(df)
        else:
            self.group_keys = _normalize_group_keys(df, group_keys)
        self.default_methods = (
            _normalize_default_methods(default_methods) if default_methods else {}
        )
        self.default_versions = (
            _normalize_default_versions(default_versions)
            if default_versions else {}
        )
        self.filter_context = filter_context
        # mhctools >=3.13.7 per-(model, kind) metadata. Optional; when
        # provided (typically from ``TopiaryPredictor.kind_support``),
        # nodes that care about allele dependence (e.g.
        # :class:`BestAlleleField`) can warn or branch on it. Shape:
        # ``{model_key: {kind_value: {"mhc_dependence", "mhc_class"}}}``.
        self.kind_support = kind_support
        self.alleles = _normalize_alleles(alleles)
        self._group_index = None
        self._key_frame = None
        self._group_tuples_cache = None
        self._group_codes_cache = None
        # Internal: when Comparison auto-aggregates across methods, it
        # binds Field(method=None, kind=K) references to a specific
        # method per iteration by setting (kind_value, method_name) here.
        self._method_override = None

    def derive(self, **overrides) -> "EvalContext":
        """A context on the same frame with some options changed.

        The expensive part of a context is frame-derived — the key
        frame, the unique group index, the row→group codes — and none of
        it depends on ``default_methods``, ``filter_context`` or
        ``kind_support``.  So a derived context that leaves ``df``,
        ``group_keys`` and ``alleles`` alone inherits those caches
        outright rather than recomputing them, which is what lets
        :func:`~topiary.ranking.apply_filter` (which needs
        ``filter_context=True``) and :func:`~topiary.ranking.apply_sort`
        (which needs it ``False``) share one grouping.

        Overriding a frame-shaping option is allowed but drops the
        caches, since they would no longer describe the result.
        """
        unknown = set(overrides) - {
            "df", "group_keys", "default_methods", "default_versions",
            "filter_context", "kind_support", "alleles",
        }
        if unknown:
            raise TypeError(
                f"Unknown EvalContext option(s): {sorted(unknown)}"
            )
        df = overrides.get("df", self.df)
        group_keys = overrides.get("group_keys", self.group_keys)
        alleles = overrides.get("alleles", self.alleles)
        derived = EvalContext(
            df,
            group_keys=group_keys,
            default_methods=overrides.get(
                "default_methods", self.default_methods or None
            ),
            default_versions=overrides.get(
                "default_versions", self.default_versions or None
            ),
            filter_context=overrides.get(
                "filter_context", self.filter_context
            ),
            kind_support=overrides.get("kind_support", self.kind_support),
            alleles=alleles,
        )
        reshaped = (
            df is not self.df
            or list(group_keys) != list(self.group_keys)
            or list(alleles or ()) != list(self.alleles or ())
        )
        if not reshaped:
            derived._key_frame = self._key_frame
            derived._group_index = self._group_index
            derived._group_tuples_cache = self._group_tuples_cache
            derived._group_codes_cache = self._group_codes_cache
        return derived

    @property
    def key_frame(self) -> pd.DataFrame:
        """The group-key columns, with every not-stated spelling collapsed.

        ``groupby(dropna=False)`` — which every node evaluates through —
        treats ``None``, ``NaN`` and ``pd.NA`` in an object column as one
        group, but a plain ``drop_duplicates`` keeps them apart.  Building
        the group index from raw values therefore produces groups no node
        result can ever key, and rows would silently score NaN.  Collapse
        them once here so the index, the row mapping and every node agree.

        The collapse covers :data:`NULL_TEXT` as well as real nulls, so
        a key that went through ``astype(str)`` somewhere — carrying
        ``"nan"`` where it carried ``NaN`` — lands in the null group
        rather than becoming an allele literally named ``"nan"``.  A
        blank string is left alone: it is a stated-but-empty value, and
        frames use it as a group of its own.
        """
        if self._key_frame is None:
            frame = self.df[self.group_keys]
            object_keys = [
                k for k in self.group_keys if frame[k].dtype == object
            ]
            replacements = {}
            for key in object_keys:
                column = frame[key]
                # Null, or the *text* of a null. Not blank: a blank cell
                # is a stated-but-empty value and stays its own group.
                missing = column.isna() | column.astype(str).str.strip(
                ).str.lower().isin(NULL_TEXT)
                if missing.any():
                    replacements[key] = column.where(~missing, np.nan)
            if replacements:
                frame = frame.assign(**replacements)
            self._key_frame = frame
        return self._key_frame

    @property
    def group_index(self) -> pd.Index:
        """Index of unique group keys, preserving row order.

        A MultiIndex of key tuples, or a flat Index of bare values when
        there is a single group key.  That mirrors what
        ``DataFrame.groupby`` produces, so node results — which are all
        groupby output reindexed onto this — align in both cases.  A
        1-level MultiIndex here would silently reindex to all-NaN
        against a groupby's flat Index.
        """
        if self._group_index is None:
            single_key = len(self.group_keys) == 1
            if self.df.empty:
                if single_key:
                    self._group_index = pd.Index([], name=self.group_keys[0])
                else:
                    self._group_index = pd.MultiIndex(
                        levels=[[] for _ in self.group_keys],
                        codes=[[] for _ in self.group_keys],
                        names=self.group_keys,
                    )
            else:
                key_df = self._declared_key_frame()
                if single_key:
                    key = self.group_keys[0]
                    self._group_index = pd.Index(key_df[key], name=key)
                else:
                    self._group_index = pd.MultiIndex.from_frame(key_df)
        return self._group_index

    def _declared_key_frame(self):
        """Unique group keys, extended by any declared allele set.

        Rows only ever produce the groups they name, so a peptide whose
        evidence is allele-free — an antigen-processing prediction, with
        nothing in its ``allele`` column — has no per-allele group for a
        consumer to read.  Declaring ``alleles`` (a patient's genotype,
        say) adds a group per peptide per allele, so
        :class:`PeptideView` has somewhere to broadcast the peptide's
        value to.  The added groups hold no rows: allele-scoped fields
        read NaN for them, which is the truth — that allele has no
        prediction of its own.
        """
        key_df = self.key_frame.drop_duplicates()
        peptide_keys = _peptide_keys(self.group_keys)
        if self.alleles is None or "allele" not in self.group_keys:
            return key_df
        per_peptide = callable(self.alleles) or isinstance(
            self.alleles, Mapping
        )
        if not per_peptide and not self.alleles:
            return key_df
        if not peptide_keys:
            if per_peptide:
                raise ValueError(
                    "Per-peptide alleles need a peptide in the group key, "
                    "but the group key is allele-only. Pass a flat "
                    "sequence instead."
                )
            return pd.DataFrame({"allele": self.alleles}).drop_duplicates()
        declared = self._declared_allele_rows(key_df, peptide_keys)
        if declared is None:
            return key_df
        if ALLELE_SET_COLUMN in self.group_keys:
            # A declared group is an ordinary per-allele group; it makes
            # no claim about a genotype the predictor scored.
            declared[ALLELE_SET_COLUMN] = ""
        # Observed groups first so row order is preserved; the declared
        # extras follow, and duplicates of observed ones fall away.
        return pd.concat(
            [key_df, declared[self.group_keys]], ignore_index=True,
        ).drop_duplicates()

    def _declared_allele_rows(self, key_df, peptide_keys):
        """Peptide keys crossed with the alleles declared for each.

        A flat sequence declares one set for every peptide. A mapping or
        callable declares a set per peptide, which is what a frame whose
        peptides were each reported against a different subset of a
        genotype needs: crossing every peptide with the union would
        invent per-allele groups for pairings that were never scored,
        and an expression reading only peptide-level evidence gives each
        of those a real number.

        Returns ``None`` when nothing was declared for any peptide.
        """
        peptides = key_df[peptide_keys].drop_duplicates()
        if not (callable(self.alleles) or isinstance(self.alleles, Mapping)):
            return peptides.merge(
                pd.DataFrame({"allele": list(self.alleles)}), how="cross",
            )

        lookup = _PeptideAlleleLookup(self.alleles, peptide_keys)
        blocks = []
        for row in peptides.to_dict("records"):
            declared = lookup.for_peptide(row)
            if not declared:
                # No declaration for this peptide: it keeps only the
                # groups its rows actually name. Silence here is the
                # point — an undeclared peptide must not inherit
                # another's genotype.
                continue
            block = pd.DataFrame({"allele": declared})
            for key in peptide_keys:
                block[key] = row[key]
            blocks.append(block)
        lookup.check_all_used()
        if not blocks:
            return None
        return pd.concat(blocks, ignore_index=True)

    def row_group_codes(self) -> np.ndarray:
        """Position within :attr:`group_index` of each row's group.

        The way to map a per-group Series back onto rows.  Prefer this
        over matching :meth:`row_group_tuples` against a set of keys:
        ``NaN`` never equals itself, and since Python 3.10 hashes by
        identity, so key lookups drop rows whose group key is null.
        Positions sidestep both.
        """
        if self._group_codes_cache is None:
            if self.df.empty:
                self._group_codes_cache = np.empty(0, dtype=int)
            else:
                if len(self.group_keys) == 1:
                    key = self.group_keys[0]
                    row_index = pd.Index(self.key_frame[key], name=key)
                else:
                    row_index = pd.MultiIndex.from_frame(self.key_frame)
                codes = self.group_index.get_indexer(row_index)
                assert (codes >= 0).all(), (
                    "internal: row group key missing from group_index"
                )
                self._group_codes_cache = codes
        return self._group_codes_cache

    def row_group_tuples(self) -> pd.Series:
        """Per-row group key, aligned to ``self.df.index``.

        A tuple of group-key values per row — or the bare value when
        there is a single group key, matching :attr:`group_index`.
        For mapping group results back onto rows, use
        :meth:`row_group_codes`: null keys make key-based lookups
        unreliable.
        """
        if self._group_tuples_cache is None:
            if self.df.empty:
                self._group_tuples_cache = pd.Series(
                    [], index=self.df.index, dtype=object
                )
            elif len(self.group_keys) == 1:
                self._group_tuples_cache = pd.Series(
                    self.key_frame[self.group_keys[0]].to_numpy(),
                    index=self.df.index,
                )
            else:
                self._group_tuples_cache = pd.Series(
                    list(zip(*[self.key_frame[k] for k in self.group_keys])),
                    index=self.df.index,
                )
        return self._group_tuples_cache

    def empty_series(self, fill=np.nan) -> pd.Series:
        """A Series of ``fill`` indexed by this context's group_index."""
        return pd.Series(fill, index=self.group_index, dtype=float)


# =============================================================================
# DSLNode — unified base class
# =============================================================================


class DSLNode:
    """Base class for all DSL nodes.

    Subclasses override :meth:`eval` to return a ``pd.Series`` indexed by
    ``ctx.group_index``.  Arithmetic and comparison/boolean operators
    produce composite nodes — the tree is built lazily and evaluated on
    demand.
    """

    # -- subclass contract --

    def eval(self, ctx: EvalContext) -> pd.Series:
        raise NotImplementedError

    def child_nodes(self) -> "list[DSLNode]":
        """Direct DSLNode children of this node.

        Leaves return ``[]``. Composite nodes return their sub-nodes in
        a stable order.  Used by generic tree walkers (e.g. column
        validation) so adding a new node type doesn't require touching
        every walker.
        """
        return []

    def to_expr_string(self) -> str:
        """Parseable DSL expression string.

        ``parse(node.to_expr_string())`` must produce a functionally
        equivalent tree for every DSLNode type.
        """
        return repr(self)

    def to_ast_string(self) -> str:
        """Canonical structural AST string for debugging / hashing."""
        return repr(self)

    # -- scalar convenience for tests and single-group frames --

    def evaluate(self, df):
        """Scalar convenience wrapper over :meth:`eval`.

        For a DataFrame with a single group, returns the scalar value.
        For empty or all-NaN inputs returns ``float("nan")``.  This is
        mainly for test-suite ergonomics; production code should build
        an :class:`EvalContext` once and call ``eval(ctx)`` directly.
        """
        if df is None:
            return float("nan")
        if isinstance(df, pd.DataFrame) and df.empty:
            return float("nan")
        result = self.eval(EvalContext(df))
        if len(result) == 0:
            return float("nan")
        val = result.iloc[0]
        if val is None:
            return float("nan")
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, float) and math.isnan(val):
            return float("nan")
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

    # -- arithmetic --

    def __add__(self, other):
        return BinOp(self, _as_node(other), operator.add)

    def __radd__(self, other):
        return BinOp(_as_node(other), self, operator.add)

    def __sub__(self, other):
        return BinOp(self, _as_node(other), operator.sub)

    def __rsub__(self, other):
        return BinOp(_as_node(other), self, operator.sub)

    def __mul__(self, other):
        return BinOp(self, _as_node(other), operator.mul)

    def __rmul__(self, other):
        return BinOp(_as_node(other), self, operator.mul)

    def __truediv__(self, other):
        return BinOp(self, _as_node(other), operator.truediv)

    def __rtruediv__(self, other):
        return BinOp(_as_node(other), self, operator.truediv)

    def __neg__(self):
        return BinOp(Const(-1), self, operator.mul)

    def __abs__(self):
        return UnaryOp(self, abs)

    def __pow__(self, other):
        return BinOp(self, _as_node(other), operator.pow)

    def __rpow__(self, other):
        return BinOp(_as_node(other), self, operator.pow)

    # -- comparison — return Comparison (a DSLNode) --

    def __le__(self, other):
        return Comparison(self, operator.le, _as_node(other))

    def __ge__(self, other):
        return Comparison(self, operator.ge, _as_node(other))

    def __lt__(self, other):
        return Comparison(self, operator.lt, _as_node(other))

    def __gt__(self, other):
        return Comparison(self, operator.gt, _as_node(other))

    # NOTE: __eq__ / __ne__ intentionally not overridden so DSLNodes
    # remain hashable and usable in sets/dicts.  Users who want an
    # equality filter can build Comparison(node, operator.eq, ...).

    # -- boolean composition --

    def __and__(self, other):
        return _combine_bool(operator.and_, self, _as_node(other))

    def __rand__(self, other):
        return _combine_bool(operator.and_, _as_node(other), self)

    def __or__(self, other):
        return _combine_bool(operator.or_, self, _as_node(other))

    def __ror__(self, other):
        return _combine_bool(operator.or_, _as_node(other), self)

    def __invert__(self):
        return BoolOp(operator.invert, [self])

    # -- transforms --

    def ascending_cdf(self, mean=0.0, std=1.0):
        """Gaussian left CDF: higher input → higher output."""
        return NormExpr(self, mean, std)

    norm = ascending_cdf

    def descending_cdf(self, mean=0.0, std=1.0):
        """Gaussian survival (1 - CDF): lower input → higher output."""
        return SurvivalExpr(self, mean, std)

    def logistic(self, midpoint=0.0, width=1.0):
        """Logistic sigmoid: lower input → higher output.

        Returns the raw sigmoid ``1/(1+exp((x-m)/w))`` whose max
        approaches 1 only as ``x → -∞``. Use
        :meth:`logistic_normalized` when you want a proper ``[0, 1]``
        binder-quality score that reaches 1 for arbitrarily good inputs.
        """
        return LogisticExpr(self, midpoint, width)

    def logistic_normalized(self, midpoint=0.0, width=1.0):
        """Logistic rescaled to ``[0, 1]``: reaches 1 as ``x → -∞``."""
        return LogisticNormalizedExpr(self, midpoint, width)

    def clip(self, lo=None, hi=None):
        """Clamp value to [lo, hi]. None = unbounded."""
        return ClipExpr(self, lo, hi)

    def hinge(self):
        """``max(0, x)``. Zeroes out negative values."""
        return ClipExpr(self, lo=0, hi=None)

    def log(self):
        return UnaryOp(self, math.log)

    def log2(self):
        return UnaryOp(self, math.log2)

    def log10(self):
        return UnaryOp(self, math.log10)

    def log1p(self):
        return UnaryOp(self, math.log1p)

    def exp(self):
        return UnaryOp(self, math.exp)

    def sqrt(self):
        return UnaryOp(self, math.sqrt)


def _as_node(x):
    """Coerce scalars / KindAccessors to DSLNodes."""
    if isinstance(x, DSLNode):
        return x
    if isinstance(x, KindAccessor):
        return x.value
    if isinstance(x, bool):
        return Const(1.0 if x else 0.0)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return Const(float(x))
    raise TypeError(
        f"Cannot convert {type(x).__name__} to DSLNode (value: {x!r})"
    )


# =============================================================================
# Const / Column / Field / Len / Count — leaves
# =============================================================================


class Const(DSLNode):
    """A constant scalar value."""

    __slots__ = ("val",)

    def __init__(self, val):
        self.val = float(val)

    def eval(self, ctx: EvalContext) -> pd.Series:
        return pd.Series(self.val, index=ctx.group_index, dtype=float)

    def __repr__(self):
        v = self.val
        if v == int(v):
            return str(int(v))
        return repr(v)

    def to_ast_string(self):
        return f"Const({_fmt_num(self.val)})"


def _fmt_num(v):
    """Format a number for repr: 500.0 → '500', 0.5 → '0.5'."""
    if v is None:
        return "None"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return repr(v)


class Column(DSLNode):
    """Reference an arbitrary column in the predictions DataFrame.

    Reads one value per peptide-allele group (first row per group).
    """

    __slots__ = ("col_name",)

    def __init__(self, col_name: str):
        self.col_name = col_name

    def eval(self, ctx: EvalContext) -> pd.Series:
        if ctx.df.empty:
            return ctx.empty_series()
        if self.col_name not in ctx.df.columns:
            raise _missing_column_error(self.col_name, ctx.df.columns)
        vals = ctx.df.groupby(
            ctx.group_keys, sort=False, dropna=False
        )[self.col_name].first()
        vals = vals.reindex(ctx.group_index)
        try:
            return vals.astype(float)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"Column {self.col_name!r} contains non-numeric values "
                f"({exc}). Only numeric columns can be used in DSL "
                f"expressions."
            ) from exc

    def __repr__(self):
        return f"column({self.col_name})"

    def to_ast_string(self):
        return f"Column({self.col_name!r})"

    # -- categorical / non-numeric equality --
    #
    # DSLNode intentionally doesn't override ``__eq__`` (keeps nodes
    # hashable for sets/dicts), and the column-eval path raises on
    # non-numeric values.  These helpers route categorical equality
    # (mhc_class, source, gene, ...) around both restrictions by
    # producing an ``IsIn`` node that reads the column raw.

    def eq(self, value) -> "IsIn":
        """Categorical equality: ``Column("mhc_class").eq("I")``."""
        return IsIn(self.col_name, [value])

    def ne(self, value) -> "IsIn":
        """Categorical inequality: ``Column("mhc_class").ne("II")``."""
        return IsIn(self.col_name, [value], negate=True)

    def includes(self, value) -> "Includes":
        """Membership in this column's delimited set — see :class:`Includes`."""
        return Includes(self.col_name, value)

    def isin(self, values) -> "IsIn":
        """Membership: ``Column("mhc_class").isin(["I", "II"])``."""
        return IsIn(self.col_name, values)


class Includes(DSLNode):
    """Membership test against a delimited set stored in one column.

    ``allele_set`` holds a comma-joined set, so asking whether a
    genotype-level prediction covers an allele is a membership question,
    not an equality one.  ``Column("allele").eq(x)`` keeps meaning
    exactly what it says — the row's ``allele`` is ``x`` — while
    ``Column("allele_set").includes(x)`` asks whether ``x`` is in the
    set the prediction was scored against.

    Comparison is by whole token, never substring: allele names prefix
    one another (``HLA-A*02:01`` is a prefix of ``HLA-A*02:010``, both
    real alleles), so a substring test reports membership that isn't
    there.  Tokens are compared as stored, so writers are responsible
    for canonical names — ``HLA-A*02:1`` will not match a stored
    ``HLA-A*02:01`` even though mhcgnomes considers them the same
    allele.
    """

    __slots__ = ("col_name", "value", "negate")

    def __init__(self, col_name: str, value, negate: bool = False):
        if not isinstance(value, str):
            raise TypeError(
                f"includes() takes one member name as a string, got "
                f"{type(value).__name__}. For several, combine with | ."
            )
        self.col_name = col_name
        self.value = value.strip()
        self.negate = negate

    def child_nodes(self):
        return []

    def __invert__(self):
        return Includes(self.col_name, self.value, negate=not self.negate)

    def eval(self, ctx: EvalContext) -> pd.Series:
        df = ctx.df
        if df.empty:
            return ctx.empty_series().astype("boolean")
        if self.col_name not in df.columns:
            raise _missing_column_error(self.col_name, df.columns)
        member = df[self.col_name].map(
            lambda cell: self.value in split_allele_set(cell)
        )
        if self.negate:
            member = ~member
        vals = member.groupby(
            [df[k] for k in ctx.group_keys], sort=False, dropna=False
        ).any()
        return vals.reindex(ctx.group_index).fillna(False).astype("boolean")

    def __repr__(self):
        prefix = "~" if self.negate else ""
        return f"{prefix}column({self.col_name}).includes({self.value!r})"

    def to_expr_string(self):
        return repr(self)

    def to_ast_string(self):
        return (
            f"Includes(col={self.col_name!r}, value={self.value!r}, "
            f"negate={self.negate})"
        )


class IsIn(DSLNode):
    """Categorical membership test against scalar values.

    Unlike :class:`Comparison`, which routes through :class:`Column`'s
    numeric-cast eval path, ``IsIn`` reads the column's raw Series and
    uses pandas ``.isin()`` directly — so it works for string columns
    (``mhc_class``, ``source``, ``gene``, ...), boolean columns, or any
    mix of dtypes.

    Construct directly or via :meth:`Column.eq` / :meth:`Column.ne` /
    :meth:`Column.isin`.  Negation through ``~`` or the ``negate=True``
    constructor kwarg.
    """

    __slots__ = ("col_name", "values", "negate")

    def __init__(self, col_name: str, values, negate: bool = False):
        if isinstance(values, (str, int, float, bool, type(None))):
            # Scalar — wrap so .isin gets a singleton.  float covers NaN
            # too (NaN is a float in Python's type system).
            values = (values,)
        else:
            try:
                values = tuple(values)
            except TypeError as exc:
                raise TypeError(
                    f"IsIn values must be a scalar or iterable, got "
                    f"{type(values).__name__}"
                ) from exc
        self.col_name = col_name
        self.values = values
        self.negate = negate

    def child_nodes(self):
        return []

    def eval(self, ctx: EvalContext) -> pd.Series:
        df = ctx.df
        if df.empty:
            # Mirror Column.eval's empty path so the result aligns with
            # ctx.group_index regardless of whether the context happens
            # to carry a stale non-empty index.
            return ctx.empty_series().astype("boolean")
        if self.col_name not in df.columns:
            raise _missing_column_error(self.col_name, df.columns)
        vals = df.groupby(
            ctx.group_keys, sort=False, dropna=False
        )[self.col_name].first()
        vals = vals.reindex(ctx.group_index)
        mask = vals.isin(self.values)
        if self.negate:
            mask = ~mask
        return mask

    def __invert__(self):
        return IsIn(self.col_name, self.values, negate=not self.negate)

    def __repr__(self):
        # Single-value: render as .eq(v) / .ne(v).  Multi-value: render
        # as .isin([...]) or ~.isin([...]) when negated.
        if len(self.values) == 1:
            method = "ne" if self.negate else "eq"
            return f"column({self.col_name}).{method}({self.values[0]!r})"
        prefix = "~" if self.negate else ""
        return f"{prefix}column({self.col_name}).isin({list(self.values)!r})"

    def to_ast_string(self):
        name = "NotIn" if self.negate else "In"
        return f"{name}({self.col_name!r}, {list(self.values)!r})"


def _filter_kind_method_version(ctx, kind, method, version):
    """Filter ``ctx.df`` to rows of a given kind/method/version.

    Returns the filtered sub-DataFrame, or ``None`` if no rows survived
    (callers should fall back to an empty Series). Raises on
    method-not-found, version-not-found, or unresolvable method
    ambiguity. Centralizes the filter logic shared between
    :class:`Field` and :class:`BestAlleleField`.
    """
    df = ctx.df
    if df.empty or "kind" not in df.columns:
        return None

    kind_val = _kind_value(kind)
    sub = df[df["kind"] == kind_val]
    if sub.empty:
        return None

    # Method binding: explicit method wins; else Comparison auto-agg
    # may have set ctx._method_override for our kind.
    effective_method = method
    if effective_method is None:
        override = ctx._method_override
        if override is not None and override[0] == kind_val:
            effective_method = override[1]

    # Filter by method substring (case-insensitive)
    if effective_method is not None:
        col = "prediction_method_name"
        if col in sub.columns:
            method_lower = effective_method.lower()
            method_mask = sub[col].str.lower().str.contains(
                method_lower, na=False, regex=False
            )
            matched = sub[method_mask]
            if matched.empty:
                available = sorted(sub[col].dropna().unique())
                raise _method_not_found_error(
                    _kind_name(kind), effective_method, available
                )
            sub = matched

    # Filter by exact version string
    if version is not None:
        col = "predictor_version"
        if col in sub.columns:
            # Only a row that names a version can be selected by one.
            # Comparing stringified values would let a missing version
            # be addressed as "nan" — the same conflation the ambiguity
            # check and resolve_default_versions were fixed for, and the
            # two must not disagree about it.
            named = _known_versions(sub[col])
            version_mask = named & (
                sub[col].astype(str).str.strip() == str(version).strip()
            )
            matched = sub[version_mask]
            if matched.empty:
                available = sorted(
                    sub[col][named].astype(str).str.strip().unique()
                )
                raise ValueError(
                    f"No {_kind_name(kind)} predictions from "
                    f"predictor_version {version!r}. "
                    f"Available: {available}"
                )
            sub = matched

    # Ambiguity: unqualified access with multiple methods in any group
    if effective_method is None and "prediction_method_name" in sub.columns:
        methods_per_group = sub.groupby(
            ctx.group_keys, sort=False, dropna=False
        )["prediction_method_name"].nunique()
        if (methods_per_group > 1).any():
            default = ctx.default_methods.get(_kind_value(kind))
            if default is not None:
                col = "prediction_method_name"
                default_lower = default.lower()
                method_mask = sub[col].str.lower().str.contains(
                    default_lower, na=False, regex=False
                )
                matched = sub[method_mask]
                if matched.empty:
                    available = sorted(sub[col].dropna().unique())
                    raise _method_not_found_error(
                        _kind_name(kind), default, available
                    )
                sub = matched
            else:
                method_list = ", ".join(
                    sorted(sub["prediction_method_name"].dropna().unique())
                )
                raise ValueError(
                    f"Ambiguous: multiple models produce "
                    f"{_kind_name(kind)} ({method_list}). "
                    f"Use {_kind_short_name(kind)}['modelname'] "
                    f"to disambiguate, or pass "
                    f"default_methods={{{_kind_value(kind)!r}: "
                    f"'modelname'}} to EvalContext."
                )

    # Same ambiguity one level down: a single method present at two
    # versions. Without this the groupby below would silently take
    # whichever row came first, which is an arbitrary choice between two
    # real predictions rather than an answer.
    # A configured default resolves the version the same way
    # ``default_methods`` resolves the model one level up.
    effective_version = version
    if (
        effective_version is None
        and ctx.default_versions
        and "prediction_method_name" in sub.columns
        and "predictor_version" in sub.columns
    ):
        methods_here = {
            str(m) for m in sub["prediction_method_name"].dropna().unique()
        }
        wanted = {
            ctx.default_versions[(kind_val, method)]
            for method in methods_here
            if (kind_val, method) in ctx.default_versions
        }
        if len(wanted) == 1:
            candidate = next(iter(wanted))
            versions = sub["predictor_version"]
            matched = sub[
                _known_versions(versions)
                & (versions.astype(str).str.strip() == candidate)
            ]
            if not matched.empty:
                sub = matched
                effective_version = candidate

    if (
        effective_version is None
        and "predictor_version" in sub.columns
        and "prediction_method_name" in sub.columns
    ):
        # Only rows that name a version can disagree about one. A frame
        # that simply does not record versions is not ambiguous, and
        # neither is one version plus rows that record none — otherwise
        # every reader that leaves predictor_version empty would start
        # raising, with nothing the caller could pass to resolve it.
        named = sub[_known_versions(sub["predictor_version"])]
        pairs = named[
            ["prediction_method_name", "predictor_version"]
        ].astype(str)
        counted = named[list(ctx.group_keys)].copy()
        counted["_pair"] = (
            pairs["prediction_method_name"] + "\x00"
            + pairs["predictor_version"]
        )
        pairs_per_group = counted.groupby(
            ctx.group_keys, sort=False, dropna=False,
        )["_pair"].nunique() if not counted.empty else pd.Series(dtype=int)
        if len(pairs_per_group) and (pairs_per_group > 1).any():
            listed = ", ".join(
                f"{m} {v}" for m, v in sorted(
                    {
                        (m, v) for m, v in zip(
                            named["prediction_method_name"],
                            named["predictor_version"].astype(str),
                        ) if pd.notna(m)
                    }
                )
            )
            raise ValueError(
                f"Ambiguous: {_kind_name(kind)} is present at more than "
                f"one predictor version ({listed}). Use "
                f"{_kind_short_name(kind)}['modelname', 'version'] "
                f"to pick one, or pass default_versions="
                f"{{{(kind_val, 'modelname')!r}: 'version'}} to "
                f"EvalContext (resolve_default_versions(df) builds one)."
            )

    return sub


class Field(DSLNode):
    """Reference to a column of a specific prediction kind.

    Parameters
    ----------
    kind : mhctools Kind
        Prediction kind (e.g. ``Kind.pMHC_affinity``).
    field : str
        Column name within the kind rows (e.g. ``"value"``,
        ``"percentile_rank"``, ``"score"``).
    method : str, optional
        Case-insensitive substring match against
        ``prediction_method_name``.
    version : str, optional
        Exact match against ``predictor_version`` (string-compared).
    scope : str
        Column-name prefix for alternate peptide contexts
        (``""``, ``"wt_"``, ``"shuffled_"``, ``"self_"``,
        ``"self_nearest_"``).
    """

    __slots__ = ("kind", "field", "method", "version", "scope")

    def __init__(self, kind, field: str, method: Optional[str] = None,
                 version: Optional[str] = None, scope: str = ""):
        self.kind = kind
        self.field = field
        self.method = method
        self.version = version
        self.scope = scope

    def eval(self, ctx: EvalContext) -> pd.Series:
        sub = _filter_kind_method_version(
            ctx, self.kind, self.method, self.version,
        )
        if sub is None:
            return ctx.empty_series()

        col_name = self.scope + self.field
        if col_name not in sub.columns:
            self._warn_missing_scope_column(ctx, col_name)
            return ctx.empty_series()

        projected = self._maybe_project_peptide_level(ctx, sub)
        if projected is not None:
            return projected

        vals = sub.groupby(
            ctx.group_keys, sort=False, dropna=False
        )[col_name].first()
        vals = vals.reindex(ctx.group_index)
        return pd.to_numeric(vals, errors="coerce")

    def _warn_missing_scope_column(self, ctx, col_name):
        """Say so when a filter reads a comparator column that isn't there.

        A scoped reference — ``wt.``, ``self_nearest.``, ``shuffled.`` —
        reads a column a producer may simply not have written.  Absent,
        it evaluates to NaN, and NaN in a filter drops every group: the
        frame comes back empty with nothing said.  Outside a filter NaN
        is a sensible answer, so this only fires where it silently
        changes the outcome.
        """
        if not self.scope or not ctx.filter_context:
            return
        import warnings
        warnings.warn(
            f"{self!r} reads {col_name!r}, which this frame does not "
            f"have, so it is NaN for every group — in a filter that "
            f"drops everything. Populate the "
            f"{self.scope.rstrip('_')}_* columns, or drop the clause.",
            UserWarning, stacklevel=4,
        )

    def _maybe_project_peptide_level(self, ctx, sub):
        """Read a peptide-level kind as the peptide-level fact it is.

        Grouping by allele puts a prediction that isn't about one allele
        in a group of its own, so a plain read leaves the peptide's
        other allele groups NaN — not a different answer to the
        question, but no answer at all, since the row can never be in
        those groups.  There is exactly one thing the reference can
        sensibly mean, so mean it: project the peptide's value across
        its groups, the same as :func:`peptide_view`.

        Both peptide-level modes qualify.  ``mhc_dependence='none'`` is
        the processing case — no allele anywhere.  ``'haplotype'`` is a
        score for a whole genotype, which mhctools stamps with the
        deconvolved best allele; reading it plainly hands the joint
        score to that one allele and leaves the rest of the genotype
        NaN, which is the same failure wearing an allele name.

        ``single_allele`` is untouched: a per-allele kind read plainly
        returns a real row, and choosing *which* row is a genuine
        decision that stays with the caller and ``best_*`` /
        ``peptide_view``.

        Returns ``None`` when no projection applies, so the caller falls
        through to the plain read.
        """
        if "allele" not in ctx.group_keys:
            return None
        if not _peptide_keys(ctx.group_keys):
            return None
        dependence = _resolve_mhc_dependence(ctx, self.kind, sub)
        if dependence not in ("none", "haplotype"):
            return None

        kind_name = _kind_short_name(self.kind)
        if dependence == "none":
            subject = f"{kind_name}, which carries no allele,"
            reading = "would leave every allele group NaN"
        else:
            subject = (
                f"{kind_name}, which is predicted for a whole genotype "
                f"rather than one allele,"
            )
            reading = (
                "would hand that joint score to the single allele "
                "mhctools deconvolved as the best one, leaving the rest "
                "of the genotype NaN"
            )
        import warnings
        warnings.warn(
            f"{self!r} reads {subject} in a grouping keyed by allele. "
            f"Reading it directly {reading}, so its peptide-level value "
            f"is projected across them — write peptide_view({self!r}) to "
            f"say so explicitly and silence this warning.",
            UserWarning, stacklevel=4,
        )
        return PeptideView(self).eval(ctx)

    def __repr__(self):
        kind_name = _kind_short_name(self.kind)
        if self.field == "percentile_rank":
            field_str = "rank"
        else:
            field_str = self.field
        if self.method is not None and self.version is not None:
            accessor = f"{kind_name}[{self.method!r}, {self.version!r}]"
        elif self.method is not None:
            accessor = f"{kind_name}[{self.method!r}]"
        else:
            accessor = kind_name
        scope_str = self.scope.rstrip("_") + "." if self.scope else ""
        return f"{scope_str}{accessor}.{field_str}"

    def to_ast_string(self):
        kind_name = _kind_short_name(self.kind)
        parts = [f"kind={kind_name}", f"field={self.field!r}"]
        if self.method is not None:
            parts.append(f"method={self.method!r}")
        if self.version is not None:
            parts.append(f"version={self.version!r}")
        if self.scope:
            parts.append(f"scope={self.scope!r}")
        return f"Field({', '.join(parts)})"



# Canonical "best direction" per (kind, field) lives upstream in
# mhctools (since 3.14.0) so consumers don't replicate the table —
# see openvax/mhctools#211.
from mhctools import best_direction as _best_direction  # noqa: E402


def _has_best_direction(kind, field) -> bool:
    """Whether mhctools defines an ordering for this (kind, field).

    Without one there is no "best" row to pick — e.g. a processing
    score has no per-allele ranking — so aggregating across alleles is
    meaningless and the value must already be one per peptide.
    """
    try:
        _best_direction(kind, field)
    except (ValueError, KeyError):
        return False
    return True


def _field_short(field: str) -> str:
    return "rank" if field == "percentile_rank" else field


class BestAlleleField(DSLNode):
    """Per-peptide aggregation of a (kind, field) value across alleles.

    For each peptide group (``ctx.group_keys`` minus ``allele``), picks
    the row with the best value of ``field`` and broadcasts either that
    value (``return_allele=False``) or the corresponding allele name
    (``return_allele=True``) to every per-(peptide, allele) entry in
    ``ctx.group_index``. Composes naturally with the rest of the DSL.

    Direction is taken from :func:`mhctools.best_direction`: ``score``
    is max, ``percentile_rank`` is min, ``value`` is per-kind (IC50 →
    min, half-life → max).

    **Semantics depend on the upstream predictor's allele mode.** With
    mhctools >=3.13.7 these are reported via ``predictor.kind_support()``:

    - ``mhc_dependence='haplotype'`` (e.g. MHCflurry presentation in
      haplotype mode): mhctools already emits one row per peptide, with
      ``allele`` set to MHCflurry's deconvolved best_allele. This
      aggregator is a no-op — it picks that single row and broadcasts.
    - ``mhc_dependence='single_allele'`` (e.g. NetMHCpan, MHCflurry in
      per-allele mode): rows are per-(peptide, allele) with independent
      scores. This aggregator returns the best per-allele score, **not**
      a true joint multi-allele aggregate — the predictor never saw the
      alleles together. Pass the topiary predictor's ``kind_support``
      to :class:`EvalContext` and a UserWarning fires to flag this.
    - ``mhc_dependence='none'`` (processing kinds): allele isn't part of
      the model — using ``best_*`` is meaningless.

    Parameters mirror :class:`Field`. ``return_allele=True`` returns a
    Series of allele-name strings (NaN for groups with no rows).
    """

    __slots__ = (
        "kind", "field", "method", "version", "scope", "return_allele",
    )

    def __init__(self, kind, field: str, method: Optional[str] = None,
                 version: Optional[str] = None, scope: str = "",
                 return_allele: bool = False):
        self.kind = kind
        self.field = field
        self.method = method
        self.version = version
        self.scope = scope
        self.return_allele = return_allele

    def _empty_result(self, ctx):
        if self.return_allele:
            return pd.Series(np.nan, index=ctx.group_index, dtype=object)
        return ctx.empty_series(fill=np.nan)

    def _maybe_warn_dependence(self, ctx, label=None, stacklevel=3):
        """Warn if any matching (model, kind) reports
        ``mhc_dependence='single_allele'``: the result is the best
        per-allele score, not a true joint multi-allele aggregate.

        *label* names the expression the user actually wrote — when this
        aggregation was reached through ``peptide_view``, saying
        ``best_score`` would name a node they never typed.
        """
        method = self.method.lower() if self.method else None
        reported = _reported_dependences(
            getattr(ctx, "kind_support", None), self.kind,
            {method} if method else None,
        )
        for model_key, dep in reported.items():
            if dep == "single_allele":
                import warnings
                default_label = (
                    f"best_{_field_short(self.field)}"
                    f"{'_allele' if self.return_allele else ''}"
                )
                warnings.warn(
                    f"{label or default_label} on "
                    f"({_kind_short_name(self.kind)}, model={model_key!r}) "
                    f"where mhc_dependence='single_allele': returns the "
                    f"best per-allele score, not a joint multi-allele "
                    f"aggregate. Use a haplotype-mode predictor (e.g. "
                    f"MHCflurry presentation in haplotype mode) for a "
                    f"true joint aggregate.",
                    UserWarning, stacklevel=stacklevel,
                )
                return  # one warning per eval is enough

    def eval(self, ctx: EvalContext, warn_label=None,
             stacklevel: int = 3) -> pd.Series:
        # Validate (kind, field) direction up front: an undefined
        # combination is a structural error and should fail loudly even
        # when the frame happens to lack matching rows.
        direction = _best_direction(self.kind, self.field)
        self._maybe_warn_dependence(ctx, warn_label, stacklevel)

        sub = _filter_kind_method_version(
            ctx, self.kind, self.method, self.version,
        )
        if sub is None:
            return self._empty_result(ctx)

        col_name = self.scope + self.field
        if col_name not in sub.columns or "allele" not in sub.columns:
            return self._empty_result(ctx)

        peptide_keys = _peptide_keys(ctx.group_keys)
        if "allele" not in ctx.group_keys or not peptide_keys:
            # No allele dimension to aggregate over. For the value form,
            # degenerate to Field. For the allele form, no meaningful
            # attribution — emit an object NaN Series so callers can
            # detect the no-op rather than misinterpret floats.
            if self.return_allele:
                return pd.Series(np.nan, index=ctx.group_index, dtype=object)
            return Field(
                self.kind, self.field,
                method=self.method, version=self.version, scope=self.scope,
            ).eval(ctx)

        # Coerce values to numeric and drop unrankable rows. Avoid
        # ``sub.copy()`` — work with index masks against the filter's
        # output (which is already a fresh slice).
        numeric = pd.to_numeric(sub[col_name], errors="coerce")
        valid_mask = numeric.notna()
        if not valid_mask.any():
            return self._empty_result(ctx)
        valid = sub.loc[valid_mask, [*peptide_keys, "allele"]].assign(
            __best_value=numeric[valid_mask],
        )

        groups = valid.groupby(
            peptide_keys, sort=False, dropna=False
        )["__best_value"]
        best_idx = groups.idxmax() if direction == "max" else groups.idxmin()

        target = "allele" if self.return_allele else "__best_value"
        per_peptide = valid.loc[best_idx].set_index(peptide_keys)[target]

        result = _broadcast_per_peptide(ctx, per_peptide, peptide_keys)
        if self.return_allele:
            return result.astype(object)
        return pd.to_numeric(result, errors="coerce")

    def __repr__(self):
        kind_name = _kind_short_name(self.kind)
        if self.return_allele:
            field_label = f"best_{_field_short(self.field)}_allele"
        else:
            field_label = f"best_{_field_short(self.field)}"
        if self.method is not None and self.version is not None:
            accessor = f"{kind_name}[{self.method!r}, {self.version!r}]"
        elif self.method is not None:
            accessor = f"{kind_name}[{self.method!r}]"
        else:
            accessor = kind_name
        scope_str = self.scope.rstrip("_") + "." if self.scope else ""
        return f"{scope_str}{accessor}.{field_label}"

    def to_ast_string(self):
        parts = [
            f"kind={_kind_short_name(self.kind)}",
            f"field={self.field!r}",
        ]
        if self.method is not None:
            parts.append(f"method={self.method!r}")
        if self.version is not None:
            parts.append(f"version={self.version!r}")
        if self.scope:
            parts.append(f"scope={self.scope!r}")
        if self.return_allele:
            parts.append("return_allele=True")
        return f"BestAlleleField({', '.join(parts)})"


#: The allele modes mhctools reports, taken from mhctools rather than
#: restated here — a copy is exactly what drifts.  Anything outside it is
#: a version skew we must not guess at.
_MHC_DEPENDENCE_VALUES = MHC_DEPENDENCE_VALUES

#: Default MHC relationship of each prediction kind, independent of any
#: model or any rows: does the kind describe a peptide-MHC pair, or the
#: peptide on its own?
#:
#: The ``pMHC_`` kinds name a peptide-MHC pair, so they are per-allele.
#: The rest are steps of antigen processing that happen before, or
#: apart from, MHC loading — cleavage, transport, trimming — so they
#: describe the peptide alone.  ``immunogenicity`` sits with the
#: per-allele kinds because every mhctools predictor that emits it
#: (DeepImmuno, PRIME, TLimmuno2) scores a peptide against an allele.
#:
#: This is the default a kind carries when nothing more specific is
#: known.  A predictor's own ``kind_support()`` overrides it — MHCflurry
#: presentation reports ``haplotype`` in haplotype mode, and a TCR model
#: that ignores the MHC reports ``none`` — and so does an ``allele_set``
#: in the rows.
KIND_MHC_DEPENDENCE = MappingProxyType({
    "pMHC_affinity": "single_allele",
    "pMHC_presentation": "single_allele",
    "pMHC_stability": "single_allele",
    "pMHC_TCR_binding": "single_allele",
    "immunogenicity": "single_allele",
    "antigen_processing": "none",
    "proteasome_cleavage": "none",
    "endolysosomal_cleavage": "none",
    "erap_trimming": "none",
    "tap_transport": "none",
})


def _model_base_name(model_key):
    """``mhcflurry__2`` -> ``mhcflurry``.

    ``TopiaryPredictor`` disambiguates same-named models with a ``__N``
    suffix, but rows only ever carry the shared
    ``prediction_method_name``.
    """
    return str(model_key).split("__")[0]


def _methods_present(sub):
    """Lower-cased ``prediction_method_name`` values in *sub*.

    ``None`` when the frame can't say — no rows, or no such column — in
    which case every reported model stays in play.
    """
    if sub is None or sub.empty or "prediction_method_name" not in sub.columns:
        return None
    return {
        str(m).lower() for m in sub["prediction_method_name"].dropna().unique()
    }


def _reported_dependences(kind_support, kind, methods=None):
    """``{model_key: mhc_dependence}`` from *kind_support* for *kind*.

    The one place that walks mhctools' per-(model, kind) metadata.
    *methods* is the set of lower-cased method names to keep, normally
    the ones the frame actually contains: metadata for a model that
    produced no rows here must not decide — or veto — this frame's
    projection.
    """
    if not kind_support:
        return {}
    kind_val = _kind_value(kind)
    reported = {}
    for model_key, kind_map in kind_support.items():
        if kind_val not in kind_map:
            continue
        if (
            methods is not None
            and _model_base_name(model_key).lower() not in methods
        ):
            continue
        dep = kind_map[kind_val].get("mhc_dependence")
        if dep:
            reported[model_key] = dep
    return reported


def _rows_for_kind(rows, kind):
    """Narrow *rows* to one kind, so callers may pass a whole frame."""
    if rows is None or rows.empty or "kind" not in rows.columns:
        return rows
    return rows[rows["kind"] == _kind_value(kind)]


def _warn_missing_allele(kind, sub):
    """Flag rows of an allele-scoped kind that arrived without an allele.

    Such a row is malformed, not peptide-level — the two are
    indistinguishable by inspection, and reading it as peptide-level
    spreads one value across alleles the model never scored, inventing
    evidence.  The kind is what tells them apart, so say so rather than
    silently keeping the row in a group of its own.
    """
    if sub is None or sub.empty or "allele" not in sub.columns:
        return
    values = sub["allele"]
    # Alleles repeat heavily, so the distinct set is small: one hashing
    # pass rules out the common (well-formed) case without touching
    # every row.  Counting happens only on the malformed path.
    if not any(_is_blank(v) for v in pd.unique(values.to_numpy())):
        return
    blank = ~stated_values(values)
    import warnings
    warnings.warn(
        f"{int(blank.sum())} {_kind_short_name(kind)} row(s) carry no "
        f"allele, but {_kind_value(kind)} describes a peptide-MHC pair. "
        f"They are kept as per-allele rows with a missing allele rather "
        f"than read as peptide-level predictions, which would spread one "
        f"value across alleles no model scored — check the producer of "
        f"this frame.",
        UserWarning, stacklevel=6,
    )


def _resolve_mhc_dependence(ctx, kind, sub):
    """:func:`mhc_dependence` for an :class:`EvalContext` and a row slice.

    *sub* is the already-selected slice — the rows this expression will
    read, after kind, method and version filtering.  Resolving from it
    rather than from the whole frame keeps one model's per-allele rows
    from reclassifying another model's allele-free ones, and keeps
    metadata for a model that contributed nothing here out of the
    decision.
    """
    return _mhc_dependence(kind, getattr(ctx, "kind_support", None), sub)


def mhc_dependence(kind, *, kind_support=None, rows=None):
    """How *kind* relates to alleles: the one resolver.

    Answers "does this prediction describe a peptide-MHC pair, or the
    peptide alone?" — as ``"single_allele"``, ``"haplotype"`` or
    ``"none"``.  Usable with nothing but a kind, which is the case on
    external-input runs where there is no predictor and therefore no
    ``kind_support`` at all.

    Evidence is consulted in order of how specific it is:

    1. *kind_support* — a model's own statement about what it emitted,
       filtered to the models the rows actually came from.
    2. An ``allele_set`` in *rows* — the data saying this prediction
       covers a genotype.
    3. :data:`KIND_MHC_DEPENDENCE` — what the kind means, absent
       anything more specific.
    4. *rows*, and only for a kind this topiary does not know.

    Row inspection is last because it cannot answer the question it
    looks like it can: a peptide-level record and an allele-scoped
    record that arrived without its allele are identical row by row.
    Only the kind separates them, so a blank allele on an allele-scoped
    kind is reported as such rather than read as peptide-level.

    Parameters
    ----------
    kind : mhctools Kind or str
    kind_support : dict, optional
        ``{model_key: {kind: {"mhc_dependence": ...}}}``, as
        :attr:`TopiaryPredictor.kind_support` produces.
    rows : pandas.DataFrame, optional
        Prediction rows.  Filtered to *kind* when they carry a ``kind``
        column, so a whole frame can be passed.
    """
    return _mhc_dependence(kind, kind_support, _rows_for_kind(rows, kind))


def _mhc_dependence(kind, kind_support, rows):
    """:func:`mhc_dependence` for rows already narrowed to *kind*."""
    reported = _reported_dependences(kind_support, kind, _methods_present(rows))
    # Validate before comparing: an unknown value can't be resolved by
    # picking one of the others, so say what it actually is.
    for dependence in set(reported.values()):
        if dependence not in _MHC_DEPENDENCE_VALUES:
            raise ValueError(
                f"peptide_view on {_kind_short_name(kind)}: unknown "
                f"mhc_dependence {dependence!r} (known: "
                f"{sorted(_MHC_DEPENDENCE_VALUES)}). This topiary is older "
                f"than the mhctools that produced the metadata; upgrade "
                f"topiary rather than guessing the projection."
            )
    found = set(reported.values())
    if len(found) > 1:
        raise _dependence_conflict_error(kind, reported)
    if found:
        return found.pop()

    if rows is not None and not rows.empty and ALLELE_SET_COLUMN in rows.columns:
        if _has_real_values(rows[ALLELE_SET_COLUMN]):
            # The rows say so themselves — no kind_support needed.
            return "haplotype"

    declared = KIND_MHC_DEPENDENCE.get(_kind_value(kind))
    if declared is not None:
        if declared != "none":
            _warn_missing_allele(kind, rows)
        return declared

    # A kind this topiary doesn't know: read what the rows show.
    if rows is None or rows.empty or "allele" not in rows.columns:
        return "none"
    return "single_allele" if _has_real_values(rows["allele"]) else "none"


def _dependence_conflict_error(kind, reported):
    """Explain disagreeing models — and whether method can separate them."""
    kind_name = _kind_short_name(kind)
    modes = sorted(set(reported.values()))
    base_names = {_model_base_name(k) for k in reported}
    if len(base_names) == 1:
        # TopiaryPredictor disambiguates same-named models as name__1 /
        # name__2, but rows only carry the shared prediction_method_name,
        # so no DSL expression can tell them apart.
        return ValueError(
            f"peptide_view on {kind_name}: models {sorted(reported)} report "
            f"different mhc_dependence ({modes}) but share the "
            f"prediction_method_name {base_names.pop()!r}, so qualifying the "
            f"kind by method cannot separate them. Score the runs "
            f"separately, or pass kind_support for only the model this "
            f"frame came from."
        )
    return ValueError(
        f"peptide_view on {kind_name}: models disagree about mhc_dependence "
        f"({modes}). Qualify the kind with a method, e.g. "
        f"{kind_name}[{sorted(base_names)[0]!r}], so one projection applies."
    )


def _broadcast_per_peptide(ctx, per_peptide, peptide_keys):
    """Spread one value per peptide across that peptide's groups.

    ``per_peptide`` is indexed by *peptide_keys*; the result is indexed
    by ``ctx.group_index``.  Shared by every node that reduces to the
    peptide level, so index shape and null-key handling have one
    definition.
    """
    allele_levels = [k for k in _ALLELE_DIMENSION_KEYS if k in ctx.group_keys]
    if allele_levels:
        peptide_index = ctx.group_index.droplevel(allele_levels)
    else:
        # Groups are already peptide-level: the "broadcast" is identity.
        peptide_index = ctx.group_index
    aligned = per_peptide.reindex(peptide_index)
    result = pd.Series(aligned.to_numpy(), index=ctx.group_index)
    return result


def _unwrap_peptide_view(node):
    """The field a projection wrapper reads.

    ``peptide_view`` changes which row is read, not which column, so
    guards and direction inference look through it.
    """
    if isinstance(node, PeptideView):
        return node.inner
    return node


class PeptideView(DSLNode):
    """One value per peptide, reduced correctly for the kind's allele mode.

    The right per-peptide reduction depends on how the predictor treats
    alleles, and until now the caller had to know which:

    ================= ========================= ========================
    ``mhc_dependence`` rows per peptide          peptide-level value
    ================= ========================= ========================
    ``single_allele``  one per (peptide, allele) best across alleles
    ``haplotype``      one per peptide           that row, read directly
    ``none``           one per peptide           that row, read directly
    ================= ========================= ========================

    ``peptide_view(Affinity.score)`` and
    ``peptide_view(Processing.score)`` therefore both mean "this
    peptide's value", and compose in one expression without the caller
    tracking which kind needs ``best_*`` and which is allele-free.

    The allele-free case is the one that could not be written before:
    a processing row carries no allele, so it lands in its own group and
    every per-allele group reads ``NaN``.  Producers worked around this
    by duplicating the row across the patient's alleles.  Here the
    peptide-level value is broadcast to each of the peptide's groups
    instead, leaving one canonical row in the frame.

    The mode comes from ``EvalContext.kind_support`` when present; see
    :func:`_resolve_mhc_dependence` for the fallback.
    """

    __slots__ = ("inner",)

    def __init__(self, inner: DSLNode):
        if getattr(inner, "return_allele", False):
            raise TypeError(
                "peptide_view returns one value per peptide, but "
                f"{inner!r} returns an allele name. Read it directly — "
                "best_*_allele already reports one allele per peptide."
            )
        if not isinstance(inner, (Field, BestAlleleField)):
            raise TypeError(
                f"peptide_view expects a kind field such as "
                f"Affinity.score or processing.score, got "
                f"{type(inner).__name__}. Wrap the field, not the "
                f"expression around it: "
                f"peptide_view(Affinity.score) * 2, not "
                f"peptide_view(Affinity.score * 2)."
            )
        self.inner = inner

    def child_nodes(self):
        return [self.inner]

    def eval(self, ctx: EvalContext) -> pd.Series:
        inner = self.inner
        peptide_keys = _peptide_keys(ctx.group_keys)
        if "allele" in ctx.group_keys and not peptide_keys:
            # Grouping by allele alone has no peptide to project onto, so
            # the contract ("one value per peptide") can't be honored —
            # and a plain read would leave every allele group but the
            # peptide-level row's own NaN.
            raise ValueError(
                f"peptide_view needs a peptide dimension, but group_keys is "
                f"{ctx.group_keys!r} — allele alone. Add the peptide "
                f"identity columns, e.g. group_keys=['peptide', 'allele']."
            )

        # Select once: the rows read here are also the rows the mode is
        # resolved from, so metadata and data can't disagree about which
        # model this expression is about.
        sub = _filter_kind_method_version(
            ctx, inner.kind, inner.method, inner.version,
        )
        dependence = _resolve_mhc_dependence(ctx, inner.kind, sub)
        aggregable = _has_best_direction(inner.kind, inner.field)

        if dependence == "single_allele" and aggregable:
            if isinstance(inner, BestAlleleField):
                return inner.eval(ctx, warn_label=repr(self), stacklevel=5)
            return BestAlleleField(
                inner.kind, inner.field, method=inner.method,
                version=inner.version, scope=inner.scope,
            ).eval(ctx, warn_label=repr(self), stacklevel=5)

        if isinstance(inner, BestAlleleField):
            raise self._nothing_to_aggregate_error(dependence, aggregable)
        return self._eval_peptide_level(ctx, sub, peptide_keys, dependence,
                                        aggregable)

    def _nothing_to_aggregate_error(self, dependence, aggregable):
        """Explain a best_* field that cannot be aggregated here."""
        inner = self.inner
        kind_name = _kind_short_name(inner.kind)
        best_label = f"best_{_field_short(inner.field)}"
        plain = f"peptide_view({kind_name}.{_field_short(inner.field)})"
        if not aggregable:
            return ValueError(
                f"peptide_view({best_label}) on {kind_name}: "
                f"{_kind_short_name(inner.kind)}.{inner.field} has no "
                f"defined best direction, so there is no best row to pick "
                f"across alleles. Use {plain} to read the peptide's value."
            )
        return ValueError(
            f"peptide_view({best_label}) on {kind_name} where "
            f"mhc_dependence={dependence!r}: there is one row per "
            f"peptide, so there is nothing to aggregate across alleles. "
            f"Use {plain} instead."
        )

    def _eval_peptide_level(self, ctx, sub, peptide_keys, dependence,
                            aggregable):
        """Read the one row per peptide and broadcast it to its groups."""
        inner = self.inner
        if sub is None:
            return ctx.empty_series()
        col_name = inner.scope + inner.field
        if col_name not in sub.columns:
            return ctx.empty_series()

        values = pd.to_numeric(sub[col_name], errors="coerce")
        keep = values.notna()
        valid = sub.loc[keep, peptide_keys].assign(
            __peptide_value=values[keep],
        )
        if valid.empty:
            return ctx.empty_series()

        stats = valid.groupby(peptide_keys, sort=False, dropna=False)[
            "__peptide_value"
        ].agg(["first", "min", "max"])
        self._check_one_value_per_peptide(stats, dependence, aggregable)
        return _broadcast_per_peptide(ctx, stats["first"], peptide_keys)

    def _check_one_value_per_peptide(self, stats, dependence, aggregable):
        """A peptide-level read must not find the peptide disagreeing.

        Compared with a tolerance: one row round-tripped through a CSV
        and one computed in-process can differ in the last bit and still
        be the same number.
        """
        spread = (stats["max"] - stats["min"]).abs()
        scale = stats[["min", "max"]].abs().max(axis=1).clip(lower=1.0)
        conflicting = stats[spread > 1e-9 * scale]
        if conflicting.empty:
            return
        inner = self.inner
        kind_name = _kind_short_name(inner.kind)
        if not aggregable:
            # We are reading per peptide because the field has no
            # ordering, not because the kind is peptide-level — say that,
            # rather than blaming a mode the rows may not be in.
            reason = (
                f"{kind_name}.{inner.field} has no defined best direction, "
                f"so its value must already be one per peptide"
            )
        else:
            reason = (
                f"mhc_dependence={dependence!r} means one row per peptide"
            )
        raise ValueError(
            f"peptide_view on {kind_name}: {reason}, but "
            f"{len(conflicting)} peptide(s) carry several different "
            f"{inner.field} values (first: {conflicting.index[0]!r}). "
            f"Filter to one row per peptide first, or qualify the kind by "
            f"method/version."
        )

    def __repr__(self):
        return f"peptide_view({repr(self.inner)})"

    def to_ast_string(self):
        return f"PeptideView({self.inner.to_ast_string()})"


def peptide_view(node):
    """One value per peptide, reduced for the kind's allele mode.

    See :class:`PeptideView`.  Accepts a kind accessor
    (``peptide_view(Affinity)``) or a field
    (``peptide_view(Affinity.score)``).
    """
    if isinstance(node, KindAccessor):
        node = node.value
    return PeptideView(node)


class Len(DSLNode):
    """Peptide length, read from a precomputed ``peptide_length`` column."""

    __slots__ = ("scope",)

    def __init__(self, scope: str = ""):
        self.scope = scope

    def eval(self, ctx: EvalContext) -> pd.Series:
        col = self.scope + "peptide_length"
        if ctx.df.empty or col not in ctx.df.columns:
            return ctx.empty_series()
        vals = ctx.df.groupby(
            ctx.group_keys, sort=False, dropna=False
        )[col].first()
        return vals.reindex(ctx.group_index).astype(float)

    def __repr__(self):
        if self.scope:
            return f"{self.scope.rstrip('_')}.len"
        return "len"


class Count(DSLNode):
    """Count occurrences of amino acid character(s) in the peptide string."""

    __slots__ = ("chars", "scope")

    def __init__(self, chars: str, scope: str = ""):
        if not chars:
            raise ValueError("count() requires at least one amino acid character")
        self.chars = chars.upper()
        self.scope = scope

    def eval(self, ctx: EvalContext) -> pd.Series:
        peptide_col = self.scope + "peptide" if self.scope else "peptide"
        if ctx.df.empty or peptide_col not in ctx.df.columns:
            return ctx.empty_series()
        peptides = ctx.df.groupby(
            ctx.group_keys, sort=False, dropna=False
        )[peptide_col].first()
        peptides = peptides.reindex(ctx.group_index)
        chars = self.chars

        def _count_chars(p):
            if not isinstance(p, str) or not p:
                return float("nan")
            return float(sum(p.count(c) for c in chars))

        return peptides.map(_count_chars).astype(float)

    def __repr__(self):
        scope_str = self.scope.rstrip("_") + "." if self.scope else ""
        return f"{scope_str}count('{self.chars}')"


class PeptideProperty(DSLNode):
    """Amino-acid property of the peptide string as a DSL node.

    The peptide string is uniform within a peptide-allele group, so
    we take ``.first()`` per group, hand the resulting Series to the
    registered compute function, and reindex back onto
    ``ctx.group_index``. Groups whose peptide is missing or empty
    come out as NaN.

    The node always recomputes from the ``peptide`` column. It does
    not pick up columns previously materialized by
    :func:`topiary.properties.add_peptide_properties` — if you've
    already written those columns and want to skip the recompute,
    reference them through :class:`Column`.
    """

    __slots__ = ("name", "compute_fn", "scope")

    def __init__(self, name: str, compute_fn, scope: str = ""):
        self.name = name
        self.compute_fn = compute_fn
        self.scope = scope

    def eval(self, ctx: EvalContext) -> pd.Series:
        peptide_col = self.scope + "peptide" if self.scope else "peptide"
        if ctx.df.empty or peptide_col not in ctx.df.columns:
            return ctx.empty_series()
        peptides = ctx.df.groupby(
            ctx.group_keys, sort=False, dropna=False
        )[peptide_col].first()
        peptides = peptides.reindex(ctx.group_index)
        valid = peptides.notna() & peptides.astype(str).str.len().gt(0)
        result = pd.Series(np.nan, index=ctx.group_index, dtype=float)
        if valid.any():
            computed = self.compute_fn(peptides[valid].astype(str))
            result.loc[valid] = pd.to_numeric(computed, errors="coerce").to_numpy()
        return result

    def __repr__(self):
        scope_str = self.scope.rstrip("_") + "." if self.scope else ""
        return f"{scope_str}{self.name}"


# =============================================================================
# BinOp / UnaryOp — arithmetic composition
# =============================================================================


_OP_SYMBOLS = {
    operator.add: "+", operator.sub: "-",
    operator.mul: "*", operator.truediv: "/",
    operator.pow: "**",
}

_OP_PREC = {
    operator.add: 1, operator.sub: 1,
    operator.mul: 2, operator.truediv: 2,
    operator.pow: 3,
}


def _op_prec(op):
    return _OP_PREC.get(op, 0)


class BinOp(DSLNode):
    """Binary arithmetic: ``left op right`` applied elementwise."""

    __slots__ = ("left", "right", "op")

    def __init__(self, left: DSLNode, right: DSLNode, op):
        self.left = left
        self.right = right
        self.op = op

    def eval(self, ctx: EvalContext) -> pd.Series:
        a = self.left.eval(ctx)
        b = self.right.eval(ctx)
        # Coerce booleans to numeric when mixing with numeric
        if a.dtype == bool and b.dtype != bool:
            a = a.astype(float)
        if b.dtype == bool and a.dtype != bool:
            b = b.astype(float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return self.op(a, b)

    def child_nodes(self):
        return [self.left, self.right]

    def __repr__(self):
        sym = _OP_SYMBOLS.get(self.op, "?")
        left_str = repr(self.left)
        right_str = repr(self.right)
        if isinstance(self.left, BinOp) and _op_prec(self.left.op) < _op_prec(self.op):
            left_str = f"({left_str})"
        if isinstance(self.right, BinOp) and _op_prec(self.right.op) < _op_prec(self.op):
            right_str = f"({right_str})"
        # Comparisons have lower precedence than arithmetic → parenthesize
        if isinstance(self.left, (Comparison, BoolOp)):
            left_str = f"({left_str})"
        if isinstance(self.right, (Comparison, BoolOp)):
            right_str = f"({right_str})"
        return f"{left_str} {sym} {right_str}"

    def to_ast_string(self):
        sym = _OP_SYMBOLS.get(self.op, "?")
        return f"BinOp({self.left.to_ast_string()}, {sym!r}, {self.right.to_ast_string()})"


_UNARY_NAMES = {
    abs: "abs", math.log: "log", math.log2: "log2",
    math.log10: "log10", math.log1p: "log1p",
    math.exp: "exp", math.sqrt: "sqrt",
}

_UNARY_NP = {
    abs: np.abs,
    math.log: np.log,
    math.log2: np.log2,
    math.log10: np.log10,
    math.log1p: np.log1p,
    math.exp: np.exp,
    math.sqrt: np.sqrt,
}


class UnaryOp(DSLNode):
    """Apply a unary function elementwise."""

    __slots__ = ("inner", "fn")

    def __init__(self, inner: DSLNode, fn):
        self.inner = inner
        self.fn = fn

    def child_nodes(self):
        return [self.inner]

    def eval(self, ctx: EvalContext) -> pd.Series:
        vals = self.inner.eval(ctx)
        npfn = _UNARY_NP.get(self.fn)
        if npfn is None:
            return vals.map(
                lambda v: float("nan") if v is None or (
                    isinstance(v, float) and math.isnan(v)
                ) else float(self.fn(v))
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            result = npfn(vals)
        return result

    def __repr__(self):
        name = _UNARY_NAMES.get(self.fn)
        if name == "abs":
            return f"abs({repr(self.inner)})"
        if name:
            return f"{repr(self.inner)}.{name}()"
        return f"{repr(self.inner)}.<?>()"

    def to_ast_string(self):
        name = _UNARY_NAMES.get(self.fn, "<?>")
        return f"UnaryOp({self.inner.to_ast_string()}, {name!r})"


# =============================================================================
# Gaussian CDF / survival / logistic / clip
# =============================================================================


_ERF_UFUNC = np.frompyfunc(math.erf, 1, 1)


def _gauss_cdf(x):
    """Standard Gaussian CDF of a scalar (kept for test compat)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _gauss_cdf_vec(values: pd.Series) -> pd.Series:
    """Vectorized Gaussian CDF of a pd.Series; preserves the index."""
    arr = values.to_numpy(dtype=float, na_value=np.nan)
    result = np.empty_like(arr)
    mask = ~np.isnan(arr)
    # frompyfunc returns object arrays; cast back to float
    if mask.any():
        transformed = _ERF_UFUNC(arr[mask] / math.sqrt(2.0))
        result[mask] = 0.5 * (1.0 + transformed.astype(float))
    result[~mask] = np.nan
    return pd.Series(result, index=values.index)


class NormExpr(DSLNode):
    """Gaussian CDF of an inner expression (ascending)."""

    __slots__ = ("inner", "mean", "std")

    def __init__(self, inner: DSLNode, mean, std):
        self.inner = inner
        self.mean = float(mean)
        self.std = float(std)

    def child_nodes(self):
        return [self.inner]

    def eval(self, ctx: EvalContext) -> pd.Series:
        vals = self.inner.eval(ctx)
        if self.std == 0:
            return pd.Series(np.nan, index=vals.index)
        z = (vals - self.mean) / self.std
        return _gauss_cdf_vec(z)

    def __repr__(self):
        return (
            f"{repr(self.inner)}.ascending_cdf"
            f"({_fmt_num(self.mean)}, {_fmt_num(self.std)})"
        )

    def to_ast_string(self):
        return (
            f"AscendingCDF({self.inner.to_ast_string()}, "
            f"mean={_fmt_num(self.mean)}, std={_fmt_num(self.std)})"
        )


class SurvivalExpr(DSLNode):
    """Gaussian survival (1 - CDF): descending."""

    __slots__ = ("inner", "mean", "std")

    def __init__(self, inner: DSLNode, mean, std):
        self.inner = inner
        self.mean = float(mean)
        self.std = float(std)

    def child_nodes(self):
        return [self.inner]

    def eval(self, ctx: EvalContext) -> pd.Series:
        vals = self.inner.eval(ctx)
        if self.std == 0:
            return pd.Series(np.nan, index=vals.index)
        z = (vals - self.mean) / self.std
        return 1.0 - _gauss_cdf_vec(z)

    def __repr__(self):
        return (
            f"{repr(self.inner)}.descending_cdf"
            f"({_fmt_num(self.mean)}, {_fmt_num(self.std)})"
        )

    def to_ast_string(self):
        return (
            f"DescendingCDF({self.inner.to_ast_string()}, "
            f"mean={_fmt_num(self.mean)}, std={_fmt_num(self.std)})"
        )


class LogisticExpr(DSLNode):
    """Logistic sigmoid: lower input → higher output."""

    __slots__ = ("inner", "midpoint", "width")

    def __init__(self, inner: DSLNode, midpoint, width):
        self.inner = inner
        self.midpoint = float(midpoint)
        self.width = float(width)

    def child_nodes(self):
        return [self.inner]

    def eval(self, ctx: EvalContext) -> pd.Series:
        vals = self.inner.eval(ctx)
        if self.width == 0:
            return pd.Series(np.nan, index=vals.index)
        z = (vals - self.midpoint) / self.width
        # Clip to avoid overflow in exp; 700 is near the float max exponent
        z_clipped = z.clip(lower=-700, upper=700)
        with np.errstate(over="ignore"):
            result = 1.0 / (1.0 + np.exp(z_clipped))
        return result

    def __repr__(self):
        return (
            f"{repr(self.inner)}.logistic"
            f"({_fmt_num(self.midpoint)}, {_fmt_num(self.width)})"
        )

    def to_ast_string(self):
        return (
            f"Logistic({self.inner.to_ast_string()}, "
            f"midpoint={_fmt_num(self.midpoint)}, width={_fmt_num(self.width)})"
        )


class LogisticNormalizedExpr(DSLNode):
    """Logistic sigmoid rescaled so the range is [0, 1].

    Standard logistic ``1/(1+exp((x-m)/w))`` caps below 1 at the
    asymptote — at ``(m=350, w=150)`` the max is ~0.912, only reaching
    1 as ``x → -∞``.  This node divides by that cap so the output
    approaches 1 as the input approaches ``-∞`` and is exactly
    ``0.5`` at ``x = m`` (as with the raw logistic), giving a proper
    binder-quality score in ``[0, 1]``.

    Equivalent to: ``raw_logistic(x, m, w) / raw_logistic(-∞, m, w)``.
    """

    __slots__ = ("inner", "midpoint", "width")

    def __init__(self, inner: DSLNode, midpoint, width):
        self.inner = inner
        self.midpoint = float(midpoint)
        self.width = float(width)

    def child_nodes(self):
        return [self.inner]

    def eval(self, ctx: EvalContext) -> pd.Series:
        vals = self.inner.eval(ctx)
        if self.width == 0:
            return pd.Series(np.nan, index=vals.index)
        z = (vals - self.midpoint) / self.width
        z_clipped = z.clip(lower=-700, upper=700)
        with np.errstate(over="ignore"):
            raw = 1.0 / (1.0 + np.exp(z_clipped))
            # Normalizer: the raw logistic's asymptotic maximum.
            # raw(-inf, m, w) = 1 / (1 + exp(-m/w)).
            norm = 1.0 / (1.0 + math.exp(-self.midpoint / self.width))
        return raw / norm

    def __repr__(self):
        return (
            f"{repr(self.inner)}.logistic_normalized"
            f"({_fmt_num(self.midpoint)}, {_fmt_num(self.width)})"
        )

    def to_ast_string(self):
        return (
            f"LogisticNormalized({self.inner.to_ast_string()}, "
            f"midpoint={_fmt_num(self.midpoint)}, width={_fmt_num(self.width)})"
        )


class ClipExpr(DSLNode):
    """Clamp an inner expression to [lo, hi]."""

    __slots__ = ("inner", "lo", "hi")

    def __init__(self, inner: DSLNode, lo, hi):
        self.inner = inner
        self.lo = lo
        self.hi = hi

    def child_nodes(self):
        return [self.inner]

    def eval(self, ctx: EvalContext) -> pd.Series:
        vals = self.inner.eval(ctx)
        result = vals
        if self.lo is not None:
            result = result.clip(lower=self.lo)
        if self.hi is not None:
            result = result.clip(upper=self.hi)
        return result

    def __repr__(self):
        if self.lo == 0 and self.hi is None:
            return f"{repr(self.inner)}.hinge()"
        return f"{repr(self.inner)}.clip({_fmt_num(self.lo)}, {_fmt_num(self.hi)})"

    def to_ast_string(self):
        return (
            f"Clip({self.inner.to_ast_string()}, "
            f"lo={_fmt_num(self.lo)}, hi={_fmt_num(self.hi)})"
        )


# =============================================================================
# AggExpr — vectorized row-wise aggregation over multiple expressions
# =============================================================================


class AggExpr(DSLNode):
    """Aggregate multiple expressions with a named reducer.

    Each child expression is evaluated to a Series indexed by
    ``ctx.group_index``; those Series are stacked column-wise and
    reduced along axis=1 with NaN-skipping semantics.

    Supported names: ``mean``, ``geomean``, ``minimum``, ``maximum``,
    ``median``.  All reducers are vectorized via pandas / numpy;
    ``geomean`` additionally treats non-positive values as missing.
    """

    __slots__ = ("exprs", "name")

    def __init__(self, exprs, name):
        self.exprs = list(exprs)
        self.name = name

    def child_nodes(self):
        return list(self.exprs)

    def eval(self, ctx: EvalContext) -> pd.Series:
        if not self.exprs:
            return ctx.empty_series()
        columns = {str(i): e.eval(ctx) for i, e in enumerate(self.exprs)}
        df_vals = pd.DataFrame(columns, index=ctx.group_index)

        if self.name == "mean":
            return df_vals.mean(axis=1, skipna=True)
        if self.name == "minimum":
            return df_vals.min(axis=1, skipna=True)
        if self.name == "maximum":
            return df_vals.max(axis=1, skipna=True)
        if self.name == "median":
            return df_vals.median(axis=1, skipna=True)
        if self.name == "geomean":
            arr = df_vals.to_numpy(dtype=float)
            # Treat non-positive values as missing so log() is well-defined.
            positive = np.where(arr > 0, arr, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_mean = np.nanmean(np.log(positive), axis=1)
            return pd.Series(np.exp(log_mean), index=df_vals.index)

        raise ValueError(f"Unknown aggregator {self.name!r}")

    def __repr__(self):
        args = ", ".join(repr(e) for e in self.exprs)
        return f"{self.name}({args})"

    def to_ast_string(self):
        args = ", ".join(e.to_ast_string() for e in self.exprs)
        return f"Agg({self.name!r}, {args})"


def mean(*exprs):
    """Arithmetic mean of expressions. NaN values are skipped."""
    return AggExpr([_as_node(e) for e in exprs], "mean")


def geomean(*exprs):
    """Geometric mean. NaN and non-positive values are skipped."""
    return AggExpr([_as_node(e) for e in exprs], "geomean")


def minimum(*exprs):
    """Minimum of expressions. NaN values are skipped."""
    return AggExpr([_as_node(e) for e in exprs], "minimum")


def maximum(*exprs):
    """Maximum of expressions. NaN values are skipped."""
    return AggExpr([_as_node(e) for e in exprs], "maximum")


def median(*exprs):
    """Median of expressions. NaN values are skipped."""
    return AggExpr([_as_node(e) for e in exprs], "median")


def _method_not_found_error(kind_name, method, available):
    msg = (
        f"No {kind_name} predictions from method matching {method!r}. "
        f"Available: {available}"
    )
    close = get_close_matches(method.lower(), [a.lower() for a in available], n=2, cutoff=0.6)
    if close:
        suggestions = [next(a for a in available if a.lower() == c) for c in close]
        msg += f". Did you mean: {suggestions}?"
    return ValueError(msg)


# =============================================================================
# Comparison — returns a boolean Series
# =============================================================================


_CMP_SYMBOLS = {
    operator.le: "<=",
    operator.ge: ">=",
    operator.lt: "<",
    operator.gt: ">",
    operator.eq: "==",
    operator.ne: "!=",
}

_AGG_OPS = (operator.lt, operator.le, operator.gt, operator.ge)


def _collect_unqualified_kinds(node):
    """Canonical kind values of every unqualified Field ref in *node*."""
    kinds = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n is None:
            continue
        if isinstance(n, Field) and n.method is None:
            kinds.add(_kind_value(n.kind))
        stack.extend(n.child_nodes())
    return kinds


class Comparison(DSLNode):
    """Pointwise comparison between two DSL nodes.

    Returns a boolean-valued Series.  Composes with arithmetic
    (``True`` → 1, ``False`` → 0) and with boolean operators.
    """

    __slots__ = ("left", "op", "right")

    def __init__(self, left: DSLNode, op, right: DSLNode):
        self.left = left
        self.op = op
        self.right = right

    def child_nodes(self):
        return [self.left, self.right]

    def eval(self, ctx: EvalContext) -> pd.Series:
        if self._should_auto_aggregate(ctx):
            result = self._auto_aggregate(ctx)
            if result is not None:
                return result
        a = self.left.eval(ctx)
        b = self.right.eval(ctx)
        # pandas comparison returns False for NaN comparisons — matches
        # the intended "missing values fail the filter" behavior.
        return self.op(a, b)

    def _should_auto_aggregate(self, ctx):
        """Gate check for the narrow auto-aggregation scope (issue #118).

        Fires only when the comparison is evaluated as a filter (not in
        sort, not in scalar score arithmetic), the operator is a
        directional inequality, and we haven't already entered the
        per-method binding loop (no override set).

        `default_methods` (issue #140) takes precedence: if every
        unqualified kind reference in this comparison has a default,
        `Field.eval` will resolve it cleanly without ambiguity — skip
        auto-agg so the explicit user choice wins.
        """
        if not (
            ctx.filter_context
            and ctx._method_override is None
            and self.op in _AGG_OPS
        ):
            return False
        if ctx.default_methods:
            kinds = (
                _collect_unqualified_kinds(self.left)
                | _collect_unqualified_kinds(self.right)
            )
            if kinds and kinds.issubset(ctx.default_methods):
                return False
        return True

    def _auto_aggregate(self, ctx):
        """Try to auto-aggregate across methods for this comparison.

        Returns a boolean ``pd.Series`` indexed by ``ctx.group_index``
        on success, or ``None`` to signal the caller should fall back
        to strict per-side eval (which raises the ambiguity error
        when warranted).
        """
        kinds = _collect_unqualified_kinds(self.left) | _collect_unqualified_kinds(
            self.right
        )
        if len(kinds) != 1:
            return None
        kind_val = next(iter(kinds))

        df = ctx.df
        if df.empty or "kind" not in df.columns:
            return None
        kind_rows = df[df["kind"] == kind_val]
        if "prediction_method_name" not in kind_rows.columns:
            return None
        methods = sorted(
            kind_rows["prediction_method_name"].dropna().unique()
        )
        if len(methods) <= 1:
            return None

        left_per_method = []
        right_per_method = []
        saved_override = ctx._method_override
        try:
            for m in methods:
                ctx._method_override = (kind_val, m)
                left_per_method.append(self.left.eval(ctx))
                right_per_method.append(self.right.eval(ctx))
        finally:
            ctx._method_override = saved_override

        left_df = pd.concat(left_per_method, axis=1)
        right_df = pd.concat(right_per_method, axis=1)

        # For "lower is better" comparisons (<, <=), aggregate the LHS
        # with nanmin and the RHS with nanmax — giving the "any method
        # satisfies" interpretation the issue specifies.  Mirrored for
        # >, >=.
        if self.op in (operator.lt, operator.le):
            left_agg = left_df.min(axis=1, skipna=True)
            right_agg = right_df.max(axis=1, skipna=True)
        else:
            left_agg = left_df.max(axis=1, skipna=True)
            right_agg = right_df.min(axis=1, skipna=True)

        return self.op(left_agg, right_agg)

    def __repr__(self):
        sym = _CMP_SYMBOLS.get(self.op, "?")
        left_str = repr(self.left)
        right_str = repr(self.right)
        # Wrap lower-precedence boolean children
        if isinstance(self.left, BoolOp):
            left_str = f"({left_str})"
        if isinstance(self.right, BoolOp):
            right_str = f"({right_str})"
        return f"{left_str} {sym} {right_str}"

    def to_ast_string(self):
        sym = _CMP_SYMBOLS.get(self.op, "?")
        return (
            f"Comparison({self.left.to_ast_string()}, {sym!r}, "
            f"{self.right.to_ast_string()})"
        )


# =============================================================================
# BoolOp — AND / OR / NOT over boolean-valued DSL nodes
# =============================================================================


_BOOL_SYMBOLS = {
    operator.and_: "&",
    operator.or_: "|",
    operator.invert: "~",
}


class BoolOp(DSLNode):
    """Boolean combinator over 1+ boolean-valued DSL nodes."""

    __slots__ = ("op", "children")

    def __init__(self, op, children):
        self.op = op
        self.children = list(children)

    def child_nodes(self):
        return list(self.children)

    def eval(self, ctx: EvalContext) -> pd.Series:
        # Policy: NaN is treated as False.  Naive `astype(bool)` coerces
        # NaN / None to True (any object is truthy), so we explicitly
        # map NaN → False per-dtype before applying the boolean op.
        if self.op is operator.invert:
            return ~_as_bool_series(self.children[0].eval(ctx))
        values = [_as_bool_series(c.eval(ctx)) for c in self.children]
        if self.op is operator.and_:
            result = values[0]
            for v in values[1:]:
                result = result & v
            return result
        if self.op is operator.or_:
            result = values[0]
            for v in values[1:]:
                result = result | v
            return result
        raise ValueError(f"Unknown boolean op: {self.op!r}")

    def __repr__(self):
        if self.op is operator.invert:
            inner = self.children[0]
            inner_str = repr(inner)
            # ~ binds tighter than comparison & boolean combinators in
            # the parser grammar, so wrap anything that isn't a bare
            # atom / unary-invert.
            if isinstance(inner, (Comparison, BoolOp)):
                if not (isinstance(inner, BoolOp) and inner.op is operator.invert):
                    inner_str = f"({inner_str})"
            return f"~{inner_str}"
        sym = _BOOL_SYMBOLS.get(self.op, "?")
        parts = []
        for c in self.children:
            s = repr(c)
            # Parenthesize lower-precedence boolean children (| inside &)
            if isinstance(c, BoolOp) and c.op is operator.or_ and self.op is operator.and_:
                s = f"({s})"
            parts.append(s)
        return f" {sym} ".join(parts)

    def to_ast_string(self):
        if self.op is operator.invert:
            return f"Not({self.children[0].to_ast_string()})"
        name = "And" if self.op is operator.and_ else "Or"
        args = ", ".join(c.to_ast_string() for c in self.children)
        return f"{name}({args})"


def _as_bool_series(s: pd.Series) -> pd.Series:
    """Coerce a Series to bool under the NaN → False policy.

    Straight ``astype(bool)`` turns NaN / None into True (non-empty
    object → truthy), which violates the policy used by ``apply_filter``.
    Dispatches per-dtype to stay numpy-native on common cases.
    """
    if s.dtype == bool:
        return s
    if s.dtype.kind == "f":
        arr = s.to_numpy()
        return pd.Series((arr != 0) & ~np.isnan(arr), index=s.index)
    if s.dtype.kind in "iu":
        return s.astype(bool)

    def _bool_of(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False
        return bool(v)
    return s.map(_bool_of).astype(bool)


def _combine_bool(op, left: DSLNode, right: DSLNode) -> BoolOp:
    """Build a BoolOp, flattening when the child has the same op."""
    children = []
    for node in (left, right):
        if isinstance(node, BoolOp) and node.op is op:
            children.extend(node.children)
        else:
            children.append(node)
    return BoolOp(op, children)


# =============================================================================
# KindAccessor — attribute-style access to Fields for a given Kind
# =============================================================================


class KindAccessor:
    """Proxy for a prediction Kind with typed field access.

    Bracket indexing supports method and optional version::

        Affinity["netmhcpan"]               # method only
        Affinity["netmhcpan", "4.1b"]       # method + version

    Scope to an alternate peptide context via :class:`Scope`::

        wt.Affinity.score
        shuffled.Affinity.value
    """

    __slots__ = ("kind", "method", "version", "scope")

    def __init__(self, kind, method: Optional[str] = None,
                 version: Optional[str] = None, scope: str = ""):
        self.kind = kind
        self.method = method
        self.version = version
        self.scope = scope

    def __getitem__(self, key) -> "KindAccessor":
        if isinstance(key, tuple):
            if len(key) == 2:
                method, version = key
            elif len(key) == 1:
                method, version = key[0], None
            else:
                raise ValueError(
                    f"KindAccessor[...] accepts 1 or 2 elements "
                    f"(method, version), got {len(key)}"
                )
        else:
            method, version = key, None
        return KindAccessor(
            self.kind, method=method, version=version, scope=self.scope,
        )

    @property
    def value(self) -> Field:
        return Field(self.kind, "value", method=self.method,
                     version=self.version, scope=self.scope)

    @property
    def rank(self) -> Field:
        return Field(self.kind, "percentile_rank", method=self.method,
                     version=self.version, scope=self.scope)

    @property
    def score(self) -> Field:
        return Field(self.kind, "score", method=self.method,
                     version=self.version, scope=self.scope)

    # -- best-allele aggregation: max/min across alleles per peptide,
    #    broadcast back to per-(peptide, allele) groups. Use these when
    #    the underlying predictor reports `mhc_dependence='haplotype'`
    #    (e.g. MHCflurry presentation in haplotype mode) so that one
    #    "presented anywhere" answer applies regardless of which
    #    allele's row a downstream node evaluates against.

    @property
    def best_value(self) -> "BestAlleleField":
        return BestAlleleField(self.kind, "value", method=self.method,
                               version=self.version, scope=self.scope)

    @property
    def best_score(self) -> "BestAlleleField":
        return BestAlleleField(self.kind, "score", method=self.method,
                               version=self.version, scope=self.scope)

    @property
    def best_rank(self) -> "BestAlleleField":
        return BestAlleleField(self.kind, "percentile_rank",
                               method=self.method, version=self.version,
                               scope=self.scope)

    @property
    def best_value_allele(self) -> "BestAlleleField":
        return BestAlleleField(self.kind, "value", method=self.method,
                               version=self.version, scope=self.scope,
                               return_allele=True)

    @property
    def best_score_allele(self) -> "BestAlleleField":
        return BestAlleleField(self.kind, "score", method=self.method,
                               version=self.version, scope=self.scope,
                               return_allele=True)

    @property
    def best_rank_allele(self) -> "BestAlleleField":
        return BestAlleleField(self.kind, "percentile_rank",
                               method=self.method, version=self.version,
                               scope=self.scope, return_allele=True)

    # -- delegations to .value so Affinity <= 500 and Affinity.norm(...) work --

    def __le__(self, other): return self.value.__le__(other)
    def __lt__(self, other): return self.value.__lt__(other)
    def __ge__(self, other): return self.value.__ge__(other)
    def __gt__(self, other): return self.value.__gt__(other)

    def ascending_cdf(self, mean=0.0, std=1.0):
        return self.value.ascending_cdf(mean, std)

    norm = ascending_cdf

    def descending_cdf(self, mean=0.0, std=1.0):
        return self.value.descending_cdf(mean, std)

    def logistic(self, midpoint=0.0, width=1.0):
        return self.value.logistic(midpoint, width)

    def logistic_normalized(self, midpoint=0.0, width=1.0):
        return self.value.logistic_normalized(midpoint, width)

    def clip(self, lo=None, hi=None):
        return self.value.clip(lo, hi)

    def hinge(self):
        return self.value.hinge()

    def log(self): return self.value.log()
    def log2(self): return self.value.log2()
    def log10(self): return self.value.log10()
    def log1p(self): return self.value.log1p()
    def exp(self): return self.value.exp()
    def sqrt(self): return self.value.sqrt()

    def __neg__(self): return -self.value
    def __abs__(self): return abs(self.value)
    def __add__(self, other): return self.value + other
    def __radd__(self, other): return other + self.value
    def __sub__(self, other): return self.value - other
    def __rsub__(self, other): return other - self.value
    def __mul__(self, other): return self.value * other
    def __rmul__(self, other): return other * self.value
    def __truediv__(self, other): return self.value / other
    def __rtruediv__(self, other): return other / self.value
    def __pow__(self, other): return self.value ** other


# Top-level accessors for common kinds
Affinity = KindAccessor(Kind.pMHC_affinity)
Presentation = KindAccessor(Kind.pMHC_presentation)
Stability = KindAccessor(Kind.pMHC_stability)
Processing = KindAccessor(Kind.antigen_processing)

# Pre-built IsIn nodes for the most common categorical filter: MHC class.
# Both require a ``mhc_class`` column in the DataFrame — present after
# :func:`topiary.read_pvacseq` and other loaders that derive it from
# alleles.  Fresh ``TopiaryPredictor`` output doesn't carry the column
# (class lives in :attr:`kind_support` at the model level); derive with
# ``df["mhc_class"] = df["allele"].map(...)`` first if you need these
# on a fresh prediction result.
class_i = IsIn("mhc_class", ["I"])
class_ii = IsIn("mhc_class", ["II"])


# =============================================================================
# Scope — alternate peptide context (wt, shuffled, self)
# =============================================================================

_CONTEXT_KEYWORDS = {"wt", "shuffled", "self", "self_nearest"}


class Scope:
    """Select an alternate peptide context for field access."""

    __slots__ = ("prefix", "name")

    def __init__(self, name: str):
        self.name = name
        self.prefix = name + "_"

    def __getattr__(self, attr):
        if attr in ("prefix", "name"):
            raise AttributeError(attr)
        if attr == "len":
            return Len(scope=self.prefix)
        attr_lower = attr.lower()
        if attr_lower in KIND_ALIASES:
            return KindAccessor(KIND_ALIASES[attr_lower], scope=self.prefix)
        # Lazy import — topiary.properties depends on this module.
        from ..properties import _PROPERTIES
        if attr_lower in _PROPERTIES:
            compute_fn, _ = _PROPERTIES[attr_lower]
            return PeptideProperty(attr_lower, compute_fn, scope=self.prefix)
        available = sorted(KIND_ALIASES.keys())
        raise AttributeError(
            f"Unknown kind {attr!r} in scope {self.name!r}. "
            f"Available: {available}"
        )

    def count(self, chars: str) -> "Count":
        return Count(chars, scope=self.prefix)

    def __repr__(self):
        return self.name


wt = Scope("wt")
shuffled = Scope("shuffled")
self_scope = Scope("self")

# Reserved DSL scope for "nearest-self healthy-tissue peptide" data.
# Topiary does not compute these columns — producers populate them
# externally (via BLAST / edit distance against a healthy-tissue
# proteome, with a producer-chosen definition of "self").  The scope
# reads ``self_nearest_*`` columns; when absent, evaluates to NaN.
# See docs/fragments.md for the reserved column namespace.
self_nearest = Scope("self_nearest")


# =============================================================================
# Kind / field name resolution — used by both the parser and callers
# =============================================================================

KIND_ALIASES = _build_kind_aliases()
"""Public mapping of every accepted kind spelling (canonical, short
name, common abbreviations like ``"ba"``/``"el"``) to the
``mhctools.Kind`` constant. Stable across topiary minor releases —
external consumers (e.g. vaxrank) can rely on this name.
"""

_FIELD_ALIASES = {
    "value": "value", "val": "value", "ic50": "value",
    "rank": "percentile_rank", "percentile_rank": "percentile_rank",
    "percentile": "percentile_rank",
    "score": "score",
}


def _resolve_kind(name):
    key = name.strip().lower()
    if key in KIND_ALIASES:
        return KIND_ALIASES[key]
    available = sorted(KIND_ALIASES.keys())
    close = get_close_matches(key, available, n=3, cutoff=0.6)
    msg = f"Unknown prediction kind {name!r}."
    if close:
        msg += f" Did you mean: {close}?"
    else:
        msg += f" Available: {available}"
    raise ValueError(msg)


def _resolve_qualified_kind(name):
    """Resolve ``tool_kind`` or plain ``kind`` to ``(Kind, method|None)``."""
    key = name.strip().lower()
    if key in KIND_ALIASES:
        return KIND_ALIASES[key], None
    parts = key.split("_")
    for i in range(1, len(parts)):
        tool = "_".join(parts[:i])
        kind_str = "_".join(parts[i:])
        if kind_str in KIND_ALIASES:
            return KIND_ALIASES[kind_str], tool
    available = sorted(KIND_ALIASES.keys())
    close = get_close_matches(key, available, n=3, cutoff=0.6)
    msg = f"Unknown prediction kind {name!r}. Use 'kind' or 'tool_kind' format."
    if close:
        msg += f" Did you mean: {close}?"
    else:
        msg += f" Available kinds: {available}"
    raise ValueError(msg)


def _resolve_field(name):
    key = name.strip().lower()
    if key in _FIELD_ALIASES:
        return _FIELD_ALIASES[key]
    available = sorted(_FIELD_ALIASES.keys())
    close = get_close_matches(key, available, n=3, cutoff=0.6)
    msg = f"Unknown field {name!r}."
    if close:
        msg += f" Did you mean: {close}?"
    else:
        msg += f" Available: {available}"
    raise ValueError(msg)
