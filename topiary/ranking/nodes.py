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
from difflib import get_close_matches
from typing import Optional

import numpy as np
import pandas as pd
from mhctools import Kind


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
_GROUP_KEYS = ["source_sequence_name", "peptide", "peptide_offset", "allele"]
_GROUP_KEYS_VARIANT = ["variant", "peptide", "peptide_offset", "allele"]


_GROUP_KEYS_FRAGMENT = ["fragment_id", "peptide", "peptide_offset", "allele"]


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
    return any(
        not pd.isna(v) and str(v).strip() != ""
        for v in pd.unique(values.to_numpy())
    )


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
    return _with_optional_sample_key(df, keys)


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
        kind = aliases.get(_kind_name(key).lower())
        if kind is None:
            # Surface the canonical kind values and every DSL short
            # alias so a user who typed 'banana' sees that 'ba' /
            # 'affinity' / 'pMHC_affinity' all map to the same kind.
            # The alias dict is lower-cased for case-insensitive
            # lookup; skip lower-case duplicates of canonicals to
            # keep the list readable.
            canonical = {_kind_value(k) for k in aliases.values()}
            canonical_lower = {c.lower() for c in canonical}
            shorts = {a for a in aliases.keys() if a not in canonical_lower}
            accepted = sorted(shorts | canonical)
            raise ValueError(
                f"default_methods key {key!r} is not a known kind. "
                f"Accepted spellings: {accepted}"
            )
        out[_kind_value(kind)] = method
    return out


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
    filter_context : bool, optional
        When true, directional ``Comparison`` nodes
        (``<``, ``<=``, ``>``, ``>=``) with unqualified same-kind refs
        auto-aggregate across methods (nanmin for ``<``/``<=``, nanmax
        for ``>``/``>=``) instead of raising on ambiguity.
        :func:`apply_filter` sets this to ``True`` automatically;
        :func:`apply_sort` leaves it ``False`` so sort stays strict.
    """

    __slots__ = (
        "df", "group_keys", "default_methods", "filter_context",
        "kind_support",
        "_group_index", "_key_frame", "_group_tuples_cache",
        "_group_codes_cache", "_method_override",
    )

    def __init__(
        self, df, group_keys=None, default_methods=None, filter_context=False,
        kind_support=None,
    ):
        self.df = df
        if group_keys is None:
            self.group_keys = _pick_group_keys(df)
        else:
            self.group_keys = _normalize_group_keys(df, group_keys)
        self.default_methods = (
            _normalize_default_methods(default_methods) if default_methods else {}
        )
        self.filter_context = filter_context
        # mhctools >=3.13.7 per-(model, kind) metadata. Optional; when
        # provided (typically from ``TopiaryPredictor.kind_support``),
        # nodes that care about allele dependence (e.g.
        # :class:`BestAlleleField`) can warn or branch on it. Shape:
        # ``{model_key: {kind_value: {"mhc_dependence", "mhc_class"}}}``.
        self.kind_support = kind_support
        self._group_index = None
        self._key_frame = None
        self._group_tuples_cache = None
        self._group_codes_cache = None
        # Internal: when Comparison auto-aggregates across methods, it
        # binds Field(method=None, kind=K) references to a specific
        # method per iteration by setting (kind_value, method_name) here.
        self._method_override = None

    @property
    def key_frame(self) -> pd.DataFrame:
        """The group-key columns, with null spellings collapsed.

        ``groupby(dropna=False)`` — which every node evaluates through —
        treats ``None``, ``NaN`` and ``pd.NA`` in an object column as one
        group, but a plain ``drop_duplicates`` keeps them apart.  Building
        the group index from raw values therefore produces groups no node
        result can ever key, and rows would silently score NaN.  Collapse
        them once here so the index, the row mapping and every node agree.
        """
        if self._key_frame is None:
            frame = self.df[self.group_keys]
            mixed = [
                k for k in self.group_keys
                if frame[k].dtype == object and frame[k].isna().any()
            ]
            if mixed:
                frame = frame.assign(**{
                    k: frame[k].where(frame[k].notna(), np.nan) for k in mixed
                })
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
                key_df = self.key_frame.drop_duplicates()
                if single_key:
                    key = self.group_keys[0]
                    self._group_index = pd.Index(key_df[key], name=key)
                else:
                    self._group_index = pd.MultiIndex.from_frame(key_df)
        return self._group_index

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

    def isin(self, values) -> "IsIn":
        """Membership: ``Column("mhc_class").isin(["I", "II"])``."""
        return IsIn(self.col_name, values)


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
            version_mask = sub[col].astype(str) == str(version)
            matched = sub[version_mask]
            if matched.empty:
                available = sorted(sub[col].dropna().astype(str).unique())
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
            return ctx.empty_series()

        vals = sub.groupby(
            ctx.group_keys, sort=False, dropna=False
        )[col_name].first()
        vals = vals.reindex(ctx.group_index)
        return pd.to_numeric(vals, errors="coerce")

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

    # (Scoped fields cannot appear in filters — guarded in Comparison.__init__)


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
            ctx, self.kind, {method} if method else None,
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

        peptide_keys = [k for k in ctx.group_keys if k != "allele"]
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


#: The allele modes mhctools reports.  Anything else is a version skew
#: we must not guess at.
_MHC_DEPENDENCE_VALUES = frozenset({"none", "single_allele", "haplotype"})


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


def _reported_dependences(ctx, kind, methods=None):
    """``{model_key: mhc_dependence}`` from ``ctx.kind_support`` for *kind*.

    The one place that walks mhctools' per-(model, kind) metadata.
    *methods* is the set of lower-cased method names to keep, normally
    the ones the frame actually contains: metadata for a model that
    produced no rows here must not decide — or veto — this frame's
    projection.
    """
    kind_support = getattr(ctx, "kind_support", None)
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


def _resolve_mhc_dependence(ctx, kind, sub):
    """How *kind* relates to alleles for the rows in *sub*.

    *sub* is the already-selected slice — the rows this expression will
    read, after kind, method and version filtering.  Resolving from it
    rather than from the whole frame keeps one model's per-allele rows
    from reclassifying another model's allele-free ones, and keeps
    metadata for a model that contributed nothing here out of the
    decision.

    Prefers ``ctx.kind_support``, mhctools' per-(model, kind) metadata —
    the only authority that can tell ``haplotype`` from
    ``single_allele``, since both put a real allele on every row.
    Without it, read what the rows show: no allele at all means
    allele-independent, anything else is per-allele, which is what the
    DSL already assumed.
    """
    reported = _reported_dependences(ctx, kind, _methods_present(sub))
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

    if sub is None or sub.empty or "allele" not in sub.columns:
        # Nothing to read, or no allele column at all: nothing here is
        # per-allele.
        return "none"
    return "single_allele" if _has_real_values(sub["allele"]) else "none"


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
    if "allele" in ctx.group_keys:
        peptide_index = ctx.group_index.droplevel("allele")
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
        peptide_keys = [k for k in ctx.group_keys if k != "allele"]
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
        for side in (left, right):
            # Look through peptide_view(): it changes which row is read,
            # not which columns, so a scoped field stays off-limits in a
            # filter however it is wrapped.
            side = _unwrap_peptide_view(side)
            if isinstance(side, (Field, BestAlleleField)) and side.scope:
                scope_name = side.scope.rstrip("_")
                raise TypeError(
                    f"Scoped fields ({scope_name}.*) can't be used in filters. "
                    f"Use them in sorting expressions instead, e.g.: "
                    f"sort_by=[Affinity.score - {scope_name}.Affinity.score]"
                )
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
