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

_GROUP_KEYS = ["source_sequence_name", "peptide", "peptide_offset", "allele"]
_GROUP_KEYS_VARIANT = ["variant", "peptide", "peptide_offset", "allele"]


_GROUP_KEYS_FRAGMENT = ["fragment_id", "peptide", "peptide_offset", "allele"]


def _pick_group_keys(df):
    # fragment_id is the most specific identity (from predict_from_fragments);
    # variant is for the legacy varcode pipeline; source_sequence_name is the
    # generic fallback.
    if "fragment_id" in df.columns:
        return list(_GROUP_KEYS_FRAGMENT)
    if "variant" in df.columns:
        return list(_GROUP_KEYS_VARIANT)
    return list(_GROUP_KEYS)


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
    MultiIndex.  Every :class:`DSLNode` ``.eval(ctx)`` returns a
    ``pd.Series`` indexed by this MultiIndex (one value per peptide
    -allele group).

    Parameters
    ----------
    df : pandas.DataFrame
        Prediction rows, long-form.
    group_keys : list of str, optional
        Override the auto-detected peptide-allele group keys.
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
    """

    __slots__ = (
        "df", "group_keys", "default_methods",
        "_group_index", "_group_tuples_cache",
    )

    def __init__(self, df, group_keys=None, default_methods=None):
        self.df = df
        self.group_keys = list(group_keys) if group_keys else _pick_group_keys(df)
        self.default_methods = (
            _normalize_default_methods(default_methods) if default_methods else {}
        )
        self._group_index = None
        self._group_tuples_cache = None

    @property
    def group_index(self) -> pd.MultiIndex:
        """MultiIndex of unique group-key tuples (preserving row order)."""
        if self._group_index is None:
            if self.df.empty:
                self._group_index = pd.MultiIndex(
                    levels=[[] for _ in self.group_keys],
                    codes=[[] for _ in self.group_keys],
                    names=self.group_keys,
                )
            else:
                key_df = self.df[self.group_keys].drop_duplicates()
                self._group_index = pd.MultiIndex.from_frame(key_df)
        return self._group_index

    def row_group_tuples(self) -> pd.Series:
        """Per-row tuple of group-key values, aligned to ``self.df.index``."""
        if self._group_tuples_cache is None:
            if self.df.empty:
                self._group_tuples_cache = pd.Series(
                    [], index=self.df.index, dtype=object
                )
            else:
                self._group_tuples_cache = pd.Series(
                    list(zip(*[self.df[k] for k in self.group_keys])),
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
            available = sorted(ctx.df.columns)
            close = get_close_matches(self.col_name, available, n=3, cutoff=0.6)
            msg = f"Column {self.col_name!r} not found in DataFrame."
            if close:
                msg += f" Did you mean: {close}?"
            else:
                msg += f" Available: {available}"
            raise ValueError(msg)
        vals = ctx.df.groupby(ctx.group_keys, sort=False)[self.col_name].first()
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
        df = ctx.df
        if df.empty or "kind" not in df.columns:
            return ctx.empty_series()

        kind_val = _kind_value(self.kind)
        sub = df[df["kind"] == kind_val]
        if sub.empty:
            return ctx.empty_series()

        # Filter by method substring (case-insensitive)
        if self.method is not None:
            col = "prediction_method_name"
            if col in sub.columns:
                method_lower = self.method.lower()
                method_mask = sub[col].str.lower().str.contains(
                    method_lower, na=False, regex=False
                )
                matched = sub[method_mask]
                if matched.empty:
                    available = sorted(sub[col].dropna().unique())
                    raise _method_not_found_error(
                        _kind_name(self.kind), self.method, available
                    )
                sub = matched

        # Filter by exact version string
        if self.version is not None:
            col = "predictor_version"
            if col in sub.columns:
                version_mask = sub[col].astype(str) == str(self.version)
                matched = sub[version_mask]
                if matched.empty:
                    available = sorted(sub[col].dropna().astype(str).unique())
                    raise ValueError(
                        f"No {_kind_name(self.kind)} predictions from "
                        f"predictor_version {self.version!r}. "
                        f"Available: {available}"
                    )
                sub = matched

        # Ambiguity: unqualified access with multiple methods in any group
        if (
            self.method is None
            and "prediction_method_name" in sub.columns
        ):
            methods_per_group = sub.groupby(
                ctx.group_keys, sort=False
            )["prediction_method_name"].nunique()
            if (methods_per_group > 1).any():
                default = ctx.default_methods.get(_kind_value(self.kind))
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
                            _kind_name(self.kind), default, available
                        )
                    sub = matched
                else:
                    method_list = ", ".join(
                        sorted(sub["prediction_method_name"].dropna().unique())
                    )
                    raise ValueError(
                        f"Ambiguous: multiple models produce "
                        f"{_kind_name(self.kind)} ({method_list}). "
                        f"Use {_kind_short_name(self.kind)}['modelname'] "
                        f"to disambiguate, or pass "
                        f"default_methods={{{_kind_value(self.kind)!r}: "
                        f"'modelname'}} to EvalContext."
                    )

        col_name = self.scope + self.field
        if col_name not in sub.columns:
            return ctx.empty_series()

        vals = sub.groupby(ctx.group_keys, sort=False)[col_name].first()
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


class Len(DSLNode):
    """Peptide length, read from a precomputed ``peptide_length`` column."""

    __slots__ = ("scope",)

    def __init__(self, scope: str = ""):
        self.scope = scope

    def eval(self, ctx: EvalContext) -> pd.Series:
        col = self.scope + "peptide_length"
        if ctx.df.empty or col not in ctx.df.columns:
            return ctx.empty_series()
        vals = ctx.df.groupby(ctx.group_keys, sort=False)[col].first()
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
        peptides = ctx.df.groupby(ctx.group_keys, sort=False)[peptide_col].first()
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


class Comparison(DSLNode):
    """Pointwise comparison between two DSL nodes.

    Returns a boolean-valued Series.  Composes with arithmetic
    (``True`` → 1, ``False`` → 0) and with boolean operators.
    """

    __slots__ = ("left", "op", "right")

    def __init__(self, left: DSLNode, op, right: DSLNode):
        for side in (left, right):
            if isinstance(side, Field) and side.scope:
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
        a = self.left.eval(ctx)
        b = self.right.eval(ctx)
        # pandas comparison returns False for NaN comparisons — matches
        # the intended "missing values fail the filter" behavior.
        return self.op(a, b)

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
        if attr_lower in _KIND_ALIASES:
            return KindAccessor(_KIND_ALIASES[attr_lower], scope=self.prefix)
        available = sorted(_KIND_ALIASES.keys())
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

_KIND_ALIASES = _build_kind_aliases()

_FIELD_ALIASES = {
    "value": "value", "val": "value", "ic50": "value",
    "rank": "percentile_rank", "percentile_rank": "percentile_rank",
    "percentile": "percentile_rank",
    "score": "score",
}


def _resolve_kind(name):
    key = name.strip().lower()
    if key in _KIND_ALIASES:
        return _KIND_ALIASES[key]
    available = sorted(_KIND_ALIASES.keys())
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
    if key in _KIND_ALIASES:
        return _KIND_ALIASES[key], None
    parts = key.split("_")
    for i in range(1, len(parts)):
        tool = "_".join(parts[:i])
        kind_str = "_".join(parts[i:])
        if kind_str in _KIND_ALIASES:
            return _KIND_ALIASES[kind_str], tool
    available = sorted(_KIND_ALIASES.keys())
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


