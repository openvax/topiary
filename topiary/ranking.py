"""
Filtering and ranking of epitope predictions across prediction kinds.

Express filter/rank heuristics via operator overloading::

    from topiary import Affinity, Presentation

    # Simple filter
    Affinity.value <= 500

    # OR / AND
    (Affinity.value <= 500) | (Presentation.rank <= 2.0)
    (Affinity.value <= 500) & (Presentation.rank <= 2.0)

    # Gaussian-normalized composite ranking
    score = (
        0.5 * Affinity.value.norm(mean=500, std=200) +
        0.5 * Presentation.score.norm(mean=0.5, std=0.3)
    )
    ranking = (Affinity.value <= 500).rank_by(score)
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Optional

import numpy as np
import pandas as pd
from mhctools import Kind


# ---------------------------------------------------------------------------
# Expression tree — composable, evaluable against a group DataFrame
# ---------------------------------------------------------------------------


def _gauss_cdf(x):
    """Standard Gaussian CDF without scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class Expr:
    """Base class for lazy numeric expressions over prediction rows.

    Supports arithmetic (``+``, ``-``, ``*``, ``/``), negation,
    ``.norm(mean, std)`` for Gaussian CDF normalization, and
    comparison operators (``<=``, ``>=``) to create filters.
    """

    def evaluate(self, group_df):
        """Return a float score for a peptide-allele group."""
        raise NotImplementedError

    # -- Gaussian normalization --

    def norm(self, mean=0.0, std=1.0):
        """Gaussian CDF normalization: maps value to ~[0, 1].

        **Higher input → higher output.** Good for "higher is better"
        fields like ``.score``. For "lower is better" fields like IC50,
        use ``1 - field.norm(...)``::

            # Presentation score (higher = better) → use directly
            Presentation.score.norm(mean=0.5, std=0.3)

            # IC50 (lower = better) → invert
            1 - Affinity.norm(mean=500, std=200)
        """
        return _NormExpr(self, mean, std)

    def logistic(self, midpoint=0.0, width=1.0):
        """Logistic sigmoid: ``1 / (1 + exp((x - midpoint) / width))``.

        **Lower input → higher output.** Good for "lower is better"
        fields like IC50. Values below the midpoint score > 0.5;
        values above score < 0.5::

            # IC50 (lower = better) → use directly
            Affinity.logistic(midpoint=350, width=150)

            # Score (higher = better) → logistic goes the wrong way,
            # use norm() instead
        """
        return _LogisticExpr(self, midpoint, width)

    # -- arithmetic --

    def __add__(self, other):
        return _BinOp(self, _as_expr(other), operator.add)

    def __radd__(self, other):
        return _BinOp(_as_expr(other), self, operator.add)

    def __sub__(self, other):
        return _BinOp(self, _as_expr(other), operator.sub)

    def __rsub__(self, other):
        return _BinOp(_as_expr(other), self, operator.sub)

    def __mul__(self, other):
        return _BinOp(self, _as_expr(other), operator.mul)

    def __rmul__(self, other):
        return _BinOp(_as_expr(other), self, operator.mul)

    def __truediv__(self, other):
        return _BinOp(self, _as_expr(other), operator.truediv)

    def __rtruediv__(self, other):
        return _BinOp(_as_expr(other), self, operator.truediv)

    def __neg__(self):
        return _BinOp(_Const(-1), self, operator.mul)

    def __abs__(self):
        return _UnaryOp(self, abs)

    def __pow__(self, other):
        return _BinOp(self, _as_expr(other), operator.pow)

    def __rpow__(self, other):
        return _BinOp(_as_expr(other), self, operator.pow)

    # -- transforms --

    def clip(self, lo=None, hi=None):
        """Clamp value to [lo, hi]. None = unbounded."""
        return _ClipExpr(self, lo, hi)

    def log(self):
        """Natural logarithm (NaN if value <= 0)."""
        return _UnaryOp(self, math.log)

    def log2(self):
        """Base-2 logarithm (NaN if value <= 0)."""
        return _UnaryOp(self, math.log2)

    def log10(self):
        """Base-10 logarithm (NaN if value <= 0)."""
        return _UnaryOp(self, math.log10)

    def log1p(self):
        """``log(1 + x)``, accurate for small x (NaN if x <= -1)."""
        return _UnaryOp(self, math.log1p)

    def exp(self):
        """Exponential (e^x)."""
        return _UnaryOp(self, math.exp)

    def sqrt(self):
        """Square root (NaN if value < 0)."""
        return _UnaryOp(self, math.sqrt)

    # -- filters (return EpitopeFilter, not Expr) --

    def __le__(self, threshold):
        raise TypeError(
            "Comparison operators are only supported on Field objects "
            "(e.g. Affinity.value <= 500), not on computed expressions."
        )

    def __ge__(self, threshold):
        raise TypeError(
            "Comparison operators are only supported on Field objects "
            "(e.g. Affinity.score >= 0.5), not on computed expressions."
        )


class _Const(Expr):
    """A constant scalar value."""
    __slots__ = ("val",)

    def __init__(self, val):
        self.val = float(val)

    def evaluate(self, group_df):
        return self.val


class _BinOp(Expr):
    """Binary arithmetic operation on two expressions."""
    __slots__ = ("left", "right", "op")

    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op

    def evaluate(self, group_df):
        a = self.left.evaluate(group_df)
        b = self.right.evaluate(group_df)
        if a is None or b is None:
            return float("nan")
        if (isinstance(a, float) and math.isnan(a)) or (
            isinstance(b, float) and math.isnan(b)
        ):
            return float("nan")
        return self.op(a, b)


class _NormExpr(Expr):
    """Gaussian CDF normalization of an inner expression."""
    __slots__ = ("inner", "mean", "std")

    def __init__(self, inner, mean, std):
        self.inner = inner
        self.mean = float(mean)
        self.std = float(std)

    def evaluate(self, group_df):
        val = self.inner.evaluate(group_df)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan")
        if self.std == 0:
            return float("nan")
        return _gauss_cdf((val - self.mean) / self.std)


class _LogisticExpr(Expr):
    """Logistic sigmoid of an inner expression."""
    __slots__ = ("inner", "midpoint", "width")

    def __init__(self, inner, midpoint, width):
        self.inner = inner
        self.midpoint = float(midpoint)
        self.width = float(width)

    def evaluate(self, group_df):
        val = self.inner.evaluate(group_df)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan")
        if self.width == 0:
            return float("nan")
        try:
            return 1.0 / (1.0 + math.exp((val - self.midpoint) / self.width))
        except OverflowError:
            return 0.0


class _UnaryOp(Expr):
    """Apply a unary function to an inner expression."""
    __slots__ = ("inner", "fn")

    def __init__(self, inner, fn):
        self.inner = inner
        self.fn = fn

    def evaluate(self, group_df):
        val = self.inner.evaluate(group_df)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan")
        try:
            return float(self.fn(val))
        except (ValueError, OverflowError):
            return float("nan")


class _ClipExpr(Expr):
    """Clamp an inner expression to [lo, hi]."""
    __slots__ = ("inner", "lo", "hi")

    def __init__(self, inner, lo, hi):
        self.inner = inner
        self.lo = lo
        self.hi = hi

    def evaluate(self, group_df):
        val = self.inner.evaluate(group_df)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan")
        if self.lo is not None and val < self.lo:
            return float(self.lo)
        if self.hi is not None and val > self.hi:
            return float(self.hi)
        return val


def _as_expr(obj):
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, (int, float)):
        return _Const(obj)
    raise TypeError(f"Cannot convert {type(obj)} to Expr")


def _method_not_found_error(kind_name, method, available):
    """Build a ValueError for an unrecognized method, suggesting close matches."""
    msg = (
        f"No {kind_name} predictions from method matching {method!r}. "
        f"Available: {available}"
    )
    close = get_close_matches(method.lower(), [a.lower() for a in available], n=2, cutoff=0.6)
    if close:
        suggestions = [next(a for a in available if a.lower() == c) for c in close]
        msg += f". Did you mean: {suggestions}?"
    return ValueError(msg)


# ---------------------------------------------------------------------------
# Column — reference any DataFrame column in an expression
# ---------------------------------------------------------------------------


class Column(Expr):
    """Reference an arbitrary column in the predictions DataFrame.

    Reads the value from the first row of the group (peptide-level columns
    are constant across kind rows within a group).

    Use for peptide properties, variant metadata, or any custom annotation::

        Column("hydrophobicity") >= -0.5
        Column("n_alt_reads").sqrt()
        0.5 * Affinity.score - 0.2 * Column("cysteine_count")
    """

    __slots__ = ("col_name",)

    def __init__(self, col_name: str):
        self.col_name = col_name

    def evaluate(self, group_df):
        if group_df.empty:
            return float("nan")
        if self.col_name not in group_df.columns:
            available = sorted(group_df.columns)
            close = get_close_matches(self.col_name, available, n=3, cutoff=0.6)
            msg = f"Column {self.col_name!r} not found in DataFrame."
            if close:
                msg += f" Did you mean: {close}?"
            else:
                msg += f" Available: {available}"
            raise ValueError(msg)
        val = group_df.iloc[0][self.col_name]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan")
        try:
            return float(val)
        except (ValueError, TypeError):
            raise TypeError(
                f"Column {self.col_name!r} contains non-numeric value "
                f"{val!r} ({type(val).__name__}). "
                f"Only numeric columns can be used in ranking expressions."
            )


# ---------------------------------------------------------------------------
# Field — a reference to one column of one prediction kind
# ---------------------------------------------------------------------------


class Field(Expr):
    """Reference to a column of a specific prediction kind.

    Created via :class:`KindAccessor` attributes::

        Affinity.value   # IC50 / value column for pMHC_affinity
        Affinity.rank    # percentile_rank column
        Affinity.score   # score column (higher = better)

    Optionally qualified by prediction method::

        Affinity["netmhcpan"].value   # only NetMHCpan's affinity
    """

    __slots__ = ("kind", "field", "method")

    def __init__(self, kind: Kind, field: str, method: Optional[str] = None):
        self.kind = kind
        self.field = field
        self.method = method

    def evaluate(self, group_df):
        if group_df.empty or "kind" not in group_df.columns:
            return float("nan")
        kind_rows = group_df[group_df["kind"] == self.kind.value]
        if kind_rows.empty:
            return float("nan")
        col = "prediction_method_name"
        if self.method is not None:
            if col in kind_rows.columns:
                method_lower = self.method.lower()
                matched = kind_rows[
                    kind_rows[col].str.lower().str.contains(method_lower, na=False)
                ]
                if matched.empty:
                    available = sorted(kind_rows[col].dropna().unique())
                    raise _method_not_found_error(
                        self.kind.name, self.method, available
                    )
                kind_rows = matched
            # If column doesn't exist, keep all rows (legacy data)
        elif col in kind_rows.columns:
            methods = kind_rows[col].dropna().unique()
            if len(methods) > 1:
                kind_name = self.kind.name
                method_list = ", ".join(sorted(methods))
                raise ValueError(
                    f"Ambiguous: multiple models produce {kind_name} "
                    f"({method_list}). Use Affinity[\"modelname\"] to "
                    f"disambiguate."
                )
        if kind_rows.empty:
            return float("nan")
        try:
            val = kind_rows.iloc[0][self.field]
        except KeyError:
            return float("nan")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan")
        return float(val)

    # -- filter comparisons (only valid on Field, not compound Expr) --

    def __le__(self, threshold):
        if self.field.startswith("wt_"):
            raise TypeError(
                "WT fields can't be used in filters directly. "
                "Use WT() in ranking expressions instead, e.g.: "
                "rank_by=[Affinity.score - WT(Affinity).score]"
            )
        if self.field == "value":
            return EpitopeFilter(kind=self.kind, max_value=threshold, method=self.method)
        if self.field == "percentile_rank":
            return EpitopeFilter(kind=self.kind, max_percentile_rank=threshold, method=self.method)
        if self.field == "score":
            return EpitopeFilter(kind=self.kind, max_score=threshold, method=self.method)
        raise ValueError(f"Cannot apply <= to field {self.field!r}")

    def __ge__(self, threshold):
        if self.field.startswith("wt_"):
            raise TypeError(
                "WT fields can't be used in filters directly. "
                "Use WT() in ranking expressions instead, e.g.: "
                "rank_by=[Affinity.score - WT(Affinity).score]"
            )
        if self.field == "score":
            return EpitopeFilter(kind=self.kind, min_score=threshold, method=self.method)
        if self.field == "value":
            return EpitopeFilter(kind=self.kind, min_value=threshold, method=self.method)
        if self.field == "percentile_rank":
            return EpitopeFilter(kind=self.kind, min_percentile_rank=threshold, method=self.method)
        raise ValueError(f"Cannot apply >= to field {self.field!r}")

    def __lt__(self, threshold):
        return self.__le__(threshold)

    def __gt__(self, threshold):
        return self.__ge__(threshold)


# ---------------------------------------------------------------------------
# KindAccessor — attribute-style access to Fields for a given Kind
# ---------------------------------------------------------------------------


class KindAccessor:
    """Proxy for a prediction Kind with typed field access.

    Pre-built instances: ``Affinity``, ``Presentation``, ``Stability``,
    ``Processing``.  Build custom ones with ``KindAccessor(Kind.foo)``.

    Qualify by prediction method with bracket syntax::

        Affinity["netmhcpan"] <= 500
        Affinity["mhcflurry"].score

    When unqualified, uses the first matching row (works automatically
    when only one model produces the kind).

    The default field is ``value``, so comparisons and Expr methods on
    the accessor itself act on the value column::

        Affinity <= 500              # same as Affinity.value <= 500
        Affinity.norm(500, 200)      # same as Affinity.value.norm(500, 200)
    """

    __slots__ = ("kind", "method")

    def __init__(self, kind: Kind, method: Optional[str] = None):
        self.kind = kind
        self.method = method

    def __getitem__(self, method: str) -> "KindAccessor":
        """Qualify by prediction method name, e.g. Affinity["netmhcpan"]."""
        return KindAccessor(self.kind, method=method)

    @property
    def value(self) -> Field:
        """Kind-specific value (e.g. IC50 nM for affinity)."""
        return Field(self.kind, "value", method=self.method)

    @property
    def rank(self) -> Field:
        """Percentile rank (lower is better)."""
        return Field(self.kind, "percentile_rank", method=self.method)

    @property
    def score(self) -> Field:
        """Continuous score (higher is better)."""
        return Field(self.kind, "score", method=self.method)

    # Default comparisons delegate to .value
    def __le__(self, threshold):
        return self.value.__le__(threshold)

    def __lt__(self, threshold):
        return self.value.__lt__(threshold)

    def __ge__(self, threshold):
        return self.value.__ge__(threshold)

    def __gt__(self, threshold):
        return self.value.__gt__(threshold)

    # Delegate Expr methods to .value so Affinity.norm(...) works
    def norm(self, mean=0.0, std=1.0):
        return self.value.norm(mean, std)

    def logistic(self, midpoint=0.0, width=1.0):
        return self.value.logistic(midpoint, width)

    def clip(self, lo=None, hi=None):
        return self.value.clip(lo, hi)

    def log(self):
        return self.value.log()

    def log2(self):
        return self.value.log2()

    def log10(self):
        return self.value.log10()

    def log1p(self):
        return self.value.log1p()

    def exp(self):
        return self.value.exp()

    def sqrt(self):
        return self.value.sqrt()

    def __neg__(self):
        return -self.value

    def __abs__(self):
        return abs(self.value)

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value


# Top-level accessors for common kinds (unqualified — use when one model per kind)
Affinity = KindAccessor(Kind.pMHC_affinity)
Presentation = KindAccessor(Kind.pMHC_presentation)
Stability = KindAccessor(Kind.pMHC_stability)
Processing = KindAccessor(Kind.antigen_processing)


# ---------------------------------------------------------------------------
# WT — wildtype comparison wrapper
# ---------------------------------------------------------------------------


class WT:
    """Wrap a KindAccessor to read wildtype prediction columns.

    The ``add_wildtype_predictions()`` step populates ``wt_value``,
    ``wt_score``, and ``wt_percentile_rank`` columns alongside the
    mutant values.  ``WT()`` reads those instead::

        WT(Affinity).value                        # WT IC50
        WT(Affinity["netmhcpan"]).score           # qualified WT
        Affinity.score - WT(Affinity).score       # differential binding
        WT(Affinity).logistic(350, 150)           # WT scoring

    WT is for ranking expressions, not filters. Use
    ``rank_by=[Affinity.score - WT(Affinity).score]``.

    When WT columns don't exist (non-variant inputs), evaluates to NaN.
    """

    __slots__ = ("_accessor",)

    def __init__(self, accessor: KindAccessor):
        if not isinstance(accessor, KindAccessor):
            raise TypeError(
                f"WT() expects a KindAccessor (e.g. Affinity, "
                f"Presentation), got {type(accessor).__name__}"
            )
        self._accessor = accessor

    def __getitem__(self, method: str) -> "WT":
        return WT(self._accessor[method])

    @property
    def value(self) -> Field:
        return Field(self._accessor.kind, "wt_value", method=self._accessor.method)

    @property
    def rank(self) -> Field:
        return Field(self._accessor.kind, "wt_percentile_rank", method=self._accessor.method)

    @property
    def score(self) -> Field:
        return Field(self._accessor.kind, "wt_score", method=self._accessor.method)

    def _filter_error(self):
        raise TypeError(
            "WT fields can't be used in filters directly. "
            "Use WT() in ranking expressions instead, e.g.: "
            "rank_by=[Affinity.score - WT(Affinity).score]"
        )

    def __le__(self, other):
        self._filter_error()

    def __lt__(self, other):
        self._filter_error()

    def __ge__(self, other):
        self._filter_error()

    def __gt__(self, other):
        self._filter_error()

    # Delegate Expr methods to .value for use in ranking expressions
    def norm(self, mean=0.0, std=1.0):
        return self.value.norm(mean, std)

    def logistic(self, midpoint=0.0, width=1.0):
        return self.value.logistic(midpoint, width)

    def clip(self, lo=None, hi=None):
        return self.value.clip(lo, hi)

    def __neg__(self):
        return -self.value

    def __abs__(self):
        return abs(self.value)

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value


# ---------------------------------------------------------------------------
# Filter / strategy dataclasses with operator support
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpitopeFilter:
    """A filter criterion on one prediction kind.

    All specified thresholds must be satisfied (AND within a single filter).
    Combine with ``|`` (OR) or ``&`` (AND) to build a :class:`RankingStrategy`.

    Optionally scoped to a specific prediction method via ``method``::

        Affinity["netmhcpan"] <= 500  # only filters NetMHCpan rows
    """

    kind: Kind
    max_value: Optional[float] = None
    min_value: Optional[float] = None
    max_percentile_rank: Optional[float] = None
    min_percentile_rank: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    method: Optional[str] = None

    def __or__(self, other):
        return _combine(self, other, require_all=False)

    def __and__(self, other):
        return _combine(self, other, require_all=True)

    def rank_by(self, *exprs: Expr) -> RankingStrategy:
        return RankingStrategy(filters=[self], sort_by=list(exprs))


@dataclass(frozen=True)
class ColumnFilter:
    """A filter criterion on an arbitrary DataFrame column.

    Created via CLI ``column(name) <= threshold`` or programmatically::

        ColumnFilter("cysteine_count", max_value=2)
        ColumnFilter("hydrophobicity", min_value=-0.5)
    """

    col_name: str
    max_value: Optional[float] = None
    min_value: Optional[float] = None

    def __or__(self, other):
        return _combine(self, other, require_all=False)

    def __and__(self, other):
        return _combine(self, other, require_all=True)

    def rank_by(self, *exprs: Expr) -> "RankingStrategy":
        return RankingStrategy(filters=[self], sort_by=list(exprs))


@dataclass
class RankingStrategy:
    """Composite filter + ranking specification.

    Built by combining :class:`EpitopeFilter` instances with ``|`` / ``&``,
    or constructed directly.  ``sort_by`` accepts :class:`Expr` objects
    (including arithmetic compositions with ``.norm()``).
    """

    filters: list = field(default_factory=list)
    require_all: bool = False
    sort_by: list = field(default_factory=list)

    def __or__(self, other):
        return _combine(self, other, require_all=False)

    def __and__(self, other):
        return _combine(self, other, require_all=True)

    def rank_by(self, *exprs: Expr) -> RankingStrategy:
        return RankingStrategy(
            filters=list(self.filters),
            require_all=self.require_all,
            sort_by=list(exprs),
        )


def _combine(left, right, require_all):
    """Merge two filters/strategies. Only flatten children whose operator
    matches; otherwise nest to preserve semantics of (A | B) & C."""
    def _as_items(obj, parent_require_all):
        if isinstance(obj, EpitopeFilter):
            return [obj]
        # Only flatten if child uses the same logic as parent
        if isinstance(obj, RankingStrategy) and obj.require_all == parent_require_all:
            return list(obj.filters)
        # Different logic → keep as nested sub-strategy
        return [obj]

    return RankingStrategy(
        filters=_as_items(left, require_all) + _as_items(right, require_all),
        require_all=require_all,
    )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def affinity_filter(ic50_cutoff=None, percentile_cutoff=None, min_score=None):
    return EpitopeFilter(
        kind=Kind.pMHC_affinity,
        max_value=ic50_cutoff,
        max_percentile_rank=percentile_cutoff,
        min_score=min_score,
    )


def presentation_filter(max_rank=None, min_score=None):
    return EpitopeFilter(
        kind=Kind.pMHC_presentation,
        max_percentile_rank=max_rank,
        min_score=min_score,
    )


# ---------------------------------------------------------------------------
# Core apply logic
# ---------------------------------------------------------------------------

_GROUP_KEYS = ["source_sequence_name", "peptide", "peptide_offset", "allele"]
_GROUP_KEYS_VARIANT = ["variant", "peptide", "peptide_offset", "allele"]


def _pick_group_keys(df):
    if "variant" in df.columns:
        return _GROUP_KEYS_VARIANT
    return _GROUP_KEYS


def _method_matches(row, method):
    """Check if a row's prediction_method_name matches the filter method."""
    if method is None:
        return True
    name = row.get("prediction_method_name", "")
    if not name:
        return False
    return method.lower() in name.lower()


def _row_passes_filter(row, filt):
    if row["kind"] != filt.kind.value:
        return False
    if not _method_matches(row, filt.method):
        return False

    def _check(field, val, op):
        if val is None:
            return True
        v = row.get(field)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        return op(v, val)

    return (
        _check("value", filt.max_value, operator.le)
        and _check("value", filt.min_value, operator.ge)
        and _check("percentile_rank", filt.max_percentile_rank, operator.le)
        and _check("percentile_rank", filt.min_percentile_rank, operator.ge)
        and _check("score", filt.min_score, operator.ge)
        and _check("score", filt.max_score, operator.le)
    )


def _group_passes(group_df, strategy):
    if not strategy.filters:
        return True

    results = []
    for item in strategy.filters:
        if isinstance(item, RankingStrategy):
            # Nested sub-strategy — recurse
            results.append(_group_passes(group_df, item))
        elif isinstance(item, ColumnFilter):
            # Column-level filter — check first row
            if group_df.empty or item.col_name not in group_df.columns:
                results.append(False)
                continue
            val = group_df.iloc[0][item.col_name]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                results.append(False)
                continue
            passed = True
            if item.max_value is not None and val > item.max_value:
                passed = False
            if item.min_value is not None and val < item.min_value:
                passed = False
            results.append(passed)
        else:
            # EpitopeFilter
            kind_rows = group_df[group_df["kind"] == item.kind.value]
            if item.method is not None and not kind_rows.empty:
                col = "prediction_method_name"
                if col in kind_rows.columns:
                    method_lower = item.method.lower()
                    matched = kind_rows[
                        kind_rows[col].str.lower().str.contains(
                            method_lower, na=False
                        )
                    ]
                    if matched.empty:
                        available = sorted(
                            kind_rows[col].dropna().unique()
                        )
                        raise _method_not_found_error(
                            item.kind.name, item.method, available
                        )
                    kind_rows = matched
            if kind_rows.empty:
                results.append(False)
                continue
            passed = any(
                _row_passes_filter(row, item)
                for _, row in kind_rows.iterrows()
            )
            results.append(passed)

    if strategy.require_all:
        return all(results)
    return any(results)


def _sort_key_for_group(group_df, sort_by):
    """Evaluate sort_by expressions against a group, first non-NaN wins."""
    for expr in sort_by:
        val = expr.evaluate(group_df)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            return val
    return float("-inf")


def apply_ranking_strategy(df, strategy):
    """Apply a :class:`RankingStrategy` to a predictions DataFrame.

    Returns a filtered (and optionally sorted) copy, keeping all kind rows
    for peptide-allele groups that pass the filters.
    """
    if df.empty:
        return df

    group_keys = _pick_group_keys(df)
    grouped = df.groupby(group_keys, sort=False)

    if strategy.filters:
        keep_mask = pd.Series(False, index=df.index)
        for _key, group_df in grouped:
            if _group_passes(group_df, strategy):
                keep_mask.loc[group_df.index] = True
        df = df[keep_mask]

    if strategy.sort_by and not df.empty:
        grouped = df.groupby(group_keys, sort=False)
        sort_keys = {}
        for key, group_df in grouped:
            sort_keys[key] = _sort_key_for_group(group_df, strategy.sort_by)
        df = df.copy()
        df["_sort_key"] = df.set_index(group_keys).index.map(
            lambda k: sort_keys.get(k, float("-inf"))
        )
        df = df.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# String parsing — for CLI / config files
# ---------------------------------------------------------------------------

# Lowercase aliases for Kind names and short names
_KIND_ALIASES = {}
for _k in Kind:
    _KIND_ALIASES[_k.name.lower()] = _k
    # short aliases: "affinity" -> pMHC_affinity, "presentation" -> pMHC_presentation
    short = _k.name.lower().replace("pmhc_", "")
    _KIND_ALIASES[short] = _k
# Extra convenience aliases
_KIND_ALIASES["el"] = Kind.pMHC_presentation
_KIND_ALIASES["ba"] = Kind.pMHC_affinity
_KIND_ALIASES["aff"] = Kind.pMHC_affinity
_KIND_ALIASES["ic50"] = Kind.pMHC_affinity

_FIELD_ALIASES = {
    "value": "value", "val": "value", "ic50": "value",
    "rank": "percentile_rank", "percentile_rank": "percentile_rank",
    "percentile": "percentile_rank",
    "score": "score",
}


def _resolve_kind(name):
    """Resolve a kind alias to a Kind enum value.

    Accepts plain kind names (``"affinity"``, ``"ba"``) or
    tool-qualified names (``"netmhcpan_affinity"``).

    Returns ``(Kind, method)`` when called via :func:`_resolve_qualified_kind`,
    or just ``Kind`` for backwards compatibility.
    """
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
    """Resolve a possibly tool-qualified kind string.

    Returns ``(Kind, method_or_None)``.

    Examples::

        "affinity"             -> (Kind.pMHC_affinity, None)
        "netmhcpan_affinity"   -> (Kind.pMHC_affinity, "netmhcpan")
        "netmhcpan_ba"         -> (Kind.pMHC_affinity, "netmhcpan")
        "mhcflurry_el"         -> (Kind.pMHC_presentation, "mhcflurry")
    """
    key = name.strip().lower()
    # Try as a plain kind first
    if key in _KIND_ALIASES:
        return _KIND_ALIASES[key], None
    # Try splitting at each underscore from left to right
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


def _parse_column_ref(text):
    """Check if text is a column(name) reference. Returns name or None."""
    text = text.strip()
    if text.startswith("column(") and text.endswith(")"):
        name = text[7:-1].strip()
        if not name:
            raise ValueError("column() requires a column name, e.g. column(charge)")
        if "(" in name or ")" in name:
            raise ValueError(
                f"Invalid column name {name!r} — must be a plain name, "
                f"e.g. column(charge)"
            )
        return name
    return None


def parse_filter(text):
    """Parse a single filter expression from a string.

    Supported formats::

        "affinity <= 500"
        "affinity.value <= 500"
        "presentation.rank <= 2.0"
        "presentation.score >= 0.5"
        "ic50 <= 500"
        "el.rank <= 2"
        "netmhcpan_affinity <= 500"
        "mhcflurry_el.rank <= 2"
        "column(cysteine_count) <= 2"
        "column(hydrophobicity) >= -0.5"

    Returns an :class:`EpitopeFilter` or :class:`ColumnFilter`.
    """
    # Strip parentheses and whitespace (but not column() parens)
    text = text.strip()
    if not text.startswith("column("):
        text = text.strip("()")
    for op_str, op_name in [("<=", "le"), (">=", "ge"), ("<", "lt"), (">", "gt")]:
        if op_str in text:
            lhs, rhs = text.split(op_str, 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            try:
                threshold = float(rhs)
            except ValueError:
                raise ValueError(
                    f"Invalid threshold {rhs!r} in {text!r}. "
                    f"Right side of {op_str} must be a number."
                ) from None

            # Check for column(name) syntax
            col_name = _parse_column_ref(lhs)
            if col_name is not None:
                if op_name in ("le", "lt"):
                    return ColumnFilter(col_name=col_name, max_value=threshold)
                else:
                    return ColumnFilter(col_name=col_name, min_value=threshold)

            if "." in lhs:
                kind_str, field_str = lhs.rsplit(".", 1)
            else:
                kind_str, field_str = lhs, "value"

            kind, method = _resolve_qualified_kind(kind_str)
            field_name = _resolve_field(field_str)

            if op_name in ("le", "lt"):
                if field_name == "value":
                    return EpitopeFilter(kind=kind, max_value=threshold, method=method)
                elif field_name == "percentile_rank":
                    return EpitopeFilter(kind=kind, max_percentile_rank=threshold, method=method)
                elif field_name == "score":
                    return EpitopeFilter(kind=kind, max_score=threshold, method=method)
            elif op_name in ("ge", "gt"):
                if field_name == "value":
                    return EpitopeFilter(kind=kind, min_value=threshold, method=method)
                elif field_name == "percentile_rank":
                    return EpitopeFilter(kind=kind, min_percentile_rank=threshold, method=method)
                elif field_name == "score":
                    return EpitopeFilter(kind=kind, min_score=threshold, method=method)
            break
    else:
        raise ValueError(f"No comparison operator found in {text!r}")


def parse_ranking(text):
    """Parse a ranking expression from a string.

    Filters separated by ``|`` (OR) or ``&`` (AND)::

        "affinity <= 500"
        "affinity <= 500 | presentation.rank <= 2"
        "affinity <= 500 & presentation.score >= 0.5"

    Mixing ``|`` and ``&`` in one expression is not supported; use the
    Python operator API for complex nesting.

    Returns a :class:`RankingStrategy` (or :class:`EpitopeFilter` for a
    single filter).
    """
    text = text.strip()
    has_and = "&" in text
    has_or = "|" in text
    if has_and and has_or:
        raise ValueError(
            "Cannot mix '|' and '&' in a single ranking string. "
            "Use the Python API for complex nesting: "
            "(Affinity <= 500 | Presentation.rank <= 2) & Stability.score >= 0.5"
        )
    if has_and:
        parts = text.split("&")
        filters = [parse_filter(p) for p in parts]
        return RankingStrategy(filters=filters, require_all=True)
    elif has_or:
        parts = text.split("|")
        filters = [parse_filter(p) for p in parts]
        return RankingStrategy(filters=filters, require_all=False)
    else:
        return parse_filter(text)
