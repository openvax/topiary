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

    def ascending_cdf(self, mean=0.0, std=1.0):
        """Gaussian left CDF: **higher input → higher output**.

        P(X ≤ x) — the area to the left under the curve.
        Use for "higher is better" fields like ``.score``::

            Presentation.score.ascending_cdf(mean=0.5, std=0.3)

        For "lower is better" fields (IC50, rank), use
        :meth:`descending_cdf` instead.
        """
        return _NormExpr(self, mean, std)

    # Keep norm as alias for ascending_cdf
    norm = ascending_cdf

    def descending_cdf(self, mean=0.0, std=1.0):
        """Gaussian right CDF (1-CDF): **lower input → higher output**.

        P(X > x) — the area to the right under the curve.
        Use for "lower is better" fields like IC50 and percentile rank::

            Affinity.descending_cdf(mean=500, std=200)
            Affinity.rank.descending_cdf(mean=5, std=3)

        For "higher is better" fields, use :meth:`ascending_cdf` instead.
        """
        return _SurvivalExpr(self, mean, std)

    def logistic(self, midpoint=0.0, width=1.0):
        """Logistic sigmoid: **lower input → higher output**.

        ``1 / (1 + exp((x - midpoint) / width))``.
        Values below the midpoint score > 0.5; above score < 0.5.
        Use for "lower is better" fields like IC50::

            Affinity.logistic(midpoint=350, width=150)
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

    def hinge(self):
        """``max(0, x)``. Zeroes out negative values."""
        return _ClipExpr(self, lo=0, hi=None)

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

    def __repr__(self):
        v = self.val
        if v == int(v):
            return str(int(v))
        return repr(v)


def _fmt_num(v):
    """Format a number for repr: 500.0 → '500', 0.5 → '0.5'."""
    if v is None:
        return "None"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return repr(v)


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

    def __repr__(self):
        sym = _OP_SYMBOLS.get(self.op, "?")
        left = repr(self.left)
        right = repr(self.right)
        # Parenthesize lower-precedence sub-expressions
        if isinstance(self.left, _BinOp) and _op_prec(self.left.op) < _op_prec(self.op):
            left = f"({left})"
        if isinstance(self.right, _BinOp) and _op_prec(self.right.op) < _op_prec(self.op):
            right = f"({right})"
        return f"{left} {sym} {right}"


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

    def __repr__(self):
        return f"{repr(self.inner)}.ascending_cdf({_fmt_num(self.mean)}, {_fmt_num(self.std)})"


class _SurvivalExpr(Expr):
    """Survival function (1 - Gaussian CDF) of an inner expression."""
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
        return 1.0 - _gauss_cdf((val - self.mean) / self.std)

    def __repr__(self):
        return f"{repr(self.inner)}.descending_cdf({_fmt_num(self.mean)}, {_fmt_num(self.std)})"


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

    def __repr__(self):
        return f"{repr(self.inner)}.logistic({_fmt_num(self.midpoint)}, {_fmt_num(self.width)})"


_UNARY_NAMES = {
    abs: "abs", math.log: "log", math.log2: "log2",
    math.log10: "log10", math.log1p: "log1p",
    math.exp: "exp", math.sqrt: "sqrt",
}


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

    def __repr__(self):
        name = _UNARY_NAMES.get(self.fn)
        if name == "abs":
            return f"abs({repr(self.inner)})"
        if name:
            return f"{repr(self.inner)}.{name}()"
        return f"{repr(self.inner)}.<?>()"


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

    def __repr__(self):
        if self.lo == 0 and self.hi is None:
            return f"{repr(self.inner)}.hinge()"
        return f"{repr(self.inner)}.clip({_fmt_num(self.lo)}, {_fmt_num(self.hi)})"


def _as_expr(obj):
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, (int, float)):
        return _Const(obj)
    raise TypeError(f"Cannot convert {type(obj)} to Expr")


# ---------------------------------------------------------------------------
# Aggregation functions — combine multiple expressions
# ---------------------------------------------------------------------------


class _AggExpr(Expr):
    """Aggregate multiple expressions with a reducing function."""
    __slots__ = ("exprs", "agg_fn", "name")

    def __init__(self, exprs, agg_fn, name="?"):
        self.exprs = exprs
        self.agg_fn = agg_fn
        self.name = name

    def evaluate(self, group_df):
        vals = []
        for e in self.exprs:
            v = e.evaluate(group_df)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
        if not vals:
            return float("nan")
        return self.agg_fn(vals)

    def __repr__(self):
        args = ", ".join(repr(e) for e in self.exprs)
        return f"{self.name}({args})"


def mean(*exprs):
    """Arithmetic mean of expressions. NaN values are skipped."""
    return _AggExpr([_as_expr(e) for e in exprs], lambda vs: sum(vs) / len(vs), "mean")


def geomean(*exprs):
    """Geometric mean of expressions. NaN and non-positive values are skipped."""
    def _geomean(vs):
        pos = [v for v in vs if v > 0]
        if not pos:
            return float("nan")
        return math.exp(sum(math.log(v) for v in pos) / len(pos))
    return _AggExpr([_as_expr(e) for e in exprs], _geomean, "geomean")


def minimum(*exprs):
    """Minimum of expressions. NaN values are skipped."""
    return _AggExpr([_as_expr(e) for e in exprs], min, "minimum")


def maximum(*exprs):
    """Maximum of expressions. NaN values are skipped."""
    return _AggExpr([_as_expr(e) for e in exprs], max, "maximum")


def median(*exprs):
    """Median of expressions. NaN values are skipped.

    For even count, returns mean of the two middle values.
    """
    def _median(vs):
        vs = sorted(vs)
        n = len(vs)
        if n % 2 == 1:
            return vs[n // 2]
        return (vs[n // 2 - 1] + vs[n // 2]) / 2.0
    return _AggExpr([_as_expr(e) for e in exprs], _median, "median")


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

    def __repr__(self):
        return f"column({self.col_name})"

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

    Optionally scoped to an alternate peptide context::

        wt.Affinity.score             # wildtype affinity score
    """

    __slots__ = ("kind", "field", "method", "scope")

    def __init__(self, kind: Kind, field: str, method: Optional[str] = None,
                 scope: str = ""):
        self.kind = kind
        self.field = field
        self.method = method
        self.scope = scope  # "" for default, "wt_", "shuffled_", "self_"

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
        col_name = self.scope + self.field
        try:
            val = kind_rows.iloc[0][col_name]
        except KeyError:
            return float("nan")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan")
        return float(val)

    def __repr__(self):
        kind_name = self.kind.name.lower().replace("pmhc_", "")
        # Map internal field names to DSL names
        if self.field == "percentile_rank":
            field_str = "rank"
        else:
            field_str = self.field
        # Build the accessor string
        if self.method:
            accessor = f"{kind_name}['{self.method}']"
        else:
            accessor = kind_name
        # Prepend scope prefix
        scope_str = self.scope.rstrip("_") + "." if self.scope else ""
        return f"{scope_str}{accessor}.{field_str}"

    # -- filter comparisons (only valid on unscoped Field) --

    def _check_scope_for_filter(self):
        if self.scope:
            scope_name = self.scope.rstrip("_")
            raise TypeError(
                f"Scoped fields ({scope_name}.*) can't be used in filters. "
                f"Use them in ranking expressions instead, e.g.: "
                f"rank_by=[Affinity.score - {scope_name}.Affinity.score]"
            )

    def __le__(self, threshold):
        self._check_scope_for_filter()
        if self.field == "value":
            return EpitopeFilter(kind=self.kind, max_value=threshold, method=self.method)
        if self.field == "percentile_rank":
            return EpitopeFilter(kind=self.kind, max_percentile_rank=threshold, method=self.method)
        if self.field == "score":
            return EpitopeFilter(kind=self.kind, max_score=threshold, method=self.method)
        raise ValueError(f"Cannot apply <= to field {self.field!r}")

    def __ge__(self, threshold):
        self._check_scope_for_filter()
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

    Scope to an alternate peptide context::

        wt.Affinity.score             # wildtype affinity score
        shuffled.Affinity.score       # shuffled decoy score

    When unqualified, uses the first matching row (works automatically
    when only one model produces the kind).

    The default field is ``value``, so comparisons and Expr methods on
    the accessor itself act on the value column::

        Affinity <= 500              # same as Affinity.value <= 500
        Affinity.norm(500, 200)      # same as Affinity.value.norm(500, 200)
    """

    __slots__ = ("kind", "method", "scope")

    def __init__(self, kind: Kind, method: Optional[str] = None,
                 scope: str = ""):
        self.kind = kind
        self.method = method
        self.scope = scope

    def __getitem__(self, method: str) -> "KindAccessor":
        """Qualify by prediction method name, e.g. Affinity["netmhcpan"]."""
        return KindAccessor(self.kind, method=method, scope=self.scope)

    @property
    def value(self) -> Field:
        """Kind-specific value (e.g. IC50 nM for affinity)."""
        return Field(self.kind, "value", method=self.method, scope=self.scope)

    @property
    def rank(self) -> Field:
        """Percentile rank (lower is better)."""
        return Field(self.kind, "percentile_rank", method=self.method, scope=self.scope)

    @property
    def score(self) -> Field:
        """Continuous score (higher is better)."""
        return Field(self.kind, "score", method=self.method, scope=self.scope)

    # Default comparisons delegate to .value
    def __le__(self, threshold):
        return self.value.__le__(threshold)

    def __lt__(self, threshold):
        return self.value.__lt__(threshold)

    def __ge__(self, threshold):
        return self.value.__ge__(threshold)

    def __gt__(self, threshold):
        return self.value.__gt__(threshold)

    # Delegate Expr methods to .value so Affinity.ascending_cdf(...) works
    def ascending_cdf(self, mean=0.0, std=1.0):
        return self.value.ascending_cdf(mean, std)

    norm = ascending_cdf  # alias

    def descending_cdf(self, mean=0.0, std=1.0):
        return self.value.descending_cdf(mean, std)

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
# Scope — alternate peptide context (wt, shuffled, self)
# ---------------------------------------------------------------------------

# Reserved context keywords — cannot be used as kind/method names
_CONTEXT_KEYWORDS = {"wt", "shuffled", "self"}


class Scope:
    """Select an alternate peptide context for field access.

    Pre-built instances: ``wt``, ``shuffled``.

    Use as a prefix to access predictions on an alternate peptide::

        wt.Affinity.score                       # wildtype affinity score
        wt.Affinity["netmhcpan"].descending_cdf(500, 200)
        Affinity.score - wt.Affinity.score      # differential binding
        shuffled.Affinity.score                 # shuffled decoy score
        wt.len                                  # wildtype peptide length

    In the string DSL, use dot-prefix syntax::

        "wt.affinity.score"
        "affinity.score - wt.affinity.score"
        "wt.len"
        "wt.count('C')"
    """

    __slots__ = ("prefix", "name")

    def __init__(self, name: str):
        self.name = name
        self.prefix = name + "_"

    def __getattr__(self, attr):
        # Avoid infinite recursion for __slots__ attrs
        if attr in ("prefix", "name"):
            raise AttributeError(attr)
        # len → Len with scope
        if attr == "len":
            return Len(scope=self.prefix)
        # Look up as kind alias
        attr_lower = attr.lower()
        if attr_lower in _KIND_ALIASES:
            return KindAccessor(_KIND_ALIASES[attr_lower], scope=self.prefix)
        # Try as a KindAccessor constant name (Affinity, Presentation, etc.)
        kind_map = {
            "affinity": Kind.pMHC_affinity,
            "presentation": Kind.pMHC_presentation,
            "stability": Kind.pMHC_stability,
            "processing": Kind.antigen_processing,
        }
        if attr_lower in kind_map:
            return KindAccessor(kind_map[attr_lower], scope=self.prefix)
        raise AttributeError(
            f"Unknown kind {attr!r} in scope {self.name!r}. "
            f"Available: affinity, presentation, stability, processing, "
            f"ba, el, aff, ic50"
        )

    def count(self, chars: str) -> "Count":
        """Count amino acid character(s) in this context's peptide."""
        return Count(chars, scope=self.prefix)

    def __repr__(self):
        return self.name


# Module-level scope instances
wt = Scope("wt")
shuffled = Scope("shuffled")
self_scope = Scope("self")


# ---------------------------------------------------------------------------
# Len — peptide length (precomputed column)
# ---------------------------------------------------------------------------


class Len(Expr):
    """Peptide length, read from a precomputed column.

    Reads ``peptide_length`` (default scope) or ``{scope}peptide_length``
    (e.g. ``wt_peptide_length``).
    """

    __slots__ = ("scope",)

    def __init__(self, scope: str = ""):
        self.scope = scope

    def evaluate(self, group_df):
        col = self.scope + "peptide_length"
        if col not in group_df.columns:
            return float("nan")
        val = group_df.iloc[0][col]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan")
        return float(val)

    def __repr__(self):
        if self.scope:
            return f"{self.scope.rstrip('_')}.len"
        return "len"


# ---------------------------------------------------------------------------
# Count — dynamic amino acid count from peptide string
# ---------------------------------------------------------------------------


class Count(Expr):
    """Count occurrences of amino acid character(s) in the peptide string.

    Reads directly from the ``peptide`` column (or ``{scope}peptide``)
    at evaluation time — no precomputed column required.

    The argument is a string of one or more amino acid single-letter codes.
    Each character is counted independently and the counts are summed::

        Count("C")     # cysteines
        Count("KR")    # basic residues (K + R)
    """

    __slots__ = ("chars", "scope")

    def __init__(self, chars: str, scope: str = ""):
        self.chars = chars.upper()
        self.scope = scope

    def evaluate(self, group_df):
        peptide_col = self.scope + "peptide" if self.scope else "peptide"
        if peptide_col not in group_df.columns:
            return float("nan")
        peptide = group_df.iloc[0][peptide_col]
        if not isinstance(peptide, str) or not peptide:
            return float("nan")
        return float(sum(peptide.count(c) for c in self.chars))

    def __repr__(self):
        scope_str = self.scope.rstrip("_") + "." if self.scope else ""
        return f"{scope_str}count('{self.chars}')"


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


# ---------------------------------------------------------------------------
# Expression parser — full transform/arithmetic DSL for --rank-by
# ---------------------------------------------------------------------------

# Maps string names to aggregation constructors
_AGGREGATION_FUNCS = {
    "mean": mean,
    "geomean": geomean,
    "minimum": minimum,
    "maximum": maximum,
    "median": median,
}

# Maps string names to Expr transform methods
_TRANSFORM_NAMES = {
    "ascending_cdf", "descending_cdf", "norm", "logistic",
    "clip", "hinge", "log", "log2", "log10", "log1p", "exp", "sqrt",
}

# Maps string names to top-level KindAccessor instances
_KIND_ACCESSOR_ALIASES = {
    "affinity": Affinity,
    "presentation": Presentation,
    "stability": Stability,
    "processing": Processing,
    "ba": Affinity,
    "aff": Affinity,
    "ic50": Affinity,
    "el": Presentation,
}


class _ExprTokenizer:
    """Simple tokenizer for the expression DSL.

    Token types: NUMBER, IDENT, OP, LPAREN, RPAREN, LBRACKET, RBRACKET,
    DOT, COMMA, STRING, EOF
    """

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.tokens = []
        self._tokenize()
        self._idx = 0

    def _tokenize(self):
        import re
        i = 0
        text = self.text
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            # Numbers (including negative literals after operators)
            m = re.match(r'(\d+\.?\d*([eE][+-]?\d+)?)', text[i:])
            if m:
                self.tokens.append(("NUMBER", float(m.group())))
                i += m.end()
                continue
            # Quoted strings
            if text[i] in ('"', "'"):
                quote = text[i]
                j = i + 1
                while j < len(text) and text[j] != quote:
                    j += 1
                if j >= len(text):
                    raise ValueError(f"Unterminated string at position {i}")
                self.tokens.append(("STRING", text[i + 1:j]))
                i = j + 1
                continue
            # Identifiers
            if text[i].isalpha() or text[i] == '_':
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                    j += 1
                self.tokens.append(("IDENT", text[i:j]))
                i = j
                continue
            # Two-char operators
            if i + 1 < len(text) and text[i:i + 2] == '**':
                self.tokens.append(("OP", "**"))
                i += 2
                continue
            # Single-char tokens
            c = text[i]
            if c in '+-*/':
                self.tokens.append(("OP", c))
            elif c == '(':
                self.tokens.append(("LPAREN", c))
            elif c == ')':
                self.tokens.append(("RPAREN", c))
            elif c == '[':
                self.tokens.append(("LBRACKET", c))
            elif c == ']':
                self.tokens.append(("RBRACKET", c))
            elif c == '.':
                self.tokens.append(("DOT", c))
            elif c == ',':
                self.tokens.append(("COMMA", c))
            else:
                raise ValueError(
                    f"Unexpected character {c!r} at position {i} in {self.text!r}"
                )
            i += 1
        self.tokens.append(("EOF", None))

    def peek(self):
        return self.tokens[self._idx]

    def advance(self):
        tok = self.tokens[self._idx]
        self._idx += 1
        return tok

    def expect(self, ttype, value=None):
        tok = self.advance()
        if tok[0] != ttype:
            raise ValueError(
                f"Expected {ttype} but got {tok[0]} ({tok[1]!r}) "
                f"in {self.text!r}"
            )
        if value is not None and tok[1] != value:
            raise ValueError(
                f"Expected {value!r} but got {tok[1]!r} in {self.text!r}"
            )
        return tok


class _ExprParser:
    """Recursive descent parser for the expression DSL.

    Grammar::

        expr     := term (('+' | '-') term)*
        term     := power (('*' | '/') power)*
        power    := unary ('**' power)?
        unary    := '-' unary | postfix
        postfix  := atom ('.' IDENT call? | '[' STRING ']')*
        atom     := NUMBER
                  | '(' expr ')'
                  | 'abs' '(' expr ')'
                  | AGGREGATION '(' expr (',' expr)* ')'
                  | 'count' '(' STRING ')'
                  | 'column' '(' IDENT ')'
                  | 'len'
                  | CONTEXT '.' scoped_atom
                  | kind_ref
        scoped_atom := 'len'
                     | 'count' '(' STRING ')'
                     | kind_ref
        kind_ref := IDENT ('[' STRING ']')?
        call     := '(' arglist? ')'
        arglist  := expr (',' expr)*

        CONTEXT  := 'wt' | 'shuffled' | 'self'
    """

    def __init__(self, text):
        self.tokenizer = _ExprTokenizer(text)
        self.text = text

    def parse(self):
        expr = self._parse_expr()
        tok = self.tokenizer.peek()
        if tok[0] != "EOF":
            raise ValueError(
                f"Unexpected token {tok[1]!r} after expression in {self.text!r}"
            )
        return expr

    def _parse_expr(self):
        """expr := term (('+' | '-') term)*"""
        left = self._parse_term()
        while self.tokenizer.peek() == ("OP", "+") or self.tokenizer.peek() == ("OP", "-"):
            op_tok = self.tokenizer.advance()
            right = self._parse_term()
            if op_tok[1] == "+":
                left = left + right
            else:
                left = left - right
        return left

    def _parse_term(self):
        """term := power (('*' | '/') power)*"""
        left = self._parse_power()
        while self.tokenizer.peek() == ("OP", "*") or self.tokenizer.peek() == ("OP", "/"):
            op_tok = self.tokenizer.advance()
            right = self._parse_power()
            if op_tok[1] == "*":
                left = left * right
            else:
                left = left / right
        return left

    def _parse_power(self):
        """power := unary ('**' power)?"""
        base = self._parse_unary()
        if self.tokenizer.peek() == ("OP", "**"):
            self.tokenizer.advance()
            exp = self._parse_power()  # right-associative
            base = base ** exp
        return base

    def _parse_unary(self):
        """unary := '-' unary | atom chain*"""
        if self.tokenizer.peek() == ("OP", "-"):
            self.tokenizer.advance()
            inner = self._parse_unary()
            return -inner
        return self._parse_postfix()

    def _parse_postfix(self):
        """atom followed by zero or more .method(args) or [method] chains"""
        node = self._parse_atom()
        while True:
            tok = self.tokenizer.peek()
            if tok[0] == "DOT":
                self.tokenizer.advance()
                name_tok = self.tokenizer.expect("IDENT")
                name = name_tok[1]
                # Check if it's a method call with parens
                if self.tokenizer.peek()[0] == "LPAREN":
                    args = self._parse_call_args()
                    node = self._apply_transform(node, name, args)
                else:
                    # Field access: .value, .rank, .score
                    node = self._apply_field_access(node, name)
            elif tok[0] == "LBRACKET":
                self.tokenizer.advance()
                method_tok = self.tokenizer.expect("STRING")
                self.tokenizer.expect("RBRACKET")
                node = self._apply_bracket(node, method_tok[1])
            else:
                break
        return node

    def _parse_atom(self):
        """Parse a single atom (number, identifier, paren group, etc.)"""
        tok = self.tokenizer.peek()

        # Number literal
        if tok[0] == "NUMBER":
            self.tokenizer.advance()
            return _Const(tok[1])

        # Parenthesized expression
        if tok[0] == "LPAREN":
            self.tokenizer.advance()
            expr = self._parse_expr()
            self.tokenizer.expect("RPAREN")
            return expr

        # Identifier-based atoms
        if tok[0] == "IDENT":
            name = tok[1].lower()

            # abs(expr)
            if name == "abs":
                self.tokenizer.advance()
                self.tokenizer.expect("LPAREN")
                inner = self._parse_expr()
                self.tokenizer.expect("RPAREN")
                return abs(inner)

            # Aggregation functions: mean(...), geomean(...), etc.
            if name in _AGGREGATION_FUNCS:
                self.tokenizer.advance()
                args = self._parse_call_args()
                return _AGGREGATION_FUNCS[name](*args)

            # len — peptide length (default scope)
            if name == "len":
                self.tokenizer.advance()
                return Len()

            # count('X') — amino acid count (default scope)
            if name == "count":
                self.tokenizer.advance()
                self.tokenizer.expect("LPAREN")
                chars_tok = self.tokenizer.expect("STRING")
                self.tokenizer.expect("RPAREN")
                return Count(chars_tok[1])

            # Context scope: wt.kind_ref, shuffled.kind_ref, self.kind_ref
            if name in _CONTEXT_KEYWORDS:
                # Check if followed by '.' — if so, it's a scope prefix
                self.tokenizer.advance()
                if self.tokenizer.peek()[0] == "DOT":
                    self.tokenizer.advance()
                    scope = name + "_"
                    return self._parse_scoped_atom(scope)
                else:
                    # Not followed by dot — try as a kind name (unlikely but safe)
                    raise ValueError(
                        f"{name!r} is a reserved context keyword. "
                        f"Use '{name}.kind.field' syntax, e.g. '{name}.affinity.score'"
                    )

            # column(name)
            if name == "column":
                self.tokenizer.advance()
                self.tokenizer.expect("LPAREN")
                col_tok = self.tokenizer.expect("IDENT")
                self.tokenizer.expect("RPAREN")
                return Column(col_tok[1])

            # Kind accessor (e.g. affinity, presentation, affinity.score)
            accessor = self._parse_kind_accessor()
            return accessor

        raise ValueError(
            f"Unexpected token {tok!r} in expression {self.text!r}"
        )

    def _parse_scoped_atom(self, scope):
        """Parse an atom after a scope prefix (wt., shuffled., self.)."""
        tok = self.tokenizer.peek()
        if tok[0] != "IDENT":
            raise ValueError(
                f"Expected identifier after scope prefix in {self.text!r}"
            )
        name = tok[1].lower()

        # wt.len
        if name == "len":
            self.tokenizer.advance()
            return Len(scope=scope)

        # wt.count('X')
        if name == "count":
            self.tokenizer.advance()
            self.tokenizer.expect("LPAREN")
            chars_tok = self.tokenizer.expect("STRING")
            self.tokenizer.expect("RPAREN")
            return Count(chars_tok[1], scope=scope)

        # wt.affinity, wt.ba, wt.netmhcpan_affinity, etc.
        accessor = self._parse_kind_accessor(scope=scope)
        return accessor

    def _parse_kind_accessor(self, scope=""):
        """Parse a kind accessor like 'affinity', 'affinity["netmhcpan"]'."""
        name_tok = self.tokenizer.expect("IDENT")
        name = name_tok[1].lower()

        if name in _CONTEXT_KEYWORDS:
            raise ValueError(
                f"{name!r} is a reserved context keyword and cannot be "
                f"used as a kind name in {self.text!r}"
            )

        if name not in _KIND_ACCESSOR_ALIASES:
            # Try as a tool-qualified kind: netmhcpan_affinity
            kind, method = _resolve_qualified_kind(name)
            accessor = KindAccessor(kind, method=method, scope=scope)
        else:
            accessor = KindAccessor(_KIND_ACCESSOR_ALIASES[name].kind, scope=scope)

        # Optional [method] qualifier
        if self.tokenizer.peek()[0] == "LBRACKET":
            self.tokenizer.advance()
            method_tok = self.tokenizer.expect("STRING")
            self.tokenizer.expect("RBRACKET")
            accessor = accessor[method_tok[1]]

        return accessor

    def _parse_call_args(self):
        """Parse '(' expr (',' expr)* ')' — returns list of Expr."""
        self.tokenizer.expect("LPAREN")
        args = []
        if self.tokenizer.peek()[0] != "RPAREN":
            args.append(self._parse_expr())
            while self.tokenizer.peek()[0] == "COMMA":
                self.tokenizer.advance()
                args.append(self._parse_expr())
        self.tokenizer.expect("RPAREN")
        return args

    def _apply_transform(self, node, name, args):
        """Apply a named transform method to a node."""
        name_lower = name.lower()
        if name_lower not in _TRANSFORM_NAMES:
            available = sorted(_TRANSFORM_NAMES)
            close = get_close_matches(name_lower, available, n=3, cutoff=0.6)
            msg = f"Unknown transform {name!r}."
            if close:
                msg += f" Did you mean: {close}?"
            else:
                msg += f" Available: {available}"
            raise ValueError(msg)

        # For KindAccessor, delegate to .value first if needed
        if isinstance(node, KindAccessor):
            method = getattr(node, name_lower)
            float_args = [a.val if isinstance(a, _Const) else a for a in args]
            return method(*float_args)

        if not isinstance(node, Expr):
            raise ValueError(
                f"Cannot apply .{name}() to {type(node).__name__}"
            )

        method = getattr(node, name_lower, None)
        if method is None:
            raise ValueError(f"Expr has no method {name!r}")
        float_args = [a.val if isinstance(a, _Const) else a for a in args]
        return method(*float_args)

    def _apply_field_access(self, node, name):
        """Apply .value, .rank, .score field access."""
        name_lower = name.lower()
        # First check if it's a transform with no args
        if name_lower in _TRANSFORM_NAMES:
            return self._apply_transform(node, name, [])

        if isinstance(node, KindAccessor):
            field_name = _FIELD_ALIASES.get(name_lower)
            if field_name is not None:
                if field_name == "value":
                    return node.value
                elif field_name == "percentile_rank":
                    return node.rank
                elif field_name == "score":
                    return node.score
            raise ValueError(
                f"Unknown field {name!r}. Available: value, rank, score"
            )
        raise ValueError(
            f"Cannot access .{name} on {type(node).__name__}"
        )

    def _apply_bracket(self, node, method):
        """Apply ["method"] qualifier."""
        if isinstance(node, KindAccessor):
            return node[method]
        raise ValueError(
            f"Cannot use ['...'] on {type(node).__name__}"
        )


def parse_expr(text):
    """Parse a ranking expression string into an :class:`Expr` tree.

    Supports the full expression DSL including transforms, arithmetic,
    aggregations, scoped contexts, and method qualification::

        "affinity.descending_cdf(500, 200)"
        "0.5 * affinity.score + 0.5 * presentation.score"
        "mean(affinity.logistic(350, 150), presentation.score)"
        "affinity['netmhcpan'].descending_cdf(500, 200)"
        "affinity.value.clip(1, 50000)"
        "affinity.value.log()"
        "column(hydrophobicity)"
        "wt.affinity.score"
        "affinity.score - wt.affinity.score"
        "len"
        "wt.len"
        "count('C')"
        "wt.count('KR')"

    Returns an :class:`Expr` (or :class:`KindAccessor` which delegates
    to ``.value`` for evaluation).
    """
    parser = _ExprParser(text)
    result = parser.parse()
    # If we got a bare KindAccessor, convert to its .value field
    if isinstance(result, KindAccessor):
        return result.value
    return result
