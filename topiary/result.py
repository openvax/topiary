"""TopiaryResult: DataFrame + provenance and pipeline metadata.

Carries all metadata fields directly (no nested ``.metadata`` indirection).
Filter/sort state is stored in both string form (for serialization) and
AST form (for programmatic use without re-parsing).
"""

import warnings
from collections import OrderedDict

import pandas as pd

from .io import Metadata
from .wide import detect_form


class TopiaryResult:
    """A prediction DataFrame bundled with its provenance and pipeline state.

    Parameters
    ----------
    df : pandas.DataFrame
        The underlying prediction data.
    topiary_version : str, optional
    form : str, optional
        "long" or "wide". Auto-detected from columns if not provided.
    models : dict, optional
        Model name → version string.
    sources : list of str, optional
        Files / tags that contributed rows.
    filter_by_str : str, optional
        Human-readable filter expression.
    filter_by_ast : object, optional
        Parsed filter (RankingStrategy, EpitopeFilter, etc.).
    sort_by_str : str, optional
    sort_by_ast : object, optional
        Parsed sort Expr.
    extra : dict, optional
        Unknown comment-block keys, preserved on round-trip.
    """

    def __init__(
        self,
        df,
        metadata=None,
        *,
        topiary_version=None,
        form=None,
        models=None,
        sources=None,
        filter_by_str=None,
        filter_by_ast=None,
        sort_by_str=None,
        sort_by_ast=None,
        extra=None,
    ):
        # Compat: accept a Metadata positionally and unpack its fields.
        if metadata is not None:
            topiary_version = topiary_version or metadata.topiary_version
            form = form or metadata.form
            models = models if models is not None else metadata.models
            sources = sources if sources is not None else metadata.sources
            filter_by_str = filter_by_str or getattr(metadata, "filter_by", None)
            sort_by_str = sort_by_str or getattr(metadata, "sort_by", None)
            extra = extra if extra is not None else metadata.extra

        self.df = df
        self.topiary_version = topiary_version
        self.form = form or detect_form(df)
        self.models = OrderedDict(models) if models else OrderedDict()
        self.sources = list(sources) if sources else []
        self.filter_by_str = filter_by_str
        self.filter_by_ast = filter_by_ast
        self.sort_by_str = sort_by_str
        self.sort_by_ast = sort_by_ast
        self.extra = OrderedDict(extra) if extra else OrderedDict()

    # -- DataFrame delegation ---------------------------------------------

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        return iter(self.df)

    def __getitem__(self, key):
        result = self.df[key]
        if isinstance(result, pd.DataFrame):
            return TopiaryResult(result, **self._field_kwargs())
        return result

    def __contains__(self, key):
        return key in self.df

    def __repr__(self):
        n = len(self.df)
        sources = ", ".join(self.sources) if self.sources else "<none>"
        return f"<TopiaryResult form={self.form} rows={n} sources=[{sources}]>"

    @property
    def columns(self):
        return self.df.columns

    @property
    def shape(self):
        return self.df.shape

    @property
    def empty(self):
        return self.df.empty

    def head(self, n=5):
        return TopiaryResult(self.df.head(n), **self._field_kwargs())

    def tail(self, n=5):
        return TopiaryResult(self.df.tail(n), **self._field_kwargs())

    def iterrows(self):
        return self.df.iterrows()

    def itertuples(self, *args, **kwargs):
        return self.df.itertuples(*args, **kwargs)

    # -- Form conversion --------------------------------------------------

    def to_wide(self):
        from .wide import to_wide as _to_wide
        wide_df = _to_wide(self.df)
        kwargs = self._field_kwargs()
        kwargs["form"] = "wide"
        return TopiaryResult(wide_df, **kwargs)

    def to_long(self):
        from .wide import from_wide
        long_df = from_wide(self.df, metadata=self._as_metadata())
        kwargs = self._field_kwargs()
        kwargs["form"] = "long"
        return TopiaryResult(long_df, **kwargs)

    # -- DSL operations ---------------------------------------------------

    def filter_by(self, expr):
        """Apply a filter expression. ANDs with any existing filter.

        Parameters
        ----------
        expr : str or DSL filter object
            A string like ``"affinity <= 500"`` or an object like
            ``Affinity <= 500``.

        Returns
        -------
        TopiaryResult
            New result with rows filtered and filter_by_str / filter_by_ast
            updated (ANDed with any previous filter).
        """
        from .ranking import (
            parse_ranking, apply_ranking_strategy,
            RankingStrategy, EpitopeFilter, ColumnFilter, ExprFilter,
        )

        if isinstance(expr, str):
            new_str = expr
            new_ast = parse_ranking(expr)
        elif isinstance(expr, (EpitopeFilter, ColumnFilter, ExprFilter, RankingStrategy)):
            new_ast = expr
            new_str = _dsl_filter_to_string(expr)
        else:
            raise TypeError(
                f"filter_by expects a string or DSL filter object, "
                f"got {type(expr).__name__}"
            )

        # Coerce to RankingStrategy for applying.
        if isinstance(new_ast, (EpitopeFilter, ColumnFilter, ExprFilter)):
            strategy = RankingStrategy(filters=[new_ast])
        else:
            strategy = new_ast

        # Apply to df.
        if self.df.empty:
            filtered_df = self.df
        else:
            filtered_df = apply_ranking_strategy(self.df, strategy)

        # Combine with existing filter.
        if self.filter_by_ast is not None:
            combined_ast = self.filter_by_ast & new_ast
            combined_str = _combine_filter_str(self.filter_by_str, new_str)
        else:
            combined_ast = new_ast
            combined_str = new_str

        kwargs = self._field_kwargs()
        kwargs["filter_by_str"] = combined_str
        kwargs["filter_by_ast"] = combined_ast
        return TopiaryResult(filtered_df, **kwargs)

    def sort_by(self, expr):
        """Sort rows by an expression. Replaces any previous sort.

        Parameters
        ----------
        expr : str or Expr object
            A string like ``"presentation.score"`` or an ``Expr`` object.

        Returns
        -------
        TopiaryResult
            New result with rows sorted and sort_by_str / sort_by_ast updated.
        """
        from .ranking import (
            Expr, RankingStrategy, apply_ranking_strategy, parse_expr,
        )

        if isinstance(expr, str):
            new_str = expr
            new_ast = parse_expr(expr)
        elif isinstance(expr, Expr):
            new_ast = expr
            new_str = repr(expr)
        else:
            raise TypeError(
                f"sort_by expects a string or Expr object, "
                f"got {type(expr).__name__}"
            )

        if self.df.empty:
            sorted_df = self.df
        else:
            strategy = RankingStrategy(filters=[], sort_by=[new_ast])
            sorted_df = apply_ranking_strategy(self.df, strategy)

        kwargs = self._field_kwargs()
        kwargs["sort_by_str"] = new_str
        kwargs["sort_by_ast"] = new_ast
        return TopiaryResult(sorted_df, **kwargs)

    # -- Serialization -----------------------------------------------------

    def to_tsv(self, path):
        from .io import to_tsv as _to_tsv
        _to_tsv(self.df, path, metadata=self._as_metadata())

    def to_csv(self, path):
        from .io import to_csv as _to_csv
        _to_csv(self.df, path, metadata=self._as_metadata())

    # -- Internal helpers --------------------------------------------------

    def _field_kwargs(self):
        """Return kwargs dict for reconstructing a copy."""
        return dict(
            topiary_version=self.topiary_version,
            form=self.form,
            models=OrderedDict(self.models),
            sources=list(self.sources),
            filter_by_str=self.filter_by_str,
            filter_by_ast=self.filter_by_ast,
            sort_by_str=self.sort_by_str,
            sort_by_ast=self.sort_by_ast,
            extra=OrderedDict(self.extra),
        )

    def _as_metadata(self):
        """Build a Metadata for serialization / legacy APIs."""
        return Metadata(
            topiary_version=self.topiary_version,
            form=self.form,
            models=OrderedDict(self.models),
            sources=list(self.sources),
            filter_by=self.filter_by_str,
            sort_by=self.sort_by_str,
            extra=OrderedDict(self.extra),
        )

    # Backward-compat: tests and older code access `.metadata`.
    @property
    def metadata(self):
        return self._as_metadata()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _combine_filter_str(old, new):
    """Combine two filter expression strings with AND.

    Parenthesizes both sides to handle cases where ``old`` or ``new``
    contains ``|`` or other operators.
    """
    if not old:
        return new
    if not new:
        return old
    return f"({old}) & ({new})"


def _dsl_filter_to_string(filt):
    """Convert a DSL filter object into a parseable string.

    Handles the common ``EpitopeFilter`` and ``RankingStrategy`` cases.
    Falls back to ``repr()`` for unknown types (not guaranteed to round-trip).
    """
    from .ranking import (
        EpitopeFilter, ColumnFilter, ExprFilter, RankingStrategy,
        _kind_short_name,
    )

    if isinstance(filt, EpitopeFilter):
        kind = _kind_short_name(filt.kind)
        prefix = f"{kind}[{filt.method!r}]" if filt.method else kind
        clauses = []
        if filt.max_value is not None:
            clauses.append(f"{prefix}.value <= {filt.max_value}")
        if filt.min_value is not None:
            clauses.append(f"{prefix}.value >= {filt.min_value}")
        if filt.max_percentile_rank is not None:
            clauses.append(f"{prefix}.rank <= {filt.max_percentile_rank}")
        if filt.min_percentile_rank is not None:
            clauses.append(f"{prefix}.rank >= {filt.min_percentile_rank}")
        if filt.max_score is not None:
            clauses.append(f"{prefix}.score <= {filt.max_score}")
        if filt.min_score is not None:
            clauses.append(f"{prefix}.score >= {filt.min_score}")
        return " & ".join(clauses) if clauses else prefix

    if isinstance(filt, RankingStrategy):
        if not filt.filters:
            return ""
        sub_strs = [_dsl_filter_to_string(f) for f in filt.filters]
        op = " & " if filt.require_all else " | "
        return op.join(f"({s})" for s in sub_strs)

    if isinstance(filt, (ColumnFilter, ExprFilter)):
        return repr(filt)

    return repr(filt)


def concat(results):
    """Concatenate TopiaryResults, preserving provenance.

    Parameters
    ----------
    results : list of TopiaryResult
        All must be in the same form (long or wide).

    Returns
    -------
    TopiaryResult
        DataFrames concatenated; metadata merged (sources concatenated,
        models union with warning on version conflicts; filter_by / sort_by
        preserved only if all inputs agree).
    """
    if not results:
        return TopiaryResult(pd.DataFrame())

    forms = {r.form for r in results}
    if len(forms) > 1:
        raise ValueError(
            f"Cannot concat TopiaryResults in different forms: {forms}"
        )
    form = results[0].form

    # Merge models with conflict detection.
    merged_models = OrderedDict()
    for r in results:
        for model, version in r.models.items():
            if model in merged_models and merged_models[model] != version:
                warnings.warn(
                    f"Model {model!r} has conflicting versions: "
                    f"{merged_models[model]!r} vs {version!r}",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                merged_models[model] = version

    merged_sources = []
    for r in results:
        merged_sources.extend(r.sources)

    merged_extra = OrderedDict()
    for r in results:
        merged_extra.update(r.extra)

    topiary_version = None
    for r in results:
        if r.topiary_version:
            topiary_version = r.topiary_version
            break

    # Preserve filter/sort only if all inputs agree.
    filter_strs = {r.filter_by_str for r in results}
    if len(filter_strs) == 1 and None not in filter_strs:
        filter_by_str = next(iter(filter_strs))
        filter_by_ast = results[0].filter_by_ast
    else:
        filter_by_str = None
        filter_by_ast = None

    sort_strs = {r.sort_by_str for r in results}
    if len(sort_strs) == 1 and None not in sort_strs:
        sort_by_str = next(iter(sort_strs))
        sort_by_ast = results[0].sort_by_ast
    else:
        sort_by_str = None
        sort_by_ast = None

    df = pd.concat([r.df for r in results], ignore_index=True)

    return TopiaryResult(
        df,
        topiary_version=topiary_version,
        form=form,
        models=merged_models,
        sources=merged_sources,
        filter_by_str=filter_by_str,
        filter_by_ast=filter_by_ast,
        sort_by_str=sort_by_str,
        sort_by_ast=sort_by_ast,
        extra=merged_extra,
    )
