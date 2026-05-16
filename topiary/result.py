"""TopiaryResult: DataFrame + provenance and pipeline metadata.

Carries all metadata fields directly (no nested ``.metadata`` indirection).
Filter/sort state is stored in both string form (for serialization) and
AST form (for programmatic use without re-parsing).
"""

import warnings
from collections import OrderedDict

import pandas as pd

from .io import Metadata, _models_from_dataframe
from .wide import detect_form


class _MissingIdentityValue:
    def __repr__(self):
        return "<NA>"


_MISSING_IDENTITY_VALUE = _MissingIdentityValue()


class TopiaryResult:
    """A prediction DataFrame bundled with its provenance and pipeline state.

    Delegates common DataFrame operations so most code that worked on a
    bare DataFrame continues to work.  **Follows pandas conventions for
    iteration / membership**:

    - ``"peptide" in result`` checks whether ``"peptide"`` is a *column*,
      not a row value.  Use ``"SIINFEKL" in result.df["peptide"].values``
      for row-value membership.
    - ``for x in result`` iterates column *names*, matching
      ``for x in df``.  Use ``result.iterrows()`` or ``result.df.values``
      for row iteration.

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
    filter_by_ast : DSLNode, optional
        Parsed filter — a :class:`topiary.ranking.DSLNode` (typically a
        :class:`Comparison` or :class:`BoolOp`). Supports ``&`` for
        ANDing with additional filters via :meth:`filter_by`.
    sort_by_str : str, optional
    sort_by_ast : DSLNode or list of DSLNode, optional
        Parsed sort expression(s).
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

        if models is None and hasattr(df, "attrs"):
            models = _models_from_dataframe(df)

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
        long_df = from_wide(self.df, metadata=self.metadata)
        kwargs = self._field_kwargs()
        kwargs["form"] = "long"
        return TopiaryResult(long_df, **kwargs)

    # -- DSL operations ---------------------------------------------------

    def filter_by(self, expr):
        """Apply a filter expression. ANDs with any existing filter.

        Parameters
        ----------
        expr : str or DSLNode
            A string like ``"affinity <= 500"`` or a DSL node like
            ``Affinity <= 500``.

        Returns
        -------
        TopiaryResult
            New result with rows filtered and filter_by_str /
            filter_by_ast updated (ANDed with any previous filter).
        """
        from .ranking import DSLNode, KindAccessor, apply_filter, parse

        if isinstance(expr, str):
            new_str = expr
            new_ast = parse(expr)
        elif isinstance(expr, KindAccessor):
            new_ast = expr.value
            new_str = new_ast.to_expr_string()
        elif isinstance(expr, DSLNode):
            new_ast = expr
            new_str = expr.to_expr_string()
        else:
            raise TypeError(
                f"filter_by expects a string or DSLNode, "
                f"got {type(expr).__name__}"
            )

        if self.df.empty:
            filtered_df = self.df
        else:
            filtered_df = apply_filter(self.df, new_ast)

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
        expr : str, DSLNode, or list of DSLNode
            Sort expression(s).

        Returns
        -------
        TopiaryResult
            New result with rows sorted.
        """
        from .ranking import DSLNode, KindAccessor, apply_sort, parse

        if isinstance(expr, str):
            new_str = expr
            new_ast = parse(expr)
            sort_nodes = [new_ast]
        elif isinstance(expr, KindAccessor):
            new_ast = expr.value
            new_str = new_ast.to_expr_string()
            sort_nodes = [new_ast]
        elif isinstance(expr, DSLNode):
            new_ast = expr
            new_str = expr.to_expr_string()
            sort_nodes = [new_ast]
        elif isinstance(expr, (list, tuple)):
            sort_nodes = [
                e.value if isinstance(e, KindAccessor) else e
                for e in expr
            ]
            for n in sort_nodes:
                if not isinstance(n, DSLNode):
                    raise TypeError(
                        f"sort_by list must contain DSLNode values, "
                        f"got {type(n).__name__}"
                    )
            new_ast = sort_nodes
            new_str = ", ".join(n.to_expr_string() for n in sort_nodes)
        else:
            raise TypeError(
                f"sort_by expects a string, DSLNode, or list, "
                f"got {type(expr).__name__}"
            )

        if self.df.empty:
            sorted_df = self.df
        else:
            sorted_df = apply_sort(self.df, sort_nodes)

        kwargs = self._field_kwargs()
        kwargs["sort_by_str"] = new_str
        kwargs["sort_by_ast"] = new_ast
        return TopiaryResult(sorted_df, **kwargs)

    # -- Serialization -----------------------------------------------------

    def to_tsv(self, path):
        from .io import to_tsv as _to_tsv
        _to_tsv(self.df, path, metadata=self.metadata)

    def to_csv(self, path):
        from .io import to_csv as _to_csv
        _to_csv(self.df, path, metadata=self.metadata)

    # -- Accessors / helpers ----------------------------------------------

    @property
    def metadata(self):
        """A fresh :class:`Metadata` built from this result's fields.

        Useful for passing to functions that expect a ``Metadata`` (e.g.
        :func:`to_tsv`, :func:`from_wide`) and for serializing the
        comment-block without touching private internals.
        """
        return Metadata(
            topiary_version=self.topiary_version,
            form=self.form,
            models=OrderedDict(self.models),
            sources=list(self.sources),
            filter_by=self.filter_by_str,
            sort_by=self.sort_by_str,
            extra=OrderedDict(self.extra),
        )

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


def _dsl_filter_to_string(node):
    """Convert a DSLNode into a parseable string via its to_expr_string."""
    from .ranking import DSLNode

    if isinstance(node, DSLNode):
        return node.to_expr_string()
    return repr(node)


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
        if any(r.filter_by_str for r in results):
            present = sorted({r.filter_by_str for r in results if r.filter_by_str})
            warnings.warn(
                "Dropping filter_by metadata: inputs to concat() have "
                f"differing filter history (found: {present}).  The rows are "
                "still filtered per their individual histories, but the "
                "combined result has no single filter expression that "
                "describes all of them.",
                UserWarning,
                stacklevel=2,
            )

    sort_strs = {r.sort_by_str for r in results}
    if len(sort_strs) == 1 and None not in sort_strs:
        sort_by_str = next(iter(sort_strs))
        sort_by_ast = results[0].sort_by_ast
    else:
        sort_by_str = None
        sort_by_ast = None
        if any(r.sort_by_str for r in results):
            present = sorted({r.sort_by_str for r in results if r.sort_by_str})
            warnings.warn(
                "Dropping sort_by metadata: inputs to concat() have "
                f"differing sort history (found: {present}).  The concatenated "
                "rows are no longer in a consistent sort order.",
                UserWarning,
                stacklevel=2,
            )

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


def combine_predictor_results(results, on=("peptide", "allele")):
    """Combine separate predictor outputs into one predictor-equivalent result.

    This is stricter than :func:`concat`: every input must cover the same
    identity key set, and no prediction method may appear in more than one
    input. It is intended for the common single-allele case where, for example,
    NetMHCpan and MHCflurry are run separately and then stacked into the same
    long-form shape produced by running both models together.

    Parameters
    ----------
    results : iterable of TopiaryResult or pandas.DataFrame
        Separate predictor outputs to combine.
    on : tuple of str
        Columns defining the strict identity set. Defaults to
        ``("peptide", "allele")``.

    Returns
    -------
    TopiaryResult
        Combined long-form result with merged model metadata.
    """
    results = [_as_topiary_result(r) for r in results]
    if not results:
        return TopiaryResult(pd.DataFrame())

    if isinstance(on, str):
        on = (on,)
    on = tuple(on)

    for i, result in enumerate(results):
        _validate_predictor_result(result, i, on)

    _validate_unique_prediction_methods(results)
    _validate_same_identity_keys(results, on)

    combined = concat(results)
    if "kind_support" in combined.extra:
        extra = OrderedDict(combined.extra)
        extra.pop("kind_support", None)
        combined.extra = extra
    return combined


def _as_topiary_result(result):
    if isinstance(result, TopiaryResult):
        return result
    if isinstance(result, pd.DataFrame):
        return TopiaryResult(result)
    raise TypeError(
        "combine_predictor_results expects TopiaryResult or pandas.DataFrame "
        f"inputs, got {type(result).__name__}"
    )


def _validate_predictor_result(result, index, on):
    if result.form != "long":
        raise ValueError(
            "combine_predictor_results only supports long-form predictor "
            f"results; result {index} has form {result.form!r}"
        )
    required = set(on) | {"kind", "prediction_method_name"}
    missing = sorted(c for c in required if c not in result.df.columns)
    if missing:
        raise ValueError(
            f"combine_predictor_results result {index} is missing required "
            f"column(s): {missing}"
        )
    if not _prediction_methods(result):
        raise ValueError(
            f"combine_predictor_results result {index} has no "
            "prediction_method_name values"
        )


def _prediction_methods(result):
    return {
        str(method)
        for method in result.df["prediction_method_name"].dropna().unique()
    }


def _validate_unique_prediction_methods(results):
    seen = {}
    for index, result in enumerate(results):
        for method in sorted(_prediction_methods(result)):
            if method in seen:
                raise ValueError(
                    "combine_predictor_results cannot combine duplicate "
                    f"prediction method {method!r}; found in results "
                    f"{seen[method]} and {index}"
                )
            seen[method] = index


def _identity_keys(result, on):
    return {
        tuple(_normalize_identity_value(value) for value in key)
        for key in (
            result.df.loc[:, list(on)]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    }


def _normalize_identity_value(value):
    try:
        if pd.isna(value):
            return _MISSING_IDENTITY_VALUE
    except (TypeError, ValueError):
        pass
    return value


def _validate_same_identity_keys(results, on):
    baseline = _identity_keys(results[0], on)
    for index, result in enumerate(results[1:], start=1):
        current = _identity_keys(result, on)
        missing = baseline - current
        extra = current - baseline
        if missing or extra:
            message = [
                "combine_predictor_results requires every input to cover the "
                f"same {on!r} keys; result {index} differs from result 0."
            ]
            if missing:
                message.append(
                    f"Missing from result {index}: {_format_key_examples(missing)}"
                )
            if extra:
                message.append(
                    f"Extra in result {index}: {_format_key_examples(extra)}"
                )
            raise ValueError(" ".join(message))


def _format_key_examples(keys, limit=5):
    ordered = sorted(keys, key=repr)
    shown = ordered[:limit]
    suffix = "" if len(ordered) <= limit else f" ... +{len(ordered) - limit} more"
    return f"{shown}{suffix}"
