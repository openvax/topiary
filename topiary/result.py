"""TopiaryResult: DataFrame + provenance and pipeline metadata.

Carries all metadata fields directly (no nested ``.metadata`` indirection).
Filter/sort state is stored in both string form (for serialization) and
AST form (for programmatic use without re-parsing).
"""

import warnings
from collections import OrderedDict

import pandas as pd

from .io import Metadata, _model_version_str, _models_from_dataframe
from .wide import detect_form


class _MissingIdentityValue:
    def __repr__(self):
        return "<NA>"


_MISSING_IDENTITY_VALUE = _MissingIdentityValue()


_SOURCE_CONTEXT_IDENTITY_COLUMNS = (
    "sample_name",
    "source_sequence_name",
    "peptide_offset",
    "peptide_length",
    "n_flank",
    "c_flank",
)


def _dataframe_fingerprint(df):
    """Return a cheap-ish fingerprint for detecting active-frame mutation."""
    columns = tuple(str(column) for column in df.columns)
    dtypes = tuple(str(dtype) for dtype in df.dtypes)
    try:
        value_hash = int(pd.util.hash_pandas_object(df, index=True).sum())
    except (TypeError, ValueError):
        # Some object columns may carry unhashable values. Fall back to a
        # stringified hash so converted views are still invalidated for common
        # in-place edits without making all DataFrame access defensive copies.
        try:
            value_hash = int(
                pd.util.hash_pandas_object(df.astype(str), index=True).sum()
            )
        except (TypeError, ValueError):
            value_hash = None
    return (id(df), df.shape, columns, dtypes, value_hash)


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
        The active prediction data view.  TopiaryResult keeps lazy long and
        wide views internally when conversion is possible, so Topiary-level
        operations can normalize representation without requiring callers to
        choose a form up front.
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
        _long_df=None,
        _wide_df=None,
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

        self.topiary_version = topiary_version
        input_form = form or detect_form(df)
        if input_form not in {"long", "wide", "unknown"}:
            raise ValueError(f"Unknown TopiaryResult form: {input_form!r}")
        self._df = df
        self._unknown_df = None
        self._long_df = _long_df
        self._wide_df = _wide_df
        if input_form == "long":
            self._long_df = df
        elif input_form == "wide":
            self._wide_df = df
        else:
            self._unknown_df = df
        self._long_source_fingerprint = None
        self._wide_source_fingerprint = None
        if self._df is self._long_df and self._wide_df is not None:
            self._wide_source_fingerprint = _dataframe_fingerprint(self._long_df)
        if self._df is self._wide_df and self._long_df is not None:
            self._long_source_fingerprint = _dataframe_fingerprint(self._wide_df)
        self.models = OrderedDict(models) if models else OrderedDict()
        self.sources = list(sources) if sources else []
        self.filter_by_str = filter_by_str
        self.filter_by_ast = filter_by_ast
        self.sort_by_str = sort_by_str
        self.sort_by_ast = sort_by_ast
        self.extra = OrderedDict(extra) if extra else OrderedDict()

    # -- DataFrame delegation ---------------------------------------------

    @property
    def form(self):
        """Form of the active ``.df`` view, kept for backward compatibility."""
        if self._df is self._long_df:
            return "long"
        if self._df is self._wide_df:
            return "wide"
        if self._df is self._unknown_df:
            return "unknown"
        return detect_form(self._df)

    @form.setter
    def form(self, value):
        value = value or "unknown"
        if value == self.form:
            return
        if value == "long":
            self._df = self.long_df
            if self._wide_df is not None:
                self._wide_source_fingerprint = _dataframe_fingerprint(self._long_df)
            return
        if value == "wide":
            self._df = self.wide_df
            if self._long_df is not None:
                self._long_source_fingerprint = _dataframe_fingerprint(self._wide_df)
            return
        if value == "unknown":
            if self._unknown_df is None:
                raise ValueError("Cannot set TopiaryResult form to 'unknown'")
            self._df = self._unknown_df
            return
        raise ValueError(f"Unknown TopiaryResult form: {value!r}")

    @property
    def df(self):
        """Active DataFrame view for backward-compatible pandas access."""
        return self._df

    @df.setter
    def df(self, value):
        """Replace the active DataFrame and invalidate converted views."""
        detected = detect_form(value)
        self._df = value
        self._unknown_df = None
        self._long_df = value if detected == "long" else None
        self._wide_df = value if detected == "wide" else None
        if detected == "unknown":
            self._unknown_df = value
        self._long_source_fingerprint = None
        self._wide_source_fingerprint = None

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
        wide_df = self.wide_df
        kwargs = self._field_kwargs()
        kwargs["form"] = "wide"
        return TopiaryResult(
            wide_df,
            **kwargs,
            _long_df=self._long_df,
            _wide_df=wide_df,
        )

    def to_long(self):
        long_df = self.long_df
        kwargs = self._field_kwargs()
        kwargs["form"] = "long"
        return TopiaryResult(
            long_df,
            **kwargs,
            _long_df=long_df,
            _wide_df=self._wide_df,
        )

    @property
    def long_df(self):
        """Long-form DataFrame view, computed lazily when needed."""
        if (
            self._df is self._wide_df
            and self._long_df is not None
            and self._long_source_fingerprint != _dataframe_fingerprint(self._wide_df)
        ):
            self._long_df = None
            self._long_source_fingerprint = None
        if self._long_df is None:
            if self._wide_df is not None:
                from .wide import from_wide
                self._long_df = from_wide(self._wide_df, metadata=self.metadata)
                self._long_source_fingerprint = _dataframe_fingerprint(self._wide_df)
            else:
                raise ValueError(
                    f"Cannot convert TopiaryResult with form {self.form!r} "
                    "to long form"
                )
        return self._long_df

    @property
    def wide_df(self):
        """Wide-form DataFrame view, computed lazily when needed."""
        if (
            self._df is self._long_df
            and self._wide_df is not None
            and self._wide_source_fingerprint != _dataframe_fingerprint(self._long_df)
        ):
            self._wide_df = None
            self._wide_source_fingerprint = None
        if self._wide_df is None:
            if self._long_df is not None:
                from .wide import to_wide as _to_wide
                self._wide_df = _to_wide(self._long_df)
                self._wide_source_fingerprint = _dataframe_fingerprint(self._long_df)
            else:
                raise ValueError(
                    f"Cannot convert TopiaryResult with form {self.form!r} "
                    "to wide form"
                )
        return self._wide_df

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

        df = self.long_df
        if df.empty:
            filtered_df = df
        else:
            filtered_df = apply_filter(df, new_ast)

        if self.filter_by_ast is not None:
            combined_ast = self.filter_by_ast & new_ast
            combined_str = _combine_filter_str(self.filter_by_str, new_str)
        else:
            combined_ast = new_ast
            combined_str = new_str

        kwargs = self._field_kwargs()
        kwargs["form"] = "long"
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

        df = self.long_df
        if df.empty:
            sorted_df = df
        else:
            sorted_df = apply_sort(df, sort_nodes)

        kwargs = self._field_kwargs()
        kwargs["form"] = "long"
        kwargs["sort_by_str"] = new_str
        kwargs["sort_by_ast"] = new_ast
        return TopiaryResult(sorted_df, **kwargs)

    # -- Serialization -----------------------------------------------------

    def to_tsv(self, path):
        from .io import to_tsv as _to_tsv
        _to_tsv(self, path)

    def to_csv(self, path):
        from .io import to_csv as _to_csv
        _to_csv(self, path)

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
        Results to concatenate.  Long and wide results may be mixed; mixed
        inputs are normalized to long form.

    Returns
    -------
    TopiaryResult
        DataFrames concatenated; metadata merged (sources concatenated,
        models union with warning on version conflicts; filter_by / sort_by
        preserved only if all inputs agree).  The active output form is the
        shared input form when all inputs match, otherwise long.
    """
    if not results:
        return TopiaryResult(pd.DataFrame())
    for result in results:
        if not isinstance(result, TopiaryResult):
            raise TypeError(
                "topiary.concat expects TopiaryResult inputs; use "
                "TopiaryResult(df) to attach Topiary semantics before "
                f"concatenating, got {type(result).__name__}"
            )

    forms = {r.form for r in results}
    if "unknown" in forms:
        raise ValueError(
            f"Cannot concat TopiaryResults with unknown form: {forms}"
        )
    form = results[0].form if len(forms) == 1 else "long"

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

    if form == "long":
        frames = [r.long_df for r in results]
    elif form == "wide":
        frames = [r.wide_df for r in results]
    else:
        raise ValueError(f"Cannot concat TopiaryResults with form {form!r}")

    df = pd.concat(frames, ignore_index=True)

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


def combine_predictor_results(results, on=("peptide", "allele"), coverage="complete"):
    """Combine separate predictor outputs into one predictor-equivalent result.

    This is stricter than :func:`concat`: duplicate predictions are rejected,
    and by default every emitted prediction method/kind must cover the same
    identity key set.  It supports both common split patterns:

    - different predictors run separately on the same peptides;
    - the same predictor run separately over disjoint allele/length shards.

    Parameters
    ----------
    results : iterable of TopiaryResult or pandas.DataFrame
        Separate predictor outputs to combine.
    on : tuple of str
        Columns defining the strict identity set. Defaults to
        ``("peptide", "allele")``.  Source context columns such as
        ``sample_name``, ``source_sequence_name``, and ``peptide_offset``
        are also checked when present so repeated peptide/allele rows
        remain distinct.
    coverage : {"complete", "partial"} or bool
        ``"complete"`` (default) requires each emitted
        ``(prediction_method_name, kind)`` group to cover the same identity
        key set, matching a normal multi-predictor run.  ``"partial"``
        allows sparse unions but still rejects duplicate predictions.
        ``True`` and ``False`` are accepted as aliases for ``"complete"``
        and ``"partial"``.

    Returns
    -------
    TopiaryResult
        Combined long-form result with merged model metadata.
    """
    results = [_as_topiary_result(r) for r in results]
    results = [r for r in results if not r.df.empty]
    if not results:
        return TopiaryResult(pd.DataFrame())
    results = [r.to_long() for r in results]

    if isinstance(on, str):
        on = (on,)
    on = tuple(on)
    coverage = _normalize_coverage_mode(coverage)

    for i, result in enumerate(results):
        _validate_predictor_result(result, i, on)

    results = [_drop_non_identity_source(result, on) for result in results]
    identity_columns = _identity_columns(results, on)
    combined = concat(results)
    _validate_no_duplicate_predictions(combined.df, identity_columns)
    if coverage == "complete":
        _validate_complete_prediction_coverage(combined.df, identity_columns)
    combined.models = _models_from_observed_rows(combined.df, combined.models)
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


def _normalize_coverage_mode(coverage):
    if coverage is True:
        return "complete"
    if coverage is False:
        return "partial"
    if coverage not in {"complete", "partial"}:
        raise ValueError(
            "combine_predictor_results coverage must be 'complete' or "
            f"'partial', got {coverage!r}"
        )
    return coverage


def _drop_non_identity_source(result, on):
    if "source" not in result.df.columns or "source" in on:
        return result
    return TopiaryResult(
        result.df.drop(columns=["source"]),
        **result._field_kwargs(),
    )


def _models_from_observed_rows(df, fallback_models):
    models = OrderedDict()
    fallback = OrderedDict(
        (str(model).strip(), _model_version_str(version))
        for model, version in fallback_models.items()
        if str(model).strip()
    )
    for method, rows in (
        df.dropna(subset=["prediction_method_name"])
        .groupby("prediction_method_name", sort=False)
    ):
        method_str = str(method).strip()
        if not method_str:
            continue
        version = _version_from_rows(rows)
        if not version:
            version = fallback.get(method_str, "")
        models[method_str] = version
    return models


def _version_from_rows(rows):
    if "predictor_version" not in rows.columns:
        return ""
    for version in rows["predictor_version"]:
        version_str = _model_version_str(version)
        if version_str:
            return version_str
    return ""


def _identity_columns(results, on):
    columns = list(on)
    for column in _SOURCE_CONTEXT_IDENTITY_COLUMNS:
        if column not in columns and any(column in r.df.columns for r in results):
            columns.append(column)
    return tuple(columns)


def _identity_frame(df, columns):
    identity_df = pd.DataFrame(index=df.index)
    for column in columns:
        if column in df.columns:
            identity_df[column] = df[column]
        else:
            identity_df[column] = pd.NA
    return identity_df


def _key_set(df):
    return {
        tuple(_normalize_identity_value(value) for value in key)
        for key in df.drop_duplicates().itertuples(index=False, name=None)
    }


def _normalize_identity_value(value):
    try:
        if pd.isna(value):
            return _MISSING_IDENTITY_VALUE
    except (TypeError, ValueError):
        pass
    return value


def _prediction_key_frame(df, identity_columns):
    return pd.concat(
        [
            _identity_frame(df, ("prediction_method_name", "kind")),
            _identity_frame(df, identity_columns),
        ],
        axis=1,
    )


def _validate_no_duplicate_predictions(df, identity_columns):
    key_df = _prediction_key_frame(df, identity_columns)
    seen = {}
    duplicates = []
    for row_index, key in zip(
        key_df.index, key_df.itertuples(index=False, name=None)
    ):
        normalized = tuple(_normalize_identity_value(value) for value in key)
        if normalized in seen:
            duplicates.append(normalized)
        else:
            seen[normalized] = row_index
    if duplicates:
        raise ValueError(
            "combine_predictor_results found duplicate predictions for "
            "(prediction_method_name, kind, identity) keys: "
            f"{_format_key_examples(set(duplicates))}"
        )


def _prediction_group_key_sets(df, identity_columns):
    key_df = _prediction_key_frame(df, identity_columns)
    key_df = key_df.rename(
        columns={
            "prediction_method_name": "_prediction_method_name",
            "kind": "_kind",
        }
    )
    groups = OrderedDict()
    for (method, kind), group in key_df.groupby(
        ["_prediction_method_name", "_kind"], dropna=False, sort=False
    ):
        method_key = _normalize_identity_value(method)
        kind_key = _normalize_identity_value(kind)
        groups[(method_key, kind_key)] = _key_set(group.loc[:, list(identity_columns)])
    return groups


def _validate_complete_prediction_coverage(df, identity_columns):
    groups = _prediction_group_key_sets(df, identity_columns)
    if not groups:
        return

    baseline_group, baseline_keys = next(iter(groups.items()))
    for group, keys in list(groups.items())[1:]:
        missing = baseline_keys - keys
        extra = keys - baseline_keys
        if missing or extra:
            message = [
                "combine_predictor_results coverage='complete' requires every "
                "(prediction_method_name, kind) group to cover the same "
                f"{identity_columns!r} keys; group {group!r} differs from "
                f"group {baseline_group!r}."
            ]
            if missing:
                message.append(
                    f"Missing from group {group!r}: {_format_key_examples(missing)}"
                )
            if extra:
                message.append(
                    f"Extra in group {group!r}: {_format_key_examples(extra)}"
                )
            raise ValueError(" ".join(message))


def _format_key_examples(keys, limit=5):
    ordered = sorted(keys, key=repr)
    shown = ordered[:limit]
    suffix = "" if len(ordered) <= limit else f" ... +{len(ordered) - limit} more"
    return f"{shown}{suffix}"
