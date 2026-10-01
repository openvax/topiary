"""Named-criterion measurements and reasons, evaluated by the Topiary DSL."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ranking import DSLNode, PeptideView, parse
from .ranking.apply import _check_boolean_like, _collect_column_names, _collect_kinds
from .selection_criteria import _References


class _MeasuredCriterion(DSLNode):
    def __init__(self, values, node):
        self.values = values
        self.prediction_kinds = _collect_kinds(node)

    def child_nodes(self):
        return []

    def eval(self, ctx):
        return self.values.reindex(ctx.group_index)


@dataclass
class CriterionEvaluation:
    """Named measurements and group-indexed audit records.

    ``values`` maps criterion names to nullable per-group series, and
    ``decisions`` contains group keys, criterion identity, role, status, value,
    reason and detail. Use ``expression`` to compose the measured terms through
    the existing DSL without introducing input columns or evaluating predictors.
    """

    values: dict
    decisions: pd.DataFrame
    definitions: dict

    def expression(self, expression):
        """Resolve an explicit DSL expression against these measured criteria."""
        return parse(expression, criteria={name: _MeasuredCriterion(value, self.definitions[name])
                                          for name, value in self.values.items()})


def evaluate_selection_criteria(policy, context, *, references):
    """Measure referenced criteria and retain distinct evidence states.

    Parameters
    ----------
    policy : SelectionPolicy
        Complete definitions, already validated for roles and cycles.
    context : EvalContext
        Context on this evaluation stage's evidence. Predicate evaluation uses
        nullable boolean semantics; the caller decides whether unknown excludes.
    references : sequence of str
        Criteria used in this stage. Dependencies are evaluated recursively.
        Unreferenced definitions are reported as not_evaluated, not failures.

    Returns
    -------
    CriterionEvaluation
        Values plus per-group records. False applicability is not_applicable;
        absent measurements remain unknown; observed numeric zero is a value.
        Missing columns, ambiguous/conflicting models and out-of-domain results
        have distinct reasons. Unrecognized DSL errors still raise. A model
        ambiguity affecting a vector expression marks that expression unknown
        for its context; use source_contexts to express independent choices.
        Empty contexts return empty records with the same columns.
    """
    ctx = context.derive(preserve_unknown=True)
    definitions = {item.name: item for item in policy.criteria}
    expanded = policy.expanded
    nodes = {name: parse(record["expression"]["expression"]) for name, record in expanded["criteria"].items()}
    values, records = {}, []
    index = ctx.group_index

    def measured(text):
        node = parse(text, criteria=_References(lambda name: _MeasuredCriterion(measure(name), nodes[name])))
        missing = sorted(_collect_column_names(node) - set(ctx.df))
        if missing:
            return pd.Series(np.nan, index=index), "missing_column", ", ".join(missing)
        try:
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                result = node.eval(ctx).reindex(index)
        except ValueError as error:
            message = str(error)
            if "Conflicting prediction measurements" in message:
                reason = "conflicting_evidence"
            elif "Ambiguous" in message:
                reason = "ambiguous_model"
            elif "predictions" in message and ("No " in message or "not found" in message):
                reason = "missing_model"
            else:
                raise
            return pd.Series(np.nan, index=index), reason, message
        return result, None, None

    def measure(name):
        if name in values:
            return values[name]
        if name not in definitions:
            raise ValueError(f"Unresolved criterion {name!r}")
        criterion = definitions[name]
        applicable = pd.Series(True, index=index, dtype="boolean")
        if criterion.applies_to is not None:
            applicable, _, _ = measured(criterion.applies_to)
            _check_boolean_like(applicable)
            applicable = applicable.astype("boolean")
        result, failure, detail = measured(criterion.expression)
        if criterion.role == "eligibility":
            _check_boolean_like(result)
            result = result.astype("boolean")
        else:
            result = pd.to_numeric(result, errors="raise")
            if pd.api.types.is_bool_dtype(result):
                raise ValueError(f"Criterion {name!r} requires numeric {criterion.role} values, not booleans")
        numeric = pd.to_numeric(result, errors="raise").to_numpy(dtype=float, na_value=np.nan)
        nonfinite = ~np.isfinite(numeric) & ~np.isnan(numeric)
        # Missing leaf inputs distinguish absent evidence from undefined math.
        leaf_missing = pd.Series(False, index=index)
        if failure is None and result.isna().any():
            node = nodes[name]
            stack = [node]
            while stack:
                child = stack.pop()
                children = [] if isinstance(child, PeptideView) else child.child_nodes()
                if children:
                    stack.extend(children)
                else:
                    try:
                        leaf_missing |= child.eval(ctx).reindex(index).isna()
                    except ValueError:
                        leaf_missing[:] = True
        result = result.mask(nonfinite).where(applicable.fillna(False))
        values[name] = result
        identities = index.to_frame(index=False).to_dict("records")
        for i, identity in enumerate(identities):
            value = result.iloc[i]
            applies = applicable.iloc[i]
            if pd.isna(applies):
                status, reason = "unknown", "unknown_applicability"
            elif not applies:
                status, reason = "not_applicable", "applicability_false"
            elif failure:
                status, reason = "unknown", failure
            elif pd.isna(value):
                status = "unknown"
                reason = "out_of_domain" if nonfinite[i] or not leaf_missing.iloc[i] else "missing_evidence"
            elif criterion.role == "eligibility":
                status, reason = ("pass", "predicate_true") if value else ("fail", "predicate_false")
            else:
                status, reason = "value", "observed_zero" if value == 0 else "observed_value"
            records.append(dict(identity, criterion=name, role=criterion.role, value=value,
                                status=status, reason=reason, detail=detail))
        return result

    for name in references:
        measure(name)
    for name, criterion in definitions.items():
        if name not in values:
            for identity in index.to_frame(index=False).to_dict("records"):
                records.append(dict(identity, criterion=name, role=criterion.role, value=None,
                                    status="not_evaluated", reason="not_referenced", detail=None))
    columns = [*ctx.group_keys, "criterion", "role", "value", "status", "reason", "detail"]
    return CriterionEvaluation(values, pd.DataFrame(records, columns=columns), nodes)
