"""Named criteria resolve into the existing DSL without a mutable registry."""

from collections.abc import Mapping
from dataclasses import dataclass
import re

from .ranking import BoolOp, Comparison, Includes, IsIn, parse
from .ranking.parser import _Tokenizer


@dataclass(frozen=True)
class SelectionCriterion:
    """A reusable, explicitly scoped criterion.

    Parameters
    ----------
    name : str
        Stable identifier, referenced only by ``criterion("name")``. A same-named
        input column continues to be a column, so references cannot shadow data.
    expression : str
        Existing Topiary DSL expression, optionally referencing other criteria.
    role : {'eligibility', 'score', 'ranking'}
        Predicate, numeric score term, or numeric ordered tie-break expression.
        Eligibility references may compose only predicates; score references
        only score terms; ranking may reuse score or ranking terms.
    applies_to : str, optional
        Eligibility expression defining applicability. False means not
        applicable; missing means applicability unknown. None applies to all
        groups. Non-applicable references remain unknown, never observed false.
    """

    name: str
    expression: str
    role: str
    applies_to: str | None = None

    def __post_init__(self):
        for label in ("name", "expression", "role", "applies_to"):
            value = getattr(self, label)
            if label == "applies_to" and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Criterion {label} must be a nonempty string")
        if self.role not in {"eligibility", "score", "ranking"}:
            raise ValueError("Criterion role must be eligibility, score, or ranking")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", self.name):
            raise ValueError("Criterion names must be stable identifiers: letters, digits, dots, underscores or hyphens")

    def to_dict(self):
        """Return the complete portable definition, including applicability."""
        return dict(name=self.name, expression=self.expression, role=self.role, applies_to=self.applies_to)


@dataclass(frozen=True)
class RankingTerm:
    """One ordered numeric tie-break expression and explicit direction.

    ``expression`` is a DSL string, optionally containing named score/ranking
    references. ``ascending`` must be a bool. Terms follow the primary policy
    score in the supplied order, missing values last, with stable ties.
    """

    expression: str
    ascending: bool = False

    def __post_init__(self):
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise ValueError("Ranking expression must be a nonempty string")
        if type(self.ascending) is not bool:
            raise ValueError("Ranking direction must be a boolean")

    def to_dict(self):
        """Return a portable expression/direction record."""
        return dict(expression=self.expression, ascending=self.ascending)


class _References(Mapping):
    def __init__(self, lookup):
        self.lookup = lookup

    def __getitem__(self, name):
        return self.lookup(name)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


def resolve_selection_expression(expression, criteria=(), *, role="score"):
    """Expand named references, validating dependencies and output roles.

    Parameters
    ----------
    expression : str
        Explicit DSL arithmetic/AND/OR expression; no combination is inferred.
    criteria : sequence of SelectionCriterion
        Complete definitions. Empty is valid for a direct expression. Duplicate
        names, unknown references, cyclic dependencies, and wrong roles raise.
    role : {'eligibility', 'score', 'ranking'}
        Expected output role. The numeric/boolean runtime type is also checked
        when a policy is evaluated, since input columns have no static type.

    Returns
    -------
    dict
        Expanded DSL ``expression`` and ordered ``references`` (including
        transitive dependencies). Resolution uses only supplied definitions;
        no registry or input column is consulted. Applicability dependencies
        are validated and included even though the expanded expression itself
        describes only the measurement, not its applicability mask.
    """
    definitions = {}
    for criterion in criteria:
        if not isinstance(criterion, SelectionCriterion):
            raise ValueError("criteria must contain SelectionCriterion definitions")
        if criterion.name in definitions:
            raise ValueError(f"Duplicate criterion identifier {criterion.name!r}")
        definitions[criterion.name] = criterion
    if role not in {"eligibility", "score", "ranking"}:
        raise ValueError("Unknown expression role")
    visited, active = [], []

    def expand(text, expected):
        expanded_references = {}
        def lookup(name):
            if name not in definitions:
                raise ValueError(f"Unresolved criterion {name!r}")
            criterion = definitions[name]
            allowed = {"ranking", "score"} if expected == "ranking" else {expected}
            if criterion.role not in allowed:
                raise ValueError(f"Criterion {name!r} has role {criterion.role}, expected {expected}")
            if name in active:
                raise ValueError(f"Cyclic criterion references: {' -> '.join([*active, name])}")
            if name not in visited:
                visited.append(name)
            active.append(name)
            node, expanded = expand(criterion.expression, criterion.role)
            expanded_references[name] = expanded
            if criterion.applies_to is not None:
                expand(criterion.applies_to, "eligibility")
            active.pop()
            return node
        node = parse(text, criteria=_References(lookup))
        boolean = isinstance(node, (Comparison, BoolOp, IsIn, Includes))
        if expected != "eligibility" and boolean and active:
            raise ValueError(f"Expected numeric {expected} expression, got a predicate")
        # Preserve source parentheses and numeric literals. repr(node) is a
        # human display, not a lossless serialization of arithmetic grouping.
        tokens = _Tokenizer(text).tokens[:-1]
        rendered, i = [], 0
        while i < len(tokens):
            if (tokens[i][0] == "IDENT" and tokens[i][1].lower() == "criterion"
                    and i + 3 < len(tokens) and tokens[i + 1][0] == "LPAREN"):
                rendered.append("(" + expanded_references[tokens[i + 2][1]] + ")")
                i += 4
            else:
                kind, value = tokens[i]
                rendered.append(repr(value) if kind == "STRING" else value)
                i += 1
        return node, " ".join(rendered)

    _, expanded = expand(expression, role)
    return dict(expression=expanded, references=visited)
