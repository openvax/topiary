# Report-wide selection terms

Report-wide transforms operate on one evaluated value per occurrence, after
the caller's explicit grouping and prediction model selection. Raw duplicate
model rows do not add observations. One evaluation partition must contain only
one named sample; evaluate reports separately or supply separate policy source
contexts. A policy's pre-filter runs before its scoring population is formed.

```python
from topiary import Column, ordinal_rank
support = Column("support_reads").log2()
lens_support = support / support.population_max()
expression_rank = Column("allele_expression").dense_rank(0, 1)
order = ordinal_rank(Column("tier_order"), expression_rank, Column("gene"))
```

The same expressions parse from strings and survive policy evidence replay.
`population_max()` broadcasts the maximum finite value, leaving an empty or
entirely missing population unknown. It does not add pseudocounts or decide
what zero-denominator normalization should mean.

`dense_rank(ascending, missing_bottom)` starts at 1 and gives equal values equal
ranks. Both arguments are 0/1; defaults are ascending with missing ranks kept
missing. Explicit bottom ranking places all missing/nonfinite values after
finite values. `ordinal_rank(key, ...)` ranks ascending lexicographic tuples,
including text Column keys; missing keys sort last and exact ties remain tied.
Contradictory per-occurrence column values raise.

`expression.fillna(value)` explicitly fills missing numeric values. For an
arbitrary Column it also defines the absent-column fallback. This is useful
when CCF is genuinely optional, not permission to fabricate required evidence.
Ambiguous models, conflicting values and nonfinite domain errors stay visible.

These are generic ranking primitives, not fitted scientific models. Consumers
must still pin published definitions, evidence units, source parameters and
normalization populations. See openvax/topiary#462 and openvax/vaxrank#497.
