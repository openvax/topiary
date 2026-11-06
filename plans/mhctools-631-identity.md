# mhctools #631 downstream compatibility

Extend the existing allele-validation PR #578 before it lands. Match canonical
parser options explicitly, reject G/P typings, and require concrete same-species
alpha/beta components. Keep Topiary's null-expression eligibility separate.
Use the shared mhctools identity corpus and exercise distinct DR beta/pair keys
through prediction normalization, caches, occurrence joins and serialized reload.
Review and run full lint/test/CI gates; ship Topiary 5.111.0 before mhctools 4.9.0.
The shared Python environment is outside this work.
