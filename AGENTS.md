# AGENTS.md — topiary

Guide for coding agents working in this repo. Read this before touching code.

---

## Golden Rules

1. **Never commit to `main`.** Always `git checkout -b <feature-branch>` before editing. Land via PR.
2. **Every PR bumps the version.** Even doc-only PRs — at minimum a patch bump. `deploy.sh <version>` handles the bump + commit + push.
3. **"Done" means merged AND deployed to PyPI** — never stop at merge. After a PR merges, run `./deploy.sh` from a clean main. Skipping deploy = task not done.
4. **File problems as issues, don't silently work around them.** If you hit a bug here or in a sibling openvax/pirl-unc repo, open a GitHub issue on the correct repo and link it from the PR.
5. **After a PR ships, look for the next block of work.** Read open issues across the relevant openvax repos, group by dependency + urgency. Prefer *foundational* changes that unblock multiple downstream improvements; otherwise chain the smallest independent improvements.

---

## Before Completing Any Task

Before telling the user a change is "complete":

1. **`./lint.sh`** — must pass
2. **`./test.sh`** — must pass
3. For a PR: **CI must be green on GitHub**, then merge, then **`./deploy.sh`**.

`deploy.sh` gates on lint + test, refuses to run off `main`/`master`, and refuses a dirty tree — don't work around these. If deploy fails, fix the root cause.

## Scripts

- `./develop.sh` — editable install (dev mode)
- `./lint.sh` — ruff check
- `./test.sh` — pytest (with coverage where configured)
- `./deploy.sh [version]` — lint → test → optional version bump → build → twine upload → tag → push

## Test the whole, not the parts

**Before saying a workflow is supported, run it.** Checking that the pieces
exist is not checking that they compose.

The failure this comes from: asked whether a downstream consumer was blocked, I
verified that the four capabilities their design needed were exported and said
they were unblocked. They were exported. They did not compose — the one
operation being asked for was silently discarded, so the feature built on them
produced identical output for every setting. A second time, I said the DSL could
reference a LENS annotation by a name that does not exist, having read that the
reader passes annotations through without running an expression.

So:

- **A claim about behavior needs something that runs behind it.** Not two files
  read and joined in your head.
- **Assert the difference, not the presence.** "The policy has an effect" means
  two configurations produce different answers. That a capability is exported,
  a column is present, or a name is in `__all__` is not evidence the workflow
  works — those pass just as happily when it doesn't.
- **`tests/test_consumer_workflows.py` is where composed workflows live.** When
  a consumer asks whether something is supported, the answer belongs there as a
  test before it goes in a message.
- The same applies to *your own* fix: after changing something, ask what the
  change enables, not only what it repairs.

## Public API over private helpers

**Logic that does real work belongs in one documented public function, not in a
private helper.** If a `_leading_underscore` function encodes a decision a user
or a sibling repo would want to make — building a fragment from a variant
effect, deciding what counts as a version, classifying an allele — it is in the
wrong place.

The failure mode this prevents: a consumer needs the behavior, cannot import
it, and reimplements it. That copy then drifts. This has happened repeatedly
across openvax/pirl-unc — vaxrank reimplemented topiary's LENS sniffer, its
MHC-dependence table, its method-preference order, and its "was a version named
at all" rule, and every copy was wrong in a way the original was not.

So:

- **Centralize.** One implementation, one place. If two functions answer the
  same question, one calls the other — `resolve_default_versions` is
  implemented in terms of `describe_default_versions` for exactly this reason.
- **Make it public.** Export it from `topiary/__init__.py` and list it in
  `__all__`. A private function a consumer needs is a bug report waiting to be
  filed.
- **Keep it simple enough to read.** Prefer a function that can be understood
  in one sitting over a clever one. If it needs a diagram, it needs splitting.
- **Document it for humans**, numpy style: what it returns, what the arguments
  mean, what it does when the input is absent or empty, and *why* it makes the
  choice it makes. State the non-obvious decision, not just the signature.
- **Give it a parseable name.** The name should say what it returns, in the
  domain's vocabulary — `fragment_from_effect`, not `_ffe` or `_build`.

Genuinely private helpers are fine: single-call-site plumbing, a formatting
detail, a cache key. The test is whether someone outside this repo would ever
want to call it. If yes, it is public.

## Code Style

- Python 3.9+
- Lint: ruff (config in `pyproject.toml`)
- Docstrings: numpy style
- Bugfixes include a regression test where feasible
- topiary's DSL (filters + rankers) is the public API vaxrank consumes — changes here need a coordinated vaxrank bump.

---

## Workflow Orchestration

### 1. Upfront Planning
- For any non-trivial task (3+ steps or architectural): write a short spec first. If something goes sideways, STOP and re-plan — don't keep pushing.

### 2. Verification Before Done
- Never claim complete without proof: tests green, CI green, PyPI version live.

### 3. Autonomous Bug Fixing
- Given a bug report: just fix it. Point at logs/errors/failing tests and resolve them without hand-holding.

### 4. Demand Elegance (Balanced)
- For non-trivial changes pause and ask "is there a more elegant way?" — skip for trivial fixes.
- Treat workarounds as bugs, not new abstractions. Rip out legacy paths decisively rather than accumulating special cases.

### 5. Issue Triage After Each Ship
- Close superseded/outdated issues as you notice them.
- New problems mid-task → file as issues (on the right repo, even if it's not this one), don't bury.

---

## Core Principles

- **Simplicity first.** Minimal diffs, minimal abstractions.
- **No laziness.** Find root causes; no temporary fixes, no empty-category fudges.
- **Minimal blast radius.** Touch only what the task requires.

## Scientific Domain Knowledge

- If a change touches immunology/genomics semantics, check primary sources (papers, UniProt, GenBank) before edits.
- If the code expresses a scientific model at odds with your understanding, flag it — don't silently "fix" it into something wrong.
- Use `mhcgnomes` for MHC allele parsing. Never `startswith("HLA-")` or other string hacks — alleles aren't always human.
