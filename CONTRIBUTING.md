# Contributing to Sequince examples

Contributions are welcome, including tutorials, bug reports, and improvements to
existing examples. These guidelines follow the conventions of
[Sequince](https://github.com/sequince-dev/sequince/blob/main/CONTRIBUTING.md)
and describe the documentation workflow used here.

## Community and responsibility

Treat other contributors with respect and follow the
[Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).
Contributions must comply with the project's [license](LICENSE). Preserve
required attribution when incorporating third-party material.

AI-assisted contributions are welcome under Sequince's
[AI policy](https://github.com/sequince-dev/sequince/blob/main/AI_POLICY.md).
Contributors remain responsible for understanding, validating, and explaining
all submitted content, including its scientific correctness and performance.

## Scope of contributions

Keep pull requests focused on one problem or a closely related set of changes.
Discuss substantial new dependencies or workflows in an issue first. Small fixes
can go directly to a pull request. Reusable sampling components generally belong
in Sequince; this repository demonstrates their use in concrete applications.

## Development setup

Fork `sequince-dev/sequince-examples`, clone your fork, and configure upstream:

```bash
git clone git@github.com:YOUR_USERNAME/sequince-examples.git
cd sequince-examples
git remote add upstream https://github.com/sequince-dev/sequince-examples.git
git fetch upstream
git switch main
git branch --set-upstream-to=upstream/main main
git pull --ff-only
git switch -c fix-descriptive-name
```

Keep local `main` for upstream changes and develop on separate branches. Install
[Pixi](https://pixi.prefix.dev/latest/installation/), then run:

```bash
make -C docs setup
pixi run --locked -e check pre-commit install
```

Dependencies, environments, and tasks are defined in `pyproject.toml`.
Unlike Sequince itself, this repository tracks `pixi.lock` for reproducible
example execution. Include intentional dependency updates with the corresponding
manifest changes; do not commit environments, caches, or generated results.

The configuration requests released `sequince` and `sequince-density` packages
from PyPI, with no fallback to Git main. Until both releases are available,
setup and CI dependency resolution are expected to fail. The current lock is an
older Git snapshot. After release, regenerate it using
`pixi update sequince sequince-density`. Existing local check environments can
still run lightweight checks with `pixi run --frozen -e check test`; this does
not validate a release installation.

## Code style and scientific correctness

Follow existing module conventions. Use Ruff for formatting and import ordering,
four-space indentation, and a 79-character line limit. Use descriptive names and
NumPy-style docstrings for public functions, including array shapes and numerical
assumptions where relevant. Explain non-obvious choices rather than restating
code in comments.

Add parameter and return annotations to new functions and helpers. Prefer modern
Python syntax, precise callback signatures, and existing backend-neutral types.
Keep optional imports optional and avoid broad type suppressions.

Preserve array backends, dtypes, RNG behavior, and intended target distributions.
Explain intentional changes to weighting, evidence estimates, sampler settings,
or diagnostics. Use fixed seeds and justified tolerances for numerical tests.
Avoid retaining unnecessary histories or adding expensive diagnostics by default.

## Checks and tests

Before requesting review, run:

```bash
make -C docs check
make -C docs hooks
```

The first command runs pytest, Ruff linting, and formatting checks. Pre-commit
also checks spelling, whitespace, YAML, and large files, and may modify files.
Review any edits and rerun the checks. Do not bypass hooks to make commits pass.
The configurations in `pyproject.toml` and `.pre-commit-config.yaml` are
authoritative; this repository does not currently configure mypy.

Tests live in `tests/`. Test observable behavior and numerical invariants, using
small fixed inputs or mocked execution for pipeline tests. Some tests require
optional flow or particle-filtering dependencies. Report checks actually run,
including skipped tests and unavailable environments. Lightweight pipeline tests
do not establish that a complete sampling run succeeds.

## Writing and building the documentation

Keep scripts and their reStructuredText tutorials together under `examples/`.
Introduce the problem and sampling method before showing plots. Include figures
directly in the tutorial, with explanations and the relevant source code. The
reading order in `docs/index.rst` starts with basic examples, then introduces
other methods, with flow preconditioning before persistent sampling.

Documentation commands run from the repository root:

```bash
make -C docs help
make -C docs example EXAMPLE=gaussian
make -C docs example EXAMPLE=flow-preconditioning ENV=flow
make -C docs examples
make -C docs docs
make -C docs docs-only
make -C docs validate
```

`example` runs one manifest entry; `examples` runs all entries sequentially in
their declared environments. Each execution batch resolves released dependencies
once and uses the same lock throughout. `docs` refreshes dependencies, reuses
valid outputs when possible, and executes examples if outputs are missing or
stale. `docs-only` validates existing outputs and builds Sphinx HTML without
sampling. `validate` runs checks, reruns all examples, then builds the docs.
HTML is written to `docs/_build/html/`.

Full example execution can consume substantial memory and time. Choose a single
example when developing, and use `docs-only` when valid results are available.
Execution has timeouts, but does not currently enforce a memory limit. Do not
assume that parallel local execution is safe.

### Adding an example

Add the script and tutorial under `examples/`, then register automated examples
in `docs/examples.json`. Declare the environment, settings, timeout, expected
outputs, and figure captions. Add the tutorial to `docs/index.rst`.

Follow existing scripts' documentation-output adapters: write outputs to the
provided directory and keep ordinary script behavior intact. Register additional
dependencies in `pyproject.toml`, using a dedicated environment when needed.
Add focused tests for new pipeline behavior and verify the tutorial with Sphinx.

The runner writes plots and metadata under `build-docs/outputs/<example-id>/`.
Validation checks the exact output set, checksums, source hashes, settings,
source commit, and dependency provenance before staging pages under
`docs/_generated/`. Changes to source files or the execution lock invalidate old
results. Do not edit metadata to make stale outputs pass, or commit generated
plots and staging directories.

### CI and hosting

The documentation workflow resolves one execution lock and shares it across
parallel example jobs. It collects outputs from that same workflow run, runs
checks, builds Sphinx, and uploads HTML and a validated output bundle.

Trusted runs on `main` can attach the bundle to a dedicated GitHub release keyed
by the full source commit and trigger a Read the Docs build. Read the Docs
installs `docs/requirements.txt`, fetches the exact matching bundle, and verifies
it before building; it does not run sampling. Freshness checks guard `latest`
builds, and there is no fallback to results from another commit.

Maintainers must configure the Read the Docs project, the repository variable
`RTD_PROJECT`, and the secret `RTD_TOKEN` before enabling this hosting workflow.
Keep credentials out of source files and pull request jobs. The configuration
uses standard Read the Docs hosting; private access requires a hosting service
and account configuration that supports it.

## Commit messages

Use Sequince's uppercase SciPy-style prefixes and optional lowercase scopes:

```text
BUG(docs): reject stale example outputs
ENH(examples): add a particle filtering tutorial
DOC: clarify local documentation builds
CI(docs): collect example artifacts
BLD: update documentation dependencies
```

Other supported prefixes include `PERF`, `API`, `DEP`, `MAINT`, `TST`, `REL`, and
`STY`. Use `BUG` rather than `FIX`. Start the summary with a lowercase imperative
verb, omit a trailing period, and aim for at most 72 characters. Use one prefix
by default, or two separated by `/` when both are needed. Make coherent commits;
pull request titles should follow the same convention for squash merging.

## Pull requests and review

Push your contribution branch to your fork and open a pull request targeting
`sequince-dev/sequince-examples:main`. Use a draft for work in progress.

Explain the problem and resulting behavior, link related issues, and describe
scientific assumptions or compatibility changes. Summarize validation with
commands and results, including sampling runs that were not performed and any
skips. For documentation changes, identify which pages and examples were built.

Respond to reviewer feedback and keep the description aligned with the final
implementation. Maintainers should wait for applicable checks and review
conversations to be resolved, and check the squash commit message before merging.
