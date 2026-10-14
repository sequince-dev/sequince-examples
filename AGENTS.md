# Agent instructions

Before making changes, read and follow [CONTRIBUTING.md](CONTRIBUTING.md) and
Sequince's [AI policy](https://github.com/sequince-dev/sequince/blob/main/AI_POLICY.md).
CONTRIBUTING.md defines development setup, code style, typing, validation,
commit-message conventions, and the documentation workflow. The AI policy
applies to AI-assisted contributions to this repository.

## Working in the repository

- Inspect the working tree before editing. Preserve existing work and do not
  overwrite or revert unrelated changes. Stage only files relevant to the task.
- Keep changes focused and follow existing interfaces and helpers. Avoid
  unrelated refactors and unnecessary dependencies or abstractions.
- Keep scripts and their reStructuredText tutorials together under `examples/`.
  Register automated examples in `docs/examples.json` and update the reading
  order in `docs/index.rst` when adding pages.
- Dependencies and environments belong in `pyproject.toml`. This repository
  intentionally tracks `pixi.lock`; change it only for intentional dependency
  updates. Preserve local environments, caches, and generated results, and
  keep them out of commits.

## Scientific correctness and validation

- Identify changes that affect target distributions, weighting, evidence
  estimates, RNG behavior, sampler settings, or backend compatibility. Explain
  intentional changes and support them with appropriate tests.
- Prefer lightweight pipeline tests and small fixed numerical inputs during
  development. Full examples can consume substantial memory. Respect execution
  constraints from the user; commands such as `make -C docs docs` can trigger
  sampling when outputs are missing or stale. `docs-only` builds from existing
  validated outputs without sampling.
- Preserve output provenance checks. Do not edit metadata or weaken validation
  to reuse stale results. Keep generated plots directly in tutorial pages using
  the existing staging workflow.
- Follow CONTRIBUTING.md for checks. Report commands actually run and their
  results, including failures, skips, sampling runs not performed, and checks
  that could not be completed. Do not describe skipped checks as passing.
- Do not bypass hooks or disable checks to make a commit succeed. Address the
  failure, or report the blocker when it cannot be resolved within the task.
