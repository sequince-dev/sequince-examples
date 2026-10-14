# Sequince examples

Example scripts and their narrative tutorials live together under `examples/`.
Start with the [basic examples](examples/basic_examples/README.rst), then explore
[adaptive mutation](examples/adaptive_mutation_steps/README.rst),
[flow preconditioning](examples/preconditioning/README.rst), and
[persistent sampling](examples/persistent_sampling/README.rst). The documentation
also covers waste-free SMC and particle filtering.

The documentation pipeline is configured to execute Gaussian, Rosenbrock,
adaptive mutation, flow preconditioning, persistent sampling, and waste-free
SMC and particle-filtering examples using the latest Sequince main, resolved
once per execution batch. Their plots appear directly in the tutorial pages.

```bash
pixi install --locked -e default -e docs
pixi run --locked examples
pixi run --locked -e docs docs
```

Use `make -C docs setup` followed by `make -C docs docs` to build the full documentation.
Valid example outputs are reused; missing or stale outputs trigger example
execution. `make -C docs docs-only` never executes examples. Use `make -C docs validate` to run
checks, rerun all examples, and build the documentation.
Run `make -C docs check` for tests, lint, and formatting checks, or `make -C docs help` for
individual targets. New example runs refresh Sequince main; all jobs in a batch
use the same resolved lock. Later, the dependency will switch to releases.

Generated results are ignored. Environments, dependencies, and tasks are defined in
`pyproject.toml`; `pixi.lock` is intentionally tracked for reproducibility.
