# Persistent sampling

This example demonstrates how to use persistent sampling in `sequince` to
sample an n-dimensional Rosenbrock target.

Persistent sampling is a variation of SMC proposed by [Karamanis 2025].

Run the example from the repository root:

```bash
python persistent_sampling/rosenbrock.py
```

For a larger run:

```bash
python persistent_sampling/rosenbrock.py \
    --dimensions 16 \
    --n-particles 1024 \
    --target-ess 2048 \
    --target-total-ess 8192 \
    --mutation-steps 32
```

The script prints diagnostics from the final weighted persistent state,
including its archive size, effective sample size, weighted mean, best
Rosenbrock potential, and log evidence. Unless `--no-plot` is supplied, it also
saves a weighted corner plot as `persistent_sampling/rosenbrock_<D>d.png`.
Use `--output PATH` to choose a different location.

Run the following for all available options:

```bash
python persistent_sampling/rosenbrock.py --help
```

## Preconditioning

This example also shows how to use the normalizing-flow-based preconditioning
described in [Karamanis 2022] and [Karamanis 2025].

The `--flow` option enables learned normalizing-flow preconditioning:

```bash
python persistent_sampling/rosenbrock.py --dimensions 6 --flow
```

The flow option requires `sequince-density` and its flow dependencies.

[Karamanis 2022]: https://arxiv.org/abs/2207.05660
[Karamanis 2025]: https://arxiv.org/abs/2407.20722
