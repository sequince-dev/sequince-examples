# Nested sampling via SMC

Runnable examples for the `ns-smc-update` branch of sequince. These use the
current target cache and step APIs, including terminal shell integration and
posterior reconstruction from all shells.

| Example | Likelihood | Exact reference |
| --- | --- | --- |
| `run_gaussian.py` | Unit-width Gaussian centred at (1, -1) | Gaussian evidence and posterior |
| `run_separated_mixture.py` | Equal mixture at (-4, 0) and (4, 0), width 0.5 | Mixture evidence and posterior |
| `run_spike_and_slab.py` | Concentric mixture with widths 0.1 and 2 | Mixture evidence and posterior |

All examples use a normalized two-dimensional Gaussian prior with standard
deviation 5. `problems.py` defines the models and their analytic reference
values. Both samplers start from the same prior particles for each seed.

## Run

Use Python 3.11+ with the local `sequince` checkout on `ns-smc-update` installed.
The examples need NumPy, SciPy, Matplotlib, and MiniPCN, all provided by
sequince's default Pixi environment. No extra datasets or flow packages are
required.

From the **sequince** repository:

```bash
pixi run python ../sequince-examples/ns-smc/run_gaussian.py
pixi run python ../sequince-examples/ns-smc/run_separated_mixture.py
pixi run python ../sequince-examples/ns-smc/run_spike_and_slab.py
```

Or, from **sequince-examples**, using an environment with the updated sequince:

```bash
python ns-smc/run_example.py --problem all
python ns-smc/run_spike_and_slab.py --particles 2048 --seed 123 \
    --mutation-steps 20 --quantile 0.25
python ns-smc/run_gaussian.py --mutation differential-evolution
```

`--tempered-mutation-steps 40` changes the tempered baseline's mutation budget
independently. `--max-iterations` caps the run; incomplete runs raise an error
rather than presenting a truncated shell sum as complete evidence.

Results go under `ns-smc/outputs/<problem>/`, ignored by Git. Use `--outdir`
for separate experiments; repeating a run in the same directory replaces its
outputs. Each example writes:

- `comparison.png`: evidence accumulation, remaining NS prior volume, and two
  posterior marginals compared with the analytic answer.
- `summary.json`: configuration, analytic answers, estimates, likelihood
  evaluations, and runtime for both methods.
- `posterior_ns_smc.npz` and `posterior_tempered.npz`: weighted posterior
  particles (`particles`, `weights`).
- `shells.npz`: NS prior volumes, evidence increments, cumulative evidence, and
  upper likelihood thresholds.
- `path_ns_smc.npz` and `path_tempered.npz`: level coordinate, weighted mean
  log likelihood, and probability of radius below 0.5 at each iteration.

The NS posterior comes from `NestedSamplingSMCResult.posterior_samples()`.
The raw final NS population is a **constrained prior**, not a posterior, and
its normalizing constant is remaining prior mass. Shell quadrature weights do
not imply independent samples or provide calibrated evidence uncertainty.

## Repeated-seed comparison

From sequince-examples:

```bash
python ns-smc/benchmark.py --seeds 20 > /tmp/ns-smc-comparison.json
python ns-smc/benchmark.py --seeds 20 --tempered-mutation-steps 40 \
    > /tmp/ns-smc-comparison-longer-tempered.json
```

From sequince, prefix the script paths with `../sequince-examples/` and use
`pixi run python`. The benchmark records each seed, log-evidence bias/RMSE,
posterior mean and second-moment errors, likelihood evaluations, and wall time.
The defaults are 512 particles and 10 mutation steps. Equal particle counts
and mutation lengths do not imply equal likelihood costs: NS usually takes
more levels. The second command provides a closer, but not identical, cost
comparison. These likelihoods are cheap, so runtime includes substantial
framework overhead.

## Phase-transition demonstration and Figure 1-style plot

`run_phase_transition.py` implements the 10D example from §6.1 of
[Salomone et al.](https://doi.org/10.1093/jrsssb/qkaf015): a uniform prior on the
unit ball and a concentric Gaussian mixture with widths 0.1 and 0.01 and
weights 0.1 and 0.9. The evidence is approximately 0.3921. Unlike the easy
2D examples, the tempered path has two widely separated radial phases.

```bash
python ns-smc/run_phase_transition.py
python ns-smc/run_phase_transition.py --plot-paths-only
```

From sequince, use `pixi run python ../sequince-examples/ns-smc/run_phase_transition.py`.
The first command runs six methods for five seeds with 1,024 particles:

- Standard `make_standard_bayesian_smc` with its default configuration:
  MiniPCN (5 mutation steps), affine transform, and target ESS ratio 0.7.
- The same standard SMC factory with a symmetric coordinate random walk
  (40 steps), no transform, and target ESS ratio 0.9.
- Persistent sampling via `make_persistent_bayesian_smc`, with default
  MiniPCN (5 steps) and affine transform.
- Persistent sampling with the same coordinate random walk (40 steps) and
  no transform.
- NS-SMC with the **same proposal kernel** (10 steps per nested level).
- NS-SMC with exact independent uniform draws from the constrained ball.

The default MiniPCN baseline uses its usual settings, not a matched budget.
`--target-ess` affects only the standard SMC random-walk baseline.
`--tempered-mutation-steps` sets the mutation budget for both standard SMC
and persistent sampling with random walks.
Persistent sampling uses an active batch of 1,024 particles and the default
archive target ESS of twice the batch size. Its posterior statistics use the
complete terminal weighted archive, and completion requires the terminal ESS
condition as well as beta=1. The archive grows during sampling; equal active
batch sizes do not mean equal memory costs.

The standard SMC and NS-SMC random-walk runs have similar, not identical,
likelihood costs; the exact NS sampler is a diagnostic control using special knowledge of this toy
problem. All methods use the same initial prior population per seed. NS uses
the **known global likelihood maximum** to bound remaining evidence when
stopping; a live-mean heuristic could terminate before discovering the spike.

Outputs under `outputs/phase_transition/` include:

- **`sampling_paths.png` and `sampling_paths.pdf`**: three panels inspired by
  [Yallup & Handley, Fig. 1(a)](https://arxiv.org/html/2609.29413v1#S2.F1),
  showing fixed annealing, adaptive annealing, and nested sampling on the same
  analytic log-likelihood curve. The horizontal coordinate is **log X** to
  resolve the spike's very small prior volume. Annealing markers are modes of
  the reference density with respect to log X; adaptive temperatures use an
  exact population ESS ratio of 0.95. NS markers are equal log-volume steps.
  These are idealized reference paths, **not measured sampler trajectories**.
- `posterior_comparison.png` / `.pdf`: six panels comparing posterior density
  in log radius with the analytic answer. This resolves the spike and slab.
- `posterior_marginals.png` / `.pdf`: the same comparison for the first
  coordinate, with the analytic marginal including the unit-ball truncation.
  Both figures show individual seeds as thin curves and their equal-weight
  mean as a thick curve. These are weighted histograms, without renormalizing
  within the displayed range; seed spread is not a confidence interval.
- `posterior_<method>_seed<seed>.npz`: complete weighted posterior particles
  and weights for each run. NS uses all shells; persistent sampling uses its
  terminal archive. `posterior_comparison.npz` and `posterior_marginals.npz`
  save the histogram bins, per-seed densities, and analytic reference curves.
- `phase_transition.png`: radial bimodality along the tempered reference path,
  observed spike mass, NS volume compression, and evidence across seeds.
- `summary.json`: actual likelihood costs, spike probabilities, and evidence
  for every method and seed, plus terminal posterior population sizes.
  `reference.npz`, `path_*.npz`, and `sampling_paths.npz` retain the numerical
  plot data.

`--plot-paths-only` regenerates the Figure 1-style PNG/PDF without running the
samplers. This is the path comparison in panel (a), not the learned-flow
schematic in panel (b); these examples do not implement neural transport.

The default MiniPCN SMC baseline missed the spike in all five seeds (mean
spike mass 0), using 65,536 likelihood evaluations per run. Its budget is
substantially smaller than the random-walk comparisons below.

Both persistent configurations also missed the spike in all five seeds
(mean spike mass 0). MiniPCN used 94,208 likelihood evaluations per run;
the random walk used 533,504. Archive recycling did not recover the missing
phase at these settings. These are measured costs, not matched budgets.

In the checked five-seed run, mean spike mass was 0.00449 for SMC with random walks,
0.89834 for NS-SMC with random walks, and 0.88699 for the exact-NS control,
against an analytic 0.90000. The random-walk likelihood costs were approximately
689,152 and 662,528, respectively. All SMC random-walk runs reached beta=1 despite
largely missing the spike. Five seeds illustrate the failure mode; they are
not a precise performance study or a universal guarantee for NS.

Use `--target-ess`, `--mutation-steps`, `--tempered-mutation-steps`,
`--dimension`, and `--seeds` to explore the distinction between the outer path
and constrained-prior mixing. Results overwrite the selected output directory;
use `--outdir` when comparing settings.

## What the two-dimensional examples establish

These are correctness and finite-budget efficiency checks in two dimensions.
The spike-and-slab is a useful stress test, but this particular setup is **not
a test of immunity to phase transitions**.

The nested path compresses prior volume using likelihood contours; it does not
need to cross the same temperature-annealing bottleneck. This is the distinction
in [Appendix J of Neural Transport Nested Sampling](https://arxiv.org/html/2609.29413v1#A10).
It remains important to establish whether the finite constrained-prior mutation
kernel mixes well. A poor result from MiniPCN at a particular budget does not
refute the path advantage of NS, nor does a good result on these toys establish
it. The path archives help inspect progression; a stronger test should vary
transition severity, dimension, particle count, and mutation length, and compare
mode occupancy as well as evidence across independent seeds.

Adaptive thresholds and stopping are not an assertion of unbiased evidence.
The stopping tolerance estimates remaining evidence; it is not a bound on
unseen modes. Likelihood ties are discarded together, and a sampled plateau
with no survivors is finalized. This is not the auxiliary-variable treatment
of arbitrary ties in the general adaptive NS-SMC construction.

For the fixed/adaptive distinction see
[Salomone et al. (2025)](https://doi.org/10.1093/jrsssb/qkaf015).
