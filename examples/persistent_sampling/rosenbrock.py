"""Sample a configurable-dimensional Rosenbrock target persistently.

For example, run a six-dimensional problem with::

    python persistent_sampling/rosenbrock.py --dimensions 6

The target is the bounded density ``exp(-Rosenbrock(x) / scale)`` under a
uniform prior. A corner plot of the final weighted persistent population is
saved unless ``--no-plot`` is supplied. Pass ``--flow`` to use optional learned
normalizing-flow preconditioning instead of the default bounded transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sequince import (
    ParticleState,
    PersistentParticleState,
    make_persistent_bayesian_smc,
)
from sequince.log import configure_logging
from sequince.plot import plot_corner
from sequince.steps import StepContext
from sequince.transforms import BoundedToUnboundedTransform

documentation_output: Path | None = None


def rosenbrock(particles: np.ndarray) -> np.ndarray:
    """Return the Rosenbrock potential for each particle."""
    values = np.asarray(particles, dtype=float)
    return np.sum(
        100.0 * (values[:, 1:] - values[:, :-1] ** 2.0) ** 2.0
        + (1.0 - values[:, :-1]) ** 2.0,
        axis=1,
    )


def make_likelihood_and_prior(
    dimensions: int,
    *,
    bound: float,
    scale: float,
) -> tuple[
    callable[[np.ndarray, StepContext], np.ndarray],
    callable[[np.ndarray, StepContext], np.ndarray],
]:
    """Construct a bounded Rosenbrock target."""
    log_prior_volume = dimensions * np.log(2.0 * bound)

    def log_prior(particles: np.ndarray, context=None) -> np.ndarray:
        del context
        values = np.asarray(particles, dtype=float)
        inside = np.all(np.abs(values) <= bound, axis=1)
        return np.where(inside, -log_prior_volume, -np.inf)

    def log_likelihood(particles: np.ndarray, context=None) -> np.ndarray:
        del context
        return -rosenbrock(particles) / scale

    return (
        log_prior,
        log_likelihood,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--n-particles", type=int, default=256)
    parser.add_argument(
        "--target-ess",
        type=float,
        default=None,
        help="Absolute persistent ESS (default: twice n-particles).",
    )
    parser.add_argument(
        "--target-total-ess",
        type=float,
        default=None,
        help="Optional final persistent ESS accumulated at beta=1.",
    )
    parser.add_argument("--mutation-steps", type=int, default=8)
    parser.add_argument(
        "--flow",
        action="store_true",
        help="Use optional learned normalizing-flow preconditioning.",
    )
    parser.add_argument("--bound", type=float, default=5.0)
    parser.add_argument(
        "--rosenbrock-scale",
        type=float,
        default=1.0,
        help="Divide the Rosenbrock potential by this positive value.",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Plot path (default: persistent_sampling/rosenbrock_<D>d.png).",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    """Run persistent SMC and report archive diagnostics."""
    parser = build_parser()
    args = parser.parse_args()
    if args.dimensions < 2:
        parser.error("--dimensions must be at least 2")
    if args.n_particles < 1:
        parser.error("--n-particles must be positive")
    if args.bound <= 0.0:
        parser.error("--bound must be positive")
    if args.rosenbrock_scale <= 0.0:
        parser.error("--rosenbrock-scale must be positive")
    if args.mutation_steps < 1:
        parser.error("--mutation-steps must be positive")
    target_ess = (
        float(args.target_ess)
        if args.target_ess is not None
        else float(2 * args.n_particles)
    )

    configure_logging()

    rng = np.random.default_rng(args.seed)
    log_prior, log_likelihood = make_likelihood_and_prior(
        args.dimensions,
        bound=args.bound,
        scale=args.rosenbrock_scale,
    )
    initial_particles = rng.uniform(
        -args.bound,
        args.bound,
        size=(args.n_particles, args.dimensions),
    )
    initial_state = ParticleState(initial_particles)

    bounds = {
        index: (-args.bound, args.bound) for index in range(args.dimensions)
    }
    bounded_transform = BoundedToUnboundedTransform(bounds=bounds)
    if args.flow:
        try:
            from sequince.transforms import FlowTransform
            from sequince_density.flows import ZukoFlow
        except ImportError as exc:
            parser.error(
                "--flow requires the optional sequince-density dependency "
                f"({exc})"
            )
        flow = ZukoFlow(
            dims=args.dimensions,
            flow_class="MAF",
            hidden_features=[64, 64],
            particle_transform=bounded_transform,
        )
        particle_transform = FlowTransform(
            flow,
            fit_kwargs={
                "n_epochs": 100,
                "lr": 1e-3,
                "batch_size": 500,
            },
        )
        transform_refit = "per-context"
    else:
        particle_transform = bounded_transform
        transform_refit = "once"

    sampler, posterior, metadata = make_persistent_bayesian_smc(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        n_particles=args.n_particles,
        target_ess=target_ess,
        target_total_ess=args.target_total_ess,
        mutation_n_steps=args.mutation_steps,
        mutation_kwargs={"transform_refit": transform_refit},
        particle_transform=particle_transform,
    )
    result = sampler.run(
        initial_state,
        rng=rng,
        metadata=metadata,
    )

    final_state = result.final_state
    if not isinstance(final_state, PersistentParticleState):
        raise TypeError("persistent SMC must return a PersistentParticleState")
    particles = np.asarray(final_state.particles)
    weights = np.asarray(final_state.weights)
    potentials = rosenbrock(particles)
    best = particles[int(np.argmin(potentials))]
    weighted_mean = np.average(particles, axis=0, weights=weights)
    weighted_mean_potential = np.average(potentials, weights=weights)
    print()
    print(f"dimensions: {args.dimensions}")
    print(f"iterations: {len(result)}")
    print(f"final beta: {float(result.context.metadata['beta']):.6f}")
    print(f"final archive size: {final_state.n_particles}")
    print(f"final persistent ESS: {final_state.effective_sample_size():.1f}")
    print(
        "weighted mean: "
        f"{np.array2string(weighted_mean, precision=3, threshold=12)}"
    )
    print(f"weighted mean Rosenbrock potential: {weighted_mean_potential:.6g}")
    print(f"best Rosenbrock potential: {float(np.min(potentials)):.6g}")
    print(f"best particle: {np.array2string(best, precision=3)}")
    print(f"final log evidence: {float(final_state.log_evidence):+.6f}")

    if documentation_output is not None:
        summary = {
            "log_evidence": float(final_state.log_evidence),
            "weighted_mean": weighted_mean.tolist(),
            "archive_size": int(final_state.n_particles),
            "effective_sample_size": float(
                final_state.effective_sample_size()
            ),
            "weighted_mean_potential": float(weighted_mean_potential),
        }
        (documentation_output / "summary.json").write_text(
            json.dumps(summary, allow_nan=False) + "\n"
        )
        if args.output is None:
            args.output = documentation_output / "corner.png"

    if not args.no_plot:
        output = args.output or Path(__file__).with_name(
            f"rosenbrock_{args.dimensions}d.png"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        posterior_samples, weights = result.posterior_samples(
            method="rejection", rng=rng
        )
        plot_corner(
            posterior_samples,
            weights=weights,
            parameter_labels=[
                rf"$x_{{{index + 1}}}$" for index in range(args.dimensions)
            ],
            filename=output,
            truths=np.ones(args.dimensions),
        )
        print(f"saved plot: {output}")


if __name__ == "__main__":
    main()
