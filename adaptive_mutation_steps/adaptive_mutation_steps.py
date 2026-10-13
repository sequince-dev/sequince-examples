"""Adapt mutation budgets for a Rosenbrock posterior. See README.rst."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sequince import SequentialMonteCarlo, TemperedPosterior
from sequince.callbacks import AdaptiveMutationPlot
from sequince.log import configure_logging
from sequince.state import ParticleState
from sequince.steps import (
    AdaptiveTemperedWeighting,
    AdaptMutationSteps,
    DifferentialEvolutionMutation,
    LoggingStep,
    StepContext,
    SystematicResampling,
)
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)

# Edit these parameters to explore the example.
n_particles = 1000
n_dimensions = 8
rosenbrock_scale = 1.0
lower_bound, upper_bound = -5.0, 5.0
seed = 1234
target_steps = 5  # Target accepted moves per particle, not attempted moves.
initial_steps = 5 * n_dimensions
min_steps, max_steps = 1, 128
outdir = Path("outdir")


def log_likelihood(particles: np.ndarray, context: StepContext) -> np.ndarray:
    """Evaluate an unnormalized Rosenbrock likelihood.

    Parameters
    ----------
    particles : numpy.ndarray, shape (n_particles, n_dimensions)
        Batch of parameter vectors, with at least two dimensions.
    context : StepContext
        Runtime context containing the positive Rosenbrock scale.

    Returns
    -------
    log_density : numpy.ndarray, shape (n_particles,)
        Negative Rosenbrock potential divided by the scale. Evidence uses
        this unnormalized likelihood convention.
    """
    scale = context.shared["rosenbrock_scale"]
    x = particles[..., :-1]
    y = particles[..., 1:]
    potential = np.sum(100.0 * (y - x**2.0) ** 2.0 + (1.0 - x) ** 2.0, axis=-1)
    return -potential / scale


def log_prior(particles: np.ndarray, context: StepContext) -> np.ndarray:
    """Evaluate a normalized uniform prior within the configured box.

    Parameters
    ----------
    particles : numpy.ndarray, shape (n_particles, n_dimensions)
        Batch of parameter vectors.
    context : StepContext
        Runtime context whose shared mapping contains the prior bounds.

    Returns
    -------
    log_density : numpy.ndarray, shape (n_particles,)
        Log prior density, with negative infinity outside the bounds.
    """
    lower = context.shared["bounds"][:, 0]
    upper = context.shared["bounds"][:, 1]
    in_bounds = np.all((particles >= lower) & (particles <= upper), axis=-1)
    log_volume = np.sum(np.log(upper - lower))
    return np.where(in_bounds, -log_volume, -np.inf)


def main() -> None:
    """Run adaptive SMC and save diagnostic plots, particles, and results."""
    logger = configure_logging(level=logging.INFO)
    rng = np.random.default_rng(seed)
    bounds = np.tile([lower_bound, upper_bound], (n_dimensions, 1))
    outdir.mkdir(parents=True, exist_ok=True)

    # Define a tempered posterior with the log prior and log likelihood
    # functions.
    posterior = TemperedPosterior(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
    )

    # Freeze each stage's budget before mutation. Its completed acceptance
    # rate determines the next stage's budget, within the configured bounds.
    fixed_length_mutation = DifferentialEvolutionMutation(posterior)
    adaptive_mutation = AdaptMutationSteps(
        fixed_length_mutation,
        policy=AcceptanceRatePolicy(
            target_steps=target_steps,
            min_steps=min_steps,
            max_steps=max_steps,
        ),
        initial_steps=initial_steps,
        min_steps=min_steps,
        max_steps=max_steps,
    )

    # Define a logging function to report the current state of the SMC
    # algorithm
    def mutation_log_message(
        state: ParticleState, context: StepContext
    ) -> str:
        """Report the completed budget and the next stage's decision."""
        del state
        return (
            f"SMC iteration {context.iteration}: "
            f"beta={context.metadata.get('beta', 0.0):.3g}, "
            f"mutation steps={adaptive_mutation.current_n_steps}, "
            f"next={adaptive_mutation.next_n_steps}"
        )

    # Define the SMC runner with the adaptive mutation step and logging
    # We use the standard SMC steps: adaptive tempering, resampling, and
    # mutation.
    # We also add a callback to plot the adaptive mutation steps over the
    # course of the run
    runner = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(posterior, target_ess_ratio=0.7),
            SystematicResampling(always_resample=True),
            adaptive_mutation,
            LoggingStep(str_fn=mutation_log_message),
        ],
        callbacks=[
            AdaptiveMutationPlot(outdir, target_steps=target_steps),
        ],
    )

    # Define the initial state, which contains particles drawn from the prior
    initial_particles = rng.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(n_particles, n_dimensions),
    )
    initial_state = ParticleState(initial_particles)

    # Run the SMC algorithm with the initial state and
    # the random number generator.
    # We also pass in a shared context that contains the bounds and the
    # Rosenbrock scale, which are used in the log prior and log likelihood
    result = runner.run(
        initial_state,
        rng=rng,
        shared_context={
            "bounds": bounds,
            "rosenbrock_scale": rosenbrock_scale,
        },
    )

    state = result.final_state
    logger.info("Estimated log evidence: %.4f", state.log_evidence)
    logger.info("Posterior mean: %s", result.weighted_mean())
    logger.info("Posterior covariance:\n%s", result.weighted_covariance())

    fig = result.plot_corner(
        parameter_labels=[f"x{i + 1}" for i in range(n_dimensions)]
    )
    fig.savefig(outdir / "corner.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    result.plot_metadata_history("beta", ax=axes[0])
    axes[0].set_ylabel("Tempering parameter beta")
    result.plot_log_evidence(ax=axes[1])
    fig.tight_layout()
    fig.savefig(outdir / "history.png")
    plt.close(fig)

    # Keep weights with particles so subsequent analysis remains valid.
    np.savez(
        outdir / "posterior.npz",
        particles=state.particles,
        weights=state.weights,
        log_evidence=state.log_evidence,
    )
    result.save_to_file(outdir / "result.h5")


if __name__ == "__main__":
    main()
