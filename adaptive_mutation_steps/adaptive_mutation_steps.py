"""Adapt the mutation length between SMC iterations using acceptance."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from orng import RandomGenerator
from sequince import SequentialMonteCarlo, TemperedPosterior
from sequince.callbacks import AdaptiveMutationPlot
from sequince.log import configure_logging
from sequince.plot import plot_corner
from sequince.state import ParticleState
from sequince.steps import (
    AdaptiveTemperedWeighting,
    AdaptMutationSteps,
    DifferentialEvolutionMutation,
    LoggingStep,
    SystematicResampling,
)
from sequince.steps.mutation.adaptive.policies.acceptance import (
    AcceptanceRatePolicy,
)


def log_likelihood(particles: np.ndarray, context) -> np.ndarray:
    """Rosenbrock potential with configurable scale."""
    scale = context.shared["rosenbrock_scale"]
    x = particles[..., :-1]
    y = particles[..., 1:]
    potential = np.sum(100.0 * (y - x**2.0) ** 2.0 + (1.0 - x) ** 2.0, axis=-1)
    return -potential / scale


def log_prior(particles: np.ndarray, context) -> np.ndarray | float:
    """Uniform density within the configured box."""
    lower = context.shared["bounds"][:, 0]
    upper = context.shared["bounds"][:, 1]
    in_bounds = np.all((particles >= lower) & (particles <= upper), axis=-1)
    log_volume = np.sum(np.log(upper - lower))
    return np.where(in_bounds, -log_volume, -np.inf)


def main() -> None:
    # Configure logging and random number generator
    logger = configure_logging(level=logging.INFO)

    seed = 1234
    rng = RandomGenerator(backend="numpy", seed=seed)

    # Define the problem parameters
    n_particles = 1000
    dims = 8
    rosenbrock_scale = 1.0
    bounds = np.array([[-5.0, 5.0]]).repeat(dims, axis=0)

    # Define output directory and parameter labels for plotting
    labels = [f"x{i + 1}" for i in range(dims)]
    outdir = Path("./")

    # Define a tempered posterior with the log prior and log likelihood
    # functions.
    posterior = TemperedPosterior(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
    )

    # Create the base mutation step
    # This will be wrapped in an adaptive mutation step that adjusts the number
    # of mutation steps.
    # We pass the posterior to the mutation step so that it can use the
    # log prior and log likelihood functions.
    fixed_length_mutation = DifferentialEvolutionMutation(posterior)

    # Define the target number of mutation steps and the minimum and maximum
    # allowed steps
    target_steps = 5
    min_steps, max_steps = 1, 128

    # Create the adaptive mutation step that will adjust the number of mutation
    # steps based on the acceptance rate of the previous iteration.
    # This includes a 'policy' that defines how to update the number of steps
    # based on the acceptance rate.
    # The policy is set to target a specific number of mutation steps, with
    # minimum and maximum bounds on the number of steps.
    # Other policies could be implemented to adapt the number of steps or
    # other parameters of the mutation step based on different criteria.
    adaptive_mutation = AdaptMutationSteps(
        fixed_length_mutation,
        policy=AcceptanceRatePolicy(
            target_steps=target_steps,
            min_steps=min_steps,
            max_steps=max_steps,
        ),
        initial_steps=5 * dims,
        min_steps=min_steps,
        max_steps=max_steps,
    )

    # Define a logging function to report the current state of the SMC
    # algorithm
    def mutation_log_message(state, context) -> str:
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
        size=(n_particles, dims),
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

    # Extract the log evidence and its standard error from the result
    log_evidence = result.log_evidence()
    log_evidence_std = result.log_evidence_error()
    logger.info(f"Log evidence: {log_evidence:.4f} ± {log_evidence_std:.4f}")

    # Plot the final posterior samples using a corner plot.
    # We use rejection sampling to obtain unweighted samples from the
    # posterior.
    posterior_samples, weights = result.posterior_samples(method="rejection")
    plot_corner(
        posterior_samples,
        weights=weights,
        parameter_labels=labels,
        filename=outdir / "adaptive_mutation_steps_corner.png",
    )


if __name__ == "__main__":
    main()
