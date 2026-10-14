"""Sample a Gaussian posterior using Sequince. See README.rst."""

import logging
from math import erf, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sequince import ParticleState, make_standard_bayesian_smc
from sequince.log import configure_logging
from sequince.steps import StepContext

# Edit these parameters to explore the example.
n_dimensions = 2
n_particles = 2000
lower_bound, upper_bound = -5.0, 5.0
mean, sigma = 0.0, 1.0
seed = 42
outdir = Path("outdir/gaussian")


def log_prior(particles: np.ndarray, context: StepContext) -> np.ndarray:
    """Evaluate a normalized uniform prior within the bounds.

    Parameters
    ----------
    particles : numpy.ndarray, shape (n_particles, n_dimensions)
        Batch of parameter vectors.
    context : StepContext
        Runtime context whose shared mapping contains the model parameters.

    Returns
    -------
    log_density : numpy.ndarray, shape (n_particles,)
        Log prior density, with negative infinity outside the bounds.
    """
    lower_bound = context.shared["lower_bound"]
    upper_bound = context.shared["upper_bound"]
    inside = np.all(
        (particles >= lower_bound) & (particles <= upper_bound), axis=-1
    )
    log_density = -particles.shape[1] * np.log(upper_bound - lower_bound)
    return np.where(inside, log_density, -np.inf)


def log_likelihood(particles: np.ndarray, context: StepContext) -> np.ndarray:
    """Evaluate a normalized Gaussian likelihood.

    Parameters
    ----------
    particles : numpy.ndarray, shape (n_particles, n_dimensions)
        Batch of parameter vectors.
    context : StepContext
        Runtime context whose shared mapping contains the model parameters.

    Returns
    -------
    log_density : numpy.ndarray, shape (n_particles,)
        Gaussian log density for each particle.
    """
    mean = context.shared["mean"]
    sigma = context.shared["sigma"]
    return -0.5 * np.sum(((particles - mean) / sigma) ** 2, axis=-1) - (
        0.5 * particles.shape[1] * np.log(2.0 * np.pi * sigma**2)
    )


def main() -> None:
    """Sample the posterior and save particles, plots, and the SMC result."""
    configure_logging(level=logging.INFO)
    rng = np.random.default_rng(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    # ParticleState supplies equal weights, appropriate for prior draws.
    initial_state = ParticleState(
        particles=rng.uniform(
            lower_bound, upper_bound, size=(n_particles, n_dimensions)
        ),
    )

    # The helper combines adaptive tempering, resampling, and mutation moves.
    smc, posterior, metadata = make_standard_bayesian_smc(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        target_ess_ratio=0.8,
        mutation="minipcn",
        mutation_n_steps=16,
    )
    result = smc.run(
        initial_state,
        rng=rng,
        shared_context={
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mean": mean,
            "sigma": sigma,
        },
        metadata=metadata,
    )

    state = result.final_state
    print(f"Estimated log evidence: {state.log_evidence:.4f}")
    # Include the Gaussian mass within the finite prior bounds.
    prior_mass = 0.5 * (
        erf((upper_bound - mean) / (sigma * sqrt(2.0)))
        - erf((lower_bound - mean) / (sigma * sqrt(2.0)))
    )
    log_evidence = n_dimensions * (
        np.log(prior_mass) - np.log(upper_bound - lower_bound)
    )
    print(f"Analytic log evidence:  {log_evidence:.4f}")
    print(f"Untruncated Gaussian mean: {mean}; variance: {sigma**2}.")
    print(f"Posterior mean: {result.weighted_mean()}")
    print(f"Posterior covariance:\n{result.weighted_covariance()}")

    fig = result.plot_corner()
    fig.savefig(outdir / "corner.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    result.plot_metadata_history("beta", ax=axes[0])
    axes[0].set_ylabel("Tempering parameter beta")
    result.plot_log_evidence(ax=axes[1])
    fig.tight_layout()
    fig.savefig(outdir / "history.png")
    plt.close(fig)

    # Keep weights with the particles so subsequent analysis remains valid.
    np.savez(
        outdir / "posterior.npz",
        particles=state.particles,
        weights=state.weights,
        log_evidence=state.log_evidence,
    )
    result.save_to_file(outdir / "result.h5")


if __name__ == "__main__":
    main()
