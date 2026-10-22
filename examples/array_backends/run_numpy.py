"""Sample a Himmelblau target with NumPy. See README.rst."""

from math import log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sequince import ParticleState, make_standard_bayesian_smc
from sequince.array_api import to_numpy
from sequince.log import configure_logging
from sequince.steps import StepContext

n_particles = 1000
seed = 42
outdir = Path("outdir/numpy")

n_dimensions = 2
lower_bound, upper_bound = -6.0, 6.0


def log_prior(particles: np.ndarray, context: StepContext) -> np.ndarray:
    """Evaluate a normalized uniform prior.

    Parameters
    ----------
    particles : array, shape (n_particles, n_dimensions)
        NumPy particle coordinates.
    context : StepContext
        Runtime context; unused for this fixed target.

    Returns
    -------
    log_density : array, shape (n_particles,)
        Normalized box log density, or negative infinity outside the box.
    """
    inside = np.all(
        (particles >= lower_bound) & (particles <= upper_bound), axis=-1
    )
    density = np.full_like(
        particles[:, 0], -n_dimensions * log(upper_bound - lower_bound)
    )
    return np.where(inside, density, np.full_like(density, -float("inf")))


def log_likelihood(particles: np.ndarray, context: StepContext) -> np.ndarray:
    """Evaluate the negative Himmelblau function.

    Parameters
    ----------
    particles : array, shape (n_particles, 2)
        NumPy coordinates (x, y).
    context : StepContext
        Runtime context; unused for this fixed target.

    Returns
    -------
    log_likelihood : array, shape (n_particles,)
        Negative Himmelblau objective, defining an unnormalized likelihood.
    """
    x, y = particles[:, 0], particles[:, 1]
    return -((x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2)


def main() -> None:
    """Initialize NumPy particles and run DE mutations through tempering."""
    configure_logging()
    # Sequince wraps this NumPy generator in orng, preserving its RNG state.
    rng = np.random.default_rng(seed)
    particles = rng.uniform(
        lower_bound, upper_bound, size=(n_particles, n_dimensions)
    )
    # Prior draws start with equal weights, supplied by ParticleState.
    initial_state = ParticleState(particles=particles)
    smc, _, metadata = make_standard_bayesian_smc(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        target_ess_ratio=0.7,
        always_resample=True,
        mutation="differential-evolution",
        mutation_n_steps=10,
        particle_transform="affine",
    )
    result = smc.run(initial_state, rng=rng, metadata=metadata)

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Estimated log evidence: {result.final_state.log_evidence:.4f}")
    print(f"Posterior mean: {to_numpy(result.weighted_mean())}")
    print(f"Posterior covariance:\n{to_numpy(result.weighted_covariance())}")
    fig = result.plot_corner()
    fig.savefig(outdir / "posterior.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    result.plot_log_evidence(ax=axes[0])
    # Read the tempering parameter from each weighting step output.
    betas = [float(to_numpy(output[0]["beta"])) for output in result.outputs]
    axes[1].plot(betas)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Tempering parameter beta")
    fig.tight_layout()
    fig.savefig(outdir / "history.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
