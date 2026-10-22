"""Run functional JAX SMC with DE and BlackJAX NUTS. See README.rst."""

from math import log
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sequince import ParticleState, make_standard_bayesian_smc
from sequince.array_api import to_numpy
from sequince.jax import compile_functional_iteration
from sequince.log import configure_logging
from sequince.steps import StepContext

n_particles = 1000
seed = 42
outdir = Path("outdir/jax")

# Change to "cuda" with GPU-enabled JAX to run on the GPU.
platform = "cpu"

n_dimensions = 2
lower_bound, upper_bound = -6.0, 6.0


def log_prior(particles: jax.Array, context: StepContext) -> jax.Array:
    """Evaluate a normalized uniform prior.

    Parameters
    ----------
    particles : array, shape (..., n_dimensions)
        A particle vector or a batch of particle coordinates.
    context : StepContext
        Runtime context; unused for this fixed target.

    Returns
    -------
    log_density : array, shape (...,)
        Normalized box log density, or negative infinity outside the box.
    """
    inside = jnp.all(
        (particles >= lower_bound) & (particles <= upper_bound), axis=-1
    )
    density = jnp.full_like(
        particles[..., 0], -n_dimensions * log(upper_bound - lower_bound)
    )
    return jnp.where(inside, density, jnp.full_like(density, -float("inf")))


def log_likelihood(particles: jax.Array, context: StepContext) -> jax.Array:
    """Evaluate the negative Himmelblau function.

    Parameters
    ----------
    particles : array, shape (..., 2)
        A coordinate vector (x, y) or a batch of such vectors.
    context : StepContext
        Runtime context; unused for this fixed target.

    Returns
    -------
    log_likelihood : array, shape (...,)
        Negative Himmelblau objective, defining an unnormalized likelihood.
    """
    x, y = particles[..., 0], particles[..., 1]
    return -((x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2)


def main() -> None:
    """Compile an SMC iteration and thread an explicit JAX random key."""
    jax.config.update("jax_platforms", platform)
    configure_logging()
    for mutation in ("differential-evolution", "blackjax"):
        output_dir = outdir / mutation
        # Functional steps use orng internally, threading a JAX key explicitly.
        # Split the key so initialization and SMC use separate random draws.
        particle_key, random_state = jax.random.split(jax.random.key(seed))
        initial_state = ParticleState(
            particles=jax.random.uniform(
                particle_key,
                shape=(n_particles, n_dimensions),
                minval=lower_bound,
                maxval=upper_bound,
            )
        )
        mutation_kwargs = {}
        if mutation == "blackjax":
            # Fixed NUTS settings for this example; no automatic warm-up is run.
            mutation_kwargs = {
                "algorithm": "nuts",
                "algorithm_kwargs": {
                    "step_size": 0.02,
                    "inverse_mass_matrix": jnp.ones(n_dimensions),
                },
            }
        smc, _, metadata = make_standard_bayesian_smc(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            target_ess_ratio=0.7,
            always_resample=True,
            mutation=mutation,
            mutation_n_steps=10,
            mutation_kwargs=mutation_kwargs,
            particle_transform="affine",
            # Keep Python logging outside the compiled iteration.
            logging_frequency=0,
        )
        # Compile one iteration without consuming a sampling transition.
        iteration = compile_functional_iteration(
            smc, initial_state, random_state, metadata=metadata
        )
        result, final_random_state = smc.run_functional(
            initial_state,
            random_state=random_state,
            metadata=metadata,
            iteration=iteration,
        )
        jax.block_until_ready(result.final_state.particles)
        # Retain the returned key for subsequent random operations.
        print(f"Final random key: {jax.random.key_data(final_random_state)}")
        print(f"JAX device: {result.final_state.particles.device}")
        print(f"Mutation: {mutation}")
        likelihood_evaluations = int(
            result.evaluation_counts.log_likelihood_evaluations
        )
        print(f"Total likelihood evaluations: {likelihood_evaluations:,}")

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Estimated log evidence: {result.final_state.log_evidence:.4f}")
        print(f"Posterior mean: {to_numpy(result.weighted_mean())}")
        print(
            f"Posterior covariance:\n{to_numpy(result.weighted_covariance())}"
        )
        fig = result.plot_corner()
        fig.savefig(output_dir / "posterior.png")
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        result.plot_log_evidence(ax=axes[0])
        # Functional runs retain beta in step outputs rather than state metadata.
        betas = [
            float(to_numpy(output[0]["beta"])) for output in result.outputs
        ]
        axes[1].plot(betas)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Tempering parameter beta")
        fig.tight_layout()
        fig.savefig(output_dir / "history.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
