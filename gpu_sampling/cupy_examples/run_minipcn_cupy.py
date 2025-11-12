import matplotlib.pyplot as plt
import cupy as cp

from sequince import ParticleState
from sequince.logging_utils import configure_logging
from sequince.posterior import draw_additional_samples
from sequince.prebuilt_samplers import make_standard_bayesian_smc
from orng import ArrayRNG

# Number of dimensions in the multivariate normal model
dims = 4
# Mean and standard deviation of the target normal distribution
mu = cp.array(0.0)
sigma = cp.array(1.0)
# Prior bounds
bounds = (cp.array(-10.0), cp.array(10.0))
# True log evidence for the model with uniform prior over the bounds
true_log_evidence = -dims * cp.log(cp.array(20.0))


def log_likelihood(particles: cp.ndarray, context) -> cp.ndarray:
    """Log-likelihood for a multivariate standard normal observation."""
    return (
        -0.5 * cp.sum(cp.square(particles - mu) / sigma**2, axis=-1)
        - (dims / 2) * cp.log(2 * cp.pi * sigma**2)
    )


def log_prior(particles: cp.ndarray, context) -> cp.ndarray:
    """Uniform prior over a bounded region."""
    lower_bound = context.shared["lower_bound"]
    upper_bound = context.shared["upper_bound"]
    valid = (particles >= lower_bound) & (particles <= upper_bound)
    # sum over dimensions; assign 0 if within bounds else -inf
    return cp.where(valid, 0, -cp.inf).sum(axis=-1)


def main() -> None:
    # Create a CuPy RNG
    rng = ArrayRNG(backend='cupy', seed=42)

    # Enable logging
    configure_logging()

    # Initialize particles from the prior
    lower_bound, upper_bound = bounds
    n_particles = 1000
    initial_particles = rng.uniform(low=lower_bound, high=upper_bound, size=(n_particles, dims))
    initial_log_weights = cp.full((n_particles,), -cp.log(cp.array(n_particles)))

    initial_state = ParticleState(
        particles=initial_particles,
        log_weights=initial_log_weights
    )

    # Use the prebuilt SMC with MiniPCN mutation
    # MiniPCN is the only sampler that supports the array API backends
    smc, posterior, shared_context = make_standard_bayesian_smc(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        target_ess_ratio=0.7,
        max_temperature=1.0,
        min_delta=1e-8,
        always_resample=True,
        mutation="minipcn",
        mutation_n_steps=10,
        particle_transform="affine",
        auto_normalize=True,
    )

    # Add prior bounds to the shared context
    shared_context["lower_bound"] = lower_bound
    shared_context["upper_bound"] = upper_bound

    result = smc.run(initial_state, rng=rng, shared_context=shared_context)

    fig, axs = plt.subplots(2, 1, figsize=(4, 12))
    result.plot_log_evidence(ax=axs[0])
    result.plot_metadata_history("temperature", ax=axs[1])
    fig.savefig("state.png")

    fig = result.plot_corner()
    fig.savefig("posterior.png")


if __name__ == "__main__":
    main()
