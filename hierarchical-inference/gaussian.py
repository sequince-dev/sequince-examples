"""Example of using SMC for hierarchical inference of a Gaussian mean parameter
with known variance.
"""

import matplotlib.pyplot as plt
import numpy as np
from sequince import ParticleState, SequentialMonteCarlo
from sequince.steps import LogLikelihoodWeighting, SystematicResampling


def log_likelihood(particles, context):
    obs_var = context.shared["obs_var"]
    obs = np.atleast_2d(context.observation)
    return -0.5 * np.sum(
        (obs - particles[np.newaxis, :]) ** 2 / obs_var, axis=0
    )


def main():
    rng = np.random.default_rng(seed=1234)

    # True 'population' parameters
    lambda_mu = 1.0
    lambda_sigma = 0.5

    # Prior parameters
    prior_mu = 0.0
    prior_sigma = 2.0

    # Number of observations
    n_observations = 150

    # Realization of the population
    true_mu = rng.normal(lambda_mu, lambda_sigma)

    # Each observation is draw from a Gussian
    obs_var = 0.5
    observations = rng.normal(true_mu, np.sqrt(obs_var), size=n_observations)

    # Compute the posterior variance analytically for comparison
    analytic_posterior_var = 1 / (
        n_observations / obs_var + 1 / prior_sigma**2
    )
    analytic_posterior_mean = analytic_posterior_var * (
        np.sum(observations) / obs_var + prior_mu / prior_sigma**2
    )

    # We incorporate the prior by drawing the initial particles from it
    initial_particles = rng.normal(
        loc=prior_mu, scale=prior_sigma, size=(1000,)
    )
    initial_log_weights = np.full(
        len(initial_particles), -np.log(len(initial_particles))
    )
    state = ParticleState(
        particles=initial_particles, log_weights=initial_log_weights
    )

    steps = [
        LogLikelihoodWeighting(log_likelihood, normalize_after=False),
        SystematicResampling(always_resample=True),
    ]

    smc = SequentialMonteCarlo(steps=steps, auto_normalize=True)

    result = smc.run(
        state,
        observations=iter(observations),
        rng=rng,
        shared_context={
            "obs_var": obs_var,
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
        },
    )

    final_state = result[-1]
    posterior_mean = np.mean(final_state.particles)
    posterior_var = np.var(final_state.particles)

    print(f"True mean: {true_mu:.3f}")
    print(f"Posterior mean: {posterior_mean:.3f}")
    print(f"Posterior std: {np.sqrt(posterior_var):.3f}")
    print(f"Analytical posterior mean: {analytic_posterior_mean:.3f}")
    print(f"Analytical posterior std: {np.sqrt(analytic_posterior_var):.3f}")

    running_mean = []
    for state in result:
        running_mean.append(np.mean(state.particles))
    running_var = []
    for state in result:
        running_var.append(np.var(state.particles))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(final_state.particles, bins=30, alpha=0.7)
    axs[0].set_xlabel(r"$\mu$")
    axs[0].set_ylabel("Frequency")
    axs[1].plot(
        observations, label="Observations", color="tab:gray", alpha=0.5
    )
    axs[1].plot(running_mean, label="Running mean", color="tab:blue")
    axs[1].fill_between(
        np.arange(len(running_mean)),
        np.array(running_mean) - np.sqrt(running_var),
        np.array(running_mean) + np.sqrt(running_var),
        alpha=0.2,
        color="tab:blue",
    )
    axs[1].axhline(true_mu, color="tab:red", linestyle="--", label="True mean")
    axs[1].legend()
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Running Mean")
    fig.savefig("gaussian_smc.png")


if __name__ == "__main__":
    main()
