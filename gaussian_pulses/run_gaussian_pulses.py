from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sequince import ParticleState, SequentialMonteCarlo
from sequince.steps import (
    AdaptiveTemperedWeighting,
    ContinueWhile,
    DimensionalityExpansion,
    EmceeMutation,
    SystematicResampling,
)
from sequince.steps.base import StepContext


def gauss(x, A, mu, sigma):
    """Single Gaussian pulse model."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def component_view(particles):
    """View particles as (n_particles, 3, n_pulses) array."""
    n_pulses = particles.shape[1] // 3
    particles = particles.reshape((particles.shape[0], n_pulses, 3))
    particles = particles.transpose((0, 2, 1))
    return particles


def log_likelihood(particles, context):
    xdata = context.static["data_x"]
    ydata = context.static["data_y"]
    sigma = context.static["sigma"]
    particles = component_view(particles)
    templates = gauss(
        xdata[:, None, None],
        particles[:, 0, :],
        particles[:, 1, :],
        particles[:, 2, :],
    ).sum(-1)
    log_norm = -0.5 * ydata.shape[0] * np.log(2 * np.pi * sigma**2)
    return (
        -0.5 * ((ydata[:, None] - templates) ** 2).sum(0) / (sigma**2)
        + log_norm
    )


def log_prior(particles, context):
    """Uniform prior on means, amplitudes and scales"""
    mean_bounds = context.static.get("mean_bounds", None)
    scale_bounds = context.static.get("scale_bounds", None)
    amplitude_bounds = context.static.get("amplitude_bounds", None)

    n_pulses = particles.shape[1] // 3

    log_norm_constant = (
        -np.log(mean_bounds[1] - mean_bounds[0]) * n_pulses
        - np.log(scale_bounds[1] - scale_bounds[0]) * n_pulses
        - np.log(amplitude_bounds[1] - amplitude_bounds[0]) * n_pulses
    )

    log_prior = np.full(particles.shape[0], log_norm_constant)

    # View of particles as (n_particles, 3, n_pulses)
    particles = component_view(particles)

    if mean_bounds is not None:
        in_mean_bounds = np.all(
            (particles[:, 1, :] >= mean_bounds[0])
            & (particles[:, 1, :] <= mean_bounds[1]),
            axis=1,
        )
        log_prior[~in_mean_bounds] = -np.inf
    if scale_bounds is not None:
        in_scale_bounds = np.all(
            (particles[:, 2, :] >= scale_bounds[0])
            & (particles[:, 2, :] <= scale_bounds[1]),
            axis=1,
        )
        log_prior[~in_scale_bounds] = -np.inf
    if amplitude_bounds is not None:
        in_amplitude_bounds = np.all(
            (particles[:, 0, :] >= amplitude_bounds[0])
            & (particles[:, 0, :] <= amplitude_bounds[1]),
            axis=1,
        )
        log_prior[~in_amplitude_bounds] = -np.inf
    return log_prior


def tempered_log_posterior(state, context):
    temperature = context.static["temperature"]
    logp = log_prior(state, context)
    logl = log_likelihood(state, context)
    return temperature * logl + logp


def component_sampler(state, context, rng):
    """Sample new components from the prior."""
    n_particles = state.particles.shape[0]
    mean_bounds = context.static["mean_bounds"]
    scale_bounds = context.static["scale_bounds"]
    amplitude_bounds = context.static["amplitude_bounds"]

    new_means = rng.uniform(
        mean_bounds[0], mean_bounds[1], size=(n_particles, 1)
    )
    new_scales = rng.uniform(
        scale_bounds[0], scale_bounds[1], size=(n_particles, 1)
    )
    new_amplitudes = rng.uniform(
        amplitude_bounds[0], amplitude_bounds[1], size=(n_particles, 1)
    )

    new_components = np.concatenate(
        [new_amplitudes, new_means, new_scales],
        axis=1,
    )
    return new_components


class TemperatureTriggeredExpansion(DimensionalityExpansion):
    def __init__(
        self,
        target_pulses: int,
        trigger_temperature: float = 0.999,
    ):
        super().__init__(
            component_sampler=component_sampler,
            expected_block_size=3,
            metadata_key="dimensions",
            context_key="dimensions",
            context_delta_key="dimension_delta",
        )
        self.target_pulses = target_pulses

        self.trigger_temperature = trigger_temperature

    def should_expand(self, context) -> bool:
        temperature = context.static["temperature"]
        trigger_temperature = self.trigger_temperature
        current_pulses = (
            context.static["dimensions"] // self._expected_block_size
        )
        return (temperature >= trigger_temperature) and (
            current_pulses < self.target_pulses
        )

    def apply(self, state, context, rng):
        if not self.should_expand(context):
            return state
        state = super().apply(state, context, rng)
        context.static["n_pulses"] = context.static["dimensions"] // 3
        state.metadata["n_pulses"] = context.static["n_pulses"]
        return state


def make_continue_annealing_predicate(
    target_pulses: int,
    trigger_temperature: float = 0.999,
) -> Callable:
    def predicate(state, context) -> bool:
        temperature = context.static["temperature"]
        current_pulses = context.static["dimensions"] // 3
        continue_annealing = (current_pulses < target_pulses) or (
            temperature < trigger_temperature
        )
        return continue_annealing

    return predicate


def main():
    rng = np.random.default_rng(1751)

    bounds = np.array(
        [
            [0.5, 1.5],
            [1, 4],
            [0.5, 1],
        ]
    ).T

    n_pulses = 3

    A_true, mu_true, sigma_true = rng.uniform(
        bounds[0], bounds[1], (n_pulses, 3)
    ).T

    print(f"True parameters: A={A_true}, mu={mu_true}, sigma={sigma_true}")

    xdata = np.linspace(0, 5, 1001)

    signals = gauss(xdata[:, None], A_true, mu_true, sigma_true)

    sigma = 1e-2

    signal_tot = signals.sum(1)
    noise = sigma * rng.normal(size=xdata.shape)

    ydata = signal_tot + noise

    plt.plot(xdata, ydata)
    plt.plot(xdata, signal_tot)
    plt.plot(xdata, signals, alpha=0.5, c="k")

    target_n_pulses = 4
    trigger_temperature = 0.999

    smc = SequentialMonteCarlo(
        steps=[
            TemperatureTriggeredExpansion(
                target_pulses=target_n_pulses,
                trigger_temperature=trigger_temperature,
            ),
            AdaptiveTemperedWeighting(
                log_likelihood,
                target_ess_ratio=0.8,
                max_temperature=1.0,
                min_delta=1e-8,
            ),
            SystematicResampling(always_resample=True),
            EmceeMutation(
                tempered_log_posterior,
                n_steps=10,
                vectorize=True,
                run_kwargs={"skip_initial_state_check": True},
            ),
            ContinueWhile(
                make_continue_annealing_predicate(
                    target_pulses=target_n_pulses,
                    trigger_temperature=trigger_temperature,
                )
            ),
        ],
        auto_normalize=True,
    )
    # Initial state with draws from the prior
    initial_state = ParticleState(
        particles=rng.uniform(
            low=bounds[0],
            high=bounds[1],
            size=(500, 3),
        ),
        log_weights=np.full(500, -np.log(500)),
        metadata={
            "temperature": 0.0,
            "n_pulses": 1,
            "dimensions": 3,
        },
    )

    static_context = {
        "data_x": xdata,
        "data_y": ydata,
        "sigma": sigma,
        "amplitude_bounds": bounds[:, 0],
        "mean_bounds": bounds[:, 1],
        "scale_bounds": bounds[:, 2],
        "n_pulses": 1,
        "dimensions": 3,
        "temperature": 0.0,
    }

    initial_log_priors = log_prior(
        initial_state.particles,
        StepContext(
            t=0,
            observation=None,
            static=static_context,
        ),
    )

    # check logl around truth
    truth_point = np.zeros((1, 3 * n_pulses))
    truth_point[0, 0::3] = A_true
    truth_point[0, 1::3] = mu_true
    truth_point[0, 2::3] = sigma_true
    true_log_likelihood = log_likelihood(
        truth_point,
        StepContext(
            t=0,
            observation=None,
            static=static_context,
        ),
    )
    print("True log likelihood:", true_log_likelihood)

    if not np.all(np.isfinite(initial_log_priors)):
        raise ValueError("Some initial particles have zero prior density.")

    result = smc.run(initial_state, rng=rng, static_context=static_context)

    print("Run summary:")
    for key, value in result.summary().items():
        print(f"  {key}: {value}")

    ln_evidences = result.log_evidence_history()
    n_pulses_history = result.metadata_history("n_pulses")
    evidence_per_n_pulses = {}
    idx_per_n_pulses = {}
    for n in range(1, target_n_pulses + 1):
        idx = np.where(n_pulses_history == n)[0]
        evidence_per_n_pulses[n] = ln_evidences[idx.max()]
        idx_per_n_pulses[n] = idx.max()

    print("Evidence per number of pulses:")
    for n, ln_evidence in evidence_per_n_pulses.items():
        print(f"  {n} pulses: ln evidence = {ln_evidence}")

    preferred_n_pulses = max(
        evidence_per_n_pulses,
        key=lambda n: evidence_per_n_pulses[n],
    )
    print(f"Preferred number of pulses: {preferred_n_pulses}")

    fig, axes = plt.subplots(2, 1)
    result.plot_metadata_history("temperature", ax=axes[0])
    result.plot_metadata_history("n_pulses", ax=axes[1])
    fig.tight_layout()
    fig.savefig("temperature_schedule.png")

    # Plot the evidence
    fig = plt.figure()
    result.plot_log_evidence(ax=plt.gca())
    fig.savefig("evidence.png")

    # Use posterior samples from preferred number of pulses
    state = result[idx_per_n_pulses[preferred_n_pulses]]
    posterior_samples = state.particles

    context = StepContext(
        t=len(result) - 1,
        observation=None,
        static=static_context,
    )

    log_likelihoods = log_likelihood(posterior_samples, context)
    log_priors = log_prior(posterior_samples, context)
    log_posteriors = log_likelihoods + log_priors

    particles_reshaped = component_view(posterior_samples)

    max_post_idx = np.argmax(log_posteriors)
    print("Max posterior particle:")
    print(f"Amplitudes: {particles_reshaped[max_post_idx, 0, :]}")
    print(f"Means: {particles_reshaped[max_post_idx, 1, :]}")
    print(f"Scales: {particles_reshaped[max_post_idx, 2, :]}")
    print("Log Likelihood:", log_likelihoods[max_post_idx])
    print("Log Prior:", log_priors[max_post_idx])
    print("Log Posterior:", log_posteriors[max_post_idx])

    max_template = gauss(
        xdata[:, None],
        particles_reshaped[max_post_idx, 0, :],
        particles_reshaped[max_post_idx, 1, :],
        particles_reshaped[max_post_idx, 2, :],
    ).sum(-1)

    fig = plt.figure()
    plt.plot(xdata, ydata, alpha=0.3, label="Data")
    plt.plot(xdata, max_template, c="C1", lw=2, label="MAP template")
    plt.legend()
    plt.title("Posterior Predictive")
    fig.savefig("fit.png")


if __name__ == "__main__":
    main()
