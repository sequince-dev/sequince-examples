import pathlib
from typing import Callable
from unittest import signals
import matplotlib.pyplot as plt
import numpy as np

import corner

from sequince import ParticleState, SequentialMonteCarlo
from sequince.logging_utils import configure_logging
from sequince.posterior import TemperedPosterior
from sequince.steps import (
    AdaptiveTemperedWeighting,
    LoggingStep,
    MiniPCNMutation,
    SystematicResampling,
    NestedSMCStep,
    SMCTransformation,
    ExpandDimensions
)
from sequince.steps.control import StopIf
from sequince.result import NestedSMCResult
from sequince.steps.base import StepContext
from sequince.transforms import AffineTransform

from hypertriangle import hypertriangulate

def component_view(particles, context):
    """View particles as (3, n_pulses, n_particles) array."""
    n_pulses = context.metadata["n_dims"] // context.shared["dims_per_pulse"]
    return particles.reshape((particles.shape[0], n_pulses, context.shared["dims_per_pulse"])).T

def gauss(x, A, mu, sigma):
    """Single Gaussian pulse model."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _log_likelihood(xdata, ydata, sigma, particles):
    templates = gauss(xdata[:, None, None], *particles)
    log_norm = -0.5 * ydata.shape[0] * np.log(2 * np.pi * sigma**2)
    logl = (
        -0.5 * ((ydata[:, None] - templates.sum(1)) ** 2).sum(0) / (sigma**2)
        + log_norm
    )
    return logl

def _log_likelihood_no_sum(xdata, ydata, sigma, particles):
    templates = gauss(xdata[:, None, None], *particles)
    log_norm = -0.5 * ydata.shape[0] * np.log(2 * np.pi * sigma**2)
    return (
        -0.5 * ((ydata[:, None, None] - templates) ** 2).sum(0) / (sigma**2)
        + log_norm
    )


def log_likelihood(particles, context, sum_over_pulses=True):
    """
    Log likelihood of Gaussian pulse model with Gaussian noise.
    """
    xdata = context.shared["data_x"]
    ydata = context.shared["data_y"]
    sigma = context.shared["sigma"]
    particles = component_view(particles, context).copy()

    # apply hypertriangle map
    # particles[1] = hypertriangulate(particles[1].T, bounds=context.shared["bounds"][:,1]).T

    if sum_over_pulses:
        return _log_likelihood(xdata, ydata, sigma, particles)
    else:
        return _log_likelihood_no_sum(xdata, ydata, sigma, particles)


def bounds_check(particles_component_view, context):
    bounds = context.shared["bounds"]
    in_bounds = np.all(np.all((
        particles_component_view >= bounds[0,:,None,None]
    ) & (
        particles_component_view <= bounds[1,:,None,None]
    ), axis=0), axis=0)
    return in_bounds

def log_prior(particles, context):
    """
    Uniform prior on means, amplitudes and scales
    """
    bounds = context.shared["bounds"]
    n_pulses = context.metadata["n_dims"] // context.shared["dims_per_pulse"]

    log_norm_constant = (
        -np.log(bounds[1] - bounds[0]) * n_pulses
    ).sum(0)

    log_prior = np.full(particles.shape[0], log_norm_constant)
    particles = component_view(particles, context)
    in_bounds = bounds_check(particles, context)
    log_prior[~in_bounds] = -np.inf
    return log_prior


def component_sampler(state, context, rng):
    """Sample new components from the prior."""
    new_components = rng.uniform(
        context.shared["bounds"][0],
        context.shared["bounds"][1],
        size=(state.particles.shape[0], context.shared["dims_per_pulse"])
    )
    return new_components


class PulseTemperedPosterior(TemperedPosterior):
    def log_increment(
        self,
        particles,
        context,
        *,
        log_proposal_override=None,
    ):
        if log_proposal_override is None:
            return super().log_increment(
                particles,
                context,
                log_proposal_override=log_proposal_override,
            )

        
        components = component_view(particles, context)
        # components[1] = hypertriangulate(components[1].T, bounds=context.shared["bounds"][:,1]).T
        last_pulse = components[:, -1][:, None, :]
        
        xdata = context.shared["data_x"]
        ydata = context.shared["data_y"]
        sigma = context.shared["sigma"]
        increments = _log_likelihood(xdata, ydata, sigma, last_pulse)

        if particles.ndim == 1:
            return np.array(increments[0], ndmin=1)
        return increments

    def log_prob(
        self,
        particles,
        context,
        *,
        log_proposal_override=None,
    ):
        if log_proposal_override is None:
            return super().log_prob(
                particles,
                context,
                log_proposal_override=log_proposal_override,
            )

        n_dims = int(context.metadata["n_dims"])
        beta = float(context.metadata.get("temperature", 0.0))

        log_prior_vals = log_prior(particles, context)
        
        if n_dims <= 0:
            result = log_prior_vals
        else:
            partial_logls = log_likelihood(particles, context, sum_over_pulses=False)
            if n_dims == context.shared["dims_per_pulse"]:
                previous_log_like = np.zeros(particles.shape[0], dtype=float)
            else:
                previous_log_like = partial_logls[:-1].sum(0)

            new_log_like = partial_logls[-1]
            result = log_prior_vals + previous_log_like + beta * new_log_like

        if particles.ndim == 1:
            return float(result[0])
        return result


class PulseAdditionTransformation(SMCTransformation):
    def forward(self, state, context, rng):
        new_block = component_sampler(state, context, rng) # (n_particles, 3)
        particles = np.asarray(state.particles)
        expanded_particles = np.concatenate([particles, new_block], axis=1) # (n_particles, old_dim + 3)
        return state.with_updates(particles=expanded_particles)

    def inverse(self, particles, context):
        return particles[:, :-3]

    def log_auxiliary_density(self, particles, context):
        new_block = particles[:, -3:]
        in_bounds = bounds_check(new_block.T[:, None, :], context)
        log_q = np.full(new_block.shape[0], -np.inf, dtype=float)

        bounds = context.shared["bounds"]

        log_norm_constant = (
            -np.log(bounds[1] - bounds[0])
        ).sum(0)


        log_q[in_bounds] = log_norm_constant
        return log_q


def stop_predicate(state, context):
    return context.metadata['n_dims'] // context.shared['dims_per_pulse'] >= context.shared['target_pulses']

def logging_str_fn(state: ParticleState, context: StepContext) -> str:
    n_pulses = context.metadata["n_dims"] // context.shared["dims_per_pulse"]
    log_evidence = state.log_evidence
    return (
        f"it {context.t}:, "
        f"Pulses: {n_pulses}, "
        f"Log Evidence: {log_evidence:.1f}"
    )

def sub_logging_str_fn(state: ParticleState, context: StepContext) -> str:
    temperature = context.metadata["temperature"]
    log_evidence = state.log_evidence
    return (
        f"sub-it {context.t}:, "
        f"Temperature: {temperature:.4f}, "
        f"Log Evidence: {log_evidence:.1f}"
    )


def main():
    configure_logging()

    rng = np.random.default_rng(1730)

    bounds = np.array(
        [
            [0.1, 1.5],
            [-4, 4],
            [0.1, 1],
        ]
    ).T

    dims_per_pulse = 3
    n_pulses = 2
    n_pulses_start = 1
    target_n_pulses = 2
    n_particles = 5_000

    outdir = pathlib.Path("outdir") / f"{n_pulses}_pulses"
    outdir.mkdir(parents=True, exist_ok=True)

    A_true, mu_true, sigma_true = rng.uniform(
        bounds[0], bounds[1], (n_pulses, 3)
    ).T

    print(f"True parameters: A={A_true}, mu={mu_true}, sigma={sigma_true}")

    xdata = np.linspace(-5, 5, 101)
    signals = gauss(xdata[:, None], A_true, mu_true, sigma_true)
    signal_tot = signals.sum(1)
    noise_sigma = 1e-3
    noise = noise_sigma * rng.normal(size=xdata.shape)
    ydata = signal_tot + noise

    plt.plot(xdata, ydata)
    plt.plot(xdata, signal_tot)
    plt.plot(xdata, signals, alpha=0.5, c="k")
    plt.savefig(outdir / "data.png")
    plt.close()

    tempered_posterior = PulseTemperedPosterior(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
    )

    metadata = {
        "n_dims": dims_per_pulse * n_pulses_start,
    }

    shared_context = {
        "data_x": xdata,
        "data_y": ydata,
        "sigma": noise_sigma,
        "bounds": bounds,
        "target_pulses": target_n_pulses,
        "dims_per_pulse": dims_per_pulse,
    }

    inner_smc = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(
                tempered_posterior,
                target_ess_ratio=0.9,
                max_temperature=1.0,
                min_delta=1e-8,
            ),
            SystematicResampling(always_resample=True),
            MiniPCNMutation(
                tempered_posterior,
                n_steps=20,
                step_fn="tpcn",
                transform=AffineTransform(),
            ),
            LoggingStep(str_fn=sub_logging_str_fn, frequency=1),
        ],
        auto_normalize=True,
    )

    td_smc = SequentialMonteCarlo(
        steps = [
            NestedSMCStep(
                inner_smc,
                posterior=tempered_posterior,
                record_outputs=True,
                output_key="inner_results",
            ),
            LoggingStep(str_fn=logging_str_fn, frequency=1),
            StopIf(stop_predicate),
            ExpandDimensions(
                transformation=PulseAdditionTransformation(),
                expected_block_size=dims_per_pulse,
                metadata_key="n_dims",
                posterior=tempered_posterior
            ),
        ]
    )

    # Initial state with draws from the prior
    initial_state = ParticleState(
        particles=rng.uniform(
            low=bounds[0],
            high=bounds[1],
            size=(n_particles, dims_per_pulse),
        ),
        log_weights=np.full(n_particles, -np.log(n_particles)),
    )

    result = td_smc.run(
        initial_state, 
        rng=rng, 
        shared_context=shared_context,
        metadata=metadata
    )

    nested_result = NestedSMCResult(result.context.outputs["inner_results"])

    print("Run summary:")
    for key, value in nested_result.summary().items():
        print(f"  {key}: {value}")

    nested_result.results[0].plot_corner(labels=["A1", "mu1", "sigma1"], filename=outdir / "posterior_1.png")
    nested_result.results[1].plot_corner(labels=["A1", "mu1", "sigma1", "A2", "mu2", "sigma2"], filename=outdir / "posterior_2.png")
    # nested_result.results[2].plot_corner(labels=["A1", "mu1", "sigma1", "A2", "mu2", "sigma2", "A3", "mu3", "sigma3"], filename=outdir / "posterior_3.png")

    nested_result.plot_log_evidence()
    plt.savefig(outdir / "evidence_history.png")
    
    # ln_evidences = result.log_evidence_history()
    # n_pulses_history = result.metadata_history("n_pulses")
    # evidence_per_n_pulses = {}
    # idx_per_n_pulses = {}
    # for n in range(1, target_n_pulses + 1):
    #     idx = np.where(n_pulses_history == n)[0]
    #     evidence_per_n_pulses[n] = ln_evidences[idx.max()]
    #     idx_per_n_pulses[n] = idx.max()

    # print("Evidence per number of pulses:")
    # for n, ln_evidence in evidence_per_n_pulses.items():
    #     print(f"  {n} pulses: ln evidence = {ln_evidence}")

    # preferred_n_pulses = max(
    #     evidence_per_n_pulses,
    #     key=lambda n: evidence_per_n_pulses[n],
    # )
    # print(f"Preferred number of pulses: {preferred_n_pulses}")

    # fig, axes = plt.subplots(2, 1)
    # result.plot_metadata_history("temperature", ax=axes[0])
    # result.plot_metadata_history("n_pulses", ax=axes[1])
    # fig.tight_layout()
    # fig.savefig(outdir / "temperature_schedule.png")

    # # Plot the evidence
    # fig = plt.figure()
    # result.plot_log_evidence(ax=plt.gca())
    # fig.savefig(outdir / "evidence.png")

    # # Use posterior samples from preferred number of pulses
    # state = result[idx_per_n_pulses[preferred_n_pulses]]
    # posterior_samples = state.particles

    # context = StepContext(
    #     t=len(result) - 1,
    #     observation=None,
    #     shared=shared_context,
    # )

    # log_likelihoods = log_likelihood(posterior_samples, context)
    # log_priors = log_prior(posterior_samples, context)
    # log_posteriors = log_likelihoods + log_priors

    # particles_reshaped = component_view(posterior_samples)

    # max_post_idx = np.argmax(log_posteriors)
    # print("Max posterior particle:")
    # print(f"Amplitudes: {particles_reshaped[max_post_idx, 0, :]}")
    # print(f"Means: {particles_reshaped[max_post_idx, 1, :]}")
    # print(f"Scales: {particles_reshaped[max_post_idx, 2, :]}")
    # print("Log Likelihood:", log_likelihoods[max_post_idx])
    # print("Log Prior:", log_priors[max_post_idx])
    # print("Log Posterior:", log_posteriors[max_post_idx])

    # max_templates = gauss(
    #     xdata[:, None],
    #     particles_reshaped[max_post_idx, 0, :],
    #     particles_reshaped[max_post_idx, 1, :],
    #     particles_reshaped[max_post_idx, 2, :],
    # )
    # max_template = max_templates.sum(-1)

    # fig = plt.figure()
    # plt.plot(xdata, ydata, alpha=0.3, label="Data")
    # plt.plot(xdata, max_template, c="C1", lw=2, label="MAP template")
    # for template in max_templates.T:
    #     plt.plot(xdata, template, c="k", alpha=0.3)
    # for signal in signals.T:
    #     plt.plot(xdata, signal, c="red", alpha=0.1, ls="--")
    # plt.legend()
    # plt.title("Posterior Predictive")
    # fig.savefig(outdir / "fit.png")

    # # Plot all parameters on the same 3D corner plot
    # fig = None
    # # Corner range, each entry is (min, max)
    # corner_range = bounds.T
    # # Colors set based on order of components
    # # Use dark blue to light blue
    # color_map = plt.get_cmap("Blues_r")
    # # Reverse order so first component is on top
    # for component_id in reversed(range(preferred_n_pulses)):
    #     component_particles = particles_reshaped[:, :, component_id]
    #     fig = corner.corner(
    #         component_particles,
    #         fig=fig,
    #         color=color_map(component_id / preferred_n_pulses),
    #         bins=30,
    #         plot_datapoints=False,
    #         plot_density=False,
    #         plot_contours=True,
    #         smooth=1.6,
    #         range=corner_range,
    #         labels=["Amplitude", "Mean", "Scale"],
    #     )
    # fig.savefig(outdir / "posterior_corner.png")


if __name__ == "__main__":
    main()
