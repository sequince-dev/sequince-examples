import math
import pathlib

import corner
import matplotlib.pyplot as plt
import numpy as np
from hypertriangle import hypercubify, hypertriangulate
from sequince import ParticleState, SequentialMonteCarlo
from sequince.logging_utils import configure_logging
from sequince.posterior import TemperedPosterior
from sequince.result import NestedSMCResult
from sequince.steps import (
    AdaptiveTemperedWeighting,
    ExpandDimensions,
    LoggingStep,
    MiniPCNMutation,
    NestedSMCStep,
    SMCTransformation,
    SystematicResampling,
)
from sequince.steps.base import StepContext
from sequince.steps.control import StopIf
from sequince.transforms import AffineTransform


def component_view(particles, context):
    """View particles as (3, n_pulses, n_particles) array."""
    n_pulses = particles.shape[1] // context.shared["dims_per_pulse"]
    return particles.reshape(
        (particles.shape[0], n_pulses, context.shared["dims_per_pulse"])
    ).T


def physical_component_view(particles, context):
    """Return pulse parameters with ordered physical means."""
    pulses = component_view(particles, context).copy()
    pulses[1] = hypertriangulate(
        pulses[1].T,
        bounds=context.shared["bounds"][:, 1],
    ).T
    return pulses


def physical_component_view_for_dims(particles, n_dims, shared_context):
    """Return pulse parameters with ordered physical means for a fixed dimension."""
    n_pulses = n_dims // shared_context["dims_per_pulse"]
    pulses = particles.reshape(
        (particles.shape[0], n_pulses, shared_context["dims_per_pulse"])
    ).T.copy()
    pulses[1] = hypertriangulate(
        pulses[1].T,
        bounds=shared_context["bounds"][:, 1],
    ).T
    return pulses


def flatten_physical_particles(particles, n_dims, shared_context):
    """Return particles flattened back to (n_particles, n_dims) in physical coordinates."""
    pulses = physical_component_view_for_dims(
        particles, n_dims, shared_context
    )
    return pulses.T.reshape(particles.shape[0], n_dims)


def sort_true_parameters_by_mean(
    true_parameters: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    """Match truth ordering to the ordered-mean parameterization used in plots."""
    amplitudes, means, sigmas = true_parameters
    order = np.argsort(means)
    return amplitudes[order], means[order], sigmas[order]


def plot_physical_corner(
    result,
    *,
    shared_context,
    filename: pathlib.Path,
    labels: list[str],
    truths: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
):
    """Corner plot in physical coordinates rather than stored hypercube coordinates."""
    state = result.final_state
    n_dims = int(state.metadata.get("n_dims", state.particles.shape[1]))
    physical_particles = flatten_physical_particles(
        np.asarray(state.particles),
        n_dims,
        shared_context,
    )

    corner_kwargs = dict(
        bins=32,
        smooth=0.9,
        color="#EFBF35",
        truth_color="#6B8E23",
        quantiles=[0.16, 0.5, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
        plot_density=True,
        plot_datapoints=True,
        fill_contours=True,
        show_titles=True,
        hist_kwargs=dict(density=True, linewidth=2),
        weights=state.weights,
        labels=labels,
    )

    if truths is not None:
        amplitudes, means, sigmas = truths
        n_pulses = n_dims // shared_context["dims_per_pulse"]
        truths_flat = np.column_stack(
            [amplitudes[:n_pulses], means[:n_pulses], sigmas[:n_pulses]]
        ).reshape(-1)
        corner_kwargs["truths"] = truths_flat

    fig = corner.corner(physical_particles, **corner_kwargs)
    fig.savefig(filename)
    plt.close(fig)


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


def log_likelihood(particles, context):
    """
    Log likelihood of Gaussian pulse model with Gaussian noise.
    """
    xdata = context.shared["data_x"]
    ydata = context.shared["data_y"]
    sigma = context.shared["sigma"]
    particles = physical_component_view(particles, context)
    return _log_likelihood(xdata, ydata, sigma, particles)

def bounds_check(particles_component_view, context):
    bounds = context.shared["bounds"]
    in_bounds = np.all(
        np.all(
            (particles_component_view >= bounds[0, :, None, None])
            & (particles_component_view <= bounds[1, :, None, None]),
            axis=0,
        ),
        axis=0,
    )
    return in_bounds


def log_prior(particles, context):
    """
    Uniform prior on means, amplitudes and scales
    """
    bounds = context.shared["bounds"]
    n_pulses = context.metadata["n_dims"] // context.shared["dims_per_pulse"]

    log_norm_constant = (-np.log(bounds[1] - bounds[0]) * n_pulses).sum(0)
    log_norm_constant += math.lgamma(n_pulses + 1)

    log_prior = np.full(particles.shape[0], log_norm_constant)
    particles = component_view(particles, context)
    in_bounds = bounds_check(particles, context)
    log_prior[~in_bounds] = -np.inf
    return log_prior


class PulseAdditionTransformation(SMCTransformation):
    def forward(self, state, context, rng):
        particles = np.asarray(state.particles)
        n_particles = particles.shape[0]
        bounds = context.shared["bounds"]

        # draw new pulse (physical parameters)
        new_pulse = rng.uniform(
            bounds[0],
            bounds[1],
            (n_particles, context.shared["dims_per_pulse"]),
        )
        
        # Transform to physical parameters to add new pulse
        pulses = physical_component_view(particles, context).copy()

        # Append new pulse to existing pulses in physical coordinates
        new_pulses = np.concatenate((pulses, new_pulse.T[:, None, :]), axis=1)

        # sort pulses by their means to enforce identifiability
        sorted_indices = np.argsort(new_pulses[1], axis=0)

        # keep track of the added pulse
        added_pulse_indices = np.where(sorted_indices == pulses.shape[1])[0]

        context.metadata["added_pulse_indices"] = added_pulse_indices

        sorted_pulses = np.take_along_axis(
            new_pulses, sorted_indices[None, :, :], axis=1
        )
        
        # Map the sorted means back to hypercube coordinates
        sorted_pulses[1] = hypercubify(
            sorted_pulses[1].T,
            bounds=context.shared["bounds"][:, 1],
        ).T

        expanded_particles = sorted_pulses.T.reshape(
            n_particles,
            particles.shape[1] + context.shared["dims_per_pulse"],
        )

        return state.with_updates(particles=expanded_particles)

    def inverse(self, particles, context):
        # Map to physical space
        pulses = physical_component_view(particles, context).copy()

        remove = context.metadata["added_pulse_indices"]

        all_pulses = np.arange(pulses.shape[1])
        row_idx = np.where(
            all_pulses[:, None] < remove[None, :],
            all_pulses[:, None],
            all_pulses[:, None] + 1
        )[:pulses.shape[1]-1]                       

        col_idx = np.arange(pulses.shape[2])[None, :]
        reduced_pulses = pulses[:, row_idx, col_idx].T.reshape(
            particles.shape[0],
            particles.shape[1] - context.shared["dims_per_pulse"],
        )
        
        # Map back to hypercube coordinates
        reduced_pulses[1] = hypercubify(
            reduced_pulses[1].T,
            bounds=context.shared["bounds"][:, 1],
        ).T

        return reduced_pulses.T.reshape(particles.shape[0], particles.shape[1] - context.shared["dims_per_pulse"])

    def log_auxiliary_density(self, particles, context):
        in_bounds = bounds_check(component_view(particles, context), context)
        log_q = np.full(particles.shape[0], -np.inf, dtype=float)
        bounds = context.shared["bounds"]
        log_q[in_bounds] =  (-np.log(bounds[1] - bounds[0])).sum(0)
        return log_q

    def log_jacobian(self, particles, context):
        # The hypertriangle reparameterization contributes a constant birth-move
        # Jacobian of k + 1 when going from k ordered means to k + 1.
        n_pulses = (
            context.metadata["n_dims"] // context.shared["dims_per_pulse"]
        )
        return np.full(particles.shape[0], np.log(n_pulses), dtype=float)


def make_inner_parameter_plot_callback(
    outdir: pathlib.Path,
    true_parameters: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    def _callback(state: ParticleState, context: StepContext, result) -> None:
        inner_results = result.context.outputs.get("inner_results")
        if not inner_results:
            return

        nested_result = NestedSMCResult(inner_results)
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        colors = plt.cm.tab10.colors
        labels = ["Amplitude", "Mean", "Sigma"]
        true_amplitudes, true_means, true_sigmas = true_parameters
        true_series = [true_amplitudes, true_means, true_sigmas]

        iteration_offset = 0
        for nested in nested_result.results:
            for local_index, inner_state in enumerate(nested.states):
                n_dims = int(
                    inner_state.metadata.get(
                        "n_dims", inner_state.particles.shape[1]
                    )
                )
                pulses = physical_component_view_for_dims(
                    np.asarray(inner_state.particles),
                    n_dims,
                    context.shared,
                )
                weights = np.asarray(inner_state.weights)
                weighted_means = np.sum(
                    pulses * weights[None, None, :],
                    axis=2,
                )
                x = iteration_offset + local_index
                for component_index in range(weighted_means.shape[1]):
                    color = colors[component_index % len(colors)]
                    for param_index, axis in enumerate(axes):
                        axis.scatter(
                            x,
                            weighted_means[param_index, component_index],
                            color=color,
                            s=18,
                        )
            iteration_offset += len(nested.states)

        max_components = max(
            res.final_state.particles.shape[1]
            // context.shared["dims_per_pulse"]
            for res in nested_result.results
        )
        for component_index in range(max_components):
            color = colors[component_index % len(colors)]
            axes[0].plot(
                [], [], color=color, label=f"Pulse {component_index + 1}"
            )

        for param_index, axis in enumerate(axes):
            truths = true_series[param_index]
            for component_index, true_value in enumerate(truths):
                color = colors[component_index % len(colors)]
                axis.axhline(
                    true_value,
                    color=color,
                    linestyle="--",
                    alpha=0.6,
                )

        for axis, label in zip(axes, labels):
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("Inner iteration")
        fig.tight_layout()
        fig.savefig(outdir / "inner_parameter_trajectories.png")
        plt.close(fig)

    return _callback


def stop_predicate(state: ParticleState, context: StepContext) -> bool:
    return (
        context.metadata["n_dims"] // context.shared["dims_per_pulse"]
        >= context.shared["target_pulses"]
    )


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
        f"Temperature: {temperature:.3e}, "
        f"Log Evidence: {log_evidence:.2e}"
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
    n_particles = 10_000

    outdir = pathlib.Path("outdir") / f"{n_pulses}_pulses"
    outdir.mkdir(parents=True, exist_ok=True)

    A_true, mu_true, sigma_true = rng.uniform(
        bounds[0], bounds[1], (n_pulses, 3)
    ).T
    true_parameters = sort_true_parameters_by_mean(
        (A_true, mu_true, sigma_true)
    )

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

    tempered_posterior = TemperedPosterior(
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
                target_ess_ratio=0.8,
                max_temperature=1.0,
                min_delta=1e-8,
            ),
            SystematicResampling(always_resample=True),
            MiniPCNMutation(
                tempered_posterior,
                n_steps=50,
                step_fn="tpcn",
                transform=AffineTransform(),
            ),
            LoggingStep(str_fn=sub_logging_str_fn, frequency=1),
        ],
        auto_normalize=True,
    )

    td_smc = SequentialMonteCarlo(
        steps=[
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
                posterior=tempered_posterior,
            ),
        ],
        callbacks=[
            make_inner_parameter_plot_callback(outdir, true_parameters)
        ],
    )

    particles = np.zeros((n_particles, dims_per_pulse * n_pulses_start))
    for i in range(n_pulses_start):
        particles[:, i*dims_per_pulse:(i+1)*dims_per_pulse] = rng.uniform(
            low=bounds[0],
            high=bounds[1],
            size=(n_particles, dims_per_pulse),
        )

    # Initial state with draws from the prior
    initial_state = ParticleState(
        particles=particles,
        log_weights=np.full(n_particles, -np.log(n_particles)),
    )

    result = td_smc.run(
        initial_state,
        rng=rng,
        shared_context=shared_context,
        metadata=metadata,
    )

    nested_result = NestedSMCResult(result.context.outputs["inner_results"])

    print("Run summary:")
    for key, value in nested_result.summary().items():
        print(f"  {key}: {value}")

    plot_physical_corner(
        nested_result.results[0],
        shared_context=shared_context,
        filename=outdir / "posterior_1.png",
        labels=["A1", "mu1", "sigma1"],
        truths=true_parameters,
    )
    plot_physical_corner(
        nested_result.results[1],
        shared_context=shared_context,
        filename=outdir / "posterior_2.png",
        labels=["A1", "mu1", "sigma1", "A2", "mu2", "sigma2"],
        truths=true_parameters,
    )
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
