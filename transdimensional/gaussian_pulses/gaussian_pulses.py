"""Fit Gaussian pulses using prior births and adaptive tempering.

Example: python gaussian_pulses.py --true-pulses 8 --max-pulses 12

Each expansion appends a pulse drawn from its prior. The new pulse prior
cancels in the density ratio, which reuses the old and new pulse signals.
Each inner SMC run bridges the transformed previous posterior to the next
pulse-count posterior. Model comparisons assume equal prior model probabilities.
"""

import argparse
import pathlib

import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sequince import ParticleState, SequentialMonteCarlo, TemperedPosterior
from sequince.log import configure_logging
from sequince.steps import (
    AdaptiveTemperedWeighting,
    DifferentialEvolutionMutation,
    LoggingStep,
    MetaSMCStep,
    SMCTransformation,
    SystematicResampling,
    TransformationStep,
)
from sequince.steps.context import LOG_PROPOSAL_OVERRIDE_KEY
from sequince.steps.control import StopIf


def gauss(x, A, mu, sigma):
    """Single Gaussian pulse model."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def component_view(particles):
    """View interleaved (amplitude, mean, width) triples as (N, 3, K)."""
    points = np.atleast_2d(particles)
    if points.shape[1] % 3:
        raise ValueError("Each pulse must contain amplitude, mean and width.")
    return points.reshape(points.shape[0], -1, 3).transpose(0, 2, 1)


def pulse_signals(particles, x):
    """Return individual signals with shape (observations, particles, pulses)."""
    components = component_view(particles)
    return gauss(
        x[:, None, None], components[:, 0], components[:, 1], components[:, 2]
    )


def log_likelihood(particles, context):
    templates = pulse_signals(particles, context.shared["data_x"]).sum(-1)
    data = context.shared["data_y"]
    sigma = context.shared["sigma"]
    values = -0.5 * np.sum((data[:, None] - templates) ** 2, axis=0) / sigma**2
    values -= 0.5 * len(data) * np.log(2 * np.pi * sigma**2)
    return values[0] if particles.ndim == 1 else values


def prior_bounds(context):
    return np.array(
        [
            context.shared[key]
            for key in ("amplitude_bounds", "mean_bounds", "scale_bounds")
        ]
    )


def log_prior(particles, context):
    """Normalised independent uniform priors on labelled pulse parameters."""
    components = component_view(particles)
    bounds = prior_bounds(context)
    in_bounds = np.all(
        (components >= bounds[None, :, :1])
        & (components <= bounds[None, :, 1:]),
        axis=(1, 2),
    )
    values = np.where(
        in_bounds,
        -components.shape[-1] * np.log(bounds[:, 1] - bounds[:, 0]).sum(),
        -np.inf,
    )
    return values[0] if particles.ndim == 1 else values


def component_sampler(state, context, rng):
    bounds = prior_bounds(context)
    return rng.uniform(
        bounds[:, 0], bounds[:, 1], size=(len(state.particles), 3)
    )


class PulseAdditionTransformation(SMCTransformation):
    """Append a pulse from its prior with a unit-Jacobian inverse."""

    def forward_with_context(self, state, context, rng):
        particles = np.concatenate(
            [state.particles, component_sampler(state, context, rng)], axis=1
        )
        metadata = dict(state.metadata)
        metadata.update(
            n_pulses=particles.shape[1] // 3, beta=0.0, delta_beta=0.0
        )
        return state.with_updates(
            particles=particles, metadata=metadata
        ), context.with_metadata(**metadata)

    def inverse(self, particles, context):
        return particles[..., :-3]

    def log_jacobian(self, particles, context):
        return 0.0

    def log_auxiliary_density(self, particles, context):
        return log_prior(particles[..., -3:], context)


def log_density_ratio(particles, context):
    """Reuse pulse signals when computing the new-minus-old likelihood.

    On expansion the new pulse prior cancels with the auxiliary proposal.
    The first model instead uses the full likelihood, including normalisation.
    """
    if context.metadata.get(LOG_PROPOSAL_OVERRIDE_KEY) is None:
        return log_likelihood(particles, context)
    signals = pulse_signals(particles, context.shared["data_x"])
    old_residual = context.shared["data_y"][:, None] - signals[..., :-1].sum(
        -1
    )
    addition = signals[..., -1]
    values = np.sum(2 * old_residual * addition - addition**2, axis=0) / (
        2 * context.shared["sigma"] ** 2
    )
    return values[0] if particles.ndim == 1 else values


class AddPulse(TransformationStep):
    def __init__(self, posterior):
        super().__init__(
            transformation=PulseAdditionTransformation(), posterior=posterior
        )

    def apply(self, state, context, rng):
        if context.t == 0:
            return state, context, {}

        # The completed inner run targets the full previous posterior. Use
        # that endpoint directly instead of recursively evaluating old bridges.
        def previous_log_posterior(particles):
            return log_prior(particles, context) + log_likelihood(
                particles, context
            )

        context = context.with_metadata(
            **{LOG_PROPOSAL_OVERRIDE_KEY: previous_log_posterior}
        )
        return super().apply(state, context, rng)


def logging_str_fn(state, context):
    return (
        f"it {context.t}: beta={context.metadata['beta']:.4f}, "
        f"pulses={context.metadata['n_pulses']}, log evidence={state.log_evidence:.2f}"
    )


def predictive_summary(state, x, noise_sigma, rng, n_draws=2000):
    """Weighted predictive mean and pointwise 90% observation intervals.

    Draw parameter samples according to their particle weights and add fresh
    measurement noise. The interval includes parameter and noise uncertainty.
    """
    templates = pulse_signals(state.particles, x).sum(-1)
    mean = templates @ state.weights
    indices = rng.choice(len(state.particles), size=n_draws, p=state.weights)
    draws = templates[:, indices] + rng.normal(
        scale=noise_sigma, size=(len(x), n_draws)
    )
    lower, upper = np.quantile(draws, [0.05, 0.95], axis=1)
    return mean, lower, upper


class PulseParametersPlot:
    """Track weighted parameter means across inner runs using fixed labels.

    History is retained when the inner iteration counter restarts. New pulse
    traces begin at their first recorded iteration; labels may swap during MCMC.
    """

    def __init__(self, filename, truth):
        self.filename = filename
        self.truth = np.asarray(truth)
        self.history = []

    def __call__(self, state, context, results):
        means = np.sum(
            component_view(state.particles) * state.weights[:, None, None],
            axis=0,
        )
        self.history.append(means)
        counts = [row.shape[1] for row in self.history]
        history = np.full((len(self.history), 3, max(counts)), np.nan)
        for iteration, values in enumerate(self.history):
            history[iteration, :, : values.shape[1]] = values
        iterations = np.arange(len(history))
        expansions = np.flatnonzero(np.diff(counts) > 0) + 1
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        colors = plt.get_cmap("tab20")
        for parameter, (ax, label) in enumerate(
            zip(axes, ("Amplitude", "Mean", "Width"))
        ):
            for pulse in range(max(counts)):
                values = history[:, parameter, pulse]
                color = colors(pulse % 20)
                ax.plot(
                    iterations, values, color=color, label=f"Pulse {pulse + 1}"
                )
                first = np.flatnonzero(np.isfinite(values))[0]
                ax.scatter(first, values[first], color=color, s=25, zorder=3)
            for value in self.truth[:, parameter]:
                ax.axhline(
                    value,
                    color="tab:red",
                    linestyle="--",
                    alpha=0.45,
                    linewidth=0.8,
                )
            for iteration in expansions:
                ax.axvline(iteration, color=".7", linestyle=":", linewidth=0.8)
            ax.set_ylabel(f"Posterior mean: {label.lower()}")
        axes[0].legend(ncols=2, loc="best")
        axes[0].set_title(
            "Labelled pulse parameters (labels may swap)\nDots: introduction; dashed red: true values"
        )
        axes[-1].set_xlabel("SMC iteration")
        fig.tight_layout()
        fig.savefig(self.filename, dpi=150)
        plt.close(fig)


def main(
    *,
    true_pulses=4,
    max_pulses=6,
    n_particles=200,
    mutation_steps=20,
    n_observations=1001,
    plot=True,
    outdir="outdir",
):
    if true_pulses < 1 or max_pulses < 1:
        raise ValueError("Pulse counts must be at least 1.")
    configure_logging(level="INFO")
    rng = np.random.default_rng(1751)
    bounds = np.array([[1e-3, 1.5], [-4.0, 4.0], [0.1, 1.0]])
    truth = rng.uniform(bounds[:, 0], bounds[:, 1], (true_pulses, 3))
    xdata = np.linspace(-5, 5, n_observations)
    signals = gauss(xdata[:, None], truth[:, 0], truth[:, 1], truth[:, 2])
    sigma = 0.1
    ydata = signals.sum(-1) + sigma * rng.normal(size=len(xdata))
    shared = dict(
        data_x=xdata,
        data_y=ydata,
        sigma=sigma,
        amplitude_bounds=bounds[0],
        mean_bounds=bounds[1],
        scale_bounds=bounds[2],
    )
    posterior = TemperedPosterior(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        log_density_ratio=log_density_ratio,
    )
    outdir = pathlib.Path(outdir) / f"{true_pulses}_pulses_prior"
    if plot:
        outdir.mkdir(parents=True, exist_ok=True)
    parameter_plot = (
        PulseParametersPlot(outdir / "pulse_parameters.png", truth)
        if plot
        else None
    )
    inner = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(
                posterior, target_ess_ratio=0.6, max_beta=1.0, min_delta=1e-8
            ),
            SystematicResampling(always_resample=True),
            DifferentialEvolutionMutation(posterior, n_steps=mutation_steps),
            LoggingStep(str_fn=logging_str_fn, frequency=5),
        ],
        callbacks=[parameter_plot] if parameter_plot is not None else [],
    )
    runner = SequentialMonteCarlo(
        steps=[
            AddPulse(posterior),
            MetaSMCStep(
                inner,
                posterior=posterior,
                record_outputs=True,
                output_key="inner_results",
            ),
            StopIf(
                lambda state, context: (
                    context.metadata["n_pulses"] >= max_pulses
                )
            ),
        ]
    )
    initial = ParticleState(
        particles=rng.uniform(bounds[:, 0], bounds[:, 1], (n_particles, 3)),
        metadata={"n_pulses": 1, "beta": 0.0},
    )
    result = runner.run(
        initial,
        rng=rng,
        shared_context=shared,
        metadata={"n_pulses": 1, "beta": 0.0},
    )
    meta = result.get_meta_results("inner_results")
    if meta is None:
        raise RuntimeError("Inner SMC results were not recorded.")
    evidence = {
        int(r.final_state.metadata["n_pulses"]): float(
            r.final_state.log_evidence
        )
        for r in meta.results
    }
    for count, logz in evidence.items():
        print(f"{count} pulses: log evidence = {logz:.3f}")
    preferred = max(evidence, key=evidence.get)
    print(f"Preferred pulse count (equal model priors): {preferred}")
    if not plot:
        return meta

    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, label="Data", alpha=0.4)
    ax.plot(xdata, signals.sum(-1), label="True signal")
    ax.legend()
    fig.savefig(outdir / "data.png")
    plt.close(fig)

    states = meta.all_states
    iterations = np.arange(len(states))
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(iterations, [s.metadata["beta"] for s in states])
    axes[0].set_ylabel("Beta")
    axes[1].step(
        iterations, [s.metadata["n_pulses"] for s in states], where="post"
    )
    axes[1].set_ylabel("Pulses")
    axes[1].set_xlabel("SMC iteration")
    fig.tight_layout()
    fig.savefig(outdir / "temperature_schedule.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(list(evidence), list(evidence.values()), marker="o")
    ax.set_xlabel("Number of pulses")
    ax.set_ylabel("Log evidence")
    fig.savefig(outdir / "evidence.png")
    plt.close(fig)

    state = next(
        r.final_state
        for r in meta.results
        if r.final_state.metadata["n_pulses"] == preferred
    )
    mean, lower, upper = predictive_summary(state, xdata, sigma, rng)
    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, alpha=0.3, label="Data")
    ax.fill_between(
        xdata, lower, upper, alpha=0.25, label="90% predictive interval"
    )
    ax.plot(xdata, mean, label="Posterior predictive mean")
    ax.plot(xdata, signals.sum(-1), "--", label="True signal")
    ax.set_title(f"Posterior predictive: {preferred} pulses")
    ax.legend()
    fig.savefig(outdir / "fit.png")
    plt.close(fig)

    components = component_view(state.particles)
    fig = None
    for index in reversed(range(preferred)):
        fig = corner.corner(
            components[:, :, index],
            weights=state.weights,
            fig=fig,
            color=plt.get_cmap("Blues_r")(index / preferred),
            bins=30,
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True,
            smooth=1.6,
            range=bounds,
            labels=["Amplitude", "Mean", "Width"],
        )
    # Overlay every true pulse, without assuming a match to inferred labels
    # or requiring the selected model to have the same pulse count.
    axes = np.asarray(fig.axes).reshape(3, 3)
    for row in range(3):
        for value in truth[:, row]:
            axes[row, row].axvline(
                value, color="tab:red", linestyle="--", linewidth=1.0
            )
        for column in range(row):
            axes[row, column].scatter(
                truth[:, column],
                truth[:, row],
                color="tab:red",
                marker="x",
                s=45,
                zorder=5,
            )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="tab:red",
                marker="x",
                linestyle="--",
                label="True pulse parameters",
            )
        ],
        loc="upper right",
        frameon=False,
    )
    fig.savefig(outdir / "posterior_corner.png")
    plt.close(fig)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-pulses", type=int, default=4)
    parser.add_argument("--max-pulses", type=int, default=6)
    args = parser.parse_args()
    if args.true_pulses < 1 or args.max_pulses < 1:
        parser.error("Pulse counts must be at least 1")
    main(
        true_pulses=args.true_pulses,
        max_pulses=args.max_pulses,
    )
