"""Compare two particle filters for a 1D linear-Gaussian state-space model.

The bootstrap filter proposes from the transition model. The locally optimal
filter instead proposes conditional on the current observation. Both are built
from Sequince's general SMC steps and checked against an exact Kalman filter.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sequince import ParticleState, SequentialMonteCarlo, SMCResult
from sequince.steps import (
    ImportanceTransition,
    IncrementalWeighting,
    SystematicResampling,
)
from sequince.steps.base import SMCStep, StepContext
from sequince.steps.resampling import ResamplingOutput

if TYPE_CHECKING:
    from orng import RandomGenerator


documentation_output: Path | None = None


@dataclass(frozen=True)
class LinearGaussianModel:
    """Parameters of the scalar state-space model."""

    a: float = 0.9
    sigma_0: float = 1.0
    sigma_x: float = 0.4
    sigma_y: float = 0.7


@dataclass(frozen=True)
class BootstrapComparison:
    """Simulated data and exact/particle filtering summaries."""

    latent: np.ndarray
    observations: np.ndarray
    kalman_means: np.ndarray
    kalman_variances: np.ndarray
    particle_means: np.ndarray
    particle_variances: np.ndarray
    ess_before_resampling: np.ndarray
    kalman_log_evidence: float
    particle_log_evidence: float
    result: SMCResult


class PropagationStep(SMCStep):
    """Sample each new state from ``p(x_t | x_{t-1})``."""

    def __init__(self, a: float, sigma_x: float) -> None:
        self.a = a
        self.sigma_x = sigma_x

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> tuple[ParticleState, StepContext, None]:
        particles = self.a * state.particles + rng.normal(
            scale=self.sigma_x,
            size=state.particles.shape,
        )
        return state.replace_particles(particles), context, None


class RecordFilteringState(SMCStep):
    """Capture the current weighted filtering state without changing it."""

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> tuple[ParticleState, StepContext, ParticleState]:
        del rng
        # SMCStep does not forbid later steps from mutating state in place, so
        # retain a backend-safe, detached snapshot rather than a reference.
        return state, context, state.copy()


def simulate_model(
    model: LinearGaussianModel,
    n_observations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``x_0, ..., x_T`` and observations ``y_1, ..., y_T``."""
    latent = np.empty(n_observations + 1)
    observations = np.empty(n_observations)
    latent[0] = rng.normal(scale=model.sigma_0)
    for t in range(n_observations):
        latent[t + 1] = model.a * latent[t] + rng.normal(scale=model.sigma_x)
        observations[t] = latent[t + 1] + rng.normal(scale=model.sigma_y)
    return latent, observations


def kalman_filter_with_log_evidence(
    observations: np.ndarray,
    model: LinearGaussianModel,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return exact filtering moments and ``log p(y_1:T)``."""
    means = np.empty(len(observations))
    variances = np.empty(len(observations))
    mean = 0.0
    variance = model.sigma_0**2
    log_evidence = 0.0

    for t, observation in enumerate(observations):
        predicted_mean = model.a * mean
        predicted_variance = model.a**2 * variance + model.sigma_x**2
        innovation_variance = predicted_variance + model.sigma_y**2
        innovation = observation - predicted_mean
        log_evidence += -0.5 * (
            innovation**2 / innovation_variance
            + np.log(2.0 * np.pi * innovation_variance)
        )
        gain = predicted_variance / innovation_variance
        mean = predicted_mean + gain * (observation - predicted_mean)
        variance = (1.0 - gain) * predicted_variance
        means[t] = mean
        variances[t] = variance

    return means, variances, float(log_evidence)


def observation_log_likelihood(
    particles: np.ndarray,
    context: StepContext,
    sigma_y: float,
) -> np.ndarray:
    """Evaluate ``log p(y_t | x_t)`` for the current observation."""
    residual = context.observation - particles[:, 0]
    return -0.5 * (residual**2 / sigma_y**2 + np.log(2.0 * np.pi * sigma_y**2))


def filtering_states(result: SMCResult) -> list[ParticleState]:
    """Extract the recorded pre-resampling states from step outputs."""
    return [
        step_output_of_type(iteration_outputs, ParticleState)
        for iteration_outputs in result.outputs
    ]


def filtering_moments(
    states: list[ParticleState],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scalar weighted moments for a sequence of particle states."""
    means = np.array(
        [
            float(np.sum(state.weights * state.particles[:, 0]))
            for state in states
        ]
    )
    variances = np.array(
        [
            float(np.sum(state.weights * (state.particles[:, 0] - mean) ** 2))
            for state, mean in zip(states, means, strict=True)
        ]
    )
    return means, variances


def step_output_of_type(iteration_outputs: tuple, output_type: type):
    """Find one typed step output in the runner's positional output tuple."""
    matches = [
        output
        for output in iteration_outputs
        if isinstance(output, output_type)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {output_type.__name__} output, found {len(matches)}."
        )
    return matches[0]


def resampling_outputs(result: SMCResult) -> list[ResamplingOutput]:
    """Extract typed resampling outputs without relying on pipeline indices."""
    return [
        step_output_of_type(iteration_outputs, ResamplingOutput)
        for iteration_outputs in result.outputs
    ]


def resampling_ess(result: SMCResult) -> np.ndarray:
    """Return ESS measured immediately before each resampling decision."""
    return np.array(
        [float(output.ess) for output in resampling_outputs(result)]
    )


def run_particle_filter(
    observations: np.ndarray,
    model: LinearGaussianModel,
    *,
    n_particles: int = 50_000,
    seed: int = 2026,
    always_resample: bool = True,
    ess_threshold: float = 0.5,
) -> tuple[SMCResult, np.ndarray, np.ndarray]:
    """Run the Sequince bootstrap filter and return its moments."""
    rng = np.random.default_rng(seed)
    initial_particles = rng.normal(
        scale=model.sigma_0,
        size=(n_particles, 1),
    )
    initial_state = ParticleState(initial_particles)

    smc = SequentialMonteCarlo(
        steps=[
            PropagationStep(model.a, model.sigma_x),
            IncrementalWeighting(
                lambda particles, context: observation_log_likelihood(
                    particles,
                    context,
                    model.sigma_y,
                )
            ),
            RecordFilteringState(),
            SystematicResampling(
                always_resample=always_resample,
                ess_threshold=ess_threshold,
            ),
        ]
    )
    result = smc.run(
        initial_state,
        observations=observations,
        rng=rng,
    )
    means, variances = filtering_moments(filtering_states(result))
    return result, means, variances


def run_bootstrap_comparison(
    *,
    model: LinearGaussianModel = LinearGaussianModel(),
    n_observations: int = 20,
    n_particles: int = 5_000,
    seed: int = 2026,
    always_resample: bool = True,
    ess_threshold: float = 0.5,
) -> BootstrapComparison:
    """Simulate data and compare bootstrap and exact filtering recursions."""
    latent, observations = simulate_model(
        model,
        n_observations,
        np.random.default_rng(seed),
    )
    kalman_means, kalman_variances, kalman_log_evidence = (
        kalman_filter_with_log_evidence(observations, model)
    )
    result, particle_means, particle_variances = run_particle_filter(
        observations,
        model,
        n_particles=n_particles,
        seed=seed + 1,
        always_resample=always_resample,
        ess_threshold=ess_threshold,
    )
    return BootstrapComparison(
        latent=latent,
        observations=observations,
        kalman_means=kalman_means,
        kalman_variances=kalman_variances,
        particle_means=particle_means,
        particle_variances=particle_variances,
        ess_before_resampling=resampling_ess(result),
        kalman_log_evidence=kalman_log_evidence,
        particle_log_evidence=float(result.final_state.log_evidence),
        result=result,
    )


@dataclass(frozen=True)
class ProposalComparison:
    """Exact, optimal-proposal, and bootstrap-filter summaries."""

    latent: np.ndarray
    observations: np.ndarray
    kalman_means: np.ndarray
    kalman_variances: np.ndarray
    proposal_means: np.ndarray
    proposal_variances: np.ndarray
    proposal_ess: np.ndarray
    bootstrap_means: np.ndarray
    bootstrap_variances: np.ndarray
    bootstrap_ess: np.ndarray
    kalman_log_evidence: float
    proposal_log_evidence: float
    bootstrap_log_evidence: float
    proposal_result: SMCResult
    bootstrap_result: SMCResult


def normal_logpdf(
    value: np.ndarray,
    mean: np.ndarray,
    variance: float,
) -> np.ndarray:
    """Evaluate a scalar Gaussian log density elementwise."""
    return -0.5 * (
        (value - mean) ** 2 / variance + np.log(2.0 * np.pi * variance)
    )


def optimal_proposal_parameters(
    parents: np.ndarray,
    observation: float,
    model: LinearGaussianModel,
) -> tuple[np.ndarray, float]:
    """Return the mean and variance of ``p(x_t | x_(t-1), y_t)``."""
    transition_variance = model.sigma_x**2
    observation_variance = model.sigma_y**2
    proposal_variance = 1.0 / (
        1.0 / transition_variance + 1.0 / observation_variance
    )
    proposal_mean = proposal_variance * (
        model.a * parents / transition_variance
        + observation / observation_variance
    )
    return proposal_mean, proposal_variance


@dataclass(frozen=True)
class OptimalProposalKernel:
    """Locally optimal transition and its importance correction."""

    model: LinearGaussianModel

    def sample(self, parents, context, rng):
        proposal_mean, proposal_variance = optimal_proposal_parameters(
            parents,
            context.observation,
            self.model,
        )
        return rng.normal(
            loc=proposal_mean,
            scale=np.sqrt(proposal_variance),
            size=parents.shape,
        )

    def log_increment(self, parents, particles, context):
        """Return transition/proposal correction plus observation density."""
        proposal_mean, proposal_variance = optimal_proposal_parameters(
            parents,
            context.observation,
            self.model,
        )
        log_transition = normal_logpdf(
            particles,
            self.model.a * parents,
            self.model.sigma_x**2,
        )[:, 0]
        log_proposal = normal_logpdf(
            particles,
            proposal_mean,
            proposal_variance,
        )[:, 0]
        return (
            log_transition
            - log_proposal
            + observation_log_likelihood(
                particles,
                context,
                self.model.sigma_y,
            )
        )


def run_optimal_proposal_filter(
    observations: np.ndarray,
    model: LinearGaussianModel,
    *,
    n_particles: int = 5_000,
    seed: int = 2027,
    always_resample: bool = True,
    ess_threshold: float = 0.5,
) -> tuple[SMCResult, np.ndarray, np.ndarray]:
    """Run the optimal-proposal filter and return filtering moments."""
    rng = np.random.default_rng(seed)
    initial_state = ParticleState(
        rng.normal(scale=model.sigma_0, size=(n_particles, 1))
    )
    smc = SequentialMonteCarlo(
        steps=[
            ImportanceTransition(OptimalProposalKernel(model)),
            RecordFilteringState(),
            SystematicResampling(
                always_resample=always_resample,
                ess_threshold=ess_threshold,
            ),
        ]
    )
    result = smc.run(initial_state, observations=observations, rng=rng)
    means, variances = filtering_moments(filtering_states(result))
    return result, means, variances


def run_comparison(
    *,
    model: LinearGaussianModel = LinearGaussianModel(),
    n_observations: int = 20,
    n_particles: int = 5_000,
    seed: int = 2026,
    always_resample: bool = True,
    ess_threshold: float = 0.5,
) -> ProposalComparison:
    """Compare the two particle filters with the exact Kalman filter."""
    bootstrap = run_bootstrap_comparison(
        model=model,
        n_observations=n_observations,
        n_particles=n_particles,
        seed=seed,
        always_resample=always_resample,
        ess_threshold=ess_threshold,
    )
    proposal_result, proposal_means, proposal_variances = (
        run_optimal_proposal_filter(
            bootstrap.observations,
            model,
            n_particles=n_particles,
            seed=seed + 1,
            always_resample=always_resample,
            ess_threshold=ess_threshold,
        )
    )
    return ProposalComparison(
        latent=bootstrap.latent,
        observations=bootstrap.observations,
        kalman_means=bootstrap.kalman_means,
        kalman_variances=bootstrap.kalman_variances,
        proposal_means=proposal_means,
        proposal_variances=proposal_variances,
        proposal_ess=resampling_ess(proposal_result),
        bootstrap_means=bootstrap.particle_means,
        bootstrap_variances=bootstrap.particle_variances,
        bootstrap_ess=bootstrap.ess_before_resampling,
        kalman_log_evidence=bootstrap.kalman_log_evidence,
        proposal_log_evidence=float(proposal_result.final_state.log_evidence),
        bootstrap_log_evidence=bootstrap.particle_log_evidence,
        proposal_result=proposal_result,
        bootstrap_result=bootstrap.result,
    )


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Return root-mean-square error."""
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def plot_comparison(comparison: ProposalComparison, output: Path) -> None:
    """Plot both filters and their pre-resampling ESS."""
    import matplotlib.pyplot as plt

    times = np.arange(1, len(comparison.observations) + 1)
    kalman_sd = np.sqrt(comparison.kalman_variances)
    proposal_sd = np.sqrt(comparison.proposal_variances)
    bootstrap_sd = np.sqrt(comparison.bootstrap_variances)

    figure, (filter_axis, ess_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    filter_axis.plot(
        np.arange(len(comparison.latent)),
        comparison.latent,
        color="black",
        label="latent state",
    )
    filter_axis.scatter(
        times,
        comparison.observations,
        color="tab:gray",
        marker="x",
        label="observations",
    )
    filter_axis.plot(
        times,
        comparison.kalman_means,
        color="tab:blue",
        label="Kalman mean",
    )
    filter_axis.fill_between(
        times,
        comparison.kalman_means - 2.0 * kalman_sd,
        comparison.kalman_means + 2.0 * kalman_sd,
        color="tab:blue",
        alpha=0.12,
        label="Kalman $\u00b12$ sd",
    )
    filter_axis.plot(
        times,
        comparison.proposal_means,
        color="tab:orange",
        linestyle="--",
        label="optimal-proposal mean",
    )
    filter_axis.fill_between(
        times,
        comparison.proposal_means - 2.0 * proposal_sd,
        comparison.proposal_means + 2.0 * proposal_sd,
        color="tab:orange",
        alpha=0.12,
        label="optimal proposal $\u00b12$ sd",
    )
    filter_axis.plot(
        times,
        comparison.bootstrap_means,
        color="tab:green",
        linestyle=":",
        label="bootstrap mean",
    )
    filter_axis.fill_between(
        times,
        comparison.bootstrap_means - 2.0 * bootstrap_sd,
        comparison.bootstrap_means + 2.0 * bootstrap_sd,
        color="tab:green",
        alpha=0.08,
        label="bootstrap $\u00b12$ sd",
    )
    filter_axis.set_ylabel("state")
    filter_axis.legend(ncol=2, fontsize="small")

    ess_axis.plot(
        times,
        comparison.proposal_ess,
        marker=".",
        label="optimal proposal",
    )
    ess_axis.plot(
        times,
        comparison.bootstrap_ess,
        marker=".",
        label="bootstrap",
    )
    ess_axis.set_xlabel("time")
    ess_axis.set_ylabel("ESS")
    ess_axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used for execution and documentation metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=1_000)
    parser.add_argument("--observations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--ess-threshold",
        type=float,
        default=None,
        help="Use ESS-triggered resampling at this ESS/n threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("particle_filter.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = run_comparison(
        n_observations=args.observations,
        n_particles=args.particles,
        seed=args.seed,
        always_resample=args.ess_threshold is None,
        ess_threshold=(
            0.5 if args.ess_threshold is None else args.ess_threshold
        ),
    )

    print(
        "mean RMSE (optimal / bootstrap): "
        f"{rmse(comparison.proposal_means, comparison.kalman_means):.5f} / "
        f"{rmse(comparison.bootstrap_means, comparison.kalman_means):.5f}"
    )
    print(
        "variance RMSE (optimal / bootstrap): "
        f"{rmse(comparison.proposal_variances, comparison.kalman_variances):.5f} / "
        f"{rmse(comparison.bootstrap_variances, comparison.kalman_variances):.5f}"
    )
    print(
        "mean pre-resampling ESS (optimal / bootstrap): "
        f"{np.mean(comparison.proposal_ess):.1f} / "
        f"{np.mean(comparison.bootstrap_ess):.1f}"
    )
    print(
        "log evidence (optimal / bootstrap / Kalman): "
        f"{comparison.proposal_log_evidence:.4f} / "
        f"{comparison.bootstrap_log_evidence:.4f} / "
        f"{comparison.kalman_log_evidence:.4f}"
    )
    if documentation_output is not None:
        args.output = documentation_output / "particle_filter.png"
        summary = {
            "mean_rmse": {
                "optimal": rmse(
                    comparison.proposal_means, comparison.kalman_means
                ),
                "bootstrap": rmse(
                    comparison.bootstrap_means, comparison.kalman_means
                ),
            },
            "variance_rmse": {
                "optimal": rmse(
                    comparison.proposal_variances, comparison.kalman_variances
                ),
                "bootstrap": rmse(
                    comparison.bootstrap_variances, comparison.kalman_variances
                ),
            },
            "mean_ess_before_resampling": {
                "optimal": float(np.mean(comparison.proposal_ess)),
                "bootstrap": float(np.mean(comparison.bootstrap_ess)),
            },
            "log_evidence": {
                "optimal": comparison.proposal_log_evidence,
                "bootstrap": comparison.bootstrap_log_evidence,
                "kalman": comparison.kalman_log_evidence,
            },
        }
        (documentation_output / "summary.json").write_text(
            json.dumps(summary, allow_nan=False) + "\n"
        )
    plot_comparison(comparison, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
