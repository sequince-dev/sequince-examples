"""Waste-free SMC for a correlated Gaussian likelihood and uniform prior."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sequince import ParticleState, SequentialMonteCarlo, TemperedPosterior
from sequince.log import configure_logging
from sequince.steps import (
    AdaptiveTemperedWeighting,
    LoggingStep,
    MultinomialResampling,
    StepContext,
    WasteFreeResampleMove,
)
from sequince.steps.mutation import (
    GaussianRandomWalk,
    MetropolisHastingsMutation,
)

dims = 4
bounds = (-10.0, 10.0)
correlation = 0.8
n_chains = 100
chain_length = 100  # Includes the resampled starting state.
seed = 123
documentation_output: Path | None = None


def log_likelihood(particles: np.ndarray, context: StepContext) -> np.ndarray:
    """Return normalized zero-mean Gaussian log densities."""
    whitened = np.linalg.solve(context.shared["cholesky"], particles.T)
    return context.shared["log_norm"] - 0.5 * np.sum(whitened**2, axis=0)


def log_prior(particles: np.ndarray, context: StepContext) -> np.ndarray:
    """Return normalized uniform log densities inside the prior box."""
    lower, upper = context.shared["bounds"]
    inside = np.all((particles >= lower) & (particles <= upper), axis=1)
    return np.where(
        inside, -particles.shape[1] * np.log(upper - lower), -np.inf
    )


def main() -> None:
    logger = configure_logging()
    indices = np.arange(dims)
    covariance = correlation ** np.abs(indices[:, None] - indices[None, :])
    cholesky = np.linalg.cholesky(covariance)
    log_norm = (
        -0.5 * dims * np.log(2 * np.pi) - np.log(np.diag(cholesky)).sum()
    )
    posterior = TemperedPosterior(log_prior, log_likelihood)
    # Truncation at +/-10 has negligible effect on the evidence reference.
    reference_log_evidence = -dims * np.log(bounds[1] - bounds[0])
    logger.info("Reference log evidence: %.4f", reference_log_evidence)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    comparison_rows: list[str] = []
    posterior_counts: dict[str, int] = {}

    summaries: dict[str, dict] = {}

    for name in ("Standard", "Waste-free"):
        mutation = MetropolisHastingsMutation(
            posterior,
            GaussianRandomWalk(scale=0.7),
            n_steps=chain_length - 1,
        )
        if name == "Waste-free":
            n_particles = n_chains * chain_length
            moves = [
                WasteFreeResampleMove(mutation, chain_length=chain_length)
            ]
        else:
            n_particles = n_chains
            moves = [MultinomialResampling(always_resample=True), mutation]

        logger.info("%s SMC: %d particles", name, n_particles)
        rng = np.random.default_rng(seed)
        initial = ParticleState(rng.uniform(*bounds, size=(n_particles, dims)))
        smc = SequentialMonteCarlo(
            steps=[
                AdaptiveTemperedWeighting(posterior, target_ess_ratio=0.85),
                *moves,
                LoggingStep(
                    str_fn=lambda state, context: (
                        f"Stage {context.t + 1}: "
                        f"beta={float(context.metadata['beta']):.4f}, "
                        f"log evidence={float(state.log_evidence):.4f}"
                    )
                ),
            ]
        )
        results = smc.run(
            initial,
            rng=rng,
            shared_context={
                "bounds": bounds,
                "cholesky": cholesky,
                "log_norm": log_norm,
            },
        )
        particles = results.final_state.particles
        logger.info(
            "%s posterior means (reference: zero): %s",
            name,
            particles.mean(axis=0),
        )
        logger.info(
            "%s posterior variances (reference: one): %s",
            name,
            particles.var(axis=0),
        )
        if documentation_output is not None:
            summaries[name] = {
                "log_evidence": float(results.final_state.log_evidence),
                "weighted_mean": results.weighted_mean().tolist(),
                "weighted_population_covariance": results.weighted_covariance().tolist(),
                "n_particles": len(particles),
                "likelihood_evaluations": int(
                    results.context.evaluation_counts.log_likelihood_evaluations
                ),
            }
        posterior_counts[name] = len(particles)
        comparison_rows.append(
            f"{name:<12} {len(particles):>10,d} {len(results.outputs):>8d} "
            f"{results.context.evaluation_counts.log_likelihood_evaluations:>16,d} "
            f"{float(results.final_state.log_evidence):>12.4f}"
        )
        stages = np.arange(1, len(results.outputs) + 1)
        ax.plot(
            stages,
            results.metadata_history("beta"),
            marker=".",
            label=f"{name}: {len(particles):,} posterior samples",
        )

    logger.info(
        "Final comparison:\n%-12s %10s %8s %16s %12s\n%s",
        "Method",
        "Samples",
        "Stages",
        "Likelihood evals",
        "Log evidence",
        "\n".join(comparison_rows),
    )
    logger.info(
        "Waste-free retains %.0f times as many posterior samples; "
        "these samples are correlated, including repeats after rejection.",
        posterior_counts["Waste-free"] / posterior_counts["Standard"],
    )

    ax.set(
        title="Adaptive tempering: standard and waste-free SMC",
        xlabel="Stage",
        ylabel=r"Inverse temperature, $\beta$",
        ylim=(0, 1.05),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend()
    output_directory = documentation_output or Path(".")
    fig.savefig(output_directory / "waste_free_history.png", dpi=180)
    if documentation_output is not None:
        (documentation_output / "summary.json").write_text(
            json.dumps(summaries, allow_nan=False) + "\n"
        )
    plt.close(fig)
    logger.info("Saved waste_free_history.png")


if __name__ == "__main__":
    main()
