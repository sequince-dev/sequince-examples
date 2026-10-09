"""Waste-free SMC for a standard Gaussian likelihood and uniform prior."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sequince import ParticleState, SequentialMonteCarlo, TemperedPosterior
from sequince.log import configure_logging
from sequince.steps import (
    AdaptiveTemperedWeighting,
    LoggingStep,
    StepContext,
    WasteFreeResampleMove,
)
from sequince.steps.mutation import (
    GaussianRandomWalk,
    MetropolisHastingsMutation,
)


def log_likelihood(particles: np.ndarray) -> np.ndarray:
    """Return normalized standard Gaussian log densities."""
    dims = particles.shape[1]
    return -0.5 * np.sum(particles**2, axis=1) - 0.5 * dims * np.log(2 * np.pi)


def log_prior(particles: np.ndarray, context: StepContext) -> np.ndarray:
    """Return normalized uniform log densities inside the prior box."""
    lower, upper = context.shared["bounds"]
    inside = np.all((particles >= lower) & (particles <= upper), axis=1)
    return np.where(
        inside, -particles.shape[1] * np.log(upper - lower), -np.inf
    )


def main() -> None:
    logger = configure_logging()
    dims = 4
    bounds = (-10.0, 10.0)
    n_chains = 100
    chain_length = 100  # Includes the resampled starting state.
    n_particles = n_chains * chain_length
    rng = np.random.default_rng(123)
    initial = ParticleState(rng.uniform(*bounds, size=(n_particles, dims)))
    posterior = TemperedPosterior(log_prior, log_likelihood)

    smc = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(posterior, target_ess_ratio=0.85),
            WasteFreeResampleMove(
                MetropolisHastingsMutation(
                    posterior, GaussianRandomWalk(scale=0.7)
                ),
                chain_length=chain_length,
            ),
            LoggingStep(
                str_fn=lambda state, context: (
                    f"Stage {context.t + 1}: "
                    f"beta={float(context.metadata['beta']):.4f}, "
                    f"log evidence={float(state.log_evidence):.4f}"
                )
            ),
        ]
    )
    results = smc.run(initial, rng=rng, shared_context={"bounds": bounds})
    particles = results.final_state.particles
    moves = [stage[1] for stage in results.outputs]
    # Gaussian truncation at +/-10 has negligible effect on this reference.
    reference_log_evidence = -dims * np.log(bounds[1] - bounds[0])
    logger.info("Final beta: %.3f", float(results.context.metadata["beta"]))
    logger.info(
        "Estimated log evidence: %.4f", results.final_state.log_evidence
    )
    logger.info("Reference log evidence: %.4f", reference_log_evidence)
    logger.info("Posterior means (reference: 0): %s", particles.mean(axis=0))
    logger.info(
        "Posterior variances (reference: 1): %s", particles.var(axis=0)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    fig.suptitle(
        f"Waste-free SMC: {n_chains} chains × {chain_length} retained states"
    )
    stages = np.arange(1, len(moves) + 1)
    axes[0].plot(stages, results.metadata_history("beta"), "o-", color="C0")
    axes[0].set(title="Adaptive tempering", ylabel=r"$\beta$", ylim=(0, 1.05))
    axes[1].plot(
        stages, results.log_evidence_history(), "o-", color="C1", label="SMC"
    )
    axes[1].axhline(
        reference_log_evidence,
        color="black",
        linestyle="--",
        label="Reference",
    )
    axes[1].legend()
    axes[1].set(title="Log evidence", ylabel=r"$\log Z$")
    for ax in axes[:2]:
        ax.set(xlabel="Stage", xlim=(0.5, len(moves) + 0.5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Waste-free output is ordered by chain position, then chain index.
    chains = particles.reshape(chain_length, n_chains, dims)
    positions = np.arange(chain_length)
    for i in range(3):
        axes[2].plot(
            positions,
            chains[:, i, 0],
            ".-",
            color=f"C{i}",
            markersize=4,
            alpha=0.8,
            label="All retained states" if i == 0 else None,
        )
    for position, marker, label in (
        (0, "s", "Resampled starts"),
        (chain_length - 1, "X", "Endpoints"),
    ):
        axes[2].scatter(
            np.full(3, position),
            chains[position, :3, 0],
            marker=marker,
            color="black",
            zorder=3,
            label=label,
        )
    axes[2].set(
        title="Final stage: three retained chains",
        xlabel="Chain position (flat segments = rejections)",
        ylabel=r"$x_0$",
    )
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].legend(fontsize="small")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(alpha=0.2)
    fig.savefig("waste_free_history.png", dpi=180)
    plt.close(fig)
    logger.info("Saved waste_free_history.png")


if __name__ == "__main__":
    main()
