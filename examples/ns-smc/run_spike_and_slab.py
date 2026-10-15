"""The 10D spike-and-slab phase-transition example of Salomone et al., §6.1.

Compare standard, persistent, and nested SMC using MiniPCN, alongside exact
constrained-prior NS draws as a diagnostic control. The known likelihood
maximum protects NS stopping against an as-yet undiscovered spike.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from scipy.special import gammainc, gammaln, logsumexp
from sequince import (
    NestedSamplingSMCResult,
    ParticleState,
    make_nested_sampling_smc,
    make_persistent_bayesian_smc,
    make_standard_bayesian_smc,
)
from sequince.steps.base import SMCStep

METHOD_LABELS = {
    "standard_smc": "SMC (MiniPCN)",
    "persistent_smc": "Persistent (MiniPCN)",
    "ns_smc": "NS-SMC (MiniPCN)",
    "ns_exact": "NS-SMC (exact)",
}


@dataclass
class SpikeAndSlab:
    dimension: int = 10
    evaluations: int = 0

    @property
    def log_volume(self) -> float:
        return self.dimension / 2 * math.log(math.pi) - float(
            gammaln(self.dimension / 2 + 1)
        )

    @property
    def split_radius(self) -> float:
        # Radius at which the two weighted likelihood components are equal.
        return math.sqrt(
            2
            * (math.log(9) + self.dimension * math.log(10))
            / (1 / 0.01**2 - 1 / 0.1**2)
        )

    def radial_log_likelihood(self, radius):
        r2 = np.asarray(radius) ** 2
        broad = (
            math.log(0.1)
            - self.dimension / 2 * math.log(2 * math.pi * 0.1**2)
            - r2 / (2 * 0.1**2)
        )
        narrow = (
            math.log(0.9)
            - self.dimension / 2 * math.log(2 * math.pi * 0.01**2)
            - r2 / (2 * 0.01**2)
        )
        return np.logaddexp(broad, narrow)

    def log_prior(self, x):
        return np.where(np.sum(x**2, axis=-1) <= 1, -self.log_volume, -np.inf)

    def log_likelihood(self, x):
        self.evaluations += 1 if x.ndim == 1 else len(x)
        return self.radial_log_likelihood(np.linalg.norm(x, axis=-1))

    def counted_radial_likelihood(self, radius):
        self.evaluations += 1
        return float(self.radial_log_likelihood(radius))

    def truth(self):
        masses = np.array([0.1, 0.9]) * gammainc(
            self.dimension / 2, 1 / (2 * np.array([0.1, 0.01]) ** 2)
        )
        central = np.array([0.1, 0.9]) * gammainc(
            self.dimension / 2,
            self.split_radius**2 / (2 * np.array([0.1, 0.01]) ** 2),
        )
        return float(np.log(masses.sum()) - self.log_volume), float(
            central.sum() / masses.sum()
        )

    def draw_ball(self, rng, n, radius=1.0):
        direction = np.asarray(rng.normal(size=(n, self.dimension)))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        radii = radius * np.asarray(rng.uniform(size=n)) ** (
            1 / self.dimension
        )
        return direction * radii[:, None]


class ExactConstrainedPrior(SMCStep):
    """Independent uniform draws in the ball defined by the likelihood level."""

    def __init__(self, model):
        self.model = model

    def apply(self, state, context, rng):
        threshold = context.metadata["nested_sampling"]["threshold"]
        radius = brentq(
            lambda r: self.model.counted_radial_likelihood(r) - threshold,
            0.0,
            1.0,
            xtol=1e-14,
        )
        particles = self.model.draw_ball(rng, len(state.particles), radius)
        return (
            state.with_updates(particles=particles),
            context,
            {"constraint_radius": radius},
        )


def reference_path(model):
    # Integrate in log radius: the uniform-ball prior has density d * r**d
    # with respect to d(log r). Include the split exactly in the grid.
    log_r = np.unique(
        np.r_[np.linspace(-14, 0, 16385), np.log(model.split_radius)]
    )
    radius = np.exp(log_r)
    ll = model.radial_log_likelihood(radius)
    betas = np.linspace(0, 1, 201)
    central = []
    densities = []
    evidence = []
    for beta in betas:
        log_density = (
            math.log(model.dimension) + model.dimension * log_r + beta * ll
        )
        offset = np.max(log_density)
        unnormalized = np.exp(log_density - offset)
        norm = trapezoid(unnormalized, x=log_r)
        densities.append(unnormalized / norm)
        inside = radius <= model.split_radius
        central.append(trapezoid(unnormalized[inside], x=log_r[inside]) / norm)
        evidence.append(offset + np.log(norm))
    # Independent analytic normalization at beta=1 checks the reference grid.
    assert abs(evidence[-1] - model.truth()[0]) < 1e-6
    return (
        log_r,
        betas,
        np.array(central),
        np.array(densities),
        np.array(evidence),
    )


def experiment(args):
    model = SpikeAndSlab(args.dimension)
    log_z, true_mass = model.truth()
    log_r, betas, central, densities, reference_z = reference_path(model)
    runs = {name: [] for name in METHOD_LABELS}
    paths = {}
    posterior_histograms = {
        name: {"log_radius": [], "x0": []} for name in runs
    }
    posterior_bins = {
        "log_radius": np.linspace(-7, 0, 101),
        "x0": np.linspace(-1, 1, 401),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        initial = model.draw_ball(np.random.default_rng(seed), args.particles)
        for name, records in runs.items():
            model.evaluations = 0
            options = dict(logging_frequency=0)
            if name == "ns_exact":
                options.update(
                    particle_transform=None,
                    mutation=ExactConstrainedPrior(model),
                )
            elif name == "ns_smc":
                options.update(
                    mutation="minipcn",
                    mutation_n_steps=args.mutation_steps,
                )
            if name.startswith("ns"):
                sampler, _, metadata = make_nested_sampling_smc(
                    model.log_prior,
                    model.log_likelihood,
                    threshold_quantile=args.quantile,
                    evidence_tolerance=1e-3,
                    log_likelihood_upper_bound=float(
                        model.radial_log_likelihood(0.0)
                    ),
                    **options,
                )
            elif name.startswith("persistent"):
                sampler, _, metadata = make_persistent_bayesian_smc(
                    model.log_prior,
                    model.log_likelihood,
                    n_particles=args.particles,
                    **options,
                )
            else:
                sampler, _, metadata = make_standard_bayesian_smc(
                    model.log_prior,
                    model.log_likelihood,
                    **options,
                )
            raw = sampler.run(
                ParticleState(initial.copy()),
                metadata=metadata,
                rng=seed + 10000,
                num_steps=args.max_iterations,
            )
            if name.startswith("ns"):
                result = NestedSamplingSMCResult(raw)
                if not result.complete:
                    raise RuntimeError(
                        f"{name} did not finish; increase --max-iterations"
                    )
                estimate = result.log_evidence
                samples, weights = result.posterior_samples()
                coordinate = result.log_prior_volume_history()
            else:
                if raw.context.metadata["beta"] != 1.0:
                    raise RuntimeError(
                        "Tempered SMC did not finish; increase --max-iterations"
                    )
                if (
                    name.startswith("persistent")
                    and not raw.context.stop_requested
                ):
                    raise RuntimeError(
                        "Persistent sampling has not met its terminal ESS; "
                        "increase --max-iterations"
                    )
                estimate = raw.final_state.log_evidence
                samples, weights = raw.posterior_samples()
                coordinate = raw.metadata_history("beta")
            mass = float(
                weights[
                    np.linalg.norm(samples, axis=1) < model.split_radius
                ].sum()
            )
            records.append(
                {
                    "seed": seed,
                    "log_evidence": estimate,
                    "central_mass": mass,
                    "likelihood_evaluations": model.evaluations,
                    "iterations": len(raw),
                    "posterior_particles": len(samples),
                }
            )
            # Preserve the complete weighted posterior: NS shells, persistent
            # archive, or standard SMC final population, respectively.
            np.savez_compressed(
                args.outdir / f"posterior_{name}_seed{seed}.npz",
                particles=samples,
                weights=weights,
            )
            posterior_values = {
                "log_radius": np.log(
                    np.maximum(
                        np.linalg.norm(samples, axis=1), np.finfo(float).tiny
                    )
                ),
                "x0": samples[:, 0],
            }
            for quantity, values in posterior_values.items():
                edges = posterior_bins[quantity]
                # Do not renormalize within the displayed range.
                counts, _ = np.histogram(values, bins=edges, weights=weights)
                posterior_histograms[name][quantity].append(
                    counts / np.diff(edges)
                )
            if seed == 0:
                radii = [
                    np.linalg.norm(state.particles, axis=1)
                    for state in raw.states
                ]
                paths[name] = {
                    "coordinate": np.asarray(coordinate),
                    "max_radius": np.array([r.max() for r in radii]),
                    "central_mass": np.array(
                        [
                            np.sum(state.weights[r < model.split_radius])
                            for state, r in zip(raw.states, radii)
                        ]
                    ),
                }
            print(
                f"seed={seed} {name:11} log Z={estimate:+.4f} (true {log_z:+.4f}), spike mass={mass:.3f} (true {true_mass:.3f}), calls={model.evaluations:,}"
            )
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "settings": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "truth": {
            "log_evidence": log_z,
            "central_mass": true_mass,
            "split_radius": model.split_radius,
        },
        "methods": {
            "standard_smc": {
                "factory": "make_standard_bayesian_smc",
                "configuration": "factory defaults, logging disabled",
            },
            "persistent_smc": {
                "factory": "make_persistent_bayesian_smc",
                "configuration": "factory defaults, logging disabled",
                "n_particles": args.particles,
                "target_ess": 2 * args.particles,
            },
            "ns_smc": {
                "factory": "make_nested_sampling_smc",
                "mutation": "minipcn",
                "mutation_steps": args.mutation_steps,
            },
            "ns_exact": {"mutation": "exact constrained-prior draws"},
        },
        "runs": runs,
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    np.savez_compressed(
        args.outdir / "reference.npz",
        log_radius=log_r,
        beta=betas,
        central_mass=central,
        radial_density=densities,
        log_evidence=reference_z,
    )
    for name, path in paths.items():
        np.savez_compressed(args.outdir / f"path_{name}.npz", **path)
    plot(args.outdir, model, log_r, betas, central, densities, paths, runs)
    plot_posteriors(args.outdir, model, posterior_bins, posterior_histograms)


def plot(outdir, model, log_r, betas, central, densities, paths, runs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    transition = int(np.argmin(abs(central - 0.5)))
    for index in sorted(
        set(
            [
                max(0, transition - 10),
                transition,
                min(200, transition + 10),
                200,
            ]
        )
    ):
        axes[0, 0].plot(
            log_r, densities[index], label=f"beta={betas[index]:.3f}"
        )
    axes[0, 0].axvline(
        math.log(model.split_radius), color="grey", linestyle=":"
    )
    axes[0, 0].set(
        xlim=(-6, 0),
        xlabel="log radius",
        ylabel="Density w.r.t. log radius",
        title="Tempered path: two radial phases",
    )
    axes[0, 0].legend()
    axes[0, 1].plot(
        betas, central, color="black", label="Reference quadrature"
    )
    for name in ("standard_smc", "persistent_smc"):
        path = paths[name]
        axes[0, 1].plot(
            path["coordinate"],
            path["central_mass"],
            ".-",
            label=f"{METHOD_LABELS[name]}, seed 0",
        )
    axes[0, 1].set(
        xlabel="beta",
        ylabel="Probability inside spike region",
        title="Does annealing discover the spike?",
    )
    axes[0, 1].legend()
    for name in ("ns_smc", "ns_exact"):
        path = paths[name]
        axes[1, 0].plot(
            -path["coordinate"],
            np.log(path["max_radius"]),
            label=METHOD_LABELS[name],
        )
    axes[1, 0].axhline(
        math.log(model.split_radius),
        color="grey",
        linestyle=":",
        label="Spike boundary",
    )
    axes[1, 0].set(
        xlabel="-log remaining prior volume",
        ylabel="log maximum live radius",
        title="NS path: continuing volume compression",
    )
    axes[1, 0].legend()
    for index, (name, records) in enumerate(runs.items()):
        axes[1, 1].scatter(
            np.full(len(records), index),
            [r["log_evidence"] for r in records],
            label=METHOD_LABELS[name],
        )
    axes[1, 1].axhline(
        model.truth()[0], color="black", linestyle="--", label="Exact log Z"
    )
    axes[1, 1].set(
        xticks=range(len(runs)),
        xticklabels=[
            METHOD_LABELS[name].replace(" (", "\n(") for name in runs
        ],
        ylabel="log evidence",
        title="Independent runs; budgets are not matched",
    )
    axes[1, 1].legend(handles=[axes[1, 1].lines[-1]])
    fig.suptitle(
        f"{model.dimension}D spike-and-slab: nested versus tempered paths"
    )
    fig.tight_layout()
    fig.savefig(outdir / "phase_transition.png", dpi=170)
    plt.close(fig)


def plot_posteriors(outdir, model, bins, histograms):
    """Compare each run's weighted posterior with the truncated mixture truth."""
    log_z, _ = model.truth()
    normalizer = math.exp(log_z + model.log_volume)
    grids = {
        "log_radius": np.linspace(-30, 0, 12001),
        "x0": np.linspace(-1, 1, 10001),
    }
    log_r = grids["log_radius"]
    reference = {
        "log_radius": np.exp(
            math.log(model.dimension)
            + model.dimension * log_r
            + model.radial_log_likelihood(np.exp(log_r))
            - log_z
        )
    }
    x = grids["x0"]
    marginal = np.zeros_like(x)
    for weight, scale in ((0.1, 0.1), (0.9, 0.01)):
        # Integrate the other d-1 coordinates within the unit prior ball.
        remaining_mass = (
            gammainc(
                (model.dimension - 1) / 2,
                np.maximum(0, 1 - x**2) / (2 * scale**2),
            )
            if model.dimension > 1
            else np.ones_like(x)
        )
        marginal += (
            weight
            * np.exp(-0.5 * (x / scale) ** 2)
            / (math.sqrt(2 * math.pi) * scale)
            * remaining_mass
        )
    reference["x0"] = marginal / normalizer
    for quantity in grids:
        assert (
            abs(trapezoid(reference[quantity], x=grids[quantity]) - 1) < 1e-5
        )
        fig, axes = plt.subplots(
            2, 3, figsize=(14, 8), sharex=True, sharey=True
        )
        for index, (ax, name) in enumerate(zip(axes.flat, histograms)):
            values = np.asarray(histograms[name][quantity])
            color = f"C{index}"
            for run in values:
                ax.stairs(
                    run, bins[quantity], color=color, alpha=0.25, linewidth=0.8
                )
            ax.stairs(
                values.mean(axis=0),
                bins[quantity],
                color=color,
                linewidth=1.8,
                label="Mean across seeds",
            )
            ax.plot(
                grids[quantity],
                reference[quantity],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Analytic posterior",
            )
            if quantity == "log_radius":
                ax.axvline(
                    math.log(model.split_radius),
                    color="grey",
                    linestyle=":",
                    label="Spike boundary",
                )
            ax.set_title(METHOD_LABELS[name])
            ax.set_xlabel(
                "log radius"
                if quantity == "log_radius"
                else "First coordinate"
            )
            ax.set_ylabel("Posterior density")
            ax.set_xlim((-6, 0) if quantity == "log_radius" else (-0.35, 0.35))
        axes.flat[0].legend(fontsize=9)
        seed_count = len(next(iter(histograms.values()))[quantity])
        fig.suptitle(
            f"{model.dimension}D posterior comparison — {seed_count} seeds, weighted samples"
        )
        fig.text(
            0.5,
            0.01,
            "Thin curves: individual seeds. Thick curves: equal-weight mean of run densities. Budgets differ; seed spread is not a confidence interval.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.96))
        stem = (
            "posterior_comparison"
            if quantity == "log_radius"
            else "posterior_marginals"
        )
        for extension in ("png", "pdf"):
            fig.savefig(outdir / f"{stem}.{extension}", dpi=170)
        plt.close(fig)
        np.savez_compressed(
            outdir / f"{stem}.npz",
            bins=bins[quantity],
            reference_grid=grids[quantity],
            reference_density=reference[quantity],
            **{
                name: np.asarray(values[quantity])
                for name, values in histograms.items()
            },
        )


def plot_sampling_paths(outdir, model):
    """Figure 1(a)-style comparison from exact radial reference distributions.

    Annealing markers are modes of the density with respect to log X, not
    empirical sampler trajectories. The adaptive schedule uses an exact
    population ESS ratio, assuming equilibrium at each temperature. NS markers
    are geometric volume levels, so no mutation kernel enters this figure.
    """
    log_x = np.linspace(-90.0, 0.0, 24001)
    likelihood = model.radial_log_likelihood(np.exp(log_x / model.dimension))
    quadrature = np.full(len(log_x), log_x[1] - log_x[0])
    quadrature[[0, -1]] *= 0.5
    log_measure = log_x + np.log(quadrature)

    def log_z(beta):
        return float(logsumexp(log_measure + beta * likelihood))

    def ess_ratio(beta, next_beta):
        return math.exp(
            min(
                0.0,
                2 * log_z(next_beta)
                - log_z(beta)
                - log_z(2 * next_beta - beta),
            )
        )

    def mode(beta):
        return log_x[np.argmax(log_x + beta * likelihood)]

    split = model.dimension * math.log(model.split_radius)
    inside = log_x <= split

    def core_probability(beta):
        return math.exp(
            float(logsumexp((log_measure + beta * likelihood)[inside]))
            - log_z(beta)
        )

    if core_probability(0.0) < 0.5 < core_probability(1.0):
        beta_c = brentq(lambda beta: core_probability(beta) - 0.5, 0.0, 1.0)
    else:
        beta_c = 1.0
    coexistence_density = log_x + beta_c * likelihood
    core_mode = log_x[inside][np.argmax(coexistence_density[inside])]
    broad_mode = log_x[~inside][np.argmax(coexistence_density[~inside])]

    fixed_betas = np.linspace(0.0, 1.0, 36)
    adaptive_betas = [0.0]
    while adaptive_betas[-1] < 1.0:
        beta = adaptive_betas[-1]
        next_beta = (
            1.0
            if ess_ratio(beta, 1.0) >= 0.95
            else brentq(
                lambda candidate: ess_ratio(beta, candidate) - 0.95, beta, 1.0
            )
        )
        adaptive_betas.append(next_beta)
        if len(adaptive_betas) > 1000:
            raise RuntimeError("Reference adaptive schedule did not converge")
    levels = [
        np.array([mode(beta) for beta in fixed_betas]),
        np.array([mode(beta) for beta in adaptive_betas]),
        np.linspace(0.0, -55.0, 36),
    ]
    titles = ["Annealing: fixed", "Annealing: adaptive", "Nested sampling"]
    subtitles = [
        "Equal temperature increments",
        "Reference ESS ratio = 0.95",
        "Equal log-volume increments",
    ]
    colors = ["#bf6a20", "#bf6a20", "#287b8e"]
    visible = log_x >= -55
    lower = float(likelihood[visible].min()) - 3
    upper = float(likelihood.max()) + 4
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharex=True, sharey=True)
    for index, (ax, positions, title, subtitle, color) in enumerate(
        zip(axes, levels, titles, subtitles, colors)
    ):
        ax.axvspan(core_mode, broad_mode, color="#f2e6c9", alpha=0.7, zorder=0)
        ax.vlines(
            positions,
            lower,
            model.radial_log_likelihood(np.exp(positions / model.dimension)),
            colors=color,
            linewidth=0.9,
            alpha=0.85,
        )
        ax.plot(
            log_x[visible], likelihood[visible], color="#252525", linewidth=2.2
        )
        ax.scatter(
            positions,
            model.radial_log_likelihood(np.exp(positions / model.dimension)),
            s=12,
            color=color,
            zorder=3,
        )
        ax.set(
            xlim=(-55, 0),
            ylim=(lower, upper),
            xlabel=r"$\log X$  (remaining prior volume)",
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=24)
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            color="#555555",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
        if index < 2:
            ax.text(
                (core_mode + broad_mode) / 2,
                lower + 7,
                "Few typical radii\nat coexistence",
                ha="center",
                fontsize=9,
                color="#725728",
            )
        else:
            ax.text(
                (core_mode + broad_mode) / 2,
                lower + 7,
                "Volume levels\ncontinue through",
                ha="center",
                fontsize=9,
                color=color,
            )
    axes[0].set_ylabel(r"$\log L(X)$", fontsize=12)
    fig.suptitle(
        f"{model.dimension}D spike-and-slab: paths through the same likelihood landscape",
        fontsize=16,
        y=1.01,
    )
    fig.text(
        0.5,
        0.015,
        f"Reference-path illustration inspired by Yallup & Handley, Fig. 1(a).  Coexistence near beta = {beta_c:.3f}.\nAnnealing ticks: modes of p(log X | beta), assuming equilibrium.  NS ticks: prescribed volume levels.  Shading: interval between radial modes.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.98))
    outdir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            outdir / f"sampling_paths.{extension}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(fig)
    np.savez_compressed(
        outdir / "sampling_paths.npz",
        log_prior_volume=log_x,
        log_likelihood=likelihood,
        fixed_betas=fixed_betas,
        adaptive_betas=np.array(adaptive_betas),
        fixed_mode_log_volume=levels[0],
        adaptive_mode_log_volume=levels[1],
        nested_log_volume=levels[2],
        coexistence_beta=beta_c,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--mutation-steps", type=int, default=10)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs") / "spike_and_slab",
    )
    parser.add_argument(
        "--plot-paths-only",
        action="store_true",
        help="Generate the Figure 1(a)-style reference plot without running samplers",
    )
    args = parser.parse_args()
    if (
        min(
            args.dimension,
            args.particles,
            args.seeds,
            args.mutation_steps,
            args.max_iterations,
        )
        <= 0
    ):
        parser.error("counts must be positive")
    plot_sampling_paths(args.outdir, SpikeAndSlab(args.dimension))
    if not args.plot_paths_only:
        experiment(args)


if __name__ == "__main__":
    main()
