"""Plot NS-SMC shell diagnostics and compare posterior samples to exact answers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from problems import PROBLEMS, CountedLikelihood, Problem
from sequince import (
    NestedSamplingSMCResult,
    ParticleState,
    make_nested_sampling_smc,
    make_standard_bayesian_smc,
)


def run(problem: Problem, args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir / problem.name
    outdir.mkdir(parents=True, exist_ok=True)
    truth_z, truth_mean, truth_second = problem.truth()
    summary: dict[str, Any] = {
        "problem": problem.name,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "truth": {
            "log_evidence": truth_z,
            "mean": truth_mean.tolist(),
            "second_moment": truth_second.tolist(),
        },
        "methods": {},
    }
    initial = np.random.default_rng(args.seed).normal(
        0, problem.prior_std, size=(args.particles, 2)
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {"ns_smc": "tab:blue", "tempered": "tab:orange"}
    for method in ("ns_smc", "tempered"):
        counter = CountedLikelihood(problem)
        settings: dict[str, Any] = dict(
            mutation=args.mutation,
            mutation_n_steps=args.mutation_steps,
            logging_frequency=0,
        )
        if method == "ns_smc":
            factory = make_nested_sampling_smc
            settings.update(
                threshold_quantile=args.quantile,
                evidence_tolerance=args.evidence_tolerance,
            )
        else:
            factory = make_standard_bayesian_smc
            settings["mutation_n_steps"] = (
                args.tempered_mutation_steps or args.mutation_steps
            )
        sampler, _, metadata = factory(problem.log_prior, counter, **settings)
        start = time.perf_counter()
        raw = sampler.run(
            ParticleState(initial.copy()),
            metadata=metadata,
            rng=args.seed + 10000,
            num_steps=args.max_iterations,
        )
        if method == "ns_smc":
            result = NestedSamplingSMCResult(raw)
            if not result.complete:
                raise RuntimeError(
                    "NS-SMC is incomplete; increase --max-iterations."
                )
            estimate = result.log_evidence
            history = result.log_evidence_history()
            levels = result.levels()
            axes[0, 1].plot(
                [level.log_prior_volume for level in levels],
                color=colors[method],
            )
            # These come from every shell, not raw.final_state.particles.
            samples, weights = result.posterior_samples()
            np.savez_compressed(
                outdir / "shells.npz",
                log_prior_volume=np.array(
                    [level.log_prior_volume for level in levels]
                ),
                log_evidence=history,
                log_evidence_increment=result.log_evidence_increment_history(),
                upper_threshold=result.shell_upper_threshold_history(),
            )
        else:
            if raw.context.metadata["beta"] != 1.0:
                raise RuntimeError(
                    "Tempered SMC is incomplete; increase --max-iterations."
                )
            estimate = raw.final_state.log_evidence
            history = raw.log_normalizing_constant_history()
            samples, weights = raw.posterior_samples()
        elapsed = time.perf_counter() - start
        coordinate = (
            result.log_prior_volume_history()
            if method == "ns_smc"
            else raw.metadata_history("beta")
        )
        mean_log_likelihood = []
        central_mass = []
        for state in raw.states:
            values = problem.log_likelihood(state.particles)
            mean_log_likelihood.append(float(np.sum(state.weights * values)))
            central_mass.append(
                float(
                    np.sum(
                        state.weights
                        * (np.linalg.norm(state.particles, axis=1) < 0.5)
                    )
                )
            )
        np.savez_compressed(
            outdir / f"path_{method}.npz",
            coordinate=coordinate,
            coordinate_name="log_prior_volume"
            if method == "ns_smc"
            else "beta",
            mean_log_likelihood=np.array(mean_log_likelihood),
            central_mass=np.array(central_mass),
        )
        np.savez_compressed(
            outdir / f"posterior_{method}.npz",
            particles=samples,
            weights=weights,
        )
        estimated_mean = np.sum(weights[:, None] * samples, axis=0)
        estimated_second = np.sum(weights[:, None] * samples**2, axis=0)
        summary["methods"][method] = {
            "log_evidence": float(estimate),
            "log_evidence_error": float(estimate - truth_z),
            "mean": estimated_mean.tolist(),
            "second_moment": estimated_second.tolist(),
            "likelihood_evaluations": counter.evaluations,
            "seconds": elapsed,
            "iterations": len(raw),
        }
        axes[0, 0].plot(history, label=method, color=colors[method])
        for dimension, ax in enumerate(axes[1]):
            ax.hist(
                samples[:, dimension],
                weights=weights,
                bins=np.linspace(-10, 10, 201),
                density=True,
                histtype="step",
                label=method,
                color=colors[method],
            )
        print(
            f"{problem.name:20} {method:9} log Z = {estimate:.5f} (error {estimate - truth_z:+.5f}), {counter.evaluations:,} likelihood evaluations"
        )
    axes[0, 0].axhline(truth_z, color="black", linestyle="--", label="exact")
    axes[0, 0].set(xlabel="SMC iteration / NS shell", ylabel="log evidence")
    axes[0, 0].legend()
    axes[0, 1].set(xlabel="NS shell", ylabel="log remaining prior volume")
    grid = np.linspace(-10, 10, 4001)
    for dimension, ax in enumerate(axes[1]):
        ax.plot(
            grid,
            problem.marginal_pdf(grid, dimension),
            color="black",
            linestyle="--",
            label="exact",
        )
        ax.set(
            xlabel=f"x{dimension}", ylabel="Posterior density", xlim=(-10, 10)
        )
        ax.legend()
    fig.suptitle(problem.name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(outdir / "comparison.png", dpi=160)
    plt.close(fig)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(default_problem: str = "all") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem", choices=["all", *PROBLEMS], default=default_problem
    )
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mutation",
        choices=["minipcn", "differential-evolution"],
        default="minipcn",
    )
    parser.add_argument("--mutation-steps", type=int, default=10)
    parser.add_argument("--tempered-mutation-steps", type=int)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--evidence-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = parser.parse_args()
    if min(args.particles, args.mutation_steps, args.max_iterations) <= 0:
        parser.error(
            "particles, mutation steps, and max iterations must be positive"
        )
    if (
        args.tempered_mutation_steps is not None
        and args.tempered_mutation_steps <= 0
    ):
        parser.error("tempered mutation steps must be positive")
    for name in PROBLEMS if args.problem == "all" else [args.problem]:
        run(PROBLEMS[name], args)


if __name__ == "__main__":
    main()
