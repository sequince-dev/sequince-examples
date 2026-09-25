"""Compare eager NS-SMC and tempered SMC against analytic Gaussian mixtures.

Run with ``python ns-smc/benchmark.py --seeds 20``. Results are
JSON on stdout. Both methods use the same prior draws and mutation settings;
likelihood counts include every evaluated particle, including rejected moves.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
from problems import PROBLEMS, CountedLikelihood
from sequince import (
    NestedSamplingSMCResult,
    ParticleState,
    make_nested_sampling_smc,
    make_standard_bayesian_smc,
)


def benchmark(
    n_particles: int,
    seeds: int,
    n_steps: int,
    mutation: str,
    tempered_steps: int | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "settings": {
            "particles": n_particles,
            "seeds": seeds,
            "mutation_steps": n_steps,
            "tempered_mutation_steps": tempered_steps or n_steps,
            "mutation": mutation,
            "prior_std": 5.0,
            "ns_quantile": 0.5,
            "ns_evidence_tolerance": 1e-3,
        },
        "problems": {},
    }
    for name, model in PROBLEMS.items():
        log_z, true_mean, true_second = model.truth()
        problem: dict[str, Any] = {
            "log_evidence": log_z,
            "mean": true_mean.tolist(),
            "second_moment": true_second.tolist(),
        }
        for method in ("tempered", "ns_smc"):
            runs = []
            for seed in range(seeds):
                likelihood = CountedLikelihood(model)
                factory = (
                    make_nested_sampling_smc
                    if method == "ns_smc"
                    else make_standard_bayesian_smc
                )
                sampler, _, metadata = factory(
                    model.log_prior,
                    likelihood,
                    mutation=mutation,
                    mutation_n_steps=(tempered_steps or n_steps)
                    if method == "tempered"
                    else n_steps,
                    logging_frequency=0,
                )
                particles = np.random.default_rng(seed).normal(
                    0, 5, size=(n_particles, 2)
                )
                start = time.perf_counter()
                raw = sampler.run(
                    ParticleState(particles),
                    metadata=metadata,
                    rng=seed + 10000,
                    num_steps=1000,
                )
                result = (
                    NestedSamplingSMCResult(raw) if method == "ns_smc" else raw
                )
                if method == "ns_smc":
                    assert isinstance(result, NestedSamplingSMCResult)
                    if not result.complete:
                        raise RuntimeError(
                            "NS-SMC exceeded benchmark iteration limit."
                        )
                    estimate = result.log_evidence
                else:
                    if raw.context.metadata["beta"] != 1.0:
                        raise RuntimeError(
                            "Tempered SMC did not reach beta=1."
                        )
                    estimate = raw.final_state.log_evidence
                samples, weights = result.posterior_samples()
                elapsed = time.perf_counter() - start
                estimated_mean = np.sum(weights[:, None] * samples, axis=0)
                estimated_second = np.sum(
                    weights[:, None] * samples**2, axis=0
                )
                runs.append(
                    {
                        "seed": seed,
                        "log_evidence": float(estimate),
                        "mean": estimated_mean.tolist(),
                        "second_moment": estimated_second.tolist(),
                        "likelihood_evaluations": likelihood.evaluations,
                        "seconds": elapsed,
                        "iterations": len(raw),
                    }
                )
            errors = np.array([r["log_evidence"] - log_z for r in runs])
            problem[method] = {
                "log_evidence_bias": float(errors.mean()),
                "log_evidence_rmse": float(np.sqrt(np.mean(errors**2))),
                "mean_rmse": float(
                    np.sqrt(
                        np.mean(
                            (np.array([r["mean"] for r in runs]) - true_mean)
                            ** 2
                        )
                    )
                ),
                "relative_second_moment_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                (
                                    np.array(
                                        [r["second_moment"] for r in runs]
                                    )
                                    - true_second
                                )
                                / true_second
                            )
                            ** 2
                        )
                    )
                ),
                "mean_likelihood_evaluations": float(
                    np.mean([r["likelihood_evaluations"] for r in runs])
                ),
                "mean_seconds": float(np.mean([r["seconds"] for r in runs])),
                "runs": runs,
            }
        results["problems"][name] = problem
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--mutation-steps", type=int, default=10)
    parser.add_argument(
        "--mutation",
        choices=["minipcn", "differential-evolution"],
        default="minipcn",
    )
    parser.add_argument("--tempered-mutation-steps", type=int, default=None)
    args = parser.parse_args()
    if (
        args.tempered_mutation_steps is not None
        and args.tempered_mutation_steps <= 0
    ):
        parser.error("tempered mutation steps must be positive")
    if min(args.particles, args.seeds, args.mutation_steps) <= 0:
        parser.error("particles, seeds, and mutation steps must be positive")
    print(
        json.dumps(
            benchmark(
                args.particles,
                args.seeds,
                args.mutation_steps,
                args.mutation,
                args.tempered_mutation_steps,
            ),
            indent=2,
        )
    )
