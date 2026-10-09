"""Minimal NS-SMC inference for a Gaussian likelihood and Gaussian prior."""

from __future__ import annotations

import numpy as np
from sequince import ParticleState, make_nested_sampling_smc
from sequince.log import configure_logging


def log_prior(x: np.ndarray) -> np.ndarray:
    """Normalized two-dimensional Gaussian prior with standard deviation 5."""
    return -0.5 * np.sum(x**2, axis=-1) / 25.0 - np.log(2.0 * np.pi * 25.0)


def log_likelihood(x: np.ndarray) -> np.ndarray:
    """Unit-width Gaussian likelihood centred on (1, -1)."""
    return -0.5 * np.sum((x - np.array([1.0, -1.0])) ** 2, axis=-1) - np.log(
        2.0 * np.pi
    )


def main() -> None:
    configure_logging()
    rng = np.random.default_rng(42)
    initial_state = ParticleState(rng.normal(0.0, 5.0, size=(1024, 2)))
    sampler, target, metadata = make_nested_sampling_smc(
        log_prior,
        log_likelihood,
        threshold_quantile=0.5,
        evidence_tolerance=1e-3,
        mutation="minipcn",
        mutation_n_steps=10,
        logging_frequency=1,
    )
    raw = sampler.run(
        initial_state, metadata=metadata, rng=rng, num_steps=1000
    )
    result = raw.as_nested_sampling_result()
    if not result.complete:
        raise RuntimeError("Run is incomplete; increase the iteration limit.")

    particles, weights = result.posterior_samples()
    mean = np.sum(weights[:, None] * particles, axis=0)
    exact_log_evidence = -np.log(2.0 * np.pi * 26.0) - 1.0 / 26.0
    exact_mean = np.array([1.0, -1.0]) * 25.0 / 26.0
    print(f"Log evidence: {result.log_evidence:.6f}")
    print(f"Exact log evidence: {exact_log_evidence:.6f}")
    print(f"Posterior mean: {mean}")
    print(f"Exact posterior mean: {exact_mean}")
    print(f"Shells: {len(result.levels())}; stop reason: {result.stop_reason}")


if __name__ == "__main__":
    main()
