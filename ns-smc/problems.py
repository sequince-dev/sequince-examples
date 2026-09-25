"""Two-dimensional Gaussian-mixture likelihoods with analytic posteriors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class Problem:
    name: str
    means: np.ndarray
    scales: np.ndarray
    mixing: np.ndarray
    prior_std: float = 5.0

    def log_prior(self, x: np.ndarray) -> np.ndarray:
        return -0.5 * np.sum(x**2, axis=-1) / self.prior_std**2 - np.log(
            2 * np.pi * self.prior_std**2
        )

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        terms = (
            -0.5
            * np.sum((x[..., None, :] - self.means) ** 2, axis=-1)
            / self.scales**2
            - np.log(2 * np.pi * self.scales**2)
            + np.log(self.mixing)
        )
        return logsumexp(terms, axis=-1)

    def posterior_components(
        self,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        prior_var = self.prior_std**2
        variances = self.scales**2
        log_component_z = -0.5 * np.sum(self.means**2, axis=1) / (
            prior_var + variances
        ) - np.log(2 * np.pi * (prior_var + variances))
        log_z = float(logsumexp(np.log(self.mixing) + log_component_z))
        mixing = np.exp(np.log(self.mixing) + log_component_z - log_z)
        means = self.means * (prior_var / (prior_var + variances))[:, None]
        variances = prior_var * variances / (prior_var + variances)
        return log_z, mixing, means, variances

    def truth(self) -> tuple[float, np.ndarray, np.ndarray]:
        log_z, mixing, means, variances = self.posterior_components()
        mean = np.sum(mixing[:, None] * means, axis=0)
        second = np.sum(
            mixing[:, None] * (variances[:, None] + means**2), axis=0
        )
        return log_z, mean, second

    def marginal_pdf(self, x: np.ndarray, dimension: int) -> np.ndarray:
        _, mixing, means, variances = self.posterior_components()
        log_pdf = (
            -0.5 * (x[:, None] - means[:, dimension]) ** 2 / variances
            - 0.5 * np.log(2 * np.pi * variances)
            + np.log(mixing)
        )
        return np.exp(logsumexp(log_pdf, axis=-1))


PROBLEMS = {
    "gaussian": Problem(
        "gaussian", np.array([[1.0, -1.0]]), np.array([1.0]), np.array([1.0])
    ),
    "separated_mixture": Problem(
        "separated_mixture",
        np.array([[-4.0, 0.0], [4.0, 0.0]]),
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
    ),
    "spike_and_slab": Problem(
        "spike_and_slab",
        np.zeros((2, 2)),
        np.array([0.1, 2.0]),
        np.array([0.5, 0.5]),
    ),
}


class CountedLikelihood:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.evaluations = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.evaluations += 1 if x.ndim == 1 else len(x)
        return self.problem.log_likelihood(x)
