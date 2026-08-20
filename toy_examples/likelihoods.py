from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from array_api_compat import array_namespace
from sequince.utils import in_bounds


@dataclass
class ToyModel:
    xp: Any
    dimensions: int
    prior_bounds: Any
    log_likelihood_fn: Callable
    log_likelihood_args: dict[str, Any] | None = None
    log_evidence: float = float("nan")
    log_prior_volume: float = field(init=False)

    def __post_init__(self):
        self.prior_bounds = self.xp.asarray(self.prior_bounds)
        if self.prior_bounds.ndim == 1:
            self.prior_bounds = self.xp.repeat(
                self.prior_bounds[:, None], repeats=self.dimensions, axis=1
            )

        if self.prior_bounds.shape != (2, self.dimensions):
            raise ValueError(
                f"prior_bounds must have shape (2, {self.dimensions})"
            )
        self.log_prior_volume = self.xp.log(
            self.xp.prod(self.prior_bounds[1] - self.prior_bounds[0])
        )

    def log_prior(self, particles):
        return self.xp.where(
            in_bounds(particles, self.prior_bounds),
            -self.log_prior_volume,
            -self.xp.inf,
        )

    def log_likelihood(self, particles):
        if self.log_likelihood_args is not None:
            return self.log_likelihood_fn(
                particles, **self.log_likelihood_args
            )
        else:
            return self.log_likelihood_fn(particles)


def _rosenbrock(x):
    xp = array_namespace(x)
    return xp.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0
        + (1.0 - x[..., :-1]) ** 2.0,
        axis=-1,
    )


def rosenbrock(x):
    return -_rosenbrock(x)


def hybrid_rosenbrock(
    x: np.ndarray,
    n1: int,
    n2: int,
    mu: float = 1.0,
    a: float = 1.0 / 20.0,
    b: float | np.ndarray = 100.0 / 20.0,
    normalized: bool = True,
) -> np.ndarray:
    """
    Hybrid Rosenbrock log-density for an input array of shape (n, dims).

    The dimension must satisfy:

        dims = 1 + n2 * (n1 - 1)

    Parameters
    ----------
    x:
        Array of points with shape (n, dims).
    n1:
        Number of variables per Rosenbrock branch, including the shared root.
    n2:
        Number of branches.
    mu:
        Mean of the shared root variable.
    a:
        Positive coefficient for the shared root term.
    b:
        Scalar or array with shape (n2, n1 - 1).
    normalized:
        Whether to include the analytic normalisation constant.

    Returns
    -------
    logp:
        Array with shape (n,).
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 2:
        raise ValueError(f"x must have shape (n, dims), got shape {x.shape}")

    n, dims = x.shape
    expected_dims = 1 + n2 * (n1 - 1)

    if dims != expected_dims:
        raise ValueError(
            f"Expected dims={expected_dims} for n1={n1}, n2={n2}, got {dims}"
        )

    if n1 < 2:
        raise ValueError("n1 must be at least 2")

    if n2 < 1:
        raise ValueError("n2 must be at least 1")

    if a <= 0:
        raise ValueError("a must be positive")

    b = np.asarray(b, dtype=float)

    if b.ndim == 0:
        b = np.full((n2, n1 - 1), float(b))
    elif b.shape != (n2, n1 - 1):
        raise ValueError(f"b must be scalar or have shape {(n2, n1 - 1)}")

    if np.any(b <= 0):
        raise ValueError("All b values must be positive")

    # Shared root variable, shape (n,)
    x0 = x[:, 0]

    logp = -a * (x0 - mu) ** 2

    # Remaining variables as branches:
    # shape (n, n2, n1 - 1)
    branches = x[:, 1:].reshape(n, n2, n1 - 1)

    # First variable in each branch depends on shared root x0.
    # branch[:, j, 0] ~ centered on x0**2
    logp -= np.sum(
        b[:, 0] * (branches[:, :, 0] - x0[:, None] ** 2) ** 2,
        axis=1,
    )

    # Later variables depend on previous variable in same branch.
    for i in range(1, n1 - 1):
        logp -= np.sum(
            b[:, i] * (branches[:, :, i] - branches[:, :, i - 1] ** 2) ** 2,
            axis=1,
        )

    if normalized:
        log_norm = (
            0.5 * np.log(a)
            + 0.5 * np.sum(np.log(b))
            - 0.5 * dims * np.log(np.pi)
        )
        logp = logp + log_norm

    return logp


def multivariate_gaussian(x, mean, cov):
    xp = array_namespace(x, mean, cov)
    x = xp.asarray(x)
    mean = xp.asarray(mean, dtype=x.dtype)
    cov = xp.asarray(cov, dtype=x.dtype)

    if mean.ndim == 0:
        mean = xp.full((x.shape[-1],), mean, dtype=x.dtype)
    if mean.shape != (x.shape[-1],):
        raise ValueError(
            f"mean must have shape ({x.shape[-1]},), received {mean.shape}"
        )

    if cov.ndim == 0:
        cov = cov * xp.eye(x.shape[-1], dtype=x.dtype)
    elif cov.ndim == 1:
        if cov.shape != (x.shape[-1],):
            raise ValueError(
                f"cov diagonal must have shape ({x.shape[-1]},), received {cov.shape}"
            )
        cov = cov[:, None] * xp.eye(x.shape[-1], dtype=x.dtype)
    if cov.shape != (x.shape[-1], x.shape[-1]):
        raise ValueError(
            "cov must have shape "
            f"({x.shape[-1]}, {x.shape[-1]}), received {cov.shape}"
        )

    d = x - mean
    solved = xp.linalg.solve(cov, d.T).T
    exponent = -0.5 * xp.sum(d * solved, axis=-1)
    log_normalization = 0.5 * (
        x.shape[-1] * xp.log(2 * xp.pi) + xp.log(xp.linalg.det(cov))
    )
    return exponent - log_normalization


@dataclass
class Rosenbrock(ToyModel):
    log_likelihood_fn: Callable = rosenbrock
    dimensions: int = 2
    prior_bounds: tuple[float, float] = (-5.0, 5.0)


@dataclass
class HybridRosenbrock(ToyModel):
    log_likelihood_fn: Callable = hybrid_rosenbrock
    dimensions: int | None = None
    prior_bounds: tuple[float, float] = (-5.0, 5.0)
    log_likelihood_args: dict[str, Any] | None = field(
        default_factory=lambda: {
            "n1": 2,
            "n2": 2,
            "mu": 1.0,
            "a": 1.0 / 20.0,
            "b": 100.0 / 20.0,
            "normalized": True,
        }
    )

    def __post_init__(self):
        if self.dimensions is None:
            n1 = self.log_likelihood_args.get("n1", 2)
            n2 = self.log_likelihood_args.get("n2", 2)
            self.dimensions = 1 + n2 * (n1 - 1)

        super().__post_init__()

        normalized = self.log_likelihood_args.get("normalized", True)

        if normalized:
            self.log_evidence = -self.log_prior_volume
        else:
            self.log_evidence = float("nan")


@dataclass
class MultivariateGaussian(ToyModel):
    log_likelihood_fn: Callable = multivariate_gaussian
    dimensions: int = 2
    prior_bounds: tuple[float, float] = (-10.0, 10.0)

    def __post_init__(self):
        super().__post_init__()
        self.log_likelihood_args = self.log_likelihood_args or {}
        if "mean" not in self.log_likelihood_args:
            self.log_likelihood_args["mean"] = self.xp.zeros(self.dimensions)
        if "cov" not in self.log_likelihood_args:
            self.log_likelihood_args["cov"] = self.xp.eye(self.dimensions)

        # log-evidence is just the normalization from the prior
        self.log_evidence = -self.log_prior_volume


models = {
    "rosenbrock": Rosenbrock,
    "hybrid_rosenbrock": HybridRosenbrock,
    "hybridrosenbrock": HybridRosenbrock,
    "multivariate_gaussian": MultivariateGaussian,
    "gaussian": MultivariateGaussian,
}
