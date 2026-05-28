from dataclasses import dataclass
from typing import Any, Callable

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

    def log_prior(self, particles):
        return self.xp.where(
            in_bounds(particles, self.prior_bounds),
            0.0,
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
        self.log_evidence = -self.xp.log(
            self.xp.prod(self.prior_bounds[1] - self.prior_bounds[0])
        )


models = {
    "rosenbrock": Rosenbrock,
    "multivariate_gaussian": MultivariateGaussian,
    "gaussian": MultivariateGaussian,
}
