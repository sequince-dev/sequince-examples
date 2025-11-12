from typing import Callable

import numpy as np
from sequince import ParticleState, SequentialMonteCarlo
from sequince.logging_utils import configure_logging
from sequince.posterior import TemperedPosterior
from sequince.steps import (
    AdaptiveTemperedWeighting,
    ContinueWhile,
    DimensionalityExpansion,
    LoggingStep,
    MiniPCNMutation,
    SystematicResampling,
)
from sequince.steps.base import StepContext
from sequince.transforms import AffineTransform


def log_likelihood(particles, context):
    """Gaussian likelihood with zero mean and identity covariance."""
    diffs = particles
    log_norm = -0.5 * particles.shape[1] * np.log(2 * np.pi)
    log_likes = -0.5 * np.sum(diffs**2, axis=-1)
    return log_likes - log_norm


def log_prior(particles, context):
    """Uniform prior between context['lower_bound'] and context['upper_bound']."""
    lower_bound = context.shared["lower_bound"]
    upper_bound = context.shared["upper_bound"]
    in_bounds = np.all(
        (particles >= lower_bound) & (particles <= upper_bound), axis=-1
    )
    log_prior_value = -np.log(upper_bound - lower_bound) * particles.shape[1]
    return np.where(in_bounds, log_prior_value, -np.inf)


def component_sampler(state, context, rng):
    """Sample new components from the prior."""
    n_particles = state.particles.shape[0]
    return rng.uniform(
        low=context.shared["lower_bound"],
        high=context.shared["upper_bound"],
        size=(n_particles, 1),
    )


class TemperatureTriggeredExpansion(DimensionalityExpansion):
    def __init__(
        self,
        log_likelihood: Callable[[np.ndarray], np.ndarray],
        log_prior: Callable[[np.ndarray], np.ndarray],
        target_dims: int,
        trigger_temperature: float = 0.999,
    ):
        super().__init__(
            component_sampler=component_sampler,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            expected_block_size=1,
            metadata_key="dimensions",
            context_key="dimensions",
            context_delta_key="dimension_delta",
        )
        self.target_dims = target_dims
        self.trigger_temperature = trigger_temperature

    def should_expand(self, context) -> bool:
        temperature = context.shared["temperature"]
        trigger_temperature = self.trigger_temperature
        current_dims = context.shared["n_dims"] // self._expected_block_size
        return (temperature >= trigger_temperature) and (
            current_dims < self.target_dims
        )

    def apply(self, state, context, rng):
        if not self.should_expand(context):
            return state
        state = super().apply(state, context, rng)
        context.shared["n_dims"] = state.particles.shape[1]
        state.metadata["n_dims"] = state.particles.shape[1]
        return state


def make_continue_annealing_predicate(
    target_dims: int,
    trigger_temperature: float = 0.999,
) -> Callable:
    def predicate(state, context) -> bool:
        dims = context.shared["n_dims"]
        current_dims = context.shared["n_dims"]
        continue_annealing = (current_dims < target_dims) or (
            dims < trigger_temperature
        )
        return continue_annealing

    return predicate


def logging_str_fn(state: ParticleState, context: StepContext) -> str:
    temperature = context.shared["temperature"]
    dims = context.shared["dimensions"]
    log_evidence = state.log_evidence
    return (
        f"it {context.t}:, "
        f"Temperature: {temperature:.4f}, "
        f"Dimensions: {dims}, "
        f"Log Evidence: {log_evidence:.1f}"
    )


def main():
    configure_logging(level="INFO")

    rng = np.random.default_rng(1235)

    initial_dims = 1
    target_dims = 4

    lower_bound = -10.0
    upper_bound = 10.0

    n_particles = 5_000
    trigger_temperature = 1 - 1e-6

    tempered_posterior = TemperedPosterior(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
    )

    smc = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(
                tempered_posterior,
                target_ess_ratio=0.95,
                max_temperature=1.0,
                min_delta=1e-8,
            ),
            SystematicResampling(always_resample=True),
            MiniPCNMutation(
                tempered_posterior,
                n_steps=50,
                step_fn="tpcn",
                transform=AffineTransform(),
            ),
            LoggingStep(str_fn=logging_str_fn, frequency=1),
            ContinueWhile(
                make_continue_annealing_predicate(
                    target_dims=target_dims,
                    trigger_temperature=trigger_temperature,
                )
            ),
            TemperatureTriggeredExpansion(
                log_likelihood=log_likelihood,
                log_prior=log_prior,
                target_dims=target_dims,
                trigger_temperature=trigger_temperature,
            ),
        ],
        auto_normalize=True,
    )
    # Initial state with draws from the prior
    initial_state = ParticleState(
        particles=rng.uniform(
            low=lower_bound,
            high=upper_bound,
            size=(n_particles, initial_dims),
        ),
        log_weights=np.full(n_particles, -np.log(n_particles)),
        metadata={
            "temperature": 0.0,
            "n_pulses": 1,
            "dimensions": 3,
        },
    )

    shared_context = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_dims": 1,
        "dimensions": 1,
        "temperature": 0.0,
    }

    result = smc.run(initial_state, rng=rng, shared_context=shared_context)

    print("Run summary:")
    for key, value in result.summary().items():
        print(f"  {key}: {value}")

    true_log_evidence = -np.log(upper_bound - lower_bound) * target_dims
    print(f"True log evidence: {true_log_evidence:.2f}")

    result.plot_corner(filename="transdimensional_gaussian_corner.png")


if __name__ == "__main__":
    main()
