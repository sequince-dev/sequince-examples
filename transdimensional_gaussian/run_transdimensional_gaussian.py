import numpy as np
from sequince import ParticleState, SequentialMonteCarlo
from sequince.logging_utils import configure_logging
from sequince.posterior import TemperedPosterior
from sequince.result import NestedSMCResult
from sequince.steps import (
    AdaptiveTemperedWeighting,
    LoggingStep,
    MiniPCNMutation,
    NestedSMCStep,
    SystematicResampling,
)
from sequince.steps.base import StepContext
from sequince.steps.control import StopIf
from sequince.steps.transformations import ExpandDimensions, SMCTransformation
from sequince.transforms import AffineTransform


def _active_particles(particles, context):
    n_dims = int(context.metadata["n_dims"])
    return particles[:, :n_dims]


def log_likelihood(particles, context):
    """Gaussian likelihood with zero mean and identity covariance."""
    active_particles = _active_particles(particles, context)
    diffs = active_particles
    log_norm = -0.5 * active_particles.shape[1] * np.log(2 * np.pi)
    log_likes = -0.5 * np.sum(diffs**2, axis=-1)
    return log_likes + log_norm


def log_prior(particles, context):
    """Uniform prior between context['lower_bound'] and context['upper_bound']."""
    active_particles = _active_particles(particles, context)
    lower_bound = context.shared["lower_bound"]
    upper_bound = context.shared["upper_bound"]
    in_bounds = np.all(
        (active_particles >= lower_bound) & (active_particles <= upper_bound),
        axis=-1,
    )
    log_prior_value = (
        -np.log(upper_bound - lower_bound) * active_particles.shape[1]
    )
    return np.where(in_bounds, log_prior_value, -np.inf)


class GaussianTemperedPosterior(TemperedPosterior):
    """Posterior with a custom increment for one-dimensional expansions.

    After an expansion from ``d-1`` to ``d`` dimensions, the auxiliary proposal
    draws the new coordinate from the prior, so the bridge increment reduces to
    the Gaussian likelihood contribution of the final coordinate only.
    """

    def log_increment(
        self,
        particles,
        context,
        *,
        log_proposal_override=None,
    ):
        if log_proposal_override is None:
            return super().log_increment(
                particles,
                context,
                log_proposal_override=log_proposal_override,
            )

        pts = np.atleast_2d(particles)
        last_dim = pts[:, int(context.metadata["n_dims"]) - 1]
        increments = -0.5 * last_dim**2 - 0.5 * np.log(2 * np.pi)
        if particles.ndim == 1:
            return np.array(increments[0], ndmin=1)
        return increments

    def log_prob(
        self,
        particles,
        context,
        *,
        log_proposal_override=None,
    ):
        if log_proposal_override is None:
            return super().log_prob(
                particles,
                context,
                log_proposal_override=log_proposal_override,
            )

        pts = np.atleast_2d(particles)
        n_dims = int(context.metadata["n_dims"])
        beta = float(context.metadata.get("temperature", 0.0))

        log_prior_vals = log_prior(pts, context)

        if n_dims <= 0:
            result = log_prior_vals
        else:
            previous = pts[:, : n_dims - 1]
            previous_log_like = np.zeros(pts.shape[0], dtype=float)
            if previous.shape[1] > 0:
                previous_log_like = -0.5 * np.sum(
                    previous**2, axis=-1
                ) - 0.5 * previous.shape[1] * np.log(2 * np.pi)

            last_dim = pts[:, n_dims - 1]
            new_log_like = -0.5 * last_dim**2 - 0.5 * np.log(2 * np.pi)
            result = log_prior_vals + previous_log_like + beta * new_log_like

        if particles.ndim == 1:
            return float(result[0])
        return result


class GaussianExpansionTransformation(SMCTransformation):
    def forward(self, state, context, rng):
        n_particles = state.particles.shape[0]
        lower_bound = context.shared["lower_bound"]
        upper_bound = context.shared["upper_bound"]
        new_block = rng.uniform(
            low=lower_bound,
            high=upper_bound,
            size=(n_particles, 1),
        )
        particles = np.asarray(state.particles)
        expanded_particles = np.concatenate([particles, new_block], axis=1)
        return state.with_updates(particles=expanded_particles)

    def inverse(self, particles, context):
        return particles[:, :-1]

    def log_auxiliary_density(self, particles, context):
        lower_bound = context.shared["lower_bound"]
        upper_bound = context.shared["upper_bound"]
        new_block = particles[:, -1:]
        in_bounds = np.all(
            (new_block >= lower_bound) & (new_block <= upper_bound),
            axis=-1,
        )
        log_q = np.full(new_block.shape[0], -np.inf, dtype=float)
        log_q[in_bounds] = -np.log(upper_bound - lower_bound)
        return log_q


def logging_str_fn(state: ParticleState, context: StepContext) -> str:
    dims = context.metadata["n_dims"]
    log_evidence = state.log_evidence
    return (
        f"it {context.t}:, "
        f"Dimensions: {dims}, "
        f"Log Evidence: {log_evidence:.1f}"
    )


def inner_logging_str_fn(state: ParticleState, context: StepContext) -> str:
    temperature = context.metadata["temperature"]
    log_evidence = state.log_evidence
    return (
        f"  inner it {context.t}:, "
        f"Temperature: {temperature:.4f}, "
        f"Log Evidence: {log_evidence:.1f}"
    )


def stop_predicate(state, context):
    return (
        context.t > 0
        and int(context.metadata["n_dims"]) >= context.shared["target_dims"]
    )


def main():
    configure_logging(level="INFO")

    rng = np.random.default_rng(1235)

    initial_dims = 0
    target_dims = 4

    lower_bound = -np.exp(3) / 2
    upper_bound = np.exp(3) / 2

    n_particles = 5_000
    tempered_posterior = GaussianTemperedPosterior(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
    )

    shared_context = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "target_dims": target_dims,
    }

    metadata = {
        "n_dims": initial_dims,
    }
    transformation = GaussianExpansionTransformation()

    inner_runner = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(
                tempered_posterior,
                target_ess_ratio=0.9,
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
            LoggingStep(str_fn=inner_logging_str_fn, frequency=1),
        ],
        auto_normalize=True,
    )
    outer_runner = SequentialMonteCarlo(
        steps=[
            ExpandDimensions(
                transformation=transformation,
                expected_block_size=1,
                metadata_key="n_dims",
                posterior=tempered_posterior,
            ),
            NestedSMCStep(
                inner_runner,
                posterior=tempered_posterior,
                record_outputs=True,
                output_key="inner_results",
            ),
            LoggingStep(str_fn=logging_str_fn, frequency=1),
            StopIf(stop_predicate),
        ],
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
            "n_dims": initial_dims,
        },
    )

    result = outer_runner.run(
        initial_state,
        rng=rng,
        shared_context=shared_context,
        metadata=metadata,
    )
    nested_result = NestedSMCResult(result.context.outputs["inner_results"])

    print("Run summary:")
    for key, value in nested_result.summary().items():
        if key != "final_temperature":
            print(f"{key}: {value}")

    true_log_evidence = -np.log(upper_bound - lower_bound) * target_dims
    print(f"True log evidence: {true_log_evidence:.5f}")

    nested_result.final_result.plot_corner(
        filename="transdimensional_gaussian_corner.png"
    )


if __name__ == "__main__":
    main()
