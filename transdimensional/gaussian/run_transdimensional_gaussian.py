import numpy as np
from sequince import ParticleState, SequentialMonteCarlo
from sequince.log import configure_logging
from sequince.posterior import TemperedPosterior
from sequince.steps import (
    AdaptiveTemperedWeighting,
    LoggingStep,
    MetaSMCStep,
    MiniPCNMutation,
    SystematicResampling,
)
from sequince.steps.base import StepContext
from sequince.steps.control import StopIf
from sequince.steps.transformations import ExpandDimensions, SMCTransformation
from sequince.transforms import AffineTransform


def _active_particles(particles, context):
    n_dims = int(context.metadata["n_dims"])
    return np.atleast_2d(particles)[:, :n_dims]


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


def log_density_ratio(particles, context):
    """Evaluate the bridge ratio using only the newly added coordinate.

    The expansion proposal already contains the previous coordinates'
    likelihood and the new coordinate's prior, so those terms cancel.
    """
    pts = np.atleast_2d(particles)
    last_dim = pts[:, int(context.metadata["n_dims"]) - 1]
    values = -0.5 * last_dim**2 - 0.5 * np.log(2 * np.pi)
    return float(values[0]) if particles.ndim == 1 else values


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
        return particles[..., :-1]

    def log_jacobian(self, particles, context):
        return 0.0

    def log_auxiliary_density(self, particles, context):
        lower_bound = context.shared["lower_bound"]
        upper_bound = context.shared["upper_bound"]
        new_block = np.atleast_2d(particles)[:, -1:]
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
    beta = context.metadata["beta"]
    log_evidence = state.log_evidence
    return (
        f"  inner it {context.t}:, "
        f"Beta: {beta:.4f}, "
        f"Log Evidence: {log_evidence:.1f}"
    )


def stop_predicate(state, context):
    return int(context.metadata["n_dims"]) >= context.shared["target_dims"]


def main(*, n_particles=5_000, target_dims=4, mutation_steps=50, plot=True):
    configure_logging(level="INFO")

    rng = np.random.default_rng(1235)

    initial_dims = 0

    lower_bound = -np.exp(3) / 2
    upper_bound = np.exp(3) / 2

    tempered_posterior = TemperedPosterior(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        log_density_ratio=log_density_ratio,
    )

    shared_context = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "target_dims": target_dims,
    }

    metadata = {
        "n_dims": initial_dims,
        # Each newly added coordinate starts its own tempering bridge.
        "beta": 0.0,
    }
    transformation = GaussianExpansionTransformation()

    inner_runner = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(
                tempered_posterior,
                target_ess_ratio=0.9,
                max_beta=1.0,
                min_delta=1e-8,
            ),
            SystematicResampling(always_resample=True),
            MiniPCNMutation(
                tempered_posterior,
                n_steps=mutation_steps,
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
            MetaSMCStep(
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
            "beta": 0.0,
            "n_dims": initial_dims,
        },
    )

    result = outer_runner.run(
        initial_state,
        rng=rng,
        shared_context=shared_context,
        metadata=metadata,
    )
    meta_result = result.get_meta_results("inner_results")
    if meta_result is None:
        raise RuntimeError("The meta SMC step did not record inner results.")

    print("Run summary:")
    for key, value in meta_result.summary().items():
        if key != "final_beta":
            print(f"{key}: {value}")

    true_log_evidence = -np.log(upper_bound - lower_bound) * target_dims
    print(f"True log evidence: {true_log_evidence:.5f}")

    if plot:
        meta_result.final_result.plot_corner(
            filename="transdimensional_gaussian_corner.png"
        )
    return meta_result


if __name__ == "__main__":
    main()
