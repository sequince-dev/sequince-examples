"""Adaptive annealing with on-the-fly Gaussian mixture expansion.

This example demonstrates sequential Monte Carlo with adaptive tempering and
MCMC mutation to fit a Gaussian mixture model to one-dimensional data.  The
number of mixture components is not fixed in advance; instead, new components
are added dynamically during the SMC run whenever the current annealing cycle
completes. Choose prior draws (the default) or a variable-width split:

    python gmm_transdimensional.py --expansion prior
    python gmm_transdimensional.py --expansion split --true-components 3 --max-components 6

The split divides the last labelled component. Use --expansion split-random
for uniform parent selection with a proposal marginalised over parent choices.
All components have independent inferred widths; labels are retained without reordering.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from orng import RandomGenerator
from scipy.special import expit, gammaln, logsumexp, softmax
from sequince import (
    ParticleState,
    SequentialMonteCarlo,
    TemperedPosterior,
)
from sequince.log import configure_logging
from sequince.steps import (
    AdaptiveTemperedWeighting,
    DifferentialEvolutionMutation,
    LoggingStep,
    SMCTransformation,
    SystematicResampling,
    TransformationStep,
)
from sequince.steps.base import StepContext
from sequince.steps.context import LOG_PROPOSAL_OVERRIDE_KEY
from sequince.steps.control import StopIf
from sequince.steps.meta import MetaSMCStep
from sequince.utils.callables import call_with_optional_context


def generate_gmm_data(
    weights: Sequence[float],
    means: Sequence[float],
    *,
    std: float | np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw observations from a 1D Gaussian mixture."""
    weights = np.array(weights, dtype=float)
    weights /= np.sum(weights)
    means = np.array(means, dtype=float)
    assignments = rng.choice(len(weights), size=n_samples, p=weights)
    widths = np.broadcast_to(np.asarray(std), means.shape)
    return rng.normal(
        loc=means[assignments], scale=widths[assignments], size=n_samples
    )


def log_width_prior(log_stds, context):
    """Normal density in log-standard-deviation coordinates."""
    loc = context.shared.get("log_std_prior_mean", np.log(0.75))
    scale = context.shared.get("log_std_prior_scale", 0.75)
    return -0.5 * ((log_stds - loc) / scale) ** 2 - np.log(
        np.sqrt(2 * np.pi) * scale
    )


def log_prior(position: np.ndarray, context: StepContext) -> np.ndarray:
    """Independent normal priors on logits/log widths and uniform means."""
    points = np.atleast_2d(position)

    bounds: tuple[float, float] = context.shared["mean_bounds"]
    n_components = points.shape[1] // 3
    logits = points[:, :n_components]
    means = points[:, n_components : 2 * n_components]
    log_stds = points[:, 2 * n_components :]

    width = bounds[1] - bounds[0]
    if width <= 0.0:
        raise ValueError("mean_bounds must define a positive-width interval.")

    scale = float(context.shared.get("logit_prior_scale", 2.0))
    if scale <= 0.0:
        raise ValueError("logit_prior_scale must be positive.")

    in_bounds = np.all((means >= bounds[0]) & (means <= bounds[1]), axis=1)
    log_prior_vals = np.full(points.shape[0], -np.inf, dtype=float)

    valid_logits = logits[in_bounds]
    if valid_logits.size > 0:
        norm_const = np.log(np.sqrt(2.0 * np.pi) * scale)
        log_prior_means = -n_components * np.log(width)
        log_prior_logits = -0.5 * np.sum((valid_logits / scale) ** 2, axis=1)
        log_prior_logits -= n_components * norm_const

        rate = float(context.shared.get("component_count_rate", 3.0))
        if rate <= 0.0:
            raise ValueError("component_count_rate must be positive.")
        log_component_count = (
            n_components * np.log(rate) - rate - gammaln(n_components + 1)
        )

        log_prior_vals[in_bounds] = (
            log_prior_means
            + log_prior_logits
            + log_component_count
            + np.sum(log_width_prior(log_stds[in_bounds], context), axis=1)
        )

    if position.ndim == 1:
        return np.array(log_prior_vals[0], ndmin=1)
    return log_prior_vals


def _blocked_gmm_log_likelihoods(
    particles: np.ndarray,
    context: StepContext,
    component_counts: Sequence[int],
) -> dict[int, np.ndarray]:
    """Evaluate mixture likelihoods using bounded observation blocks."""
    points = np.atleast_2d(np.asarray(particles, dtype=float))
    n_particles, param_dim = points.shape
    if param_dim % 3 != 0:
        raise ValueError(
            "Particle array must contain logits, means, then log standard deviations."
        )

    available_components = param_dim // 3
    counts = tuple(dict.fromkeys(int(count) for count in component_counts))
    if not counts:
        raise ValueError("At least one component count must be requested.")
    if any(count < 1 or count > available_components for count in counts):
        raise ValueError(
            "Requested component counts must be between one and the number "
            "of components represented by the particle array."
        )

    data = np.asarray(context.shared["data"], dtype=float)
    block_size = int(context.shared.get("likelihood_block_size", 64))
    if block_size < 1:
        raise ValueError("likelihood_block_size must be positive.")

    max_components = max(counts)
    # Layout: [all logits, all means, all log standard deviations].
    logits = points[:, :max_components]
    means = points[
        :, available_components : available_components + max_components
    ]
    log_stds = points[
        :, 2 * available_components : 2 * available_components + max_components
    ]
    inverse_variances = np.exp(-2 * log_stds)
    log_normalizers = {
        count: logsumexp(logits[:, :count], axis=1) for count in counts
    }
    totals = {count: np.zeros(n_particles, dtype=float) for count in counts}
    log_gaussian_normalizer = 0.5 * np.log(2 * np.pi) + log_stds

    # Reuse each bounded data block for every requested mixture order.
    for start in range(0, data.size, block_size):
        block = data[start : start + block_size]
        log_terms = block[np.newaxis, :, np.newaxis] - means[:, np.newaxis, :]
        np.square(log_terms, out=log_terms)
        log_terms *= -0.5 * inverse_variances[:, np.newaxis, :]
        log_terms -= log_gaussian_normalizer[:, np.newaxis, :]
        log_terms += logits[:, np.newaxis, :]

        for count in counts:
            log_mix = logsumexp(log_terms[:, :, :count], axis=2)
            log_mix -= log_normalizers[count][:, np.newaxis]
            totals[count] += np.sum(log_mix, axis=1)

    return totals


def gmm_log_likelihood(
    particles: np.ndarray,
    context: StepContext,
    *,
    active_components: int | None = None,
) -> np.ndarray:
    """Log likelihood of the dataset under a mixture with component-specific widths."""
    param_dim = particles.shape[-1]
    if active_components is None:
        n_components = int(context.metadata.get("active_components", 0))
        if n_components <= 0:
            if param_dim % 3 != 0:
                raise ValueError(
                    "Particle array must contain logits, means, then log standard deviations."
                )
            n_components = param_dim // 3
    else:
        n_components = active_components
    return _blocked_gmm_log_likelihoods(particles, context, (n_components,))[
        n_components
    ]


class GMMExpansionTransformation(SMCTransformation):
    """Add one mixture component using the component prior as proposal."""

    def forward_with_context(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> tuple[ParticleState, StepContext]:
        current_components = int(context.metadata["active_components"])
        particles = np.asarray(state.particles)
        if particles.ndim == 1:
            particles = particles[:, np.newaxis]

        expanded_particles = self._expand_particles(particles, context, rng)

        next_components = current_components + 1
        # The nested SMC run anneals from the old posterior to this larger model.
        metadata = dict(state.metadata)
        metadata["previous_active_components"] = current_components
        metadata["active_components"] = next_components
        metadata["beta"] = 0.0
        metadata.pop("delta_beta", None)

        context = context.with_metadata(
            previous_active_components=current_components,
            active_components=next_components,
            beta=0.0,
            delta_beta=0.0,
        )

        return (
            state.with_updates(
                particles=expanded_particles,
                metadata=metadata,
            ),
            context,
        )

    def _expand_particles(self, particles, context, rng):
        current_components = int(context.metadata["active_components"])
        n = particles.shape[0]
        bounds: tuple[float, float] = context.shared["mean_bounds"]
        logit_prior_scale = float(context.shared["logit_prior_scale"])

        # Drawing the added parameters from their priors makes their prior and
        # auxiliary-proposal terms cancel in the subsequent weight update.
        new_logits = rng.normal(loc=0.0, scale=logit_prior_scale, size=(n, 1))
        new_means = rng.uniform(low=bounds[0], high=bounds[1], size=(n, 1))

        # Restore the [all logits, all means] layout after appending the pair.
        existing_logits = particles[:, :current_components]
        existing_means = particles[
            :, current_components : 2 * current_components
        ]
        existing_log_stds = particles[:, 2 * current_components :]
        logits = np.concatenate([existing_logits, new_logits], axis=1)
        means = np.concatenate([existing_means, new_means], axis=1)
        new_log_stds = rng.normal(
            loc=context.shared.get("log_std_prior_mean", np.log(0.75)),
            scale=context.shared.get("log_std_prior_scale", 0.75),
            size=(n, 1),
        )
        log_stds = np.concatenate([existing_log_stds, new_log_stds], axis=1)
        return np.concatenate([logits, means, log_stds], axis=1)

    def inverse(
        self,
        particles: np.ndarray,
        context: StepContext,
    ) -> np.ndarray:
        pts = np.atleast_2d(particles)
        current_components = int(context.metadata["active_components"])
        previous_components = int(
            context.metadata.get(
                "previous_active_components", current_components
            )
        )
        # Removing the appended pair maps an expanded particle back to the
        # particle on which the previous posterior is evaluated.
        logits = pts[:, :current_components]
        means = pts[:, current_components : 2 * current_components]
        old_logits = logits[:, :previous_components]
        old_means = means[:, :previous_components]
        old_log_stds = pts[
            :,
            2 * current_components : 2 * current_components
            + previous_components,
        ]
        inverse_particles = np.concatenate(
            [old_logits, old_means, old_log_stds], axis=1
        )
        if particles.ndim == 1:
            return inverse_particles[0]
        return inverse_particles

    def log_jacobian(self, particles, context):
        """Adding and reordering coordinates has unit Jacobian."""
        return 0.0

    def log_auxiliary_density(
        self,
        particles: np.ndarray,
        context: StepContext,
    ) -> np.ndarray:
        """The log density for the new component parameters."""
        pts = np.atleast_2d(particles)
        current_components = int(context.metadata["active_components"])
        previous_components = int(
            context.metadata.get(
                "previous_active_components", current_components
            )
        )
        if previous_components >= current_components:
            return np.zeros(pts.shape[0], dtype=float)

        bounds: tuple[float, float] = context.shared["mean_bounds"]
        width = bounds[1] - bounds[0]
        scale = float(context.shared["logit_prior_scale"])
        if width <= 0.0 or scale <= 0.0:
            raise ValueError(
                "Invalid prior configuration for expansion proposal."
            )

        logits = pts[:, :current_components]
        means = pts[:, current_components : 2 * current_components]
        # These are exactly the random variables introduced by forward().
        new_logits = logits[:, previous_components]
        new_means = means[:, previous_components]

        norm_const = np.log(np.sqrt(2.0 * np.pi) * scale)
        logit_density = -0.5 * (new_logits / scale) ** 2 - norm_const

        in_bounds = (new_means >= bounds[0]) & (new_means <= bounds[1])
        mean_density = np.full(pts.shape[0], -np.inf, dtype=float)
        mean_density[in_bounds] = -np.log(width)
        density = (
            logit_density + mean_density + log_width_prior(pts[:, -1], context)
        )
        if particles.ndim == 1:
            return np.array(density[0], ndmin=1)
        return density


class GMMSplitTransformation(GMMExpansionTransformation):
    """Split the last labelled component, preserving its weight and mean.

    Draw z ~ Normal(0, 1) and separation d ~ Normal(0, split_mean_scale).
    With u = sigmoid(z), replace (a, mu) by logits
    (a + log(u), a + log(1-u)) and means (mu - (1-u)*d, mu + u*d).
    All other logits and means stay unchanged. The forward map has absolute Jacobian 2
    (from the log-width split) in these logit coordinates, including auxiliary (z, d, width contrast).

    Child log widths are the parent's log width plus/minus an independent
    Normal(0, split_log_std_scale) contrast. Fixed component labels give a
    unique inverse without route metadata. Weight and first moment are
    preserved; variance is not. Draws outside the mean prior receive zero target
    density; the auxiliary normal distribution is not truncated.
    """

    def _expand_particles(self, particles, context, rng):
        n, dim = particles.shape
        count = dim // 3
        logits = particles[:, :count]
        means = particles[:, count : 2 * count]
        log_stds = particles[:, 2 * count :]
        z = rng.normal(size=n)
        separation = rng.normal(
            scale=context.shared["split_mean_scale"], size=n
        )
        contrast = rng.normal(
            scale=context.shared.get("split_log_std_scale", 0.25), size=n
        )
        fraction = expit(z)
        left_logit = logits[:, -1] - np.logaddexp(0.0, -z)
        right_logit = logits[:, -1] - np.logaddexp(0.0, z)
        left_mean = means[:, -1] - expit(-z) * separation
        right_mean = means[:, -1] + fraction * separation
        return np.column_stack(
            (
                logits[:, :-1],
                left_logit,
                right_logit,
                means[:, :-1],
                left_mean,
                right_mean,
                log_stds[:, :-1],
                log_stds[:, -1] + contrast,
                log_stds[:, -1] - contrast,
            )
        )

    def inverse(self, particles, context):
        points = np.atleast_2d(particles)
        count = points.shape[1] // 3
        logits, means = points[:, :count], points[:, count : 2 * count]
        log_stds = points[:, 2 * count :]
        merged_logit = np.logaddexp(logits[:, -2], logits[:, -1])
        z = logits[:, -2] - logits[:, -1]
        merged_mean = expit(z) * means[:, -2] + expit(-z) * means[:, -1]
        result = np.column_stack(
            (
                logits[:, :-2],
                merged_logit,
                means[:, :-2],
                merged_mean,
                log_stds[:, :-2],
                0.5 * (log_stds[:, -2] + log_stds[:, -1]),
            )
        )
        return result[0] if particles.ndim == 1 else result

    def log_auxiliary_density(self, particles, context):
        points = np.atleast_2d(particles)
        count = points.shape[1] // 3
        z = points[:, count - 2] - points[:, count - 1]
        separation = points[:, 2 * count - 1] - points[:, 2 * count - 2]
        contrast = 0.5 * (points[:, -2] - points[:, -1])
        contrast_scale = context.shared.get("split_log_std_scale", 0.25)
        scale = context.shared["split_mean_scale"]
        values = -0.5 * (z**2 + (separation / scale) ** 2) - np.log(
            2 * np.pi * scale
        )
        values += -0.5 * (contrast / contrast_scale) ** 2 - np.log(
            np.sqrt(2 * np.pi) * contrast_scale
        )
        return values[0] if particles.ndim == 1 else values

    def log_jacobian(self, particles, context):
        # (log_sigma, contrast) -> (log_sigma + contrast, log_sigma - contrast).
        return np.log(2.0)


class GMMRandomSplitTransformation(GMMSplitTransformation):
    """Choose a parent uniformly and marginalise its label in the proposal.

    The retained child keeps the parent's label and the other child is appended.
    Thus each output has K possible parents, each with a inverse with absolute Jacobian 1/2.
    Its proposal density is the average of these K route densities. This sums
    parent choices in our labelled, variable-width model; it does not reorder
    components or sum over all pairs as in the paper's ordered model.
    """

    @staticmethod
    def _swap_parent(particles, parent, last):
        points = np.atleast_2d(particles).copy()
        count = points.shape[1] // 3
        rows = np.arange(points.shape[0])
        for offset in (0, count, 2 * count):
            points[rows, parent + offset], points[rows, last + offset] = (
                points[rows, last + offset].copy(),
                points[rows, parent + offset].copy(),
            )
        return points[0] if particles.ndim == 1 else points

    def _expand_particles(self, particles, context, rng):
        count = particles.shape[1] // 3
        parents = np.asarray(rng.choice(count, size=particles.shape[0]))
        reordered = self._swap_parent(particles, parents, count - 1)
        expanded = super()._expand_particles(reordered, context, rng)
        return self._swap_parent(expanded, parents, count - 1)

    def inverse(self, particles, context):
        raise NotImplementedError(
            "A random split has multiple inverses; use inverse_for_parent."
        )

    def inverse_for_parent(self, particles, context, parent):
        last_parent = particles.shape[-1] // 3 - 2
        reordered = self._swap_parent(particles, parent, last_parent)
        merged = super().inverse(reordered, context)
        return self._swap_parent(merged, parent, last_parent)

    def log_auxiliary_density(self, particles, context):
        raise NotImplementedError(
            "Use the marginal proposal from make_log_proposal."
        )

    def make_log_proposal(self, base_log_proposal, context):
        def log_proposal(particles, inner_context):
            count = particles.shape[-1] // 3 - 1
            routes = []
            for parent in range(count):
                old = self.inverse_for_parent(particles, inner_context, parent)
                reordered = self._swap_parent(particles, parent, count - 1)
                log_aux = super(
                    GMMRandomSplitTransformation, self
                ).log_auxiliary_density(reordered, inner_context)
                base = call_with_optional_context(
                    base_log_proposal, (old,), inner_context
                )
                routes.append(
                    base
                    + log_aux
                    - self.log_jacobian(particles, inner_context)
                )
            return logsumexp(np.stack(routes), axis=0) - np.log(count)

        return log_proposal


def log_density_ratio(particles: np.ndarray, context: StepContext):
    """Compute both mixture likelihoods in one pass for an expansion bridge.

    The new parameters' priors cancel against the auxiliary proposal. The
    Poisson prior on component count still contributes log(rate / new_count).
    The first model uses ordinary prior-to-posterior tempering.
    """
    current = int(context.metadata["active_components"])
    if context.metadata.get(LOG_PROPOSAL_OVERRIDE_KEY) is None:
        values = gmm_log_likelihood(particles, context)
    else:
        previous = int(context.metadata["previous_active_components"])
        if previous != current - 1 or current <= 1:
            raise ValueError(
                "The expansion bridge must add exactly one component."
            )
        likelihoods = _blocked_gmm_log_likelihoods(
            particles, context, (previous, current)
        )
        rate = float(context.shared.get("component_count_rate", 3.0))
        if rate <= 0.0:
            raise ValueError("component_count_rate must be positive.")
        values = (
            likelihoods[current]
            - likelihoods[previous]
            + np.log(rate / current)
        )
    return values[0] if particles.ndim == 1 else values


def posterior_summary(state: ParticleState) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior mean weights and component means."""
    particles = np.asarray(state.particles)
    if particles.ndim == 1:
        particles = particles[:, np.newaxis]
    weights = state.weights[:, np.newaxis]
    n_components = particles.shape[1] // 3

    logits = particles[:, :n_components]
    means = particles[:, n_components : 2 * n_components]
    component_weights = softmax(logits, axis=1)

    avg_weights = np.sum(weights * component_weights, axis=0)
    avg_weights /= avg_weights.sum()
    avg_means = np.sum(weights * means, axis=0)
    return avg_weights, avg_means


class ComponentMeansPlot:
    """Update a component-mean trace after each inner SMC iteration.

    Use one instance across the inner runs so history continues across model
    expansions even though their iteration counters restart. Only weighted
    mean summaries are retained; absent components are padded with NaN.
    """

    def __init__(self, filename="gmm_component_means.png"):
        self.filename = filename
        self.mean_history = []

    def __call__(self, state, context, results):
        _, means = posterior_summary(state)
        self.mean_history.append(means.copy())
        component_counts = [len(row) for row in self.mean_history]
        mean_history = np.full(
            (len(self.mean_history), max(component_counts)), np.nan
        )
        for iteration, row in enumerate(self.mean_history):
            mean_history[iteration, : len(row)] = row
        iterations = np.arange(len(self.mean_history))
        new_component_iters = np.flatnonzero(np.diff(component_counts) > 0) + 1
        fig, ax_means = plt.subplots(figsize=(10, 4))
        for component in range(mean_history.shape[1]):
            values = mean_history[:, component]
            (line,) = ax_means.plot(
                iterations,
                values,
                label=f"Component {component + 1}",
                linewidth=1.5,
            )
            first = np.flatnonzero(np.isfinite(values))[0]
            ax_means.scatter(
                iterations[first],
                values[first],
                color=line.get_color(),
                s=30,
                zorder=3,
            )
        for iteration in new_component_iters:
            ax_means.axvline(
                iteration, color="0.7", linestyle=":", linewidth=0.8
            )
        ax_means.set_title(
            "Labelled component means (dots mark introduction; labels may swap)"
        )
        ax_means.set_xlabel("SMC iteration")
        ax_means.set_ylabel("Posterior mean of component mean")
        ax_means.legend(loc="best", ncols=2)

        fig.tight_layout()
        fig.savefig(self.filename, dpi=150)
        plt.close(fig)


def mixture_density(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    std: float | np.ndarray,
) -> np.ndarray:
    """Evaluate mixture density on grid x."""
    variance = std**2
    normal = np.exp(
        -0.5 * ((x[:, np.newaxis] - means[np.newaxis, :]) ** 2) / variance
    ) / np.sqrt(2.0 * np.pi * variance)
    return normal @ weights


def posterior_predictive_density(x, state):
    """Average mixture densities, preserving modes when labels swap.

    Evaluating a mixture at averaged parameters is not the posterior
    predictive density: different particles can assign different labels to
    the same modes.
    """
    particles = np.asarray(state.particles)
    count = particles.shape[1] // 3
    component_weights = softmax(particles[:, :count], axis=1)
    means = particles[:, count : 2 * count]
    stds = np.exp(particles[:, 2 * count :])
    density = np.zeros_like(x, dtype=float)
    for particle_weight, weights, locations, widths in zip(
        state.weights, component_weights, means, stds
    ):
        density += particle_weight * mixture_density(
            x, weights, locations, widths
        )
    return density


def logging_str_fn(state: ParticleState, context: StepContext) -> str:
    beta = context.metadata["beta"]
    components = context.metadata["active_components"]
    ess = state.effective_sample_size()
    n_particles = state.log_weights.shape[0]
    ess_ratio = ess / n_particles if n_particles else 0.0
    log_evidence = state.log_evidence
    return (
        f"it {context.t}:, "
        f"Beta: {beta:.4f}, "
        f"Components: {components}, "
        f"ESS: {ess:.1f} ({ess_ratio:.3f}), "
        f"Log Evidence: {log_evidence:.1f}"
    )


class ExpandComponents(TransformationStep):
    """Increase the mixture order between outer SMC iterations."""

    def __init__(
        self,
        target_components: int,
        posterior: TemperedPosterior,
        transformation: SMCTransformation,
    ):
        super().__init__(
            transformation=transformation,
            posterior=posterior,
        )
        self._target_components = target_components

    def apply(
        self,
        state: ParticleState,
        context: StepContext,
        rng: RandomGenerator,
    ) -> tuple[ParticleState, StepContext, dict[str, Any]]:
        current_components = int(context.metadata["active_components"])

        if context.t == 0:
            return state, context, {}

        if current_components >= self._target_components:
            return state, context.request_stop(), {}

        # Each completed inner run ends at the full current-model posterior.
        # Evaluate that endpoint directly, avoiding recursive evaluation of
        # all earlier marginalised bridges when forming the next proposal.
        previous_context = context.with_metadata(
            active_components=current_components
        )

        def previous_log_posterior(particles):
            return log_prior(particles, previous_context) + gmm_log_likelihood(
                particles, previous_context
            )

        context = context.with_metadata(
            **{LOG_PROPOSAL_OVERRIDE_KEY: previous_log_posterior}
        )
        return super().apply(state, context, rng)


def main(
    *,
    n_particles=500,
    n_samples=400,
    target_components=6,
    true_components=3,
    mutation_steps=10,
    expansion="prior",
    plot=True,
):
    if true_components < 1:
        raise ValueError("true_components must be at least 1.")
    if target_components < 1:
        raise ValueError("target_components must be at least 1.")
    transformations = {
        "prior": GMMExpansionTransformation,
        "split": GMMSplitTransformation,
        "split-random": GMMRandomSplitTransformation,
    }
    if expansion not in transformations:
        raise ValueError(
            "expansion must be 'prior', 'split', or 'split-random'."
        )
    transformation = transformations[expansion]()
    configure_logging()

    rng = np.random.default_rng(seed=2048)

    if true_components == 3:
        # Preserve the original example's data-generating mixture.
        true_weights = np.array([0.2, 0.5, 0.3])
        true_means = np.array([-4.0, 0.5, 3.5])
    else:
        # Keep modes separated as the true component count increases.
        true_weights = np.full(true_components, 1.0 / true_components)
        true_means = 3.75 * (
            np.arange(true_components) - (true_components - 1) / 2
        )
    true_stds = (
        np.array([0.5, 0.75, 1.0])
        if true_components == 3
        else np.linspace(0.5, 1.0, true_components)
    )
    split_mean_scale = 0.75  # Separation scale for split proposals only.
    observations = generate_gmm_data(
        true_weights,
        true_means,
        std=true_stds,
        n_samples=n_samples,
        rng=rng,
    )

    mean_bounds = (
        observations.min() - 3.0 * max(true_stds),
        observations.max() + 3.0 * max(true_stds),
    )

    logit_prior_scale = 2.0
    initial_logits = rng.normal(
        loc=0.0, scale=logit_prior_scale, size=(n_particles, 1)
    )
    initial_means = rng.uniform(
        low=mean_bounds[0], high=mean_bounds[1], size=(n_particles, 1)
    )
    initial_log_stds = rng.normal(
        loc=np.log(0.75), scale=0.75, size=(n_particles, 1)
    )
    initial_particles = np.concatenate(
        [initial_logits, initial_means, initial_log_stds], axis=1
    )
    initial_state = ParticleState(
        particles=initial_particles,
        metadata={"active_components": 1},
    )

    posterior = TemperedPosterior(
        log_prior=log_prior,
        log_likelihood=gmm_log_likelihood,
        # Prior cancellation applies only to prior-draw births. For splits,
        # use the full transformed proposal, including its auxiliary density.
        log_density_ratio=log_density_ratio if expansion == "prior" else None,
    )

    # Share the callback across inner runs to retain the full expansion history.
    mean_plot = (
        ComponentMeansPlot(filename=f"gmm_component_means_{expansion}.png")
        if plot
        else None
    )

    # The inner loop anneals and rejuvenates particles for one model size.
    inner_runner = SequentialMonteCarlo(
        steps=[
            AdaptiveTemperedWeighting(
                posterior,
                target_ess_ratio=0.6,
                max_beta=1.0,
                min_delta=1e-8,
            ),
            SystematicResampling(always_resample=True),
            DifferentialEvolutionMutation(posterior, n_steps=mutation_steps),
            LoggingStep(logging_str_fn),
        ],
        callbacks=[mean_plot] if mean_plot is not None else [],
    )

    shared_context = {
        "data": observations,
        "split_mean_scale": split_mean_scale,
        "log_std_prior_mean": np.log(0.75),
        "log_std_prior_scale": 0.75,
        "split_log_std_scale": 0.25,
        "mean_bounds": mean_bounds,
        "logit_prior_scale": logit_prior_scale,
        "component_count_rate": 3.0,
        "likelihood_block_size": 256,
    }

    # The outer loop adds one component, runs the inner loop, then repeats.
    outer_runner = SequentialMonteCarlo(
        steps=[
            ExpandComponents(target_components, posterior, transformation),
            MetaSMCStep(
                inner_runner,
                posterior=posterior,
                record_outputs=True,
                output_key="inner_results",
            ),
            StopIf(
                lambda state, context: (
                    context.metadata["active_components"] >= target_components
                )
            ),
        ]
    )

    metadata = {"active_components": 1}

    outer_result = outer_runner.run(
        initial_state,
        rng=rng,
        shared_context=shared_context,
        metadata=metadata,
    )
    meta_result = outer_result.get_meta_results("inner_results")
    if meta_result is None:
        raise RuntimeError("The meta SMC step did not record inner results.")

    all_states = meta_result.all_states
    final_state = meta_result.final_state
    est_weights, est_means = posterior_summary(final_state)
    print("Posterior mixture weights:", np.round(est_weights, 3))
    print("Posterior component means:", np.round(est_means, 3))
    count = final_state.particles.shape[1] // 3
    est_stds = np.average(
        np.exp(final_state.particles[:, 2 * count :]),
        axis=0,
        weights=final_state.weights,
    )
    print("Posterior component widths:", np.round(est_stds, 3))
    print(f"Final log evidence: {final_state.log_evidence:.2f}")

    iterations = np.arange(len(all_states))
    component_counts = [
        s.metadata.get("active_components", s.particles.shape[1] // 3)
        for s in all_states
    ]
    betas = [s.metadata.get("beta", 0.0) for s in all_states]
    delta_betas = [s.metadata.get("delta_beta", 0.0) for s in all_states]

    x_grid = np.linspace(mean_bounds[0], mean_bounds[1], 400)
    true_density = mixture_density(x_grid, true_weights, true_means, true_stds)
    posterior_density = posterior_predictive_density(x_grid, final_state)
    rmse = np.sqrt(np.mean((posterior_density - true_density) ** 2))
    print(f"Density RMSE vs. true mixture: {rmse:.4f}")

    new_component_iters = np.where(np.diff(component_counts) > 0)[0] + 1
    if new_component_iters.size > 0:
        print(
            "New components introduced after iterations:",
            new_component_iters.tolist(),
        )
        precursor_evidence = [
            {
                "iteration": int(iter_idx - 1),
                "components": int(component_counts[iter_idx - 1]),
                "log_evidence": float(all_states[iter_idx - 1].log_evidence),
            }
            for iter_idx in new_component_iters
            if iter_idx > 0
        ]
        print("Pre-expansion log evidence:", precursor_evidence)
    else:
        print("No component expansions detected during run.")

    evidence_by_components: dict[int, float] = {}
    for inter_result in meta_result.results:
        comp = int(
            inter_result.final_state.metadata.get(
                "active_components",
                inter_result.final_state.particles.shape[1] // 3,
            )
        )
        evidence_by_components[comp] = float(
            inter_result.final_state.log_evidence
        )
    if evidence_by_components:
        print("Log evidence by active component count:")
        for comp in sorted(evidence_by_components):
            print(
                f"  components={comp}: log_evidence={evidence_by_components[comp]:.3f}"
            )

    log_bayes_factors = {}
    base_evidence = evidence_by_components.get(1, None)
    if base_evidence is not None:
        for comp, log_evid in evidence_by_components.items():
            if comp == 1:
                continue
            log_bayes_factors[comp] = log_evid - base_evidence
        print("Bayes factors vs. 1-component model:")
        for comp in sorted(log_bayes_factors):
            print(f"  components={comp}: ln BF={log_bayes_factors[comp]:.3f}")
        best_comp = max(
            evidence_by_components, key=lambda c: evidence_by_components[c]
        )
        print(f"Favoured model by log evidence: {best_comp} components")

    if not plot:
        return meta_result

    fig, axes = plt.subplots(nrows=3, figsize=(10, 9))

    axes[0].hist(
        observations,
        bins=30,
        density=True,
        alpha=0.4,
        color="tab:gray",
        label="Observations",
    )
    axes[0].plot(
        x_grid,
        posterior_density,
        color="tab:blue",
        linewidth=2.0,
        label="Posterior predictive",
    )
    axes[0].plot(
        x_grid,
        true_density,
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
        label="True mixture",
    )
    axes[0].set_title("Adaptive mixture fit")
    axes[0].set_xlabel("Observation")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left")

    axes[1].step(
        iterations,
        component_counts,
        where="post",
        color="tab:purple",
        linewidth=2.0,
        label="Active components",
    )
    axes[1].scatter(iterations, component_counts, color="tab:purple", s=15)
    axes[1].set_title("Component count over time")
    axes[1].set_xlabel("SMC iteration")
    axes[1].set_ylabel("Components")
    axes[1].legend(loc="upper left")

    ax_beta = axes[2]
    ax_beta.plot(
        iterations,
        betas,
        color="tab:cyan",
        linewidth=2.0,
        label="Beta",
    )
    ax_delta = ax_beta.twinx()
    ax_delta.step(
        iterations,
        delta_betas,
        where="post",
        color="tab:orange",
        linewidth=1.5,
        label="Beta increment",
    )
    ax_beta.set_title("Adaptive annealing")
    ax_beta.set_xlabel("SMC iteration")
    ax_beta.set_ylabel("Beta")
    ax_delta.set_ylabel("Increment")
    ax_beta.legend(loc="upper left")
    ax_delta.legend(loc="upper right")

    fig.tight_layout()
    filename = f"gmm_transdimensional_{expansion}.png"
    fig.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close(fig)
    return meta_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expansion",
        choices=("prior", "split", "split-random"),
        default="prior",
    )
    parser.add_argument(
        "--max-components",
        "--target-components",
        dest="target_components",
        type=int,
        default=6,
        help="Maximum number of fitted components (default: 6; starts at 1).",
    )
    parser.add_argument(
        "--true-components",
        type=int,
        default=3,
        help="Number of data-generating components (default: 3). Other counts use equal weights and means spaced 3.75 apart.",
    )
    args = parser.parse_args()
    if args.true_components < 1:
        parser.error("--true-components must be at least 1")
    if args.target_components < 1:
        parser.error("--max-components must be at least 1")
    main(
        expansion=args.expansion,
        target_components=args.target_components,
        true_components=args.true_components,
    )
