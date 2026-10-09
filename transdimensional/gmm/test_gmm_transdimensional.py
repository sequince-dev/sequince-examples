"""Regression checks for component-specific widths and expansion proposals."""

from unittest.mock import Mock

import numpy as np
import pytest
from gmm_transdimensional import (
    GMMExpansionTransformation,
    GMMRandomSplitTransformation,
    GMMSplitTransformation,
    _blocked_gmm_log_likelihoods,
    gmm_log_likelihood,
    log_density_ratio,
    log_prior,
    main,
    posterior_predictive_density,
    posterior_summary,
)
from scipy.special import logsumexp, softmax
from scipy.stats import norm
from sequince import ParticleState, TemperedPosterior
from sequince.steps import StepContext
from sequince.steps.context import LOG_PROPOSAL_OVERRIDE_KEY


def context(count=2):
    return StepContext(
        t=0,
        shared=dict(
            data=np.array([-2.0, 0.1, 1.0]),
            mean_bounds=(-10.0, 10.0),
            split_mean_scale=0.75,
            logit_prior_scale=2.0,
            likelihood_block_size=2,
        ),
        metadata={"active_components": count, "beta": 1.0},
    )


def test_likelihood_and_prediction_with_distinct_widths():
    points = np.array(
        [[np.log(0.25), np.log(0.75), -2.0, 2.0, np.log(0.3), np.log(1.2)]]
    )
    swapped = points[:, [1, 0, 3, 2, 5, 4]].copy()
    swapped[:, :2] += 10
    state = ParticleState(
        np.concatenate([points, swapped]), log_weights=np.log([0.8, 0.2])
    )
    ctx = context()
    data = ctx.shared["data"]
    expected = 0.25 * norm.pdf(data, -2.0, 0.3) + 0.75 * norm.pdf(
        data, 2.0, 1.2
    )
    np.testing.assert_allclose(
        gmm_log_likelihood(points, ctx), np.log(expected).sum()
    )
    np.testing.assert_allclose(
        gmm_log_likelihood(points[0], ctx), np.log(expected).sum()
    )
    np.testing.assert_allclose(
        posterior_predictive_density(data, state), expected
    )
    np.testing.assert_allclose(posterior_summary(state)[0], [0.35, 0.65])
    both = _blocked_gmm_log_likelihoods(points, ctx, (1, 2))
    np.testing.assert_allclose(both[1], norm.logpdf(data, -2.0, 0.3).sum())


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize(
    "cls", [GMMExpansionTransformation, GMMSplitTransformation]
)
def test_expansion_inverse_density_and_jacobian(count, cls):
    ctx = context(count)
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(5, 3 * count))
    transform = cls()
    state, nctx = transform.forward_with_context(ParticleState(pts), ctx, rng)
    x = state.particles
    np.testing.assert_allclose(transform.inverse(x, nctx), pts, atol=1e-14)
    np.testing.assert_allclose(
        transform.inverse(x[0], nctx), pts[0], atol=1e-14
    )
    if cls is GMMSplitTransformation:
        k = count + 1
        old_w = softmax(pts[:, :count], axis=1)
        new_w = softmax(x[:, :k], axis=1)
        np.testing.assert_allclose(new_w[:, -2:].sum(axis=1), old_w[:, -1])
        np.testing.assert_allclose(
            (new_w * x[:, k : 2 * k]).sum(axis=1),
            (old_w * pts[:, count : 2 * count]).sum(axis=1),
        )

        def inverse_with_aux(y):
            aux = [
                y[k - 2] - y[k - 1],
                y[2 * k - 1] - y[2 * k - 2],
                0.5 * (y[-2] - y[-1]),
            ]
            return np.r_[transform.inverse(y, nctx), aux]

        eps = 1e-5
        jac = np.column_stack(
            [
                (inverse_with_aux(x[0] + e) - inverse_with_aux(x[0] - e))
                / (2 * eps)
                for e in eps * np.eye(len(x[0]))
            ]
        )
        np.testing.assert_allclose(abs(np.linalg.det(jac)), 0.5, rtol=1e-8)
        z, d, v = inverse_with_aux(x[0])[-3:]
        expected = (
            norm.logpdf(z)
            + norm.logpdf(d, scale=0.75)
            + norm.logpdf(v, scale=0.25)
        )
        np.testing.assert_allclose(
            transform.log_auxiliary_density(x[0], nctx), expected
        )
    posterior = TemperedPosterior(log_prior, gmm_log_likelihood)
    proposal = transform.make_log_proposal(
        posterior.freeze_log_prob(ctx), nctx
    )
    expected = (
        posterior.log_prob(pts, ctx)
        + transform.log_auxiliary_density(x, nctx)
        - transform.log_jacobian(x, nctx)
    )
    np.testing.assert_allclose(proposal(x, nctx), expected)
    if cls is GMMExpansionTransformation:
        nctx = nctx.with_metadata(**{LOG_PROPOSAL_OVERRIDE_KEY: proposal})
        np.testing.assert_allclose(
            log_density_ratio(x, nctx),
            posterior.log_increment(x, nctx),
            atol=1e-11,
        )


def test_random_split_marginal_density():
    ctx = context(3)
    transform = GMMRandomSplitTransformation()
    pts = np.array([[0.1, -0.2, 0.4, -3.0, 0.0, 3.0, -0.2, 0.1, 0.3]] * 3)
    rng = Mock()
    rng.choice.return_value = np.arange(3)
    rng.normal.side_effect = [
        np.array([-0.3, 0.2, 0.6]),
        np.array([0.4, -0.2, 0.8]),
        np.array([0.1, -0.2, 0.3]),
    ]
    expanded = transform._expand_particles(pts, ctx, rng)
    for i in range(3):
        np.testing.assert_allclose(
            transform.inverse_for_parent(expanded[i], ctx, i),
            pts[i],
            atol=1e-14,
        )

    def base(x):
        return -0.5 * np.sum(x * x, axis=-1)

    routes = []
    for parent in range(3):
        a, mu, s = np.split(expanded, 3, axis=1)
        old_a, old_mu, old_s = (
            a[:, :3].copy(),
            mu[:, :3].copy(),
            s[:, :3].copy(),
        )
        merged = np.logaddexp(a[:, parent], a[:, -1])
        u = np.exp(a[:, parent] - merged)
        old_a[:, parent] = merged
        old_mu[:, parent] = u * mu[:, parent] + (1 - u) * mu[:, -1]
        old_s[:, parent] = 0.5 * (s[:, parent] + s[:, -1])
        z = a[:, parent] - a[:, -1]
        d = mu[:, -1] - mu[:, parent]
        v = 0.5 * (s[:, parent] - s[:, -1])
        aux = (
            norm.logpdf(z)
            + norm.logpdf(d, scale=0.75)
            + norm.logpdf(v, scale=0.25)
        )
        routes.append(
            base(np.column_stack([old_a, old_mu, old_s])) + aux - np.log(2)
        )
    expected = logsumexp(routes, axis=0) - np.log(3)
    proposal = transform.make_log_proposal(base, ctx)
    np.testing.assert_allclose(proposal(expanded, ctx), expected)
    np.testing.assert_allclose(proposal(expanded[0], ctx), expected[0])


@pytest.mark.parametrize("expansion", ["prior", "split", "split-random"])
def test_small_run(expansion):
    result = main(
        n_particles=64,
        n_samples=20,
        target_components=3,
        mutation_steps=2,
        expansion=expansion,
        plot=False,
    )
    assert result.final_state.particles.shape == (64, 9)
    assert np.isfinite(result.final_state.log_evidence)
    assert result.final_state.metadata["beta"] == 1.0
    assert np.std(result.final_state.particles[:, 6:]) > 0
