"""Check the flow example's host summary without constructing a sampler."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_flow_documentation_summary(backend: str) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("sequince_density")
    from sequince import ParticleState
    from sequince.result import SMCResult
    from sequince.steps import StepContext

    path = (
        Path(__file__).resolve().parents[1]
        / "examples/preconditioning/flow_preconditioning.py"
    )
    spec = importlib.util.spec_from_file_location("flow_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    particles = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    log_weights = np.log([0.5, 0.25, 0.25])
    if backend == "torch":
        particles = torch.as_tensor(particles)
        log_weights = torch.as_tensor(log_weights)
    state = ParticleState(particles, log_weights=log_weights)
    result = SMCResult([state], StepContext(t=0))

    summary = module._documentation_summary(result)

    np.testing.assert_allclose(summary["weighted_mean"], [0.5, 1.0])
    np.testing.assert_allclose(
        summary["weighted_population_covariance"],
        [[0.75, -0.5], [-0.5, 3.0]],
    )
    assert summary["n_particles"] == 3
