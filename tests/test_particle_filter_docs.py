"""Check filtering documentation exports without running either filter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("documentation", [False, True])
def test_particle_filter_outputs_without_sampling(
    documentation: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("sequince")
    source = (
        Path(__file__).resolve().parents[1]
        / "examples/particle_filtering/particle_filter.py"
    )
    spec = importlib.util.spec_from_file_location("filter_example", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    calls = []
    comparison = SimpleNamespace(
        proposal_means=np.array([1.0, 2.0]),
        bootstrap_means=np.array([0.0, 3.0]),
        kalman_means=np.array([1.0, 1.0]),
        proposal_variances=np.array([1.0, 1.0]),
        bootstrap_variances=np.array([1.0, 2.0]),
        kalman_variances=np.array([1.0, 1.0]),
        proposal_ess=np.array([10.0, 20.0]),
        bootstrap_ess=np.array([5.0, 10.0]),
        proposal_log_evidence=-1.0,
        bootstrap_log_evidence=-2.0,
        kalman_log_evidence=-0.5,
    )
    monkeypatch.setattr(
        module,
        "run_comparison",
        lambda **kwargs: (calls.append(kwargs) or comparison),
    )
    monkeypatch.setattr(
        module,
        "plot_comparison",
        lambda result, output: (output.write_bytes(b"mock plot")),
    )
    requested = tmp_path / "custom.png"
    monkeypatch.setattr(sys, "argv", [str(source), "--output", str(requested)])
    if documentation:
        module.documentation_output = tmp_path
    module.main()
    assert calls == [
        {
            "n_observations": 20,
            "n_particles": 1000,
            "seed": 2026,
            "always_resample": True,
            "ess_threshold": 0.5,
        }
    ]
    if documentation:
        assert (tmp_path / "particle_filter.png").is_file()
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["log_evidence"]["kalman"] == -0.5
        assert summary["mean_ess_before_resampling"]["optimal"] == 15.0
    else:
        assert requested.is_file()
        assert not (tmp_path / "summary.json").exists()
