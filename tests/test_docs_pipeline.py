"""Exercise provenance and transport failures independently of sampling."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import docs_pipeline as pipeline  # noqa: E402


@pytest.fixture
def outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make a complete, checksummed output set for the current source."""
    monkeypatch.setattr(pipeline, "OUTPUT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "git", lambda *args: "a" * 40)
    for example in pipeline.examples():
        directory = pipeline.OUTPUT / example["id"]
        directory.mkdir(parents=True)
        for name in example["figures"]:
            (directory / name).write_bytes(b"plot")
        pipeline.write_json(directory / "summary.json", {"log_evidence": -1.0})
        pipeline.write_json(
            directory / "metadata.json",
            {
                "schema": 1,
                "example": example["id"],
                "source_commit": "a" * 40,
                "source_hashes": pipeline.source_hashes(example),
                "sequince_version": "0.1.dev1",
                "sequince_commit": pipeline.dependency_commit(),
                "git_dependencies": pipeline.git_dependencies(example),
                "dependency_versions": {
                    name: "test-version"
                    for name in pipeline.git_dependencies(example)
                },
                "environment": example["environment"],
                "seed": 42,
                "parameters": {key: 42 for key in example["parameters"]},
            },
        )
        pipeline.write_json(
            directory / "checksums.json",
            {path.name: pipeline.digest(path) for path in directory.iterdir()},
        )
    return pipeline.OUTPUT


def test_complete_outputs(outputs: Path) -> None:
    pipeline.validate(outputs)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "commit", "input"])
def test_invalid_outputs_fail(outputs: Path, failure: str) -> None:
    directory = outputs / "gaussian"
    if failure == "missing":
        (directory / "corner.png").unlink()
    elif failure == "corrupt":
        (directory / "corner.png").write_bytes(b"corrupt")
    else:
        path = directory / "metadata.json"
        metadata = json.loads(path.read_text())
        if failure == "commit":
            metadata["source_commit"] = "b" * 40
        else:
            metadata["source_hashes"]["pixi.lock"] = "wrong"
        pipeline.write_json(path, metadata)
        checksums = json.loads((directory / "checksums.json").read_text())
        checksums[path.name] = pipeline.digest(path)
        pipeline.write_json(directory / "checksums.json", checksums)
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.validate(outputs)


def test_bundle_roundtrip(
    outputs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "BUNDLES", tmp_path)
    path = pipeline.bundle()
    checksum = pipeline.digest(path)
    assert pipeline.digest(pipeline.bundle()) == checksum
    destination = tmp_path / "restored"
    pipeline.unpack(path, "a" * 40, destination)
    assert (destination / "gaussian/summary.json").read_bytes() == (
        outputs / "gaussian/summary.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="commit mismatch"):
        pipeline.unpack(path, "b" * 40, tmp_path / "wrong")
    path.write_bytes(b"existing different bundle")
    with pytest.raises(ValueError, match="Refusing to replace"):
        pipeline.bundle()


def test_unsafe_archive_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="Unsafe archive"):
        pipeline.unpack(path, "a" * 40, tmp_path / "output")
    assert not (tmp_path / "escape").exists()


def test_latest_checkout_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("READTHEDOCS_VERSION", "latest")
    monkeypatch.setattr(
        pipeline,
        "git",
        lambda *args: (
            "b" * 40 + "\trefs/heads/main"
            if args[0] == "ls-remote"
            else "a" * 40
        ),
    )
    with pytest.raises(ValueError, match="Stale latest checkout"):
        pipeline.check_latest()


def test_missing_output_root_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing example outputs:.*gaussian"):
        pipeline.validate(tmp_path / "absent")


def test_missing_example_is_named(outputs: Path) -> None:
    import shutil

    shutil.rmtree(outputs / "adaptive-mutation")
    with pytest.raises(
        ValueError, match="missing example outputs: adaptive-mutation"
    ):
        pipeline.validate(outputs)


def test_unexpected_output_is_named(outputs: Path) -> None:
    (outputs / "stray.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="unexpected entries: stray.txt"):
        pipeline.validate(outputs)


def test_prepare_preserves_narrative_and_source_paths(
    outputs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "generated"
    monkeypatch.setattr(pipeline, "GENERATED", generated)
    pipeline.prepare()
    for example in pipeline.examples():
        for key in ("script", "narrative"):
            relative = example[key]
            assert (generated / relative).read_bytes() == (
                pipeline.ROOT / relative
            ).read_bytes()
        assert (
            generated / "results" / example["id"] / "summary.json"
        ).is_file()
    assert not (generated / "index.rst").exists()
    for example in pipeline.examples():
        narrative = (generated / example["narrative"]).read_text()
        for name in example["figures"]:
            assert (
                f".. figure:: /_generated/results/{example['id']}/{name}"
                in narrative
            )
        assert "Numerical summary" not in narrative
        assert "/summary.json" not in narrative


@pytest.mark.parametrize(
    "name",
    [
        "flow-preconditioning",
        "persistent-sampling",
        "waste-free",
        "particle-filtering",
    ],
)
def test_prepared_worker_without_sampling(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise adapters and provenance using a fake main, never a sampler."""
    from types import SimpleNamespace

    example = pipeline.entry(name)
    parameters = {key: 42 for key in example["parameters"]}
    module = SimpleNamespace(**parameters)
    if name in {"persistent-sampling", "particle-filtering"}:
        module.build_parser = lambda: SimpleNamespace(
            parse_args=lambda argv: SimpleNamespace(**parameters)
        )
    original_argv = sys.argv

    def main() -> None:
        assert sys.argv == [str(pipeline.ROOT / example["script"])]
        for figure in example["figures"]:
            (module.documentation_output / figure).write_bytes(b"mock plot")
        pipeline.write_json(
            module.documentation_output / "summary.json",
            {
                "log_evidence": -1.0,
            },
        )

    module.main = main
    monkeypatch.setitem(sys.modules, "documentation_example", None)
    spec = SimpleNamespace(
        name="documentation_example",
        loader=SimpleNamespace(exec_module=lambda module: None),
    )
    monkeypatch.setattr(
        pipeline.importlib.util, "spec_from_file_location", lambda *args: spec
    )
    monkeypatch.setattr(
        pipeline.importlib.util, "module_from_spec", lambda spec: module
    )
    revisions = pipeline.git_dependencies(example)
    monkeypatch.setattr(
        pipeline.importlib.metadata,
        "distribution",
        lambda package: SimpleNamespace(
            version="test-version",
            read_text=lambda filename: json.dumps(
                {"vcs_info": {"commit_id": revisions[package]}}
            ),
        ),
    )
    monkeypatch.setattr(pipeline, "git", lambda *args: "a" * 40)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    destination = tmp_path / "outputs"
    pipeline.worker(name, tmp_path / "raw", destination)
    assert sys.argv is original_argv
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["parameters"] == parameters
    assert metadata["git_dependencies"] == revisions
    for figure in example["figures"]:
        assert (destination / figure).read_bytes() == b"mock plot"
    assert (
        json.loads((destination / "summary.json").read_text())["log_evidence"]
        == -1.0
    )


def test_bundle_restores_execution_lock(
    outputs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RTD restores the execution snapshot even when its checkout lock differs."""
    import shutil

    monkeypatch.setattr(pipeline, "BUNDLES", tmp_path)
    archive = pipeline.bundle()
    checkout = tmp_path / "checkout"
    source_paths = set()
    for example in pipeline.examples():
        source_paths.update(pipeline.source_hashes(example))
    for relative in source_paths:
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pipeline.ROOT / relative, target)
    archived_lock = (checkout / "pixi.lock").read_bytes()
    (checkout / "pixi.lock").write_bytes(
        archived_lock + b"\n# Different checkout lock\n"
    )
    monkeypatch.setattr(pipeline, "ROOT", checkout)
    with pytest.raises(ValueError, match="execution lock differs"):
        pipeline.unpack(archive, "a" * 40, tmp_path / "refused")
    pipeline.unpack(
        archive, "a" * 40, tmp_path / "restored", restore_lock=True
    )
    assert (checkout / "pixi.lock").read_bytes() == archived_lock


def test_execution_batch_refreshes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All child executions share one main resolution instead of racing updates."""
    calls = []
    monkeypatch.delenv("SEQUINCE_LOCK_READY", raising=False)
    monkeypatch.setattr(sys, "argv", ["docs_pipeline.py", "run-all"])
    monkeypatch.setattr(pipeline, "refresh", lambda: calls.append("refresh"))
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda command, **kwargs: calls.append(kwargs),
    )
    pipeline.main()
    assert calls[0] == "refresh"
    assert len(calls) == len(pipeline.examples()) + 1
    assert all(call["env"]["SEQUINCE_LOCK_READY"] == "1" for call in calls[1:])
