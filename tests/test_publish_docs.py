"""Check publication ordering and refusal paths without external writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_docs as publisher  # noqa: E402


@pytest.fixture
def publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[tuple]:
    """Replace GitHub and RTD calls with a recorded successful publication."""
    calls: list[tuple] = []
    sha = "a" * 40
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    monkeypatch.setattr(
        publisher, "git", lambda *args: ("" if args[0] == "diff" else sha)
    )
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"validated outputs")
    Path(str(bundle) + ".sha256").write_text("checksum")
    monkeypatch.setattr(publisher, "bundle", lambda: bundle)
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *args, **kwargs: (SimpleNamespace(returncode=1, stdout="")),
    )
    monkeypatch.setattr(
        publisher, "gh", lambda *args: calls.append(args) or ""
    )
    monkeypatch.setenv("RTD_PROJECT", "examples")

    def api(path: str, method: str = "GET") -> dict:
        calls.append((method, path))
        if "running=true" in path:
            return {"count": 0}
        if method == "POST":
            return {"build": {"id": 123}}
        return {"state": {"code": "finished"}, "success": True, "commit": sha}

    monkeypatch.setattr(publisher, "rtd", api)
    return calls


def test_publish_before_trigger(publication: list[tuple]) -> None:
    publisher.main()
    uploaded = publication.index(
        (
            "release",
            "edit",
            "docs-output-" + "a" * 40,
            "--draft=false",
            "--latest=false",
        )
    )
    triggered = publication.index(
        (
            "POST",
            "projects/examples/versions/latest/builds/",
        )
    )
    assert uploaded < triggered
    assert publication[-1] == ("GET", "projects/examples/builds/123/")


def test_stale_run_does_not_publish(
    publication: list[tuple], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        publisher,
        "git",
        lambda *args: ("b" * 40 if args[0] == "ls-remote" else "a" * 40),
    )
    publisher.main()
    assert not publication


def test_different_existing_asset_fails(
    publication: list[tuple], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *args, **kwargs: (
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "assets": [{"name": "bundle.tar.gz"}],
                        "isDraft": False,
                    }
                ),
            )
        ),
    )
    (tmp_path / "build-docs").mkdir()

    def download(*args: str) -> str:
        publication.append(args)
        target = Path(args[args.index("--dir") + 1]) / "bundle.tar.gz"
        target.write_bytes(b"different uploaded content")
        return ""

    monkeypatch.setattr(publisher, "gh", download)
    with pytest.raises(ValueError, match="Refusing to replace"):
        publisher.main()
    assert not any(call[0] == "POST" for call in publication)


def test_wait_idle_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(publisher, "rtd", lambda path: {"count": 1})
    monkeypatch.setattr(publisher.time, "monotonic", lambda: 10.0)
    with pytest.raises(TimeoutError, match="RTD still running"):
        publisher.wait_idle("examples", 0.0)
