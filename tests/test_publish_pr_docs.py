"""Exercise PR publication without credentials, sampling or external writes."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_pr_docs as publisher  # noqa: E402

SHA = "a" * 40


def make_bundle(path: Path, sha: str = SHA, unsafe: bool = False) -> None:
    """Write a small archive with the real transport manifest format."""
    files = {
        "execution-pixi.lock": b"lock",
        "gaussian/metadata.json": json.dumps({"source_commit": sha}).encode(),
    }
    if unsafe:
        files["../escape"] = b"bad"
    files["bundle-manifest.json"] = json.dumps(
        {
            "schema": 1,
            "source_commit": sha,
            "files": {
                name: hashlib.sha256(data).hexdigest()
                for name, data in files.items()
            },
        }
    ).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("failure", [None, "commit", "path", "size"])
def test_archive_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    path = tmp_path / "outputs.tar.gz"
    make_bundle(
        path, "b" * 40 if failure == "commit" else SHA, failure == "path"
    )
    if failure == "size":
        monkeypatch.setattr(publisher, "MAX_BYTES", 1)
    if failure:
        with pytest.raises(ValueError):
            publisher.verify_bundle(path, SHA)
    else:
        publisher.verify_bundle(path, SHA)
    assert not (tmp_path.parent / "escape").exists()


@pytest.mark.parametrize("failure", [None, "zip-path", "checksum"])
def test_download_from_exact_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    source = tmp_path / f"docs-outputs-{SHA}.tar.gz"
    make_bundle(source)
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        name = "../escape" if failure == "zip-path" else source.name
        archive.writestr(name, source.read_bytes())
        archive.writestr(
            source.name + ".sha256",
            f"{'wrong' if failure == 'checksum' else publisher.digest(source)}"
            f"  {source.name}\n",
        )

    def api(path: str) -> dict:
        assert path == "actions/runs/123/artifacts?per_page=100"
        return {
            "artifacts": [
                {
                    "id": 456,
                    "name": "validated-bundle",
                    "expired": False,
                    "size_in_bytes": len(zipped.getvalue()),
                }
            ]
        }

    def download(args: list, **kwargs: object) -> None:
        assert args[-1].endswith("actions/artifacts/456/zip")
        kwargs["stdout"].write(zipped.getvalue())

    monkeypatch.setattr(publisher, "api", api)
    monkeypatch.setattr(publisher.subprocess, "run", download)
    destination = tmp_path / "download"
    destination.mkdir()
    if failure:
        with pytest.raises(ValueError):
            publisher.download_bundle(123, SHA, destination)
    else:
        result = publisher.download_bundle(123, SHA, destination)
        assert result.read_bytes() == source.read_bytes()
    assert not (tmp_path / "escape").exists()


@pytest.fixture
def publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list:
    """Record the separate trusted publication path for a successful PR run."""
    calls = []
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"workflow_run": {"id": 123}}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("RTD_PROJECT", "examples")
    monkeypatch.setattr(
        publisher,
        "api",
        lambda path: {
            "event": "pull_request",
            "conclusion": "success",
            "path": ".github/workflows/documentation.yml",
            "repository": {"full_name": publisher.REPOSITORY},
            "head_sha": SHA,
        },
    )
    monkeypatch.setattr(publisher, "current_pr", lambda sha: {"number": 17})
    monkeypatch.setattr(publisher, "git", lambda *args: "b" * 40)
    monkeypatch.setattr(
        publisher,
        "download_bundle",
        lambda run_id, sha, directory: calls.append(("download", run_id, sha))
        or directory / "bundle.tar.gz",
    )
    monkeypatch.setattr(
        publisher,
        "publish_assets",
        lambda path, tag, target: calls.append(("publish", tag, target)),
    )
    monkeypatch.setattr(
        publisher,
        "wait_idle",
        lambda *args: calls.append(("wait",)),
    )
    monkeypatch.setattr(
        publisher,
        "build_version",
        lambda project, version, sha, deadline: calls.append(
            ("build", project, version, sha)
        ),
    )
    return calls


def test_publish_pr_before_preview(publication: list) -> None:
    publisher.main()
    assert publication == [
        ("download", 123, SHA),
        ("publish", f"docs-pr-17-{SHA}", "b" * 40),
        ("wait",),
        ("build", "examples", "17", SHA),
    ]


@pytest.mark.parametrize("advance_at", [0, 1, 2])
def test_stale_or_closed_pr(
    publication: list, monkeypatch: pytest.MonkeyPatch, advance_at: int
) -> None:
    responses = iter([{"number": 17}] * advance_at + [None])
    monkeypatch.setattr(publisher, "current_pr", lambda sha: next(responses))
    publisher.main()
    assert not any(call[0] == "build" for call in publication)
    if advance_at < 2:
        assert not any(call[0] == "publish" for call in publication)


@pytest.mark.parametrize(
    ("field", "value"),
    [("event", "push"), ("conclusion", "failure"), ("path", "other.yml")],
)
def test_unexpected_run_rejected(
    publication: list,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    original = publisher.api
    monkeypatch.setattr(
        publisher, "api", lambda path: original(path) | {field: value}
    )
    with pytest.raises(ValueError, match="Unexpected workflow"):
        publisher.main()
    assert not publication


def test_current_pr_uses_github_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "number": 17,
            "state": "open",
            "head": {"sha": SHA},
            "base": {"repo": {"full_name": publisher.REPOSITORY}},
        }
    ]
    monkeypatch.setattr(publisher, "api", lambda path: records)
    assert publisher.current_pr(SHA)["number"] == 17
    assert publisher.current_pr("b" * 40) is None
    records[0]["state"] = "closed"
    assert publisher.current_pr(SHA) is None
