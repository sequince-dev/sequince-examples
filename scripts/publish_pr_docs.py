"""Prepare PR artifacts as data using only trusted default-branch code."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

from docs_pipeline import REPOSITORY, ROOT, digest

MAX_BYTES = 128 * 1024 * 1024


def gh(*args: str) -> str:
    """Read GitHub data without executing files from the PR artifact."""
    return subprocess.check_output(["gh", *args], cwd=ROOT, text=True).strip()


def api(path: str) -> dict | list:
    """Read GitHub metadata from the fixed examples repository."""
    return json.loads(gh("api", f"repos/{REPOSITORY}/{path}"))


def current_pr(sha: str) -> dict | None:
    """Find the open PR whose current head is the completed run's commit."""
    matches = [
        pr
        for pr in api(f"commits/{sha}/pulls?per_page=100")
        if pr["state"] == "open"
        and pr["head"]["sha"] == sha
        and pr["base"]["repo"]["full_name"] == REPOSITORY
    ]
    if len(matches) > 1:
        raise ValueError("Multiple open PRs for the same head commit")
    return matches[0] if matches else None


def verify_bundle(path: Path, sha: str) -> None:
    """Check untrusted archive data without extracting or executing files."""
    hashes = {}
    manifest = None
    size = 0
    # Bound decompression before parsing tar/PAX headers as well as file data.
    with gzip.open(path, "rb") as compressed:
        contents = compressed.read(MAX_BYTES + 1)
    if len(contents) > MAX_BYTES:
        raise ValueError("Oversized decompressed PR bundle")
    with tarfile.open(fileobj=io.BytesIO(contents), mode="r:") as archive:
        for member in archive:
            name = Path(member.name)
            size += member.size
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or str(name) != member.name
                or member.name in hashes
                or member.size < 0
                or size > MAX_BYTES
                or len(hashes) >= 2000
            ):
                raise ValueError("Unsafe or oversized PR bundle")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable PR bundle member")
            data = stream.read()
            hashes[member.name] = hashlib.sha256(data).hexdigest()
            if member.name == "bundle-manifest.json":
                manifest = json.loads(data)
            elif member.name.endswith("/metadata.json"):
                if json.loads(data)["source_commit"] != sha:
                    raise ValueError("PR metadata commit mismatch")
    hashes.pop("bundle-manifest.json", None)
    if (
        not manifest
        or manifest["schema"] != 1
        or manifest["source_commit"] != sha
        or manifest["files"] != hashes
        or "execution-pixi.lock" not in hashes
        or not any(name.endswith("/metadata.json") for name in hashes)
    ):
        raise ValueError("PR bundle manifest/commit/checksum mismatch")


def download_artifact(run_id: int, name: str, destination: Path) -> Path:
    """Download one bounded artifact from the exact completed workflow run."""
    artifacts = api(f"actions/runs/{run_id}/artifacts?per_page=100")
    matches = [
        item
        for item in artifacts["artifacts"]
        if item["name"] == name and not item["expired"]
    ]
    if len(matches) != 1 or matches[0]["size_in_bytes"] > MAX_BYTES:
        raise ValueError("Missing, ambiguous or oversized PR artifact")
    artifact_id = int(matches[0]["id"])
    zipped = destination / f"{name}.zip"
    with zipped.open("wb") as stream:
        subprocess.run(
            [
                "gh",
                "api",
                f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip",
            ],
            cwd=ROOT,
            stdout=stream,
            check=True,
        )
    if zipped.stat().st_size > MAX_BYTES:
        raise ValueError("Oversized PR artifact download")
    return zipped


def download_bundle(run_id: int, sha: str, destination: Path) -> Path:
    """Validate the example provenance bundle from the completed run."""
    zipped = download_artifact(run_id, "validated-bundle", destination)
    path = destination / f"docs-outputs-{sha}.tar.gz"
    expected = {path.name, path.name + ".sha256"}
    with zipfile.ZipFile(zipped) as archive:
        members = archive.infolist()
        if (
            len(members) != 2
            or {member.filename for member in members} != expected
            or sum(member.file_size for member in members) > MAX_BYTES
        ):
            raise ValueError("Unexpected files in PR artifact")
        for name in expected:
            (destination / name).write_bytes(archive.read(name))
    checksum = Path(str(path) + ".sha256").read_text().split()
    if checksum != [digest(path), path.name]:
        raise ValueError("PR artifact checksum mismatch")
    verify_bundle(path, sha)
    return path


def download_html(run_id: int, destination: Path) -> Path:
    """Extract static HTML as data, rejecting unsafe or oversized archives."""
    zipped = download_artifact(run_id, "documentation-html", destination)
    html = destination / "html"
    with zipfile.ZipFile(zipped) as archive:
        members = archive.infolist()
        names = set()
        if (
            len(members) > 10000
            or sum(member.file_size for member in members) > MAX_BYTES
        ):
            raise ValueError("Oversized PR HTML artifact")
        # Validate all paths before writing. Never preserve executable modes,
        # create links, or extract files into the trusted source checkout.
        for member in members:
            name = Path(member.filename)
            mode = member.external_attr >> 16
            if (
                name.is_absolute()
                or ".." in name.parts
                or str(name) != member.filename.rstrip("/")
                or "\\" in member.filename
                or str(name) in names
                or stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}
            ):
                raise ValueError("Unsafe PR HTML artifact")
            names.add(str(name))
        for member in members:
            target = html / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    if not (html / "index.html").is_file():
        raise ValueError("PR HTML artifact is missing index.html")
    return html


def main() -> None:
    """Prepare a current PR's HTML and metadata for the official RTD action."""
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    run_id = int(event["workflow_run"]["id"])
    run = api(f"actions/runs/{run_id}")
    if (
        run["event"] != "pull_request"
        or run["conclusion"] != "success"
        or run["path"] != ".github/workflows/documentation.yml"
        or run["repository"]["full_name"] != REPOSITORY
    ):
        raise ValueError("Unexpected workflow run for PR publication")
    sha = run["head_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid workflow head commit")
    pr = current_pr(sha)
    if pr is None:
        print("PR closed or advanced; skipping stale publication")
        return
    number = int(pr["number"])
    # Keep the HTML until the following action finishes; the hosted runner
    # removes its temporary directory after the job.
    directory = Path(
        tempfile.mkdtemp(prefix="pr-docs-", dir=os.environ["RUNNER_TEMP"])
    )
    download_bundle(run_id, sha, directory)
    html = download_html(run_id, directory)
    current = current_pr(sha)
    if current is None or current["number"] != number:
        print("PR changed; skipping stale publication")
        return
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(
            f"ready=true\nversion={number}\ncommit={sha}\nhtml={html}\n"
        )


if __name__ == "__main__":
    main()
