"""Trusted publication only: immutable-by-policy assets, then serialized RTD."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from docs_pipeline import ROOT, bundle, digest, git


def gh(*args: str) -> str:
    """Call the GitHub CLI, failing on API or permission errors."""
    return subprocess.check_output(["gh", *args], cwd=ROOT, text=True).strip()


def rtd(path: str, method: str = "GET") -> dict:
    """Call the documented Community API without logging credentials."""
    request = urllib.request.Request(
        "https://app.readthedocs.org/api/v3/" + path,
        headers={"Authorization": "Token " + os.environ["RTD_TOKEN"]},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def wait_idle(project: str, deadline: float) -> None:
    """Wait through the RTD upload before allowing another publication."""
    while rtd(f"projects/{project}/builds/?running=true")["count"]:
        if time.monotonic() > deadline:
            raise TimeoutError(
                "RTD still running; inspect builds before retrying"
            )
        time.sleep(20)


def main() -> None:
    """Publish only main, verify assets, and monitor the triggered build."""
    sha = git("rev-parse", "HEAD")
    if git("ls-remote", "origin", "refs/heads/main").split()[0] != sha:
        print("Main has advanced; skipping stale publication")
        return
    if set(git("diff", "--name-only", "HEAD").splitlines()) - {"pixi.lock"}:
        raise ValueError("Refusing publication from a modified checkout")
    path = bundle()
    tag = f"docs-output-{sha}"
    # Dedicated prereleases in the examples repository cannot become a library
    # release or GitHub's latest software release.
    view = subprocess.run(
        ["gh", "release", "view", tag, "--json", "assets,isDraft"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if view.returncode:
        gh(
            "release",
            "create",
            tag,
            "--target",
            sha,
            "--title",
            tag,
            "--notes",
            f"Validated example documentation outputs for {sha}.",
            "--prerelease",
            "--latest=false",
            "--draft",
        )
        release = {"assets": [], "isDraft": True}
    else:
        release = json.loads(view.stdout)
    names = {asset["name"] for asset in release["assets"]}
    for asset in (path, Path(str(path) + ".sha256")):
        if asset.name in names:
            existing = ROOT / "build-docs" / "existing"
            existing.mkdir(exist_ok=True)
            gh(
                "release",
                "download",
                tag,
                "--pattern",
                asset.name,
                "--dir",
                str(existing),
                "--clobber",
            )
            if digest(existing / asset.name) != digest(asset):
                raise ValueError(
                    f"Refusing to replace different asset: {asset.name}"
                )
        else:
            gh("release", "upload", tag, str(asset))
    if release["isDraft"]:
        gh("release", "edit", tag, "--draft=false", "--latest=false")
    # The workflow's non-cancelling concurrency lock covers this whole wait.
    project = urllib.parse.quote(os.environ["RTD_PROJECT"], safe="")
    deadline = time.monotonic() + 2400
    wait_idle(project, deadline)
    if git("ls-remote", "origin", "refs/heads/main").split()[0] != sha:
        print("Bundle uploaded; main advanced, skipping latest trigger")
        return
    result = rtd(f"projects/{project}/versions/latest/builds/", "POST")
    build_id = result["build"]["id"]
    while True:
        build = rtd(f"projects/{project}/builds/{build_id}/")
        if build["state"]["code"] in {"finished", "cancelled"}:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(
                "RTD build timed out; inspect it before retrying"
            )
        time.sleep(20)
    if not build["success"] or build["commit"] != sha:
        raise ValueError(
            f"RTD build {build_id} failed or checked out another commit; "
            "inspect RTD and rerun the workflow for current main"
        )
    print(f"RTD build {build_id} succeeded for {sha}")


if __name__ == "__main__":
    main()
