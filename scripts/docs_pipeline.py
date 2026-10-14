"""Execute, validate and transport commit-specific example documentation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "build-docs"
OUTPUT = BUNDLES / "outputs"
GENERATED = ROOT / "docs" / "_generated"
REPOSITORY = "sequince-dev/sequince-examples"


def git(*args: str) -> str:
    """Run a read-only Git command in the source checkout."""
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def digest(path: Path) -> str:
    """Hash file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write stable, readable JSON and reject nonfinite numerical values."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def examples() -> list[dict]:
    """Read the explicit publication manifest."""
    return json.loads((ROOT / "docs/examples.json").read_text())["examples"]


def entry(name: str) -> dict:
    """Find an example or fail with a useful error."""
    for example in examples():
        if example["id"] == name:
            return example
    raise ValueError(f"Unknown example: {name}")


def dependency_commit() -> str:
    """Read Sequince's resolved commit from the execution lock."""
    return git_dependencies({})["sequince"]


def git_dependencies(example: dict) -> dict[str, str]:
    """Read configured Git dependencies' exact resolved lock revisions."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"][
        "pixi"
    ]
    lock = (ROOT / "pixi.lock").read_text()
    revisions = {}
    for feature in ["cpu", *example.get("dependency_features", [])]:
        for name, specification in (
            config["feature"][feature].get("pypi-dependencies", {}).items()
        ):
            if (
                not isinstance(specification, dict)
                or "git" not in specification
            ):
                continue
            pattern = (
                re.escape("git+" + specification["git"])
                + r"\?[^#\s]*#([0-9a-f]{40})"
            )
            matches = set(re.findall(pattern, lock))
            if len(matches) != 1:
                raise ValueError(
                    f"Missing or inconsistent locked Git commit: {name}"
                )
            revision = matches.pop()
            if "rev" in specification and specification["rev"] != revision:
                raise ValueError(
                    f"Configured and locked revisions differ: {name}"
                )
            revisions[name] = revision
    return revisions


def refresh() -> None:
    """Resolve current Sequince main once before a new execution batch."""
    subprocess.run(["pixi", "update", "sequince"], cwd=ROOT, check=True)


def source_hashes(example: dict) -> dict[str, str]:
    """Fingerprint execution inputs as well as the checkout commit."""
    paths = [
        example["script"],
        example["narrative"],
        "pyproject.toml",
        "pixi.lock",
        "docs/examples.json",
        "scripts/docs_pipeline.py",
    ]
    return {path: digest(ROOT / path) for path in paths}


def worker(name: str, raw: Path, destination: Path) -> None:
    """Run a script with only its output destination changed."""
    import numpy as np

    example = entry(name)
    revisions = git_dependencies(example)
    versions = {}
    for package, revision in revisions.items():
        distribution = importlib.metadata.distribution(package)
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        resolved = direct.get("vcs_info", {}).get("commit_id")
        if resolved != revision:
            raise ValueError(
                f"Installed {package} commit {resolved} != pinned {revision}"
            )
        versions[package] = distribution.version
    spec = importlib.util.spec_from_file_location(
        "documentation_example", ROOT / example["script"]
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {example['script']}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses with postponed annotations resolve their module by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    defaults = (
        vars(module.build_parser().parse_args([]))
        if hasattr(module, "build_parser")
        else vars(module)
    )
    parameters = {key: defaults[key] for key in example["parameters"]}
    # Keep the script's main function, sampler, RNG and constants intact.
    module.outdir = raw
    module.documentation_output = raw
    raw.mkdir(parents=True, exist_ok=True)
    original_argv = sys.argv
    try:
        sys.argv = [str(ROOT / example["script"])]
        module.main()
    finally:
        sys.argv = original_argv
    for name in example["required_outputs"]:
        path = raw / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing required output: {path}")
    if (raw / "summary.json").is_file():
        summary = json.loads((raw / "summary.json").read_text())
    else:
        with np.load(raw / "posterior.npz", allow_pickle=False) as posterior:
            particles = posterior["particles"]
            weights = posterior["weights"]
            if not np.all(np.isfinite(particles)):
                raise ValueError("Nonfinite posterior particles")
            if (
                not np.all(np.isfinite(weights))
                or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.0)
            ):
                raise ValueError("Invalid posterior weights")
            mean = np.sum(particles * weights[:, None], axis=0)
            centered = particles - mean
            # This is the population covariance of the weighted empirical measure.
            covariance = (centered * weights[:, None]).T @ centered
            summary = {
                "log_evidence": float(posterior["log_evidence"]),
                "weighted_mean": mean.tolist(),
                "weighted_population_covariance": covariance.tolist(),
                "effective_sample_size": float(1 / np.sum(weights**2)),
                "n_particles": len(particles),
            }
    destination.mkdir(parents=True, exist_ok=False)
    for name in example["figures"]:
        shutil.copyfile(raw / name, destination / name)
    write_json(destination / "summary.json", summary)
    write_json(
        destination / "metadata.json",
        {
            "schema": 1,
            "example": example["id"],
            "source_commit": git("rev-parse", "HEAD"),
            "source_hashes": source_hashes(example),
            "sequince_version": importlib.metadata.version("sequince"),
            "sequince_commit": revisions["sequince"],
            "git_dependencies": revisions,
            "dependency_versions": versions,
            "python_version": sys.version.split()[0],
            "environment": example["environment"],
            "parameters": parameters,
            "seed": parameters["seed"],
        },
    )
    write_json(
        destination / "checksums.json",
        {path.name: digest(path) for path in sorted(destination.iterdir())},
    )


def run(name: str) -> None:
    """Execute with a hard timeout, a noninteractive backend and fresh outputs."""
    example = entry(name)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / name
    if target.exists():
        shutil.rmtree(target)
    log = ROOT / "build-docs" / f"{name}.log"
    with tempfile.TemporaryDirectory(prefix=f"sequince-{name}-") as tmp:
        env = os.environ | {"MPLBACKEND": "Agg", "MPLCONFIGDIR": tmp + "/mpl"}
        try:
            with log.open("w") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "worker",
                        name,
                        "--raw",
                        tmp + "/raw",
                        "--destination",
                        str(target),
                    ],
                    cwd=ROOT / example["working_directory"],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=example["timeout"],
                    check=True,
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(log.read_text(), file=sys.stderr)
            raise
    print(f"Completed {name}: {target}; execution log: {log}")


def validate(base: Path | None = None) -> None:
    """Reject incomplete, corrupt, mixed-commit or incompatible outputs."""
    base = OUTPUT if base is None else base
    expected = {example["id"] for example in examples()}
    actual = {path.name for path in base.iterdir()} if base.is_dir() else set()
    missing = expected - {name for name in actual if (base / name).is_dir()}
    unexpected = actual - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                "missing example outputs: " + ", ".join(sorted(missing))
            )
        if unexpected:
            details.append(
                "unexpected entries: " + ", ".join(sorted(unexpected))
            )
        raise ValueError(
            f"Invalid documentation outputs in {base}: "
            + "; ".join(details)
            + ". Generate or restore outputs for this source revision "
            "before building documentation."
        )
    for example in examples():
        directory = base / example["id"]
        required = {*example["figures"], "summary.json", "metadata.json"}
        checksums = json.loads((directory / "checksums.json").read_text())
        if set(checksums) != required:
            raise ValueError(f"Incomplete checksum manifest: {directory}")
        if {p.name for p in directory.iterdir()} != required | {
            "checksums.json"
        }:
            raise ValueError(f"Unexpected files: {directory}")
        for name, expected in checksums.items():
            path = directory / name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing or empty output: {path}")
            if digest(path) != expected:
                raise ValueError(f"Checksum mismatch: {path}")
        metadata = json.loads((directory / "metadata.json").read_text())
        expected_metadata = {
            "schema": 1,
            "example": example["id"],
            "source_commit": git("rev-parse", "HEAD"),
            "source_hashes": source_hashes(example),
            "sequince_commit": dependency_commit(),
            "git_dependencies": git_dependencies(example),
            "environment": example["environment"],
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(f"Provenance mismatch: {directory}: {key}")
        if set(metadata["parameters"]) != set(example["parameters"]):
            raise ValueError(f"Missing parameters: {directory}")
        if metadata["seed"] != metadata["parameters"]["seed"]:
            raise ValueError(f"Seed mismatch: {directory}")
        versions = metadata.get("dependency_versions", {})
        if set(versions) != set(git_dependencies(example)) or not all(
            versions.values()
        ):
            raise ValueError(
                f"Missing installed dependency versions: {directory}"
            )
        if not metadata.get("sequince_version"):
            raise ValueError(f"Missing installed version: {directory}")
        json.loads((directory / "summary.json").read_text())
    print(f"Validated all outputs for {git('rev-parse', 'HEAD')}")


def prepare() -> None:
    """Stage narrative, script source and validated results separately."""
    validate()
    if GENERATED.exists():
        shutil.rmtree(GENERATED)
    GENERATED.mkdir(parents=True)
    narratives: set[str] = set()
    for example in examples():
        narrative = example["narrative"]
        folder = GENERATED / str(Path(narrative).parent)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            ROOT / example["script"], folder / Path(example["script"]).name
        )
        if narrative not in narratives:
            shutil.copyfile(ROOT / narrative, GENERATED / narrative)
            narratives.add(narrative)
        target = GENERATED / "results" / example["id"]
        shutil.copytree(OUTPUT / example["id"], target)


def bundle() -> Path:
    """Create a deterministic archive containing only validated public outputs."""
    validate()
    sha = git("rev-parse", "HEAD")
    path = BUNDLES / f"docs-outputs-{sha}.tar.gz"
    files = {
        str(p.relative_to(OUTPUT)): p.read_bytes()
        for p in sorted(OUTPUT.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "schema": 1,
        "source_commit": sha,
        "files": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in files.items()
        },
    }
    files["execution-pixi.lock"] = (ROOT / "pixi.lock").read_bytes()
    manifest["files"]["execution-pixi.lock"] = hashlib.sha256(
        files["execution-pixi.lock"]
    ).hexdigest()
    files["bundle-manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer, mode="wb", mtime=0, filename=""
    ) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
    if path.exists() and path.read_bytes() != buffer.getvalue():
        raise ValueError(
            f"Refusing to replace different existing bundle: {path}"
        )
    path.write_bytes(buffer.getvalue())
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest(path)}  {path.name}\n"
    )
    print(path)
    return path


def unpack(
    path: Path,
    expected_sha: str,
    destination: Path,
    *,
    restore_lock: bool = False,
) -> None:
    """Validate archive paths, completeness and checksums before extraction."""
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        files = {}
        for member in members:
            name = Path(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or str(name) != member.name
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read {member.name}")
            files[member.name] = stream.read()
    manifest = json.loads(files.pop("bundle-manifest.json"))
    if manifest["schema"] != 1 or manifest["source_commit"] != expected_sha:
        raise ValueError("Bundle source commit mismatch")
    hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in files.items()
    }
    if manifest["files"] != hashes:
        raise ValueError("Bundle completeness/checksum mismatch")
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    execution_lock = files.pop("execution-pixi.lock")
    if restore_lock:
        (ROOT / "pixi.lock").write_bytes(execution_lock)
    elif execution_lock != (ROOT / "pixi.lock").read_bytes():
        raise ValueError("Bundle execution lock differs from the active lock")
    for name, data in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    validate(destination)


def check_latest() -> None:
    """Stop stale latest builds both before Sphinx and before RTD upload."""
    if os.environ.get("READTHEDOCS_VERSION") == "latest":
        remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
        if git("rev-parse", "HEAD") != remote:
            raise ValueError(
                "Stale latest checkout: main advanced; rebuild latest"
            )


def fetch() -> None:
    """Fetch only the durable bundle for RTD's actual checked-out commit."""
    check_latest()
    sha = git("rev-parse", "HEAD")
    # No credentials: RTD Community and these assets must be publicly readable.
    name = f"docs-outputs-{sha}.tar.gz"
    base = (
        f"https://github.com/{REPOSITORY}/releases/download/docs-output-{sha}"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        for suffix in ("", ".sha256"):
            with urllib.request.urlopen(
                base + "/" + name + suffix, timeout=60
            ) as response:
                Path(str(path) + suffix).write_bytes(response.read())
        expected = Path(str(path) + ".sha256").read_text().split()
        if expected != [digest(path), name]:
            raise ValueError("Downloaded bundle checksum mismatch")
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        unpack(path, sha, OUTPUT, restore_lock=True)
    prepare()


def main() -> None:
    """Dispatch local, Actions and RTD commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "matrix",
            "run",
            "run-all",
            "worker",
            "validate",
            "prepare",
            "bundle",
            "fetch",
            "check-latest",
            "refresh",
        ],
    )
    parser.add_argument("name", nargs="?")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "matrix":
        print(
            json.dumps(
                {
                    "include": [
                        {"id": e["id"], "environment": e["environment"]}
                        for e in examples()
                    ]
                }
            )
        )
    elif args.command == "worker":
        worker(args.name, args.raw, args.destination)
    elif args.command == "run":
        if os.environ.get("SEQUINCE_LOCK_READY") != "1":
            refresh()
            subprocess.run(
                [
                    "pixi",
                    "install",
                    "--locked",
                    "-e",
                    entry(args.name)["environment"],
                ],
                cwd=ROOT,
                check=True,
            )
        run(args.name)
    elif args.command == "run-all":
        if os.environ.get("SEQUINCE_LOCK_READY") != "1":
            refresh()
        for example in examples():
            subprocess.run(
                [
                    "pixi",
                    "run",
                    "--locked",
                    "-e",
                    example["environment"],
                    "example",
                    example["id"],
                ],
                cwd=ROOT,
                env=os.environ | {"SEQUINCE_LOCK_READY": "1"},
                check=True,
            )
    else:
        {
            "refresh": refresh,
            "validate": validate,
            "prepare": prepare,
            "bundle": bundle,
            "fetch": fetch,
            "check-latest": check_latest,
        }[args.command]()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Documentation pipeline error: {error}") from None
