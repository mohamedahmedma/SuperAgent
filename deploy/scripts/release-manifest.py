#!/usr/bin/env python3
"""Freeze incremental image versions once, then promote that exact set unchanged."""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

SERVICES = ("backend", "frontend", "identity", "records", "sis")


def validate(manifest: dict) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("release", "")):
        raise ValueError("Manifest release must be a commit SHA")
    if not re.fullmatch(r"ghcr\.io/[a-z0-9_.-]+/superagent", manifest.get("prefix", "")):
        raise ValueError("Unexpected image registry prefix")
    images = manifest.get("images", {})
    if set(images) != set(SERVICES):
        raise ValueError("Manifest must contain exactly the five application services")
    if not all(
        isinstance(tag, str) and re.fullmatch(r"[0-9a-f]{40}", tag) for tag in images.values()
    ):
        raise ValueError("Every service must use an explicit commit tag; stable is not promotable")
    return manifest


def prepare(release: str, prefix: str, changed: list[str], baseline: dict | None) -> dict:
    if set(changed) - set(SERVICES):
        raise ValueError("Unknown changed service")
    images = validate(baseline)["images"].copy() if baseline else {}
    if baseline and baseline["prefix"] != prefix:
        raise ValueError("Baseline belongs to another registry")
    images.update({service: release for service in changed})
    return validate({"release": release, "prefix": prefix, "images": images})


def api(path: str) -> bytes:
    result = subprocess.run(["gh", "api", path], capture_output=True, check=True)
    return result.stdout


def baseline(repo: str, commit: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("No verified production baseline")
    artifacts = json.loads(api(f"repos/{repo}/actions/artifacts?name=release-manifest-{commit}"))
    for artifact in artifacts["artifacts"]:
        run = artifact.get("workflow_run", {})
        if artifact["expired"] or run.get("head_sha") != commit or run.get("head_branch") != "main":
            continue
        details = json.loads(api(f"repos/{repo}/actions/runs/{run['id']}"))
        if (
            details["conclusion"] != "success"
            or details["path"] != ".github/workflows/deploy.yml"
            or details["head_repository"]["full_name"].lower() != repo.lower()
            or details["event"] not in ("push", "workflow_dispatch")
        ):
            continue
        archive = api(f"repos/{repo}/actions/artifacts/{artifact['id']}/zip")
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            # Read one known file; do not extract untrusted archive paths.
            manifest = validate(json.loads(zipped.read("release-manifest.json")))
        if manifest["release"] != commit:
            raise ValueError("Artifact release does not match the production baseline")
        return manifest
    raise ValueError("Production manifest absent/expired: rebuild all services safely")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    base = commands.add_parser("baseline")
    base.add_argument("--repo", required=True)
    base.add_argument("--commit", required=True)
    base.add_argument("--output", required=True)
    make = commands.add_parser("prepare")
    make.add_argument("--release", required=True)
    make.add_argument("--prefix", required=True)
    make.add_argument("--changed", default="")
    make.add_argument("--baseline", default="baseline-manifest.json")
    make.add_argument("--output", default="release-manifest.json")
    check = commands.add_parser("validate")
    check.add_argument("manifest")
    check.add_argument("--release", required=True)
    check.add_argument("--prefix", required=True)
    check.add_argument("--tags-output")
    args = parser.parse_args()
    if args.command == "baseline":
        manifest = baseline(args.repo, args.commit)
    elif args.command == "prepare":
        path = Path(args.baseline)
        previous = json.loads(path.read_text()) if path.exists() else None
        manifest = prepare(args.release, args.prefix, args.changed.split(), previous)
    else:
        manifest = validate(json.loads(Path(args.manifest).read_text()))
        if manifest["release"] != args.release or manifest["prefix"] != args.prefix:
            raise ValueError("Promotion artifact does not match this release")
        if args.tags_output:
            content = f"REGISTRY_IMAGE_PREFIX={manifest['prefix']}\n"
            content += "".join(
                f"{service.upper()}_IMAGE_TAG={manifest['images'][service]}\n"
                for service in SERVICES
            )
            Path(args.tags_output).write_text(content, encoding="utf-8", newline="\n")
        return
    Path(args.output).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error
