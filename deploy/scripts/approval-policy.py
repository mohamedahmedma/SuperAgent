#!/usr/bin/env python3
"""Copy or verify the existing Production approval policy for dev and test."""

from __future__ import annotations

import argparse
import json
import subprocess


def api(path: str, payload: dict | None = None) -> dict:
    command = ["gh", "api", path]
    if payload is not None:
        command += ["--method", "PUT", "--input", "-"]
    result = subprocess.run(
        command,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot access {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def policy(environment: dict) -> dict:
    rules = environment.get("protection_rules", [])
    unsupported = {rule["type"] for rule in rules} - {"required_reviewers", "wait_timer"}
    if unsupported:
        raise ValueError(f"Unsupported protection rules require administrator setup: {unsupported}")
    approval = next((rule for rule in rules if rule["type"] == "required_reviewers"), {})
    reviewers = sorted(
        (
            {"type": reviewer["type"], "id": reviewer["reviewer"]["id"]}
            for reviewer in approval.get("reviewers", [])
        ),
        key=lambda item: (item["type"], item["id"]),
    )
    if not reviewers:
        raise ValueError(f"{environment.get('name', 'Environment')} has no required reviewers")
    return {
        "reviewers": reviewers,
        "prevent_self_review": approval.get("prevent_self_review", False),
        "wait_timer": next(
            (rule["wait_timer"] for rule in rules if rule["type"] == "wait_timer"), 0
        ),
        "can_admins_bypass": environment.get("can_admins_bypass", True),
        "deployment_branch_policy": environment.get("deployment_branch_policy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    expected = policy(api(f"repos/{args.repo}/environments/Production"))
    if args.plan:
        print(json.dumps({"dev": expected, "test": expected}, indent=2))
        return
    if args.apply:
        permissions = api(f"repos/{args.repo}").get("permissions", {})
        if not permissions.get("admin"):
            raise SystemExit("GitHub repository Admin access is required to configure approvals")
        if expected["deployment_branch_policy"]:
            raise SystemExit("Copy the production branch policies in Settings before applying")
        for name in ("dev", "test"):
            api(f"repos/{args.repo}/environments/{name}", expected)
    for name in ("dev", "test", "Production"):
        actual = policy(api(f"repos/{args.repo}/environments/{name}"))
        if actual != expected:
            raise SystemExit(f"{name} must have the same approval policy as Production")
        print(f"PASS {name}: matching Production reviewers and approval protections")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
