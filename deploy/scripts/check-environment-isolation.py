#!/usr/bin/env python3
"""Check resolved Compose resources across all estates without launching services."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    resources: dict[str, set[str]] = {
        "containers": set(),
        "networks": set(),
        "volumes": set(),
        "ports": set(),
    }
    for target in ("production", "dev", "test"):
        env = os.environ.copy()
        # CI fixtures; do not parse or print an operator's credential file.
        for key in ("STACK_NAME", "COMPOSE_PROJECT_NAME"):
            env.pop(key, None)
        env.update(
            {
                "POSTGRES_PASSWORD": "ci-fixture-password",
                "MINIO_ROOT_USER": "ci-fixture-user",
                "MINIO_ROOT_PASSWORD": "ci-fixture-password",
                "LOCAL_SERVICE_KEY": "ci-fixture-service-key",
                "REGISTRY_IMAGE_PREFIX": "example.invalid/superagent",
                **{
                    f"{service}_IMAGE_TAG": "ci"
                    for service in (
                        "BACKEND",
                        "FRONTEND",
                        "IDENTITY",
                        "RECORDS",
                        "SIS",
                    )
                },
            }
        )
        # Test the profile itself rather than machine overrides of its values.
        for line in (ROOT / f"deploy/environments/{target}.env").read_text().splitlines():
            if line and not line.startswith("#"):
                env.pop(line.split("=", 1)[0], None)
        process = subprocess.run(
            [
                "docker",
                "compose",
                "--env-file",
                f"deploy/environments/{target}.env",
                "-f",
                "docker-compose.yml",
                "-f",
                "docker-compose.prod.yml",
                "config",
                "--format",
                "json",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        config = json.loads(process.stdout)
        expected = "superagent" if target == "production" else f"superagent-{target}"
        assert config["name"] == expected, f"{target}: wrong Compose project"
        current = {
            "containers": {service["container_name"] for service in config["services"].values()},
            "networks": {network["name"] for network in config["networks"].values()},
            "volumes": {volume["name"] for volume in config["volumes"].values()},
            "ports": {
                str(port["published"])
                for service in config["services"].values()
                for port in service.get("ports", [])
            },
        }
        assert all(name.startswith(expected + "-") for name in current["containers"])
        assert all(name.startswith(expected + "-") for name in current["networks"])
        assert all(name.startswith(expected + "_") for name in current["volumes"])
        assert all(
            port["host_ip"] == "127.0.0.1"
            for service in config["services"].values()
            for port in service.get("ports", [])
        ), f"{target}: public host port"
        assert not config["services"]["frontend"]["build"]["args"]["VITE_IDENTITY_BASE_URL"]
        for kind, names in current.items():
            assert not (names & resources[kind]), f"{target}: shared {kind}"
            resources[kind].update(names)
        print(f"PASS {target}: isolated containers, networks, volumes and loopback ports")


if __name__ == "__main__":
    main()
