"""Promotion cannot skip an approval or change the images accepted by the tester."""

import importlib.util
import io
import json
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"deploy/scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest = load_script("release-manifest")
approvals = load_script("approval-policy")
PREFIX = "ghcr.io/mohamedahmedma/superagent"
OLD = "a" * 40
NEW = "b" * 40


def previous_release():
    return {
        "release": OLD,
        "prefix": PREFIX,
        "images": {service: OLD for service in manifest.SERVICES},
    }


class FrozenImageTests(unittest.TestCase):
    def test_incremental_candidate_keeps_unchanged_versions(self):
        baseline = previous_release()
        baseline["images"]["sis"] = "c" * 40
        candidate = manifest.prepare(NEW, PREFIX, ["backend"], baseline)
        self.assertEqual(candidate["images"]["backend"], NEW)
        for service in ("frontend", "identity", "records", "sis"):
            self.assertEqual(candidate["images"][service], baseline["images"][service])
        self.assertEqual(baseline["images"]["backend"], OLD)

    def test_missing_baseline_requires_building_every_service(self):
        with self.assertRaises(ValueError):
            manifest.prepare(NEW, PREFIX, ["backend"], None)
        candidate = manifest.prepare(NEW, PREFIX, list(manifest.SERVICES), None)
        self.assertEqual(set(candidate["images"].values()), {NEW})

    def test_mutable_or_injected_tag_cannot_be_promoted(self):
        for tag in ("stable", "latest", "$(echo injected)", OLD + "\nEXTRA=value"):
            with self.subTest(tag=tag):
                baseline = previous_release()
                baseline["images"]["sis"] = tag
                with self.assertRaises(ValueError):
                    manifest.prepare(NEW, PREFIX, ["backend"], baseline)

    def test_different_registry_cannot_supply_the_baseline(self):
        with self.assertRaises(ValueError):
            manifest.prepare(NEW, "ghcr.io/another/superagent", ["backend"], previous_release())

    def test_untrusted_workflow_artifacts_cannot_supply_image_versions(self):
        artifact = {
            "id": 123,
            "expired": False,
            "workflow_run": {"head_sha": OLD, "head_branch": "main", "id": 456},
        }
        for event, repo, path in (
            ("pull_request", "mohamedahmedma/SuperAgent", ".github/workflows/deploy.yml"),
            ("push", "someone/Fork", ".github/workflows/deploy.yml"),
            ("push", "mohamedahmedma/SuperAgent", ".github/workflows/other.yml"),
        ):
            with self.subTest(event=event, repo=repo, path=path):
                run = {
                    "conclusion": "success",
                    "path": path,
                    "event": event,
                    "head_repository": {"full_name": repo},
                }
                with patch.object(
                    manifest,
                    "api",
                    side_effect=[
                        json.dumps({"artifacts": [artifact]}).encode(),
                        json.dumps(run).encode(),
                    ],
                ) as api:
                    with self.assertRaises(ValueError):
                        manifest.baseline("mohamedahmedma/SuperAgent", OLD)
                    self.assertEqual(api.call_count, 2)

    def test_successful_production_artifact_is_reused_without_extracting_paths(self):
        artifact = {
            "id": 123,
            "expired": False,
            "workflow_run": {"head_sha": OLD, "head_branch": "main", "id": 456},
        }
        run = {
            "conclusion": "success",
            "path": ".github/workflows/deploy.yml",
            "event": "push",
            "head_repository": {"full_name": "mohamedahmedma/SuperAgent"},
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zipped:
            zipped.writestr("release-manifest.json", json.dumps(previous_release()))
            zipped.writestr("../../untrusted-file", "must never be extracted")
        with patch.object(
            manifest,
            "api",
            side_effect=[
                json.dumps({"artifacts": [artifact]}).encode(),
                json.dumps(run).encode(),
                buffer.getvalue(),
            ],
        ):
            self.assertEqual(
                manifest.baseline("mohamedahmedma/SuperAgent", OLD), previous_release()
            )


class ApprovalTests(unittest.TestCase):
    def test_an_environment_without_reviewers_is_not_a_valid_gate(self):
        for rules in ([], [{"type": "wait_timer", "wait_timer": 10}]):
            with self.assertRaises(ValueError):
                approvals.policy({"name": "test", "protection_rules": rules})

    def test_copy_payload_retains_reviewers_self_review_timer_and_bypass(self):
        payload = approvals.policy(
            {
                "name": "Production",
                "can_admins_bypass": False,
                "deployment_branch_policy": None,
                "protection_rules": [
                    {
                        "type": "required_reviewers",
                        "prevent_self_review": True,
                        "reviewers": [
                            {"type": "User", "reviewer": {"id": 22}},
                            {"type": "Team", "reviewer": {"id": 11}},
                        ],
                    },
                    {"type": "wait_timer", "wait_timer": 15},
                ],
            }
        )
        self.assertEqual(
            payload["reviewers"], [{"type": "Team", "id": 11}, {"type": "User", "id": 22}]
        )
        self.assertTrue(payload["prevent_self_review"])
        self.assertFalse(payload["can_admins_bypass"])
        self.assertEqual(payload["wait_timer"], 15)

    def test_unknown_protection_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            approvals.policy({"protection_rules": [{"type": "branch_policy"}]})


class PromotionGraphTests(unittest.TestCase):
    def test_test_and_production_require_the_previous_environment_success(self):
        workflow = yaml.load(
            (ROOT / ".github/workflows/deploy.yml").read_text(), Loader=yaml.BaseLoader
        )
        jobs = workflow["jobs"]
        self.assertEqual(workflow["on"]["push"]["branches"], ["main"])
        self.assertNotIn("inputs", workflow["on"]["workflow_dispatch"] or {})
        for job, previous, target in (
            ("deploy-test", "deploy-development", "test"),
            ("deploy", "deploy-test", "production"),
        ):
            self.assertEqual(jobs[job]["needs"], previous)
            self.assertIn(f"needs.{previous}.result == 'success'", jobs[job]["if"])
            self.assertEqual(jobs[job]["with"]["environment"], target)
        self.assertEqual(jobs["deploy-development"]["with"]["environment"], "dev")
        self.assertEqual(jobs["mark-deployed"]["needs"], "deploy")

    def test_original_granular_cd_jobs_and_stages_remain(self):
        jobs = yaml.load(
            (ROOT / ".github/workflows/deploy.yml").read_text(), Loader=yaml.BaseLoader
        )["jobs"]
        self.assertEqual(jobs["ci"]["name"], "Stage 1 - CI")
        self.assertEqual(jobs["changes"]["name"], "Stage 2 - Detect changed services")
        for service in manifest.SERVICES:
            job = jobs[f"cd-publish-{service}"]
            self.assertEqual(job["needs"], "changes")
            self.assertIn(f"needs.changes.outputs.{service}", job["if"])
        self.assertEqual(jobs["deploy"]["name"], "Stage 3 - Deploy production (approval required)")
        self.assertEqual(
            jobs["mark-deployed"]["name"], "Stage 4 - Record this as the last successful deploy"
        )

    def test_every_deployment_uses_the_protected_environment_and_same_artifact(self):
        child = yaml.load(
            (ROOT / ".github/workflows/deploy-environment.yml").read_text(), Loader=yaml.BaseLoader
        )
        self.assertNotIn("workflow_dispatch", child["on"])
        job = child["jobs"]["deploy"]
        self.assertEqual(job["environment"]["name"], "${{ inputs.environment }}")
        artifact = next(
            step for step in job["steps"] if step.get("uses") == "actions/download-artifact@v4"
        )
        self.assertEqual(artifact["with"]["name"], "release-manifest-${{ github.sha }}")


if __name__ == "__main__":
    unittest.main()
