"""Source-only tests for the guarded three-way Cosmo generation runner."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training import cosmo_evaluation_runner as runner


class CosmoEvaluationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.cases, _ = runner.evaluation.load_plan()
        _, self.prompt, _ = runner.pilot.load()

    def test_current_runner_is_blocked_before_runtime_or_dependencies(self):
        with patch.object(runner, "require_ready",
                          side_effect=AssertionError("runtime called")), \
             patch.object(runner, "verify_installed_software",
                          side_effect=AssertionError("dependencies called")):
            with self.assertRaises(runner.EvaluationRunnerError):
                runner.authorize()

    def test_environment_confirmation_cannot_override_source_gates(self):
        with patch.dict(os.environ, {runner.CONFIRMATION_ENV: "YES"}), \
             patch.object(runner, "require_ready",
                          side_effect=AssertionError("runtime called")):
            with self.assertRaises(runner.EvaluationRunnerError):
                runner.authorize()

    def test_released_source_still_requires_fresh_runtime_evidence(self):
        recipe = deepcopy(runner.load_recipe())
        recipe["account_gates"]["eastus_ncast4_quota_verified"] = True
        recipe["account_gates"]["runtime_compatibility_verified"] = True
        with patch.object(runner, "load_recipe", return_value=recipe), \
             patch.dict(os.environ, {runner.CONFIRMATION_ENV: "YES",
                                     runner.SOURCE_COMMIT_ENV: "a" * 40}), \
             patch.object(runner, "verify_source_checkout"), \
             patch.object(runner, "require_ready",
                          side_effect=runner.EvaluationRunnerError("runtime")), \
             patch.object(runner, "verify_installed_software",
                          side_effect=AssertionError("dependencies called")):
            with self.assertRaisesRegex(runner.EvaluationRunnerError, "runtime"):
                runner.authorize()

    def test_variants_apply_only_the_declared_system_prompt(self):
        ordinary = self.cases["validation-001"]
        base = runner.messages_for_variant(ordinary, "base", self.prompt)
        configured = runner.messages_for_variant(
            ordinary, "configured_base", self.prompt
        )
        trained = runner.messages_for_variant(
            ordinary, "trained_adapter", self.prompt
        )
        self.assertEqual([item["role"] for item in base], ["user"])
        self.assertEqual([item["role"] for item in configured], ["system", "user"])
        self.assertEqual(configured, trained)
        self.assertEqual(configured[0]["content"], self.prompt)

        provenance = self.cases["validation-011"]
        messages = runner.messages_for_variant(
            provenance, "trained_adapter", self.prompt
        )
        self.assertIn("Offline evaluation fixture only", messages[0]["content"])

    def test_attempt_records_are_schema_compatible_and_scores_stay_pending(self):
        case = self.cases["validation-001"]
        runtime = {
            "base_model": self.plan["base_model"],
            "base_revision": self.plan["base_revision"],
            "adapter_sha256": None,
            "adapter_receipt_sha256": "a" * 64,
            "software_lock_sha256": "b" * 64,
            "runtime_evidence_sha256": "c" * 64,
            "lifecycle_id": "lifecycle-001",
            "lifecycle_grant_id": "grant-evaluation-001",
            "lifecycle_ledger_commit_id": "ledger-commit-003",
            "lifecycle_phase_grant_sha256": "d" * 64,
            "hardware": "fixture",
            "precision": "fp16",
            "quantization": "none",
        }
        success = runner.attempt_record(
            case=case, variant="base", runtime=runtime, answer="fixture",
            latency_ms=1, input_tokens=2, output_tokens=3,
        )
        failed = runner.attempt_record(
            case=case, variant="configured_base", runtime=runtime
        )
        self.assertEqual(success["outcome"], "success")
        self.assertEqual(failed["outcome"], "failed")
        self.assertTrue(all(value == "pending"
                            for value in success["scores"].values()))
        self.assertIsNone(failed["answer_sha256"])

    def test_external_paths_reject_existing_output_and_repository_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            self.assertEqual(runner.external_existing(str(parent)), parent)
            new = parent / "new"
            self.assertEqual(runner.external_new(str(new)), new)
            new.mkdir()
            with self.assertRaises(runner.EvaluationRunnerError):
                runner.external_new(str(new))

        with tempfile.TemporaryDirectory(dir=runner.ROOT) as directory:
            internal = Path(directory)
            with self.assertRaises(runner.EvaluationRunnerError):
                runner.external_existing(str(internal))
            with self.assertRaises(runner.EvaluationRunnerError):
                runner.external_new(str(internal / "new"))

    def test_dry_run_declares_all_attempts_without_execution_claims(self):
        report = runner.dry_run()
        self.assertEqual(report["expected_attempts"], 36)
        self.assertEqual(report["variants"], [
            "base", "configured_base", "trained_adapter",
        ])
        self.assertEqual(report["rubric_sha256"], self.plan["rubric_sha256"])
        self.assertFalse(report["model_outputs_generated"])
        self.assertFalse(report["actual_model_outputs_evaluated"])
        self.assertFalse(report["phase_b_ready"])

    def test_evaluation_reserves_remote_phase_before_heavy_imports(self):
        recipe = deepcopy(runner.load_recipe())
        recipe["account_gates"]["eastus_ncast4_quota_verified"] = True
        recipe["account_gates"]["runtime_compatibility_verified"] = True
        runtime = {
            "runtime_evidence_sha256": "e" * 64,
            "deadline_utc": "2026-09-19T19:10:00Z",
            "lifecycle_id": "lifecycle-001",
            "preflight_ledger_sequence": 1,
        }
        receipt = {
            "source_commit": "a" * 40,
            "receipt_sha256": "b" * 64,
            "adapter_sha256": "c" * 64,
            "lifecycle_id": "lifecycle-001",
        }
        grant = {
            "lifecycle_id": "lifecycle-001",
            "phase_grant_sha256": "d" * 64,
            "ledger_sequence": 3,
            "grant_id": "grant-evaluation-001",
            "ledger_commit_id": "ledger-commit-003",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            adapter = root / "adapter"
            output = root / "evaluation"
            signing_key = root / "signing.key"
            snapshot.mkdir()
            adapter.mkdir()
            signing_key.write_text("00" * 32, encoding="ascii")
            signing_key.chmod(0o600)
            environment = {
                runner.CONFIRMATION_ENV: "YES",
                runner.SNAPSHOT_ENV: str(snapshot),
                runner.ADAPTER_OUTPUT_ENV: str(adapter),
                runner.EVALUATION_OUTPUT_ENV: str(output),
                runner.SOURCE_COMMIT_ENV: "a" * 40,
                runner.GENERATION_SIGNING_KEY_ENV: str(signing_key),
            }
            with patch.object(runner, "load_recipe", return_value=recipe), \
                 patch.dict(os.environ, environment, clear=True), \
                 patch.object(runner, "require_ready", return_value=runtime), \
                 patch.object(runner, "verify_installed_software"), \
                 patch.object(runner, "verify_snapshot"), \
                 patch.object(runner, "verify_source_checkout"), \
                 patch.object(runner, "verify_receipt", return_value=receipt), \
                 patch.object(
                     runner, "load_generation_trust_policy",
                     return_value={
                         "status": "runner_signing_public_key_pinned",
                         "public_key_hex": "1" * 64,
                         "public_key_sha256": "2" * 64,
                     },
                 ), \
                 patch.object(
                     runner, "load_signing_key", return_value=object()
                 ), \
                 patch.object(
                     runner, "acquire_phase_grant", return_value=grant
                 ) as acquire:
                authorized = runner.authorize()
        self.assertEqual(authorized[-1], grant)
        self.assertEqual(acquire.call_args.kwargs["phase"], "evaluation")
        self.assertEqual(
            acquire.call_args.kwargs["runtime_evidence_sha256"], "e" * 64
        )

    def test_execute_cli_is_sanitized_and_nonzero_while_blocked(self):
        error = io.StringIO()
        with redirect_stderr(error):
            self.assertEqual(runner.main(["--execute"]), 1)
        self.assertEqual(
            error.getvalue(), "kova cosmo evaluation runner rejected\n"
        )


if __name__ == "__main__":
    unittest.main()
