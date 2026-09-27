"""Synthetic evidence tests for the three-way Cosmo evaluation contract."""
from copy import deepcopy
import inspect
from pathlib import Path
import socket
import subprocess
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from training import cosmo_sft_evaluation as evaluation
from training.cosmo_generation_attestation import (
    create_attestation,
    public_key_hex,
)


class CosmoSftEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.cases, self.plan_sha256 = evaluation.load_plan()
        self.adapter_sha256 = "a" * 64
        self.adapter_receipt_sha256 = "d" * 64
        self.signing_key = Ed25519PrivateKey.from_private_bytes(b"k" * 32)
        self.public_key_hex = public_key_hex(self.signing_key)
        trust = {
            "schema_version": 1,
            "status": "runner_signing_public_key_pinned",
            "algorithm": "ed25519",
            "public_key_hex": self.public_key_hex,
            "public_key_sha256": "unused-by-mocked-loader",
            "runner_private_key_environment_variable":
                "KOVA_COSMO_GENERATION_SIGNING_KEY_FILE",
            "verifier_private_key_access_allowed": False,
            "checked_in_private_key_allowed": False,
        }
        trust_patch = patch.object(
            evaluation, "load_generation_trust_policy",
            return_value=trust,
        )
        trust_patch.start()
        self.addCleanup(trust_patch.stop)

    def bundle(self, kind="synthetic_fixture"):
        return {
            "schema_version": 1,
            "kind": kind,
            "plan_sha256": self.plan_sha256,
            "source_commit": "b" * 40,
            "adapter_sha256": self.adapter_sha256,
            "adapter_receipt_sha256": self.adapter_receipt_sha256,
            "runner_attestation": None,
            "attempts": [],
        }

    def attempt(self, case_id, variant, **changes):
        answer = "Synthetic fixture output for " + case_id + " and " + variant
        row = {
            "id": case_id + "-" + variant,
            "case_id": case_id,
            "case_sha256": evaluation.digest(self.cases[case_id]),
            "variant": variant,
            "outcome": "success",
            "answer": answer,
            "answer_sha256": evaluation.answer_digest(answer),
            "latency_ms": 10,
            "input_tokens": 20,
            "output_tokens": 8,
            "runtime": {
                "base_model": self.plan["base_model"],
                "base_revision": self.plan["base_revision"],
                "adapter_sha256": self.adapter_sha256
                if variant == "trained_adapter" else None,
                "adapter_receipt_sha256": self.adapter_receipt_sha256,
                "software_lock_sha256": "c" * 64,
                "runtime_evidence_sha256": "e" * 64,
                "lifecycle_id": "lifecycle-001",
                "lifecycle_grant_id": "grant-evaluation-001",
                "lifecycle_ledger_commit_id": "ledger-commit-003",
                "lifecycle_phase_grant_sha256": "f" * 64,
                "hardware": "synthetic-no-gpu-fixture",
                "precision": "float32-fixture",
                "quantization": "none",
            },
            "scores": {dimension: "pass" for dimension in evaluation.DIMENSIONS},
        }
        row.update(changes)
        return row

    def complete_bundle(self, kind="synthetic_fixture"):
        value = self.bundle(kind)
        value["attempts"] = [
            self.attempt(case_id, variant)
            for case_id in self.cases
            for variant in evaluation.VARIANTS
        ]
        if kind == "measured":
            for row in value["attempts"]:
                row["scores"] = {
                    dimension: "pending"
                    for dimension in evaluation.DIMENSIONS
                }
            value["runner_attestation"] = create_attestation(
                value, self.signing_key
            )
        return value

    def rejected(self, value):
        with self.assertRaisesRegex(evaluation.EvaluationError,
                                    "^cosmo evaluation evidence rejected$"):
            evaluation.analyze(value)

    def test_source_plan_requires_same_twelve_cases_across_three_variants(self):
        self.assertEqual(len(self.cases), 12)
        self.assertEqual(tuple(item["id"] for item in self.plan["variants"]),
                         evaluation.VARIANTS)
        self.assertEqual(self.plan["validation_ids"], list(self.cases))
        self.assertFalse(self.plan["actual_model_outputs_evaluated"])
        self.assertFalse(self.plan["phase_b_ready"])

    def test_human_scoring_rubric_is_hash_bound_and_nonrelease(self):
        rubric, digest = evaluation.load_rubric()
        self.assertEqual(self.plan["rubric_path"], evaluation.RUBRIC_PATH)
        self.assertEqual(self.plan["rubric_sha256"], digest)
        self.assertEqual(tuple(rubric["dimensions"]), evaluation.DIMENSIONS)
        self.assertFalse(rubric["review_rules"]["automatic_release_allowed"])
        self.assertFalse(rubric["deployment_authorized"])
        self.assertFalse(rubric["phase_b_ready"])

    def test_complete_synthetic_fixture_never_claims_real_evaluation(self):
        result = evaluation.analyze(self.complete_bundle(), require_complete=True)
        self.assertEqual(result["status"], "comparison_complete")
        self.assertEqual(result["provided_attempts"], 36)
        self.assertTrue(result["comparison_complete"])
        self.assertFalse(result["actual_model_outputs_evaluated"])
        self.assertFalse(result["human_reviewer_identity_verified"])
        self.assertFalse(result["automatic_release_allowed"])
        self.assertFalse(result["phase_b_ready"])

    def test_measured_generation_requires_hash_bound_review_receipt(self):
        bundle = self.complete_bundle("measured")
        receipt = {
            "adapter_sha256": self.adapter_sha256,
            "receipt_sha256": self.adapter_receipt_sha256,
            "lifecycle_id": "lifecycle-001",
        }
        with patch.object(evaluation, "verify_adapter_receipt", return_value=receipt) as verify:
            result = evaluation.analyze(
                bundle, require_complete=False,
                adapter_output=Path("/external/adapter-run"),
            )
        verify.assert_called_once_with(
            Path("/external/adapter-run"),
            expected_source_commit=bundle["source_commit"],
            root=evaluation.pilot.ROOT,
        )
        self.assertFalse(result["comparison_complete"])
        self.assertFalse(result["actual_model_outputs_evaluated"])
        self.assertFalse(result["review_receipt_verified"])
        self.assertTrue(result["runner_attestation_verified"])
        self.assertFalse(result["human_reviewer_identity_verified"])
        self.assertFalse(result["automatic_release_allowed"])
        self.assertEqual(result["closed_checklist_ids"], [])
        with patch.object(
            evaluation, "verify_adapter_receipt", return_value=receipt
        ), self.assertRaises(evaluation.EvaluationError):
            evaluation.analyze(
                bundle, require_complete=True,
                adapter_output=Path("/external/adapter-run"),
            )

    def test_in_place_human_scores_cannot_bypass_review_finalization(self):
        bundle = self.complete_bundle("measured")
        receipt = {
            "adapter_sha256": self.adapter_sha256,
            "receipt_sha256": self.adapter_receipt_sha256,
            "lifecycle_id": "lifecycle-001",
        }
        for row in bundle["attempts"]:
            row["scores"] = {
                dimension: "pass" for dimension in evaluation.DIMENSIONS
            }
        # Scores are deliberately excluded from the runner signature, so the
        # attestation remains valid.  The evaluation entrypoint must still
        # reject this and require the separate hash-bound review receipt.
        with patch.object(
            evaluation, "verify_adapter_receipt", return_value=receipt
        ), self.assertRaises(evaluation.EvaluationError):
            evaluation.analyze(
                bundle, require_complete=True,
                adapter_output=Path("/external/adapter-run"),
            )

    def test_measured_bundle_requires_matching_verified_adapter_receipt(self):
        bundle = self.complete_bundle("measured")
        with self.assertRaises(evaluation.EvaluationError):
            evaluation.analyze(bundle)
        for field, wrong in (("adapter_sha256", "0" * 64),
                             ("receipt_sha256", "0" * 64)):
            verified = {
                "adapter_sha256": self.adapter_sha256,
                "receipt_sha256": self.adapter_receipt_sha256,
                "lifecycle_id": "lifecycle-001",
            }
            verified[field] = wrong
            with self.subTest(field=field), patch.object(
                evaluation, "verify_adapter_receipt", return_value=verified
            ), self.assertRaises(evaluation.EvaluationError):
                evaluation.analyze(
                    bundle, adapter_output=Path("/external/adapter-run"),
                )

    def test_measured_bundle_rejects_hand_authored_or_tampered_answers(self):
        receipt = {
            "adapter_sha256": self.adapter_sha256,
            "receipt_sha256": self.adapter_receipt_sha256,
            "lifecycle_id": "lifecycle-001",
        }
        hand_authored = self.complete_bundle("measured")
        hand_authored["runner_attestation"] = None
        with patch.object(
            evaluation, "verify_adapter_receipt", return_value=receipt
        ), self.assertRaises(evaluation.EvaluationError):
            evaluation.analyze(
                hand_authored, adapter_output=Path("/external/adapter-run"),
            )

        tampered = self.complete_bundle("measured")
        tampered["attempts"][0]["answer"] = "hand-authored replacement"
        tampered["attempts"][0]["answer_sha256"] = evaluation.answer_digest(
            tampered["attempts"][0]["answer"]
        )
        with patch.object(
            evaluation, "verify_adapter_receipt", return_value=receipt
        ), self.assertRaises(evaluation.EvaluationError):
            evaluation.analyze(
                tampered, adapter_output=Path("/external/adapter-run"),
            )

    def test_measured_bundle_rejects_wrong_independent_verification_key(self):
        bundle = self.complete_bundle("measured")
        receipt = {
            "adapter_sha256": self.adapter_sha256,
            "receipt_sha256": self.adapter_receipt_sha256,
            "lifecycle_id": "lifecycle-001",
        }
        wrong_trust = dict(evaluation.load_generation_trust_policy())
        wrong_trust["public_key_hex"] = public_key_hex(
            Ed25519PrivateKey.from_private_bytes(b"w" * 32)
        )
        with patch.object(
            evaluation, "load_generation_trust_policy",
            return_value=wrong_trust,
        ), patch.object(
            evaluation, "verify_adapter_receipt", return_value=receipt
        ), self.assertRaises(evaluation.EvaluationError):
            evaluation.analyze(
                bundle, adapter_output=Path("/external/adapter-run"),
            )

    def test_verifier_api_has_no_private_key_parameter(self):
        self.assertNotIn(
            "generation_auth_key", inspect.signature(evaluation.analyze).parameters
        )

    def test_measured_bundle_rejects_unpinned_source_trust_policy(self):
        bundle = self.complete_bundle("measured")
        receipt = {
            "adapter_sha256": self.adapter_sha256,
            "receipt_sha256": self.adapter_receipt_sha256,
            "lifecycle_id": "lifecycle-001",
        }
        unpinned = {
            "schema_version": 1,
            "status": "signing_public_key_provisioning_required",
            "algorithm": "ed25519",
            "public_key_hex": None,
            "public_key_sha256": None,
            "runner_private_key_environment_variable":
                "KOVA_COSMO_GENERATION_SIGNING_KEY_FILE",
            "verifier_private_key_access_allowed": False,
            "checked_in_private_key_allowed": False,
        }
        with patch.object(
            evaluation, "load_generation_trust_policy",
            return_value=unpinned,
        ), patch.object(
            evaluation, "verify_adapter_receipt", return_value=receipt
        ), self.assertRaises(evaluation.EvaluationError):
            evaluation.analyze(
                bundle, adapter_output=Path("/external/adapter-run"),
            )

    def test_missing_failed_or_pending_attempt_blocks_completion(self):
        values = []
        missing = self.complete_bundle()
        missing["attempts"].pop()
        values.append(missing)
        failed = self.complete_bundle()
        failed["attempts"][0].update(
            outcome="failed", answer=None, answer_sha256=None,
            latency_ms=None, input_tokens=None, output_tokens=None,
            scores={dimension: "pending" for dimension in evaluation.DIMENSIONS},
        )
        values.append(failed)
        pending = self.complete_bundle()
        pending["attempts"][0]["scores"]["identity"] = "pending"
        values.append(pending)
        for value in values:
            with self.subTest(value=value["attempts"][0]["outcome"]):
                result = evaluation.analyze(value)
                self.assertFalse(result["comparison_complete"])
                with self.assertRaises(evaluation.EvaluationError):
                    evaluation.analyze(value, require_complete=True)

    def test_wrong_case_hash_answer_hash_and_duplicate_pair_are_rejected(self):
        mutations = [
            lambda value: value["attempts"][0].update(case_sha256="0" * 64),
            lambda value: value["attempts"][0].update(answer_sha256="0" * 64),
            lambda value: value["attempts"].__setitem__(1, deepcopy(value["attempts"][0])),
        ]
        for mutate in mutations:
            value = self.complete_bundle()
            mutate(value)
            with self.subTest(mutate=mutate):
                self.rejected(value)

    def test_adapter_is_present_only_on_trained_variant(self):
        for variant, wrong in (("base", self.adapter_sha256),
                               ("configured_base", self.adapter_sha256),
                               ("trained_adapter", None)):
            value = self.complete_bundle()
            row = next(item for item in value["attempts"]
                       if item["variant"] == variant)
            row["runtime"]["adapter_sha256"] = wrong
            with self.subTest(variant=variant):
                self.rejected(value)

    def test_lineage_and_runtime_must_be_identical_except_adapter(self):
        for field, wrong in (
            ("base_model", "unapproved/model"),
            ("base_revision", "0" * 40),
            ("adapter_receipt_sha256", "e" * 64),
            ("software_lock_sha256", "d" * 64),
            ("runtime_evidence_sha256", "f" * 64),
            ("lifecycle_phase_grant_sha256", "0" * 64),
            ("hardware", "different-hardware"),
            ("precision", "different-precision"),
            ("quantization", "different-quantization"),
        ):
            value = self.complete_bundle()
            value["attempts"][-1]["runtime"][field] = wrong
            with self.subTest(field=field):
                self.rejected(value)

    def test_invalid_numeric_types_unknown_fields_and_nonfinite_json_fail_closed(self):
        for field, wrong in (("latency_ms", True), ("input_tokens", -1),
                             ("output_tokens", 0)):
            value = self.complete_bundle()
            value["attempts"][0][field] = wrong
            with self.subTest(field=field):
                self.rejected(value)
        value = self.complete_bundle()
        value["attempts"][0]["release"] = True
        self.rejected(value)
        with self.assertRaises(evaluation.EvaluationError):
            evaluation.canonical({"value": float("nan")})

    def test_no_network_subprocess_or_model_loading_occurs(self):
        with patch.object(socket, "socket", side_effect=AssertionError("network")), \
             patch.object(subprocess, "Popen", side_effect=AssertionError("process")):
            result = evaluation.analyze(self.complete_bundle())
        self.assertFalse(result["phase_b_ready"])


if __name__ == "__main__":
    unittest.main()
