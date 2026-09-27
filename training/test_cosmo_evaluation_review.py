"""Offline tests for the score-only Cosmo human review overlay."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training import cosmo_evaluation_review as review


class CosmoEvaluationReviewTests(unittest.TestCase):
    def bundle(self):
        attempts = []
        for index in range(36):
            attempts.append({
                "id": f"attempt-{index:02d}",
                "variant": review.evaluation.VARIANTS[index // 12],
                "outcome": "success",
                "answer": "immutable answer",
                "scores": {dimension: "pending"
                           for dimension in review.evaluation.DIMENSIONS},
            })
        return {"kind": "measured", "attempts": attempts}

    def overlay(self, bundle, digest="a" * 64, rubric_digest="b" * 64):
        return {
            "schema_version": 1,
            "kind": "human_score_overlay",
            "generation_bundle_sha256": digest,
            "rubric_sha256": rubric_digest,
            "reviewer_label": "owner-review-session",
            "attestation": review.ATTESTATION,
            "scores": [{
                "attempt_id": attempt["id"],
                **{dimension: "pass"
                   for dimension in review.evaluation.DIMENSIONS},
            } for attempt in bundle["attempts"]],
        }

    def test_overlay_changes_only_scores_and_preserves_answers(self):
        bundle = self.bundle()
        scored = review.apply_scores(
            bundle, self.overlay(bundle), "a" * 64, "b" * 64
        )
        self.assertEqual(
            [row["answer"] for row in scored["attempts"]],
            [row["answer"] for row in bundle["attempts"]],
        )
        self.assertTrue(all(
            all(value == "pass" for value in row["scores"].values())
            for row in scored["attempts"]
        ))
        self.assertTrue(all(
            all(value == "pending" for value in row["scores"].values())
            for row in bundle["attempts"]
        ))

    def test_overlay_rejects_wrong_hash_order_missing_and_unknown_fields(self):
        mutations = []
        wrong_hash = self.overlay(self.bundle())
        wrong_hash["generation_bundle_sha256"] = "b" * 64
        mutations.append(wrong_hash)
        wrong_rubric = self.overlay(self.bundle())
        wrong_rubric["rubric_sha256"] = "c" * 64
        mutations.append(wrong_rubric)
        wrong_order = self.overlay(self.bundle())
        wrong_order["scores"][0]["attempt_id"] = "attempt-01"
        mutations.append(wrong_order)
        missing = self.overlay(self.bundle())
        missing["scores"].pop()
        mutations.append(missing)
        unknown = self.overlay(self.bundle())
        unknown["release"] = True
        mutations.append(unknown)
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(review.ReviewError):
                review.apply_scores(
                    self.bundle(), value, "a" * 64, "b" * 64
                )

    def test_overlay_rejects_pending_verdict_failed_attempt_and_edited_source(self):
        bundle = self.bundle()
        pending = self.overlay(bundle)
        pending["scores"][0]["identity"] = "pending"
        with self.assertRaises(review.ReviewError):
            review.apply_scores(bundle, pending, "a" * 64, "b" * 64)

        failed = deepcopy(bundle)
        failed["attempts"][0]["outcome"] = "failed"
        with self.assertRaises(review.ReviewError):
            review.apply_scores(
                failed, self.overlay(failed), "a" * 64, "b" * 64
            )

        already_scored = deepcopy(bundle)
        already_scored["attempts"][0]["scores"]["identity"] = "pass"
        with self.assertRaises(review.ReviewError):
            review.apply_scores(
                already_scored, self.overlay(already_scored),
                "a" * 64, "b" * 64
            )

    def test_reviewer_label_is_not_identity_verification_or_release(self):
        report = review.dry_run()
        self.assertFalse(report["human_reviewer_identity_verified"])
        self.assertFalse(report["automatic_release_allowed"])
        self.assertFalse(report["deployment_authorized"])
        self.assertFalse(report["phase_b_ready"])

    def test_finalize_writes_separate_bundle_and_nonrelease_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.bundle()
            generation = root / "generation.json"
            generation.write_text(json.dumps(bundle), encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(generation.read_bytes()).hexdigest()
            overlay = root / "scores.json"
            _, rubric_digest = review.evaluation.load_rubric()
            overlay.write_text(
                json.dumps(self.overlay(bundle, digest, rubric_digest)),
                encoding="utf-8"
            )
            adapter = root / "adapter-run"
            adapter.mkdir()
            output = root / "reviewed"
            generation_report = {
                "kind": "measured",
                "runner_attestation_verified": True,
                "review_receipt_verified": False,
                "comparison_complete": False,
                "actual_model_outputs_evaluated": False,
                "measurement_sha256": "c" * 64,
            }
            with patch.object(
                review.evaluation, "analyze", return_value=generation_report
            ):
                report = review.finalize(
                    generation, overlay, adapter, output,
                )
            self.assertTrue(
                (output / "reviewed-evaluation-bundle.v1.json").is_file()
            )
            receipt = json.loads(
                (output / "review-receipt.v1.json").read_text()
            )
            self.assertTrue(report["actual_model_outputs_evaluated"])
            self.assertTrue(report["review_receipt_verified"])
            self.assertFalse(receipt["human_reviewer_identity_verified"])
            self.assertFalse(receipt["automatic_release_allowed"])
            self.assertFalse(receipt["phase_b_ready"])

    def test_finalized_verification_rejects_tampered_receipt_or_scores(self):
        generation_report = {
            "kind": "measured",
            "runner_attestation_verified": True,
            "review_receipt_verified": False,
            "comparison_complete": False,
            "actual_model_outputs_evaluated": False,
            "measurement_sha256": "c" * 64,
        }
        for target in ("receipt", "reviewed"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = self.bundle()
                generation = root / "generation.json"
                generation.write_text(json.dumps(bundle), encoding="utf-8")
                import hashlib
                digest = hashlib.sha256(generation.read_bytes()).hexdigest()
                overlay = root / "scores.json"
                _, rubric_digest = review.evaluation.load_rubric()
                overlay.write_text(
                    json.dumps(self.overlay(bundle, digest, rubric_digest)),
                    encoding="utf-8",
                )
                adapter = root / "adapter-run"
                adapter.mkdir()
                output = root / "reviewed"
                with patch.object(
                    review.evaluation, "analyze",
                    return_value=generation_report,
                ):
                    review.finalize(
                        generation, overlay, adapter, output,
                    )
                    if target == "receipt":
                        path = output / review.REVIEW_RECEIPT_NAME
                        value = json.loads(path.read_text())
                        value["reviewed_bundle_sha256"] = "0" * 64
                        path.write_text(json.dumps(value), encoding="utf-8")
                    else:
                        path = output / review.REVIEWED_BUNDLE_NAME
                        value = json.loads(path.read_text())
                        value["attempts"][0]["scores"]["identity"] = "fail"
                        path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(review.ReviewError):
                        review.verify_finalized(
                            generation, overlay, adapter, output,
                        )


if __name__ == "__main__":
    unittest.main()
