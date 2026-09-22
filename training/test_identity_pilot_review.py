"""Checks for the dataset-bound human review ledger; no review is fabricated."""
from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from training import identity_pilot as pilot
from training import identity_pilot_review as review


class IdentityPilotReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "source"
        for relative in (pilot.PLAN_PATH, pilot.PROMPT_PATH,
                         pilot.DATA_PATH, pilot.REVIEW_PATH):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((pilot.ROOT / relative).read_bytes())

    def ledger(self):
        return json.loads((self.root / pilot.REVIEW_PATH).read_text())

    def write(self, value, *, refresh_hash=True):
        path = self.root / pilot.REVIEW_PATH
        path.write_text(json.dumps(value, indent=2) + "\n")
        if refresh_hash:
            plan_path = self.root / pilot.PLAN_PATH
            plan = json.loads(plan_path.read_text())
            plan["review_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            plan_path.write_text(json.dumps(plan))

    def rejected(self, value):
        self.write(value)
        with self.assertRaisesRegex(review.ReviewError,
                                    "^identity pilot review rejected$"):
            review.validate(self.root)

    def test_committed_ledger_is_exactly_approved_and_non_authorizing(self):
        report = review.validate(self.root)
        self.assertEqual(report["status"], "approved")
        self.assertEqual(report["records"], 36)
        self.assertEqual(report["verdict_counts"], {
            "pending": 0, "approve": 36, "reject": 0,
        })
        self.assertTrue(report["human_review_complete"])
        for field in ("training_authorized", "model_weights_downloaded",
                      "training_started", "phase_b_ready"):
            self.assertIs(report[field], False)
        self.assertEqual(report["closed_checklist_ids"], [])

    def test_complete_approval_can_be_validated_without_authorizing_training(self):
        value = self.ledger()
        value.update(status="approved", reviewer="owner-fixture",
                     reviewed_at="2026-09-19T15:00:00Z",
                     human_review_complete=True)
        for item in value["records"]:
            item["verdict"] = "approve"
        self.write(value)
        result = review.validate(self.root)
        self.assertTrue(result["human_review_complete"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["phase_b_ready"])

    def test_rejection_blocks_completion_and_requires_final_metadata(self):
        value = self.ledger()
        value.update(status="changes_requested", reviewer="owner-fixture",
                     reviewed_at="2026-09-19T15:00:00Z",
                     human_review_complete=False)
        for item in value["records"]:
            item["verdict"] = "approve"
        value["records"][4].update(verdict="reject", notes="Incorrect target.")
        self.write(value)
        result = review.validate(self.root)
        self.assertEqual(result["verdict_counts"]["reject"], 1)
        self.assertFalse(result["human_review_complete"])

    def test_partial_review_is_explicitly_in_progress(self):
        value = self.ledger()
        value.update(status="in_progress", reviewer="owner-fixture",
                     reviewed_at=None, human_review_complete=False)
        for item in value["records"]:
            item["verdict"] = "pending"
        value["records"][0]["verdict"] = "approve"
        self.write(value)
        result = review.validate(self.root)
        self.assertEqual(result["status"], "in_progress")
        self.assertEqual(result["verdict_counts"]["pending"], 35)

    def test_stale_hash_and_dataset_binding_are_rejected(self):
        value = self.ledger()
        value["records"][0]["notes"] = "changed without updating the plan"
        self.write(value, refresh_hash=False)
        with self.assertRaises(review.ReviewError):
            review.validate(self.root)
        value = self.ledger()
        value["dataset_sha256"] = "0" * 64
        self.rejected(value)

    def test_missing_duplicate_or_reordered_source_ids_are_rejected(self):
        for edit in (
            lambda records: records.pop(),
            lambda records: records.__setitem__(1, deepcopy(records[0])),
            lambda records: records.reverse(),
        ):
            value = self.ledger()
            edit(value["records"])
            with self.subTest(edit=edit):
                self.rejected(value)

    def test_invalid_status_verdict_notes_and_completion_claims_are_rejected(self):
        mutations = [
            lambda value: value.update(status="awaiting_owner_review"),
            lambda value: value.update(human_review_complete=False),
            lambda value: value["records"][0].update(verdict="pass"),
            lambda value: value["records"][0].update(notes=""),
            lambda value: value["records"][0].update(notes="x" * 2001),
            lambda value: value.update(reviewer=""),
            lambda value: value.update(reviewed_at="yesterday"),
        ]
        for mutate in mutations:
            value = self.ledger()
            mutate(value)
            with self.subTest(mutate=mutate):
                self.rejected(value)

    def test_unknown_fields_duplicate_keys_and_nonfinite_values_fail_closed(self):
        value = self.ledger()
        value["execute"] = True
        self.rejected(value)
        path = self.root / pilot.REVIEW_PATH
        plan_path = self.root / pilot.PLAN_PATH
        for raw in (b'{"schema_version":1,"schema_version":1}',
                    b'{"schema_version":NaN}', b"\xff"):
            path.write_bytes(raw)
            plan = json.loads(plan_path.read_text())
            plan["review_sha256"] = hashlib.sha256(raw).hexdigest()
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(review.ReviewError):
                review.validate(self.root)

    def test_no_network_subprocess_or_model_loader_is_used(self):
        with patch.object(socket, "socket", side_effect=AssertionError("network")), \
             patch.object(subprocess, "Popen", side_effect=AssertionError("process")):
            report = review.validate(self.root)
        self.assertFalse(report["training_started"])

    def test_cli_require_complete_passes_for_approved_ledger(self):
        error = io.StringIO()
        original = review.validate
        with patch.object(review, "validate", side_effect=lambda: original(self.root)), \
             redirect_stderr(error):
            self.assertEqual(review.main(["--require-complete"]), 0)
        self.assertEqual(error.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
