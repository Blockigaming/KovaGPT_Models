"""Check draft-label integrity, distinct held-out inputs and no benchmark repair."""

import json
import unittest
from unittest.mock import patch

from evaluation import a35_correction_data as data
from evaluation.quality_evidence import EvidenceRejected


class CorrectionDraftTests(unittest.TestCase):
    def test_checked_in_draft_targets_and_pins_are_validated_without_model_calls(self):
        result = data.validate_draft()
        self.assertGreater(result["reference_labels_verified"], 0)
        self.assertFalse(result["active_training_dataset_changed"])
        self.assertFalse(result["semantic_overlap_review_complete"])
        self.assertEqual(result["model_calls_made"], 0)

    def test_reference_filter_preserves_order_duplicates_negatives_and_zero(self):
        self.assertEqual(data.reference_answer("filter_map", {
            "values": [-8, 5, -8, 0, 6, 3], "divisor": 2, "factor": 3}),
            [-24, -24, 0, 18])
        self.assertEqual(data.reference_answer("copy_or_alias", {
            "values": [4], "append": 9, "copy": True}), [4])
        self.assertEqual(data.reference_answer("copy_or_alias", {
            "values": [4], "append": 9, "copy": False}), [4, 9])

    def test_draft_cannot_promote_itself_to_training_approval(self):
        original = json.loads(data.MANIFEST.read_text())
        for flag in ("training_authorized", "dataset_owner_approved", "prompt_owner_approved",
                     "paid_execution_authorized", "integration_into_training_authorized"):
            changed = dict(original, **{flag: True})
            with patch.object(type(data.MANIFEST), "read_text", return_value=json.dumps(changed)):
                with self.assertRaises(EvidenceRejected):
                    data.validate_draft()


if __name__ == "__main__":
    unittest.main()
