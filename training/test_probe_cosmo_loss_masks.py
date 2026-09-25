"""Dependency-light regression checks for the real TRL loss-mask probe."""
import unittest

from training import probe_cosmo_loss_masks as probe


class CosmoLossMaskProbeTests(unittest.TestCase):
    def setUp(self):
        self.full_ids = [10, 11, 12, 20, 21]
        self.prompt_ids = [10, 11, 12]
        self.record = {
            "input_ids": list(self.full_ids),
            "labels": [-100, -100, -100, 20, 21],
        }

    def test_valid_completion_only_labels_are_accepted(self):
        boundary = probe.verify_record(
            self.record, self.full_ids, self.prompt_ids, max_length=8
        )
        self.assertEqual(boundary, 3)

    def test_prompt_token_contributing_to_loss_is_rejected(self):
        self.record["labels"][1] = 11
        with self.assertRaisesRegex(probe.ProbeError, "prompt contributes to loss"):
            probe.verify_record(
                self.record, self.full_ids, self.prompt_ids, max_length=8
            )

    def test_masked_completion_token_is_rejected(self):
        self.record["labels"][3] = -100
        with self.assertRaisesRegex(probe.ProbeError, "completion labels lost"):
            probe.verify_record(
                self.record, self.full_ids, self.prompt_ids, max_length=8
            )

    def test_preprocessing_token_change_is_rejected(self):
        self.record["input_ids"][4] = 999
        with self.assertRaisesRegex(probe.ProbeError, "preprocessing changed"):
            probe.verify_record(
                self.record, self.full_ids, self.prompt_ids, max_length=8
            )


if __name__ == "__main__":
    unittest.main()
