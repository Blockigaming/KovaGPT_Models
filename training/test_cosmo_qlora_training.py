"""The approved QLoRA data path and source-only execution boundary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from training import cosmo_qlora_training as job


class QloraTrainingTests(unittest.TestCase):
    def test_training_stops_at_watchdog_cleanup_trigger(self):
        admission = {"watchdog_cleanup_trigger_utc": "2026-09-24T13:15:00Z"}
        self.assertEqual(job.require_before_cleanup(
            admission, now=datetime(2026, 9, 24, 13, 14, 59, tzinfo=timezone.utc)),
            datetime(2026, 9, 24, 13, 15, tzinfo=timezone.utc))
        with self.assertRaisesRegex(job.TrainingRejected, "insufficient time"):
            job.require_before_cleanup(
                admission, now=datetime(2026, 9, 24, 13, 15, tzinfo=timezone.utc))

    def test_training_needs_full_job_and_preservation_window(self):
        admission = {"watchdog_cleanup_trigger_utc": "2026-09-24T13:15:00Z"}
        earlier = datetime(2026, 9, 24, 12, 34, 59, tzinfo=timezone.utc)
        lead = timedelta(minutes=40)
        self.assertEqual(job.require_before_cleanup(admission, now=earlier,
            minimum_remaining=lead), datetime(2026, 9, 24, 13, 15, tzinfo=timezone.utc))
        for current, required in ((datetime(2026, 9, 24, 12, 35, tzinfo=timezone.utc), lead),
                                  (datetime(2026, 9, 24, 13, 6, tzinfo=timezone.utc),
                                   job.PRESERVATION_LEAD)):
            with self.subTest(current=current), self.assertRaisesRegex(
                    job.TrainingRejected, "insufficient time"):
                job.require_before_cleanup(admission, now=current,
                                           minimum_remaining=required)

    def test_output_cannot_overlap_verified_model_snapshot(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model"
            model.mkdir()
            (root / "outputs").mkdir()
            self.assertEqual(job.external_paths(model, root / "outputs" / "adapter"), model)
            for output in (model / "adapter", root / "adapter"):
                with self.subTest(output=output), \
                     self.assertRaisesRegex(job.TrainingRejected, "snapshot trees"):
                    job.external_paths(model, output)

    def test_missing_or_drifted_installed_qlora_dependency_fails_before_weights(self):
        from training import three_family_contract as contract
        stack = contract.load_json(job.ROOT / "config/kova-three-family-training-stack.v1.json")
        with patch.object(job.sys, "version_info", type("Info", (), {"major": 3, "minor": 12})()), \
             patch.object(job, "version", side_effect=lambda name:
                          "0.0.0" if name == "bitsandbytes" else
                          {**stack["packages"], "datasets": "5.0.1",
                           "cryptography": "50.0.1"}[name]), \
             self.assertRaisesRegex(job.TrainingRejected, "bitsandbytes"):
            job.verify_installed_stack()
        with patch.object(job.sys, "version_info", type("Info", (), {"major": 3, "minor": 12})()), \
             patch.object(job, "version", side_effect=PackageNotFoundError), \
             self.assertRaisesRegex(job.TrainingRejected, "missing QLoRA dependency"):
            job.verify_installed_stack()

    def test_preparation_uses_only_pinned_27_plus_15_record_split(self):
        plan = job.source_plan()
        train, validation = job.prepared_rows()
        self.assertEqual((len(train), len(validation)), (27, 15))
        self.assertEqual(plan["maximum_optimizer_steps"], 7)
        self.assertEqual(plan["maximum_training_seconds"], 1800)
        self.assertFalse(plan["paid_actions_enabled"])
        self.assertTrue(all(row["prompt"][0]["role"] == "system" and
                            row["prompt"][1]["role"] == "user" and
                            row["completion"][0]["role"] == "assistant"
                            for row in train + validation))
        self.assertTrue(all("Trusted test runtime" not in row["prompt"][0]["content"]
                            for row in train))
        self.assertEqual(sum("Trusted test runtime" in row["prompt"][0]["content"]
                             for row in validation), 3)

    def test_full_corpus_token_check_rejects_truncation(self):
        train, validation = job.prepared_rows()

        class FakeTokenizer:
            def __init__(self, too_long=False):
                self.calls = 0
                self.too_long = too_long

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                                    return_tensors):
                self.calls += 1
                if add_generation_prompt:
                    return [1, 2]
                return [1, 2] + [3] * (1024 if self.too_long else 1)

        tokenizer = FakeTokenizer()
        job.validate_token_masks(tokenizer, train + validation, 1024)
        self.assertEqual(tokenizer.calls, 84)
        with self.assertRaises(ValueError):
            job.validate_token_masks(FakeTokenizer(too_long=True), train + validation, 1024)

    def test_paid_entrypoint_rejects_before_model_or_azure_calls(self):
        with patch.object(job.launch, "assess_signed_quote") as quote, \
             patch.object(job, "verify_snapshot") as snapshot:
            with self.assertRaisesRegex(job.TrainingRejected, "owner release"):
                job.execute(snapshot=Path("/tmp/unavailable"),
                            output=Path("/tmp/new-output"),
                            quote=Path("/tmp/no-quote"),
                            subscription_id="12345678-1234-1234-1234-123456789abc")
            quote.assert_not_called()
            snapshot.assert_not_called()
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(job.main([]), 0)
        self.assertFalse(json.loads(output.getvalue())["paid_actions_enabled"])


if __name__ == "__main__":
    unittest.main()
