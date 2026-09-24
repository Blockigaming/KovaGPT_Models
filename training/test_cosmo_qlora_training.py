"""The approved QLoRA data path and source-only execution boundary."""

from __future__ import annotations

from io import StringIO
from contextlib import redirect_stdout
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from training import cosmo_qlora_training as job


class QloraTrainingTests(unittest.TestCase):
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
