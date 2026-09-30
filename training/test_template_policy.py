"""SFT rows and token-boundary checks must declare the same generation mode."""

import unittest

from training.cosmo_qlora_training import prepared_rows
from training.cosmo_runtime_probe import completion_tokens, RuntimeProbeError
from training.template_policy import CHAT_TEMPLATE_KWARGS


class TemplatePolicyTests(unittest.TestCase):
    def test_each_approved_row_declares_the_non_thinking_mode_for_trl(self):
        train, validation = prepared_rows()
        for row in train + validation:
            self.assertEqual(row["chat_template_kwargs"], CHAT_TEMPLATE_KWARGS)
        train[0]["chat_template_kwargs"]["enable_thinking"] = True
        self.assertIs(validation[0]["chat_template_kwargs"]["enable_thinking"], False)

    def test_rendered_prefix_and_completion_masks_use_the_explicit_row_mode(self):
        calls = []
        class Tokenizer:
            def apply_chat_template(self, messages, *, enable_thinking, add_generation_prompt,
                                    tokenize, return_tensors):
                calls.append(enable_thinking)
                return [10, 11] if add_generation_prompt else [10, 11, 12, 2]
        row = {"prompt": [], "completion": [], "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS)}
        self.assertEqual(completion_tokens(Tokenizer(), row, 8),
                         ([10, 11, 12, 2], [-100, -100, 12, 2]))
        self.assertEqual(calls, [False, False])
        row["chat_template_kwargs"]["enable_thinking"] = True
        with self.assertRaises(RuntimeProbeError):
            completion_tokens(Tokenizer(), row, 8)


if __name__ == "__main__":
    unittest.main()
