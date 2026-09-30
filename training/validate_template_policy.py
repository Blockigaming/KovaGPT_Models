"""Exercise real pinned TRL preprocessing with a tiny, local tokenizer.

No pretrained weights, hub downloads, inference or training. This verifies the
row-level API and completion masks, not the unavailable immutable Qwen assets.
"""

from tempfile import TemporaryDirectory


def validate() -> None:
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from training.template_policy import template_row
    from training.cosmo_runtime_probe import completion_tokens

    words = ["<unk>", "<pad>", "<eos>", "system", "user", "assistant",
             "QUIET", "THINK", "context", "question", "answer"]
    local = Tokenizer(models.WordLevel(dict((word, i) for i, word in enumerate(words)),
                                       unk_token="<unk>"))
    local.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=local, unk_token="<unk>",
                                       pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = (
        "{{ 'QUIET ' if enable_thinking is defined and not enable_thinking else 'THINK ' }}"
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} "
        "{% if m['role'] == 'assistant' %}{{ eos_token }} {% endif %}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}")
    row = template_row([{"role": "system", "content": "context"},
                        {"role": "user", "content": "question"}],
                       [{"role": "assistant", "content": "answer"}])
    full, labels = completion_tokens(tokenizer, row, 32)
    with TemporaryDirectory(prefix="kova-template-api-") as temporary:
        settings = SFTConfig(output_dir=temporary, use_cpu=True,
                             fp16=False, bf16=False, max_length=32, packing=False,
                             completion_only_loss=True, report_to="none")
        preparer = object.__new__(SFTTrainer)
        preparer._tokenizer = tokenizer
        preparer.chat_template = tokenizer.chat_template
        preparer.completion_only_loss = True
        dataset = preparer._prepare_dataset(Dataset.from_list([row]), tokenizer,
                                             settings, False, None, "train")
        record = dataset[0]
        if record["input_ids"] != full or full[0] != words.index("QUIET"):
            raise ValueError("TRL ignored explicit row template mode")
        batch = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)([record])
        if batch["labels"][0].tolist() != labels or labels[-1] != tokenizer.eos_token_id:
            raise ValueError("TRL completion mask or EOS changed")
    print("verified real TRL row template kwargs and completion-only masks without model weights")


if __name__ == "__main__":
    validate()
