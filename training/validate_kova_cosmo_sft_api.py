"""Installed dependency/API contract check; never loads a model or starts training."""
from __future__ import annotations

import inspect
from importlib import metadata

from training.kova_cosmo_sft import EXPECTED_SOFTWARE, load_recipe, need


def main() -> int:
    value = load_recipe()
    for distribution, expected in EXPECTED_SOFTWARE.items():
        if distribution == "python":
            continue
        need(metadata.version(distribution) == expected)

    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    need(torch.__version__.split("+")[0] == EXPECTED_SOFTWARE["torch"])
    sft_parameters = inspect.signature(SFTConfig).parameters
    trainer_parameters = inspect.signature(SFTTrainer).parameters
    for parameter in ("model_init_kwargs", "max_length", "completion_only_loss", "packing"):
        need(parameter in sft_parameters)
    for parameter in ("model", "args", "peft_config", "train_dataset", "eval_dataset"):
        need(parameter in trainer_parameters)

    training = value["training"]
    args = SFTConfig(
        output_dir=value["output"]["directory"],
        model_init_kwargs={
            "revision": value["base_revision"],
            "dtype": torch.float16,
            "trust_remote_code": False,
        },
        fp16=True,
        bf16=False,
        max_length=training["max_length"],
        completion_only_loss=True,
        packing=False,
        report_to="none",
        push_to_hub=False,
    )
    need(args.completion_only_loss is True)
    need(args.packing is False)
    need(args.model_init_kwargs["revision"] == value["base_revision"])
    need(args.model_init_kwargs["dtype"] is torch.float16)

    lora = value["lora"]
    peft = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
    )
    need(peft.r == 16 and peft.lora_alpha == 32)
    need(set(peft.target_modules) == set(value["lora"]["target_modules"]))
    print("validated pinned installed SFT/PEFT APIs without model loading or training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
