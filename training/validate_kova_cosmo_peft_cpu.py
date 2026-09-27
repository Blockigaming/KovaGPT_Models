"""Validate the pinned Qwen3/PEFT LoRA path with a tiny synthetic CPU model.

This deliberately creates a random miniature model from configuration. It does
not download or load the selected Cosmo checkpoint and does not train Kova.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from training.cosmo_adapter_receipt import validate_safetensors
from training.kova_cosmo_sft import EXPECTED_TARGETS, load_recipe
from training.probe_cosmo_loss_masks import offline_cpu, verify_environment


class CompatibilityError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompatibilityError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> dict:
    installed = verify_environment()
    recipe = load_recipe()
    lora = recipe["lora"]

    with offline_cpu():
        import torch
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            get_peft_model_state_dict,
        )
        from transformers import Qwen3Config, Qwen3ForCausalLM

        require(torch.version.cuda is None, "CPU-only PyTorch is required")
        torch.set_num_threads(1)
        torch.manual_seed(recipe["training"]["seed"])

        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=False,
        )

        def new_base():
            torch.manual_seed(recipe["training"]["seed"])
            return Qwen3ForCausalLM(config)

        base = new_base()
        base_targets = sorted(
            name for name, _ in base.named_modules()
            if any(name.endswith("." + target) for target in EXPECTED_TARGETS)
        )
        require(len(base_targets) == 2 * len(EXPECTED_TARGETS),
                "tiny Qwen3 model does not expose every LoRA target in every layer")
        for target in EXPECTED_TARGETS:
            require(sum(name.endswith("." + target) for name in base_targets) == 2,
                    "missing or duplicate LoRA target: " + target)

        adapter_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            bias="none",
            target_modules=list(EXPECTED_TARGETS),
        )
        model = get_peft_model(base, adapter_config)
        trainable = {name: parameter for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        require(len(trainable) == 4 * len(EXPECTED_TARGETS),
                "unexpected LoRA trainable tensor inventory")
        require(all("lora_" in name for name in trainable),
                "a non-LoRA parameter is trainable")

        input_ids = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([[-100, -100, -100, -100, 7, 8, 9, 2]],
                              dtype=torch.long)
        before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        require(torch.isfinite(output.loss).item() and output.loss.item() > 0,
                "synthetic loss is not finite and positive")
        output.loss.backward()
        require(all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                    for parameter in trainable.values()),
                "missing or non-finite LoRA gradient")
        require(all(parameter.grad is None for parameter in model.parameters()
                    if not parameter.requires_grad),
                "frozen base parameter received a gradient")

        optimizer = torch.optim.SGD(trainable.values(), lr=0.01)
        optimizer.step()
        updated = sum(not torch.equal(before[name], parameter.detach())
                      for name, parameter in trainable.items())
        require(updated > 0, "synthetic optimizer step changed no LoRA tensor")

        with tempfile.TemporaryDirectory(prefix="kova-peft-cpu-") as directory:
            adapter_dir = Path(directory) / "adapter"
            model.save_pretrained(adapter_dir, safe_serialization=True)
            config_path = adapter_dir / "adapter_config.json"
            weights_path = adapter_dir / "adapter_model.safetensors"
            require(config_path.is_file() and weights_path.is_file(),
                    "safe adapter files were not written")
            validate_safetensors(weights_path, recipe, expected_layers=2)
            written = json.loads(config_path.read_text())
            require(written["r"] == lora["r"] and written["lora_alpha"] == lora["alpha"],
                    "saved adapter rank or alpha drifted")
            require(written["lora_dropout"] == lora["dropout"],
                    "saved adapter dropout drifted")
            require(set(written["target_modules"]) == set(EXPECTED_TARGETS),
                    "saved adapter targets drifted")

            reloaded = PeftModel.from_pretrained(
                new_base(), adapter_dir, is_trainable=True, local_files_only=True
            )
            original_state = get_peft_model_state_dict(model)
            reloaded_state = get_peft_model_state_dict(reloaded)
            require(original_state.keys() == reloaded_state.keys(),
                    "reloaded adapter tensor names drifted")
            require(all(torch.equal(original_state[name], reloaded_state[name])
                        for name in original_state),
                    "reloaded adapter tensors changed")

            model.eval()
            reloaded.eval()
            with torch.no_grad():
                original_logits = model(input_ids=input_ids,
                                        attention_mask=attention_mask).logits
                reloaded_logits = reloaded(input_ids=input_ids,
                                            attention_mask=attention_mask).logits
            require(torch.equal(original_logits, reloaded_logits),
                    "adapter reload changed deterministic logits")
            adapter_sha256 = sha256(weights_path)
            adapter_bytes = weights_path.stat().st_size

    return {
        "status": "synthetic_qwen3_peft_cpu_roundtrip_verified",
        "software": installed,
        "base_revision_declared": recipe["base_revision"],
        "synthetic_model": True,
        "synthetic_layers": 2,
        "matched_target_modules": len(base_targets),
        "trainable_lora_tensors": len(trainable),
        "trainable_lora_parameters": sum(parameter.numel()
                                         for parameter in trainable.values()),
        "updated_lora_tensors": updated,
        "synthetic_optimizer_steps": 1,
        "adapter_tensors_roundtripped": len(original_state),
        "receipt_tensor_inventory_compatible": True,
        "temporary_adapter_sha256": adapter_sha256,
        "temporary_adapter_bytes": adapter_bytes,
        "model_weights_downloaded": False,
        "selected_checkpoint_loaded": False,
        "kova_training_started": False,
        "adapter_retained": False,
        "gpu_verified": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def main() -> int:
    print(json.dumps(validate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
