"""Offline tests for the post-training Cosmo adapter receipt."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from training import cosmo_adapter_receipt as receipt
from training.kova_cosmo_sft import EXPECTED_TARGETS


class CosmoAdapterReceiptTests(unittest.TestCase):
    source_commit = "a" * 40
    runtime_evidence_sha256 = "e" * 64
    lifecycle_phase_grant_sha256 = "f" * 64
    lifecycle_id = "lifecycle-001"
    lifecycle_grant_id = "grant-training-001"
    lifecycle_ledger_commit_id = "ledger-commit-002"

    def write_receipt(self, output: Path):
        return receipt.write_receipt(
            output, self.source_commit,
            runtime_evidence_sha256=self.runtime_evidence_sha256,
            lifecycle_phase_grant_sha256=
                self.lifecycle_phase_grant_sha256,
            lifecycle_id=self.lifecycle_id,
            lifecycle_grant_id=self.lifecycle_grant_id,
            lifecycle_ledger_commit_id=self.lifecycle_ledger_commit_id,
            global_steps=18,
            training_loss=1.25,
        )

    def expected_receipt(self, output: Path):
        return receipt.expected_receipt(
            output, self.source_commit,
            runtime_evidence_sha256=self.runtime_evidence_sha256,
            lifecycle_phase_grant_sha256=
                self.lifecycle_phase_grant_sha256,
            lifecycle_id=self.lifecycle_id,
            lifecycle_grant_id=self.lifecycle_grant_id,
            lifecycle_ledger_commit_id=self.lifecycle_ledger_commit_id,
            global_steps=18,
            training_loss=1.25,
        )

    def make_output(self, root: Path) -> Path:
        output = root / "run"
        adapter = output / receipt.ADAPTER_DIRECTORY
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": EXPECTED_TARGETS,
            "base_model_name_or_path": "/external/verified-snapshot",
        }), encoding="utf-8")
        (adapter / "README.md").write_text(
            "Synthetic receipt test fixture; not a trained adapter.\n",
            encoding="utf-8",
        )
        self.write_safetensors(adapter / "adapter_model.safetensors")
        return output

    def write_safetensors(self, path: Path, *, omit_last: bool = False) -> None:
        header = {}
        offset = 0
        for layer in range(28):
            for target in EXPECTED_TARGETS:
                for side in ("A", "B"):
                    name = (f"base_model.model.model.layers.{layer}.{target}."
                            f"lora_{side}.weight")
                    header[name] = {
                        "dtype": "F16", "shape": [1, 1],
                        "data_offsets": [offset, offset + 2],
                    }
                    offset += 2
        if omit_last:
            header.popitem()
            offset -= 2
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))

    def test_receipt_binds_source_lineage_recipe_lock_and_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            report = self.write_receipt(output)
            value = json.loads((output / receipt.RECEIPT_NAME).read_text())
            self.assertEqual(report["status"], "trained_adapter_receipt_verified")
            self.assertEqual(report["source_commit"], self.source_commit)
            self.assertEqual(report["adapter_sha256"], value["adapter_sha256"])
            self.assertRegex(report["receipt_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(value["lineage"]["base_revision"],
                             "c1899de289a04d12100db370d81485cdf75e47ca")
            for field in ("dataset_sha256", "prompt_sha256", "review_sha256",
                          "recipe_sha256", "runtime_guard_sha256",
                          "lifecycle_trust_sha256",
                          "runtime_evidence_sha256",
                          "lifecycle_phase_grant_sha256",
                          "evaluation_plan_sha256", "software_lock_sha256"):
                self.assertRegex(value["lineage"][field], r"^[0-9a-f]{64}$")
            self.assertEqual(value["lineage"]["lifecycle_id"], self.lifecycle_id)
            self.assertEqual(
                report["lifecycle_ledger_commit_id"],
                self.lifecycle_ledger_commit_id,
            )
            self.assertFalse(value["actual_model_outputs_evaluated"])
            self.assertFalse(value["deployment_authorized"])
            self.assertFalse(value["phase_b_ready"])

    def test_adapter_mutation_after_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            self.write_receipt(output)
            weights = output / "adapter/adapter_model.safetensors"
            raw = bytearray(weights.read_bytes())
            raw[-1] ^= 1
            weights.write_bytes(raw)
            with self.assertRaises(receipt.ReceiptError):
                receipt.verify_receipt(output)

    def test_receipt_mutation_and_wrong_expected_commit_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            self.write_receipt(output)
            path = output / receipt.RECEIPT_NAME
            value = json.loads(path.read_text())
            value["phase_b_ready"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(receipt.ReceiptError):
                receipt.verify_receipt(output)

        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            self.write_receipt(output)
            with self.assertRaises(receipt.ReceiptError):
                receipt.verify_receipt(output, expected_source_commit="b" * 40)

    def test_existing_receipt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            self.write_receipt(output)
            original = (output / receipt.RECEIPT_NAME).read_bytes()
            with self.assertRaises(receipt.ReceiptError):
                self.write_receipt(output)
            self.assertEqual((output / receipt.RECEIPT_NAME).read_bytes(), original)

    def test_unknown_files_and_unsafe_weight_formats_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            (output / "adapter/pytorch_model.bin").write_bytes(b"pickle")
            with self.assertRaises(receipt.ReceiptError):
                self.expected_receipt(output)

        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            (output / "adapter/adapter_model.safetensors").write_bytes(b"not-safe")
            with self.assertRaises(receipt.ReceiptError):
                self.expected_receipt(output)

    def test_incomplete_lora_tensor_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            self.write_safetensors(
                output / "adapter/adapter_model.safetensors", omit_last=True
            )
            with self.assertRaises(receipt.ReceiptError):
                self.expected_receipt(output)

    def test_symlinked_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self.make_output(root)
            config = output / "adapter/adapter_config.json"
            target = root / "outside.json"
            target.write_bytes(config.read_bytes())
            config.unlink()
            config.symlink_to(target)
            with self.assertRaises(receipt.ReceiptError):
                self.expected_receipt(output)

    def test_receipt_path_needs_no_network_or_process(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            with patch.object(socket, "socket", side_effect=AssertionError("network")), \
                 patch.object(subprocess, "Popen", side_effect=AssertionError("process")):
                report = self.write_receipt(output)
            self.assertFalse(report["phase_b_ready"])


if __name__ == "__main__":
    unittest.main()
