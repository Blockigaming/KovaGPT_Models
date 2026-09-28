"""Archive integrity checks before any experimental model loading."""

from contextlib import redirect_stdout
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from training import gpu_candidate_inference_smoke as candidate
from training import three_family_contract as contract


class ExperimentalArchiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.archive = Path(self.directory.name) / "adapter.zip"
        self.family = "kova-cosmo"
        self.source = candidate.ARCHIVES[self.family][2]
        self.config = {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16,
            "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        }
        self.weights = b"nonloadable safetensors fixture"
        self.receipt = {
            "kind": "kova_cosmo_gpu_t4_nf4_lora_experiment", "family": self.family,
            "status": "complete", "source_commit": self.source,
            "base_revision": contract.load_json(
                candidate.ROOT / "config/kova-private-lineage.v1.json"
            )["families"][self.family]["immutable_revision"],
            "dataset_sha256": contract.APPROVED_DATASET_SHA256,
            "optimizer_steps": 7, "train_records": 27, "validation_records": 15,
            "signed_controller_pilot": False, "production_routing_approved": False,
        }

    def write_archive(self, *, extra=None, duplicate=False, symlink=False):
        self.receipt["adapter_sha256"] = {
            "adapter_config.json": hashlib.sha256(json.dumps(self.config).encode()).hexdigest(),
            "adapter_model.safetensors": hashlib.sha256(self.weights).hexdigest(),
        }
        members = {
            "adapter/adapter_config.json": json.dumps(self.config).encode(),
            "adapter/adapter_model.safetensors": self.weights,
            "adapter/README.md": b"synthetic experiment, not loadable",
            "receipt.json": json.dumps(self.receipt).encode(),
        }
        if extra:
            members.update(extra)
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for name, data in members.items():
                entry = zipfile.ZipInfo(name)
                entry.create_system = 3
                entry.external_attr = (0o120777 if symlink and name ==
                                       "adapter/adapter_model.safetensors"
                                       else 0o100600) << 16
                bundle.writestr(entry, data)
            if duplicate:
                bundle.writestr("receipt.json", members["receipt.json"])
        raw = self.archive.read_bytes()
        pins = dict(candidate.ARCHIVES)
        pins[self.family] = (len(raw), hashlib.sha256(raw).hexdigest(), self.source)
        return patch.object(candidate, "ARCHIVES", pins)

    def test_complete_receipt_is_validated_and_cli_verify_only_does_not_load_model(self):
        with self.write_archive(), redirect_stdout(StringIO()) as output, \
             patch.object(candidate, "generate", side_effect=AssertionError("model loaded")):
            receipt, members = candidate.inspect_archive(self.family, self.archive)
            self.assertEqual(receipt["adapter_sha256"]["adapter_model.safetensors"],
                             hashlib.sha256(self.weights).hexdigest())
            self.assertEqual(set(members), candidate.MEMBERS)
            self.assertEqual(candidate.main(["--verify-only", "--family", self.family,
                                             "--archive", str(self.archive)]), 0)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["status"], "experimental_archive_verified")
            self.assertFalse(summary["production_routing_approved"])

    def test_any_byte_changed_rejects_before_receipt_or_model_access(self):
        with self.write_archive():
            raw = bytearray(self.archive.read_bytes())
            raw[35] ^= 1
            self.archive.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "archive rejected"):
                candidate.inspect_archive(self.family, self.archive)

    def test_wrong_family_receipt_dataset_or_training_source_is_rejected(self):
        for change in ({"family": "kova-nova"}, {"source_commit": "a" * 40},
                       {"dataset_sha256": "b" * 64}, {"optimizer_steps": True},
                       {"signed_controller_pilot": True},
                       {"production_routing_approved": True}):
            with self.subTest(change=change):
                self.receipt.update(change)
                with self.write_archive(), self.assertRaisesRegex(ValueError,
                        "archive rejected"):
                    candidate.inspect_archive(self.family, self.archive)
                self.receipt = {**self.receipt, **{key: (
                    self.family if key == "family" else self.source if key == "source_commit"
                    else contract.APPROVED_DATASET_SHA256 if key == "dataset_sha256"
                    else 7 if key == "optimizer_steps" else False)
                    for key in change}}

    def test_recipe_drift_and_unsafe_members_rejected_even_with_rehashed_zip(self):
        self.config["r"] = 32
        with self.write_archive(), self.assertRaisesRegex(ValueError, "archive rejected"):
            candidate.inspect_archive(self.family, self.archive)
        self.config["r"] = 16
        for options in ({"extra": {"../other": b"unsafe"}},
                        {"duplicate": True}, {"symlink": True}):
            with self.subTest(options=options), self.write_archive(**options), \
                 self.assertRaisesRegex(ValueError, "archive rejected"):
                candidate.inspect_archive(self.family, self.archive)

    def test_symlink_archive_is_rejected(self):
        with self.write_archive():
            alias = self.archive.with_name("alias.zip")
            alias.symlink_to(self.archive)
            with self.assertRaisesRegex(ValueError, "archive rejected"):
                candidate.inspect_archive(self.family, alias)


if __name__ == "__main__":
    unittest.main()
