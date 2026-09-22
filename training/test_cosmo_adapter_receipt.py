"""Offline tests for the post-training Cosmo adapter receipt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from training import cosmo_adapter_receipt as receipt
from training import cosmo_lifecycle_authority as authority
from training.cosmo_generation_attestation import public_key_hex
from training import cosmo_generation_attestation as generation
from training.kova_cosmo_sft import EXPECTED_TARGETS


class CosmoAdapterReceiptTests(unittest.TestCase):
    source_commit = "a" * 40
    runtime_evidence_sha256 = "e" * 64
    lifecycle_phase_grant_sha256 = "f" * 64
    lifecycle_id = "lifecycle-001"
    lifecycle_grant_id = "grant-training-001"
    lifecycle_ledger_commit_id = "ledger-commit-002"

    def setUp(self):
        self.authority_key = Ed25519PrivateKey.from_private_bytes(b"g" * 32)
        public = public_key_hex(self.authority_key)
        self.trust = {
            "schema_version": 1,
            "status": "authority_pinned",
            "issuer": authority.ISSUER,
            "algorithm": "ed25519",
            "endpoint": "https://authority.example/v1/pilot/grants",
            "public_key_hex": public,
            "public_key_sha256": hashlib.sha256(
                bytes.fromhex(public)
            ).hexdigest(),
            "bearer_token_file_environment_variable": authority.TOKEN_ENV,
            "azure_managed_identity_token_audience":
                "api://kova-cosmo-lifecycle-authority",
            "append_only_remote_ledger_required": True,
            "independent_azure_reader_required": True,
            "runner_ledger_mutation_allowed": False,
            "checked_in_private_key_allowed": False,
        }
        self.receipt_signing_key = Ed25519PrivateKey.from_private_bytes(b"r" * 32)
        receipt_public = public_key_hex(self.receipt_signing_key)
        self.generation_trust = {
            "schema_version": 1,
            "status": "runner_signing_public_key_pinned",
            "algorithm": "ed25519",
            "public_key_hex": receipt_public,
            "public_key_sha256": hashlib.sha256(
                bytes.fromhex(receipt_public)
            ).hexdigest(),
            "runner_private_key_environment_variable": generation.PRIVATE_KEY_ENV,
            "verifier_private_key_access_allowed": False,
            "checked_in_private_key_allowed": False,
        }

    def trust_patch(self):
        return patch.object(
            authority, "load_trust_policy", return_value=self.trust
        )

    def grant_context(self, output: Path) -> dict:
        return {
            "operation": "single_lora_sft_run",
            "base_model": "Qwen/Qwen3-0.6B",
            "base_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "output_directory": str(output),
            "snapshot_inventory": {
                name: {
                    "bytes": receipt.EXPECTED_BYTES[name],
                    "sha256": digest,
                }
                for name, digest in receipt.EXPECTED_SHA256.items()
            },
        }

    def grant_envelope(self, context: dict) -> dict:
        azure_instance = {
            "resource_id": (
                "/subscriptions/11111111-2222-3333-4444-555555555555/"
                "resourceGroups/kova-cosmo-pilot/providers/"
                "Microsoft.Compute/virtualMachines/kova-cosmo-t4"
            ),
            "vm_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "system_assigned_identity_principal_id":
                "99999999-8888-7777-6666-555555555555",
        }
        payload = {
            "schema_version": 1,
            "kind": "kova_cosmo_paid_phase_grant",
            "issuer": authority.ISSUER,
            "pilot_id": authority.PILOT_ID,
            "lifecycle_id": self.lifecycle_id,
            "preflight_ledger_sequence": 1,
            "ledger_sequence": 3,
            "ledger_commit_id": self.lifecycle_ledger_commit_id,
            "ledger_append_only": True,
            "ledger_status": "grant_committed_before_response",
            "grant_id": self.lifecycle_grant_id,
            "phase": "training",
            "source_commit": self.source_commit,
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "context_sha256": hashlib.sha256(
                authority.canonical(context)
            ).hexdigest(),
            "request_nonce": "d" * 64,
            "runtime_deadline_utc": "2026-09-19T19:40:00Z",
            "azure_instance": azure_instance,
            "azure_instance_identity": {
                "verification_method":
                    "microsoft_entra_system_assigned_managed_identity_token",
                "token_sha256": "c" * 64,
                "token_audience": self.trust[
                    "azure_managed_identity_token_audience"
                ],
                "verified_at_utc": "2026-09-19T19:00:00Z",
                "token_expires_at_utc": "2026-09-19T20:00:00Z",
                "verified": True,
            },
            "issued_at_utc": "2026-09-19T19:00:00Z",
            "expires_at_utc": "2026-09-19T19:40:00Z",
            "grant_reserved_seconds": 2400,
            "grant_reserved_cost_usd": "0.3507",
            "phase_grants_committed": {
                "runtime_probe": 1,
                "training": 1,
                "evaluation": 0,
            },
            "training_runs_consumed": 1,
            "aggregate_reserved_seconds": 3000,
            "aggregate_reserved_cost_usd": "0.4384",
            "approved_budget_usd": "2.0000",
            "deployment_authorized": False,
        }
        return {
            "payload": payload,
            "signature": self.authority_key.sign(
                authority.canonical(payload)
            ).hex(),
        }

    def write_receipt(self, output: Path):
        with self.trust_patch(), patch.object(
            receipt, "load_generation_trust_policy",
            return_value=self.generation_trust,
        ):
            return receipt.write_receipt(
                output, self.source_commit,
                global_steps=18,
                signing_key=self.receipt_signing_key,
            )

    def expected_receipt(self, output: Path):
        with self.trust_patch():
            return receipt.expected_receipt(
                output, self.source_commit,
                global_steps=18,
            )

    def verify_receipt(self, output: Path, **arguments):
        with self.trust_patch(), patch.object(
            receipt, "load_generation_trust_policy",
            return_value=self.generation_trust,
        ):
            return receipt.verify_receipt(output, **arguments)

    def make_output(self, root: Path) -> Path:
        output = root / "run"
        context = self.grant_context(output)
        with self.trust_patch():
            receipt.persist_training_grant(
                output,
                source_commit=self.source_commit,
                runtime_evidence_sha256=self.runtime_evidence_sha256,
                lifecycle_id=self.lifecycle_id,
                grant_context=context,
                grant_envelope=self.grant_envelope(context),
            )
        adapter = output / receipt.ADAPTER_DIRECTORY
        adapter.mkdir()
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
                    block = ("self_attn" if target in
                             ("q_proj", "k_proj", "v_proj", "o_proj") else
                             "mlp")
                    name = (f"base_model.model.model.layers.{layer}.{block}."
                            f"{target}."
                            f"lora_{side}.weight")
                    rank = 16
                    hidden = 1024
                    intermediate = 3072
                    input_width = {
                        "q_proj": hidden, "k_proj": hidden,
                        "v_proj": hidden, "o_proj": hidden,
                        "gate_proj": hidden, "up_proj": hidden,
                        "down_proj": intermediate,
                    }[target]
                    output_width = {
                        "q_proj": 2048, "k_proj": 1024,
                        "v_proj": 1024, "o_proj": hidden,
                        "gate_proj": intermediate, "up_proj": intermediate,
                        "down_proj": hidden,
                    }[target]
                    shape = ([rank, input_width] if side == "A" else
                             [output_width, rank])
                    size = shape[0] * shape[1] * 2
                    header[name] = {
                        "dtype": "F16", "shape": shape,
                        "data_offsets": [offset, offset + size],
                    }
                    offset += size
        if omit_last:
            header.popitem()
            removed = next(reversed(header.values()), None)
            # popitem above removed the final tensor; trim its encoded bytes.
            if removed is not None:
                offset = max(item["data_offsets"][1]
                             for item in header.values())
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
                self.verify_receipt(output)

    def test_rebuilt_receipt_for_replaced_adapter_needs_original_signer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            self.write_receipt(output)
            weights = output / "adapter/adapter_model.safetensors"
            raw = bytearray(weights.read_bytes())
            raw[-1] ^= 1
            weights.write_bytes(raw)
            rebuilt = self.expected_receipt(output)
            (output / receipt.RECEIPT_NAME).write_bytes(receipt.serialize(rebuilt))
            with self.assertRaises(receipt.ReceiptError):
                self.verify_receipt(output)

    def test_qwen3_attention_projection_widths_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            self.expected_receipt(output)
            weights = output / "adapter/adapter_model.safetensors"
            raw = weights.read_bytes()
            header_size = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8:8 + header_size])
            q = header[
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
            ]
            k = header[
                "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"
            ]
            self.assertEqual(q["shape"], [2048, 16])
            self.assertEqual(k["shape"], [1024, 16])

    def test_receipt_uses_deterministic_steps_and_omits_unauthenticated_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            self.write_receipt(output)
            path = output / receipt.RECEIPT_NAME
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["training"]["global_steps"], 18)
            self.assertNotIn("training_loss", value["training"])
            value["training"]["global_steps"] = 19
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(receipt.ReceiptError):
                self.verify_receipt(output)

    def test_fake_or_wrong_phase_training_grant_is_rejected(self):
        for variant in (
            "fake_signer", "wrong_phase", "wrong_pilot", "uncommitted",
            "stale_sequence", "under_reserved_aggregate",
        ):
            with self.subTest(variant=variant), \
                 tempfile.TemporaryDirectory() as directory:
                output = self.make_output(Path(directory))
                path = output / receipt.TRAINING_GRANT_NAME
                evidence = json.loads(path.read_text(encoding="utf-8"))
                payload = evidence["grant_envelope"]["payload"]
                if variant == "fake_signer":
                    payload["grant_id"] = "invented-grant"
                    fake = Ed25519PrivateKey.from_private_bytes(b"f" * 32)
                    signer = fake
                elif variant == "wrong_phase":
                    payload["phase"] = "evaluation"
                    signer = self.authority_key
                elif variant == "wrong_pilot":
                    payload["pilot_id"] = "other-pilot"
                    signer = self.authority_key
                elif variant == "uncommitted":
                    payload["ledger_status"] = "pending_commit"
                    signer = self.authority_key
                elif variant == "stale_sequence":
                    payload["preflight_ledger_sequence"] = 3
                    signer = self.authority_key
                else:
                    payload["aggregate_reserved_cost_usd"] = "0.3507"
                    signer = self.authority_key
                evidence["grant_envelope"]["signature"] = signer.sign(
                    authority.canonical(payload)
                ).hex()
                path.write_text(json.dumps(evidence), encoding="utf-8")
                with self.assertRaises(receipt.ReceiptError):
                    self.write_receipt(output)

    def test_grant_context_or_vm_lineage_tampering_is_rejected(self):
        for variant in ("context", "vm"):
            with self.subTest(variant=variant), \
                 tempfile.TemporaryDirectory() as directory:
                output = self.make_output(Path(directory))
                path = output / receipt.TRAINING_GRANT_NAME
                evidence = json.loads(path.read_text(encoding="utf-8"))
                if variant == "context":
                    evidence["context"]["output_directory"] += "-other"
                else:
                    evidence["grant_envelope"]["payload"][
                        "azure_instance"
                    ]["vm_id"] = "00000000-1111-2222-3333-444444444444"
                path.write_text(json.dumps(evidence), encoding="utf-8")
                with self.assertRaises(receipt.ReceiptError):
                    self.write_receipt(output)

    def test_receipt_mutation_and_wrong_expected_commit_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            self.write_receipt(output)
            path = output / receipt.RECEIPT_NAME
            value = json.loads(path.read_text())
            value["phase_b_ready"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(receipt.ReceiptError):
                self.verify_receipt(output)

        with tempfile.TemporaryDirectory() as directory:
            output = self.make_output(Path(directory))
            self.write_receipt(output)
            with self.assertRaises(receipt.ReceiptError):
                self.verify_receipt(
                    output, expected_source_commit="b" * 40
                )

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

    def test_duplicate_side_or_wrong_lora_shape_is_rejected(self):
        for variant in ("duplicate_side", "wrong_shape"):
            with self.subTest(variant=variant), \
                 tempfile.TemporaryDirectory() as directory:
                output = self.make_output(Path(directory))
                path = output / "adapter/adapter_model.safetensors"
                raw = path.read_bytes()
                size = struct.unpack("<Q", raw[:8])[0]
                header = json.loads(raw[8:8 + size])
                names = list(header)
                if variant == "duplicate_side":
                    old = next(name for name in names if ".lora_B." in name)
                    new = old.replace(".lora_B.", ".lora_A_duplicate.")
                    header[new] = header.pop(old)
                else:
                    header[names[0]]["shape"][0] += 1
                encoded = json.dumps(header, separators=(",", ":")).encode()
                path.write_bytes(struct.pack("<Q", len(encoded)) + encoded +
                                 raw[8 + size:])
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
