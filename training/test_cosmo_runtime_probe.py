"""Source-only tests for the guarded selected-checkpoint GPU probe."""
from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training import cosmo_runtime_probe as probe
from training import cosmo_artifacts as artifacts
from training import kova_cosmo_sft as recipe


class FakeModel:
    def __init__(self, layers=2, missing=None):
        self.layers = layers
        self.missing = missing

    def named_modules(self):
        yield "", self
        for layer in range(self.layers):
            for target in recipe.EXPECTED_TARGETS:
                if target != self.missing:
                    yield f"model.layers.{layer}.{target}", object()


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt,
                            return_tensors, tokenize):
        self.last = (messages, return_tensors, tokenize)
        return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]


class CosmoRuntimeProbeTests(unittest.TestCase):
    def snapshot(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        bodies = {name: (name + "\n").encode() for name in probe.REQUIRED_ASSETS}
        bodies["config.json"] = json.dumps({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_hidden_layers": 28,
            "hidden_size": 1024,
            "vocab_size": 151936,
        }).encode()
        bodies["LICENSE"] = b"Apache License\nVersion 2.0, January 2004\n"
        for name, body in bodies.items():
            (directory / name).write_bytes(body)
        expected_hashes = {
            name: hashlib.sha256(bodies[name]).hexdigest()
            for name in probe.EXPECTED_SHA256
        }
        expected_bytes = {
            name: len(bodies[name]) for name in probe.REQUIRED_ASSETS
        }
        return directory, expected_hashes, expected_bytes

    def test_current_probe_reaches_runtime_guard_before_dependencies(self):
        with patch.dict(os.environ, {probe.CONFIRMATION_ENV: "YES",
                                     "KOVA_SOURCE_COMMIT": "a" * 40}), \
             patch.object(probe, "verify_source_checkout"), \
             patch.object(probe, "require_ready",
                          side_effect=probe.RuntimeGuardError("runtime called")), \
             patch.object(probe, "verify_installed_software",
                          side_effect=AssertionError("dependencies called")):
            with self.assertRaisesRegex(probe.RuntimeGuardError, "runtime called"):
                probe.authorize_probe()

    def test_operator_confirmation_cannot_bypass_runtime_guard(self):
        with patch.object(probe, "require_ready",
                          side_effect=probe.RuntimeGuardError("runtime called")), \
             patch.object(probe, "verify_installed_software",
                          side_effect=AssertionError("dependencies called")), \
             patch.dict(os.environ, {probe.CONFIRMATION_ENV: "YES",
                                     "KOVA_SOURCE_COMMIT": "a" * 40}), \
             patch.object(probe, "verify_source_checkout"):
            with self.assertRaisesRegex(probe.RuntimeGuardError, "runtime called"):
                probe.authorize_probe()

    def test_released_source_must_still_pass_external_runtime_guard(self):
        value = deepcopy(recipe.load_recipe())
        value["account_gates"]["eastus_ncast4_quota_verified"] = True
        with patch.object(probe, "load_recipe", return_value=value), \
             patch.dict(os.environ, {probe.CONFIRMATION_ENV: "YES",
                                     "KOVA_SOURCE_COMMIT": "a" * 40}), \
             patch.object(probe, "verify_source_checkout"), \
             patch.object(probe, "require_ready",
                          side_effect=probe.RuntimeGuardError("runtime called")), \
             patch.object(probe, "verify_installed_software",
                          side_effect=AssertionError("dependencies called")):
            with self.assertRaisesRegex(probe.RuntimeGuardError, "runtime called"):
                probe.authorize_probe()

    def test_probe_is_only_for_unverified_runtime(self):
        value = deepcopy(recipe.load_recipe())
        value["account_gates"]["eastus_ncast4_quota_verified"] = True
        value["account_gates"]["runtime_compatibility_verified"] = True
        with patch.object(probe, "load_recipe", return_value=value), \
             patch.dict(os.environ, {probe.CONFIRMATION_ENV: "YES"}):
            with self.assertRaises(probe.RuntimeProbeError):
                probe.authorize_probe()

    def test_probe_reserves_remote_phase_before_dependency_or_model_access(self):
        value = deepcopy(recipe.load_recipe())
        value["account_gates"]["eastus_ncast4_quota_verified"] = True
        runtime = {
            "runtime_evidence_sha256": "e" * 64,
            "deadline_utc": "2026-09-19T19:10:00Z",
            "lifecycle_id": "lifecycle-001",
            "preflight_ledger_sequence": 1,
            "azure_instance": {
                "resource_id": "/subscriptions/11111111-2222-3333-4444-555555555555/resourceGroups/kova-cosmo-pilot/providers/Microsoft.Compute/virtualMachines/kova-cosmo-t4",
                "vm_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "system_assigned_identity_principal_id":
                    "99999999-8888-7777-6666-555555555555",
            },
        }
        with patch.object(probe, "load_recipe", return_value=value), \
             patch.dict(os.environ, {
                 probe.CONFIRMATION_ENV: "YES",
                 "KOVA_SOURCE_COMMIT": "a" * 40,
             }, clear=True), \
             patch.object(probe, "verify_source_checkout"), \
             patch.object(probe, "require_ready", return_value=runtime), \
             patch.object(
                 probe, "acquire_phase_grant",
                 side_effect=RuntimeError("grant boundary reached"),
             ) as acquire, \
             patch.object(
                 probe, "verify_installed_software",
                 side_effect=AssertionError("dependencies probed before grant"),
             ):
            with self.assertRaisesRegex(RuntimeError, "grant boundary reached"):
                probe.authorize_probe()
        self.assertEqual(acquire.call_args.kwargs["phase"], "runtime_probe")
        self.assertEqual(
            acquire.call_args.kwargs["runtime_evidence_sha256"], "e" * 64
        )
        self.assertEqual(
            acquire.call_args.kwargs["azure_instance"],
            runtime["azure_instance"],
        )

    def test_real_qwen_target_inventory_requires_every_target_per_layer(self):
        self.assertEqual(probe.target_inventory(FakeModel()), {
            target: 2 for target in recipe.EXPECTED_TARGETS
        })
        with self.assertRaises(probe.RuntimeProbeError):
            probe.target_inventory(FakeModel(missing="o_proj"))

    def test_snapshot_inventory_binds_weights_tokenizer_config_and_license(self):
        directory, expected, sizes = self.snapshot()
        with patch.object(artifacts, "EXPECTED_SHA256", expected), \
             patch.object(artifacts, "EXPECTED_BYTES", sizes):
            inventory = probe.verify_snapshot(directory)
        self.assertEqual(set(inventory), set(probe.REQUIRED_ASSETS))
        self.assertEqual(inventory["model.safetensors"]["sha256"],
                         expected["model.safetensors"])

    def test_every_snapshot_asset_is_hash_and_size_pinned(self):
        self.assertEqual(set(artifacts.REQUIRED_ASSETS),
                         set(artifacts.EXPECTED_SHA256))
        self.assertEqual(set(artifacts.REQUIRED_ASSETS),
                         set(artifacts.EXPECTED_BYTES))
        self.assertEqual(artifacts.DOWNLOAD_MANIFEST["model"],
                         "Qwen/Qwen3-0.6B")
        self.assertEqual(artifacts.DOWNLOAD_MANIFEST["revision"],
                         "c1899de289a04d12100db370d81485cdf75e47ca")

    def test_snapshot_rejects_drift_in_each_runtime_asset(self):
        for name in probe.REQUIRED_ASSETS:
            with self.subTest(name=name):
                directory, expected, sizes = self.snapshot()
                body = (directory / name).read_bytes()
                (directory / name).write_bytes(body[:-1] + bytes([body[-1] ^ 1]))
                with patch.object(artifacts, "EXPECTED_SHA256", expected), \
                     patch.object(artifacts, "EXPECTED_BYTES", sizes):
                    with self.assertRaises(probe.ArtifactError):
                        probe.verify_snapshot(directory)

    def test_snapshot_rejects_unpinned_top_level_asset(self):
        directory, expected, sizes = self.snapshot()
        (directory / "special_tokens_map.json").write_text(
            "{}", encoding="utf-8"
        )
        with patch.object(artifacts, "EXPECTED_SHA256", expected), \
             patch.object(artifacts, "EXPECTED_BYTES", sizes), \
             self.assertRaises(probe.ArtifactError):
            probe.verify_snapshot(directory)

    def test_completion_mask_excludes_prompt_and_preserves_completion(self):
        row = {"prompt": [{"role": "user", "content": "x"}],
               "completion": [{"role": "assistant", "content": "y"}]}
        tokens, labels = probe.completion_tokens(FakeTokenizer(), row, 16)
        self.assertEqual(tokens, [1, 2, 3, 4, 5])
        self.assertEqual(labels, [-100, -100, -100, 4, 5])

    def test_dry_run_makes_no_runtime_claims(self):
        report = probe.dry_run()
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["quota_verified"])
        self.assertFalse(report["model_weights_downloaded"])
        self.assertFalse(report["pilot_training_started"])
        self.assertFalse(report["phase_b_ready"])

    def test_cli_execute_is_sanitized_and_nonzero_while_blocked(self):
        error = io.StringIO()
        with redirect_stderr(error):
            self.assertEqual(probe.main(["--execute"]), 1)
        self.assertEqual(error.getvalue(),
                         "kova cosmo runtime probe rejected\n")


if __name__ == "__main__":
    unittest.main()
