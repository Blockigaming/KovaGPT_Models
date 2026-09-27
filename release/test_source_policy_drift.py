import json
import shutil
import tempfile
import unittest
from pathlib import Path

from release.source_policy_drift import ROOT, validate


class SourcePolicyDriftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for directory in ("config", "router", "release", "docs", "core", "worker", "scripts", "prompts", "ultra"):
            shutil.copytree(ROOT / directory, self.root / directory)

    def tearDown(self):
        self.temp.cleanup()

    def test_current_tree_is_consistent(self):
        self.assertEqual(validate(self.root)["status"], "authoritative_three_family_policy_no_drift")

    def test_nova_chat_drift_fails(self):
        path = self.root / "config/current-product-policy.v3.json"
        value = json.loads(path.read_text())
        value["entitlements"]["chat"]["pro"]["nova"] = ["light"]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "nova_must_be_work_only"):
            validate(self.root)

    def test_same_total_entitlement_cell_swap_fails(self):
        path = self.root / "config/current-product-policy.v3.json"
        value = json.loads(path.read_text())
        value["entitlements"]["chat"]["plus"]["cosmo"] = ["light", "medium", "max"]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "entitlement_runtime_drift:chat:plus"):
            validate(self.root)

    def test_legacy_file_cannot_regain_authority(self):
        path = self.root / "config/product-surface.v1.json"
        value = json.loads(path.read_text())
        value["must_not_drive_current_routing"] = False
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "legacy_file_can_drive_routing"):
            validate(self.root)

    def test_legacy_identity_must_remain_superseded_by_approved_prompt(self):
        path = self.root / "config/identity.v1.json"
        original = path.read_text()
        for field in ("status", "superseded_by"):
            with self.subTest(field=field):
                value = json.loads(original)
                value.pop(field)
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "legacy_identity_source_not_superseded"):
                    validate(self.root)
        path.write_text(original)

    def test_runtime_identity_prompt_must_match_owner_approved_bytes(self):
        path = self.root / "prompts/kova-identity.v3.txt"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "approved_runtime_identity_prompt_drift"):
            validate(self.root)

    def test_runtime_identity_prompt_must_be_available(self):
        (self.root / "prompts/kova-identity.v3.txt").unlink()
        with self.assertRaisesRegex(ValueError, "approved_runtime_identity_prompt_unavailable"):
            validate(self.root)

    def test_runtime_identity_loader_must_point_to_approved_prompt(self):
        path = self.root / "core/identity.py"
        source = path.read_text()
        self.assertIn('"kova-identity.v3.txt"', source)
        path.write_text(source.replace('"kova-identity.v3.txt"', '"kova-identity.v2.txt"'))
        with self.assertRaisesRegex(ValueError, "runtime_identity_loader_drift"):
            validate(self.root)

    def test_active_runtime_cannot_read_legacy_identity(self):
        for relative in ("core/identity.py", "core/adapter.py", "ultra/orchestrator.py", "worker/handler.py"):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_text()
                path.write_text(original + '\nlegacy = "config/identity.v1.json"\n')
                with self.assertRaisesRegex(ValueError, f"runtime_reads_archived_identity:{relative}"):
                    validate(self.root)
                path.write_text(original)

    def test_active_runtime_must_call_approved_identity_loader(self):
        removed_calls = {
            "core/adapter.py": ("identity = load_runtime_identity()", "identity = IDENTITY"),
            "ultra/orchestrator.py": ("identity = load_runtime_identity()", "identity = IDENTITY"),
            "worker/handler.py": (
                "_require(load_runtime_identity() == TRUSTED_SYSTEM_IDENTITY,",
                "_require(TRUSTED_SYSTEM_IDENTITY == TRUSTED_SYSTEM_IDENTITY,",
            ),
        }
        for relative, (original_call, removed_call) in removed_calls.items():
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_text()
                self.assertIn(original_call, original)
                path.write_text(original.replace(original_call, removed_call, 1))
                with self.assertRaisesRegex(ValueError, f"runtime_identity_loader_drift:{relative}"):
                    validate(self.root)
                path.write_text(original)

    def test_public_upstream_name_fails(self):
        path = self.root / "router/policy.py"
        path.write_text(path.read_text() + "\n# Qwen/private\n")
        with self.assertRaisesRegex(ValueError, "private_upstream_leak"):
            validate(self.root)

    def test_runtime_candidate_paths_cannot_read_archived_history(self):
        for relative in ("core/adapter.py", "worker/handler.py"):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_text()
                path.write_text(original + '\nlegacy = "config/core-serving.v1.json"\n')
                with self.assertRaisesRegex(ValueError, f"runtime_reads_archived_routing:{relative}"):
                    validate(self.root)
                path.write_text(original)

    def test_active_inference_contract_cannot_select_archived_candidates(self):
        path = self.root / "config/inference-contract.v1.json"
        value = json.loads(path.read_text())
        value["candidate_selection"]["allowlist_source"] = "config/core-serving.v1.json:candidates"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "runtime_reads_archived_routing:config/inference-contract.v1.json"):
            validate(self.root)

    def test_validator_must_check_current_candidate_registry(self):
        path = self.root / "scripts/validate-stack.mjs"
        path.write_text(path.read_text().replace(
            'inference.candidate_selection.allowlist_source !== "core/current_candidates.py:CORE_SERVING.candidates"',
            'inference.candidate_selection.allowlist_source !== "config/core-serving.v1.json:candidates"',
        ))
        with self.assertRaisesRegex(ValueError, "inference_validator_candidate_source_drift"):
            validate(self.root)

    def test_benchmark_must_not_read_archived_capability_allowlists(self):
        path = self.root / "scripts/summarize-core-benchmark.mjs"
        source = path.read_text()
        path.write_text(source.replace(
            "const servingCapabilities = current.capabilities;",
            'const servingCapabilities = JSON.parse(readFileSync(new URL("../config/core-serving.v1.json", import.meta.url), "utf8"));',
        ))
        with self.assertRaisesRegex(ValueError, "runtime_reads_archived_routing:scripts/summarize-core-benchmark.mjs"):
            validate(self.root)

    def test_benchmark_capabilities_must_follow_active_worker(self):
        path = self.root / "scripts/summarize-core-benchmark.mjs"
        source = path.read_text()
        path.write_text(source.replace(
            "from worker.handler import ALLOWED_SERVING_ENGINES, ALLOWED_ENDPOINT_TYPES",
            "ALLOWED_SERVING_ENGINES = ('old-vllm',); ALLOWED_ENDPOINT_TYPES = ('old-queue',)",
        ))
        with self.assertRaisesRegex(ValueError, "benchmark_capability_source_drift"):
            validate(self.root)


if __name__ == "__main__":
    unittest.main()
