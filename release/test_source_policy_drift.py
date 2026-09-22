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
        for directory in ("config", "router", "release", "docs"):
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

    def test_public_upstream_name_fails(self):
        path = self.root / "router/policy.py"
        path.write_text(path.read_text() + "\n# Qwen/private\n")
        with self.assertRaisesRegex(ValueError, "private_upstream_leak"):
            validate(self.root)


if __name__ == "__main__":
    unittest.main()
