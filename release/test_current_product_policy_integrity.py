from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from release import current_product_policy as policy


class CurrentPolicyIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads(policy.POLICY_PATH.read_text(encoding="utf-8"))

    def reject(self, value=None, raw=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            path.write_bytes(raw if raw is not None else json.dumps(value).encode())
            with patch.object(policy, "POLICY_PATH", path), self.assertRaisesRegex(
                    policy.CurrentPolicyError, "current product policy rejected"):
                policy.validate()

    def test_open_gates_unknown_fields_and_entitlement_drift_rejected(self):
        changes = []
        a = deepcopy(self.source); a["spending_authorized"] = True; changes.append(a)
        b = deepcopy(self.source); b["unknown"] = False; changes.append(b)
        c = deepcopy(self.source); c["entitlements"]["chat"]["pro"]["nova"] = ["light"]; changes.append(c)
        d = deepcopy(self.source); d["entitlements"]["work"]["free"]["cosmo"] = ["light"]; changes.append(d)
        e = deepcopy(self.source); e["processing_levels_are_separate_models"] = 0; changes.append(e)
        for value in changes:
            with self.subTest(value=value):
                self.reject(value=value)

    def test_invalid_utf8_duplicates_nonfinite_and_oversize_rejected(self):
        samples = [b"\xff", b'{"schema_version":3,"schema_version":3}', b'{"x":NaN}',
                   b" " * (64 * 1024 + 1), b"null"]
        for raw in samples:
            with self.subTest(raw=raw[:30]):
                self.reject(raw=raw)


if __name__ == "__main__":
    unittest.main()
