import unittest

from release.product_policy import validate_checked_in
from router.entitlements import CHAT_ALLOWED_BY_TIER, WORK_ALLOWED_BY_TIER


class ProductPolicyTests(unittest.TestCase):
    def test_exact_matrices_and_closed_execution(self):
        report = validate_checked_in()
        self.assertEqual(tuple(len(CHAT_ALLOWED_BY_TIER[t]) for t in ("free","plus","pro")), (1,6,12))
        self.assertEqual(tuple(len(WORK_ALLOWED_BY_TIER[t]) for t in ("free","plus","pro")), (0,18,18))
        self.assertFalse(report["execution_integration_verified"])
        self.assertFalse(report["phase_b_ready"])


if __name__ == "__main__":
    unittest.main()
