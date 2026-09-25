"""Policy validation is not execution, independent review or checklist closure."""

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from release import product_policy


class PolicyReportingTests(unittest.TestCase):
    def test_resolved_decisions_are_not_reported_as_completed_deliverables(self):
        report = product_policy.validate_checked_in()
        self.assertEqual(report["closed_checklist_ids"], [])
        self.assertEqual(report["resolved_product_decision_ids"], ["A36", "A37"])
        self.assertEqual(report["phase_a_total"], 40)
        self.assertIs(report["product_policy_ready"], True)

    def test_execution_and_independent_review_are_not_inferred_from_policy(self):
        report = product_policy.validate_checked_in()
        self.assertIs(report["execution_integration_verified"], False)
        self.assertIs(report["independent_review_verified"], False)
        self.assertIs(report["phase_b_ready"], False)

    def test_cli_policy_readiness_preserves_explicit_evidence_scope(self):
        root = Path(__file__).resolve().parents[1]
        for args in ([], ["--require-ready"]):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-m", "release.product_policy", *args],
                    cwd=root, capture_output=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                report = json.loads(result.stdout)
                self.assertEqual(report["closed_checklist_ids"], [])
                self.assertEqual(report["resolved_product_decision_ids"], ["A36", "A37"])
                self.assertFalse(report["execution_integration_verified"])
                self.assertFalse(report["phase_b_ready"])

    def test_policy_check_does_not_run_jobs_network_or_external_review(self):
        with patch("socket.socket", side_effect=AssertionError("network")), \
             patch("subprocess.Popen", side_effect=AssertionError("external execution")), \
             patch("execution.store.LocalJobStore.create", side_effect=AssertionError("job submission")):
            report = product_policy.validate_checked_in()
        self.assertEqual(report["closed_checklist_ids"], [])
        self.assertFalse(report["independent_review_verified"])


if __name__ == "__main__":
    unittest.main()
