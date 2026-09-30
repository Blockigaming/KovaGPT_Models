"""The ledger cannot omit a failure or promote an analyst observation to approval."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from evaluation import preserved_nova_audit as audit
from evaluation.quality_evidence import EvidenceRejected


class PreservedLedgerTests(unittest.TestCase):
    def test_source_classification_covers_all_cases_without_claiming_raw_access(self):
        result = audit.audit()
        self.assertEqual((result["strict_failures_classified"], result["manual_cases_classified"],
                          result["manual_criteria_annotated"]), (13, 14, 48))
        self.assertFalse(result["raw_report_hash_checked"])
        self.assertFalse(result["human_quality_review_complete"])

    def test_omission_or_unverified_manual_credit_is_rejected(self):
        source = json.loads(audit.LEDGER.read_text())
        missing = deepcopy(source); missing["strict_failures"].pop()
        credited = deepcopy(source); credited["manual_cases"][0]["quality_credit_awarded"] = True
        for altered in (missing, credited):
            with patch.object(type(audit.LEDGER), "read_text", return_value=json.dumps(altered)):
                with self.assertRaises(EvidenceRejected):
                    audit.audit()


if __name__ == "__main__":
    unittest.main()
