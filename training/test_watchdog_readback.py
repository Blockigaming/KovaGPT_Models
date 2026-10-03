"""Offline admission checks; no Azure resources or model execution."""

from copy import deepcopy
import unittest

from training import a35_screen_control as control
from training.cosmo_controller_ledger import LedgerRejected
from training.test_a35_nova_screen import CleanupTests
from training.watchdog_readback import fingerprint, verify_watchdog_properties


class WatchdogReadbackTests(unittest.TestCase):
    def setUp(self):
        self.fixture = CleanupTests()
        self.docs = self.fixture.documents()
        self.url = next(u for u in self.docs if u.endswith("?api-version=2019-05-01") and "/triggers/" not in u)
        self.props = self.docs[self.url]["properties"]
        self.original = deepcopy(self.props)
        self.deadline = self.props["parameters"]["deadlineUtc"]["value"]
        self.definition = deepcopy(self.props["definition"])

    def admit(self):
        return control.verify_before_vm(self.fixture.sub, self.fixture.run_id, 1000,
                                        self.docs.__getitem__, now=1100)

    def reject(self, path):
        with self.assertLogs("training.watchdog_readback", level="ERROR") as logs:
            with self.assertRaises(LedgerRejected) as failure:
                self.admit()
        evidence = failure.exception.evidence
        self.assertIn(path, [m["path"] for m in evidence["mismatches"]])
        self.assertEqual(evidence["observed_sha256"], fingerprint(self.props))
        self.assertEqual(len(evidence["normalized_actual_sha256"]), 64)
        for mismatch in evidence["mismatches"]:
            self.assertIn("expected", mismatch)
            self.assertIn("actual", mismatch)
            self.assertTrue(mismatch["comparison_rule"])
        self.assertIn(path, logs.output[0])
        return logs.output[0]

    def test_exact_readback_admits(self):
        self.assertTrue(self.admit()["live_watchdog_verified"])

    def test_object_key_order_does_not_change_admission(self):
        def reverse(value):
            if type(value) is dict:
                return {key: reverse(value[key]) for key in reversed(value)}
            if type(value) is list:
                return [reverse(item) for item in value]
            return value
        self.docs[self.url]["properties"] = reverse(self.props)
        self.assertTrue(self.admit()["live_watchdog_verified"])

    def test_redundant_string_type_and_evaluated_recurrence_admit_without_mutating_readback(self):
        self.props["parameters"]["deadlineUtc"]["type"] = "String"
        trigger = self.props["definition"]["triggers"]["every_minute"]
        trigger["evaluatedRecurrence"] = deepcopy(trigger["recurrence"])
        before = deepcopy(self.docs)
        self.assertTrue(self.admit()["live_watchdog_verified"])
        self.assertEqual(self.docs, before)

    def test_changed_deadline_or_type_fails_and_records_exact_field(self):
        for key, value in (("value", "2099-01-01T00:00:00Z"), ("type", "SecureString")):
            with self.subTest(key=key):
                self.props["parameters"]["deadlineUtc"] = {"value": self.deadline, key: value}
                self.reject("/properties/parameters/deadlineUtc/" + key)

    def test_disabled_or_unready_workflow_fails(self):
        for key, value in (("state", "Disabled"), ("provisioningState", "Updating")):
            with self.subTest(key=key):
                self.props.update(state="Enabled", provisioningState="Succeeded")
                self.props[key] = value
                self.reject("/properties/" + key)

    def test_foreign_cleanup_uri_fails_without_logging_secret_url(self):
        self.props["definition"]["actions"]["delete_pilot_group"]["inputs"]["uri"] = "https://foreign.invalid/?secret=do-not-log"
        log = self.reject("/properties/definition/actions/delete_pilot_group/inputs/uri")
        self.assertNotIn("do-not-log", log)
        self.assertNotIn("foreign.invalid", log)

    def test_missing_cleanup_action_fails(self):
        self.props["definition"]["actions"].pop("delete_pilot_group")
        self.reject("/properties/definition/actions/delete_pilot_group")

    def test_modified_condition_fails_despite_redundant_metadata(self):
        trigger = self.props["definition"]["triggers"]["every_minute"]
        trigger["evaluatedRecurrence"] = deepcopy(trigger["recurrence"])
        trigger["conditions"][0]["expression"] = "@true"
        self.reject("/properties/definition/triggers/every_minute/conditions/0/expression")

    def test_changed_evaluated_recurrence_never_ignored(self):
        self.props["definition"]["triggers"]["every_minute"]["evaluatedRecurrence"] = {"frequency": "Hour", "interval": 1}
        self.reject("/properties/definition/triggers/every_minute/evaluatedRecurrence")

    def test_bool_is_not_integer_recurrence(self):
        self.props["definition"]["triggers"]["every_minute"]["recurrence"]["interval"] = True
        self.reject("/properties/definition/triggers/every_minute/recurrence/interval")

    def test_unknown_metadata_is_not_silently_ignored(self):
        self.props["parameters"]["deadlineUtc"]["metadata"] = {"secret": "do-not-log"}
        log = self.reject("/properties/parameters/deadlineUtc/metadata")
        self.assertNotIn("do-not-log", log)

    def test_missing_properties_fail_with_persistable_evidence(self):
        with self.assertLogs("training.watchdog_readback", level="ERROR"):
            with self.assertRaises(LedgerRejected) as result:
                verify_watchdog_properties(None, self.deadline, self.definition)
        self.assertEqual(result.exception.evidence["mismatches"][0]["path"], "/properties")


if __name__ == "__main__":
    unittest.main()
