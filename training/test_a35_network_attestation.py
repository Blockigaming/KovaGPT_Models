"""The captured Azure attachment casing must pass without accepting drift."""
from copy import deepcopy
import unittest

from training.a35_network_attestation import private_nic_attachment_matches

NIC = ("/subscriptions/12345678-1234-1234-1234-123456789abc"
       "/resourceGroups/kova-a35-nova-0123456789/providers/Microsoft.Network"
       "/networkInterfaces/kova-t4-nic-0123456789")
EXPECTED = NIC + "/ipConfigurations/private"
# Exact observed shape; private account/run identifiers replaced with fixtures.
OBSERVED = [{"id": EXPECTED.replace("kova-a35-nova-0123456789", "KOVA-A35-NOVA-0123456789")
             .replace("kova-t4-nic-0123456789", "KOVA-T4-NIC-0123456789")
             .replace("/private", "/PRIVATE")}]


class PrivateNicAttachmentTests(unittest.TestCase):
    def test_reproduce_original_case_sensitive_failure_and_corrected_pass(self):
        self.assertNotEqual(OBSERVED, [{"id": EXPECTED}])
        self.assertTrue(private_nic_attachment_matches(OBSERVED, NIC))

    def test_original_exact_case_remains_valid(self):
        self.assertTrue(private_nic_attachment_matches([{"id": EXPECTED}], NIC))

    def test_complete_ascii_case_variants_pass(self):
        for value in (EXPECTED.lower(), EXPECTED.upper(), OBSERVED[0]["id"]):
            with self.subTest(value=value):
                self.assertTrue(private_nic_attachment_matches([{"id": value}], NIC))

    def test_wrong_subscription_group_nic_or_configuration_fails(self):
        changes = (("123456789abc", "123456789abd"),
                   ("kova-a35-nova-0123456789", "kova-a35-nova-9876543210"),
                   ("kova-t4-nic-0123456789", "kova-t4-nic-9876543210"),
                   ("/private", "/other"), ("Microsoft.Network", "Microsoft.Compute"))
        for old, new in changes:
            with self.subTest(change=(old,new)):
                self.assertFalse(private_nic_attachment_matches([{"id": EXPECTED.replace(old,new)}], NIC))

    def test_missing_duplicate_or_extra_reference_fields_fail(self):
        for value in (None, [], {}, [None], [{}], [{"id": None}],
                      OBSERVED * 2, [{"id": EXPECTED, "properties": {}}]):
            with self.subTest(value=value):
                self.assertFalse(private_nic_attachment_matches(value, NIC))

    def test_uri_and_unicode_ambiguities_fail(self):
        for value in (EXPECTED+"/", EXPECTED+"?x=1", EXPECTED+"#x", " "+EXPECTED,
                      "https://management.azure.com"+EXPECTED,
                      EXPECTED.replace("private", "%70rivate"),
                      EXPECTED.replace("private", "prıvate"),
                      EXPECTED.replace("/private", "/../private")):
            with self.subTest(value=value):
                self.assertFalse(private_nic_attachment_matches([{"id": value}], NIC))

    def test_malformed_expected_identity_fails_closed(self):
        for value in (None, "", "nic", NIC+"/", NIC+"?x=1", NIC.replace("123456789abc","wrong")):
            with self.subTest(value=value):
                self.assertFalse(private_nic_attachment_matches(OBSERVED, value))

    def test_evidence_is_not_rewritten(self):
        before=deepcopy(OBSERVED)
        self.assertTrue(private_nic_attachment_matches(OBSERVED,NIC))
        self.assertEqual(OBSERVED,before)


if __name__ == '__main__':
    unittest.main()
