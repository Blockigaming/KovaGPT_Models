"""No credentials, Azure requests or compute: deterministic operator auth tests."""

import base64
import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from training import a35_operator_auth as a


class RuntimePreflightTests(unittest.TestCase):
    def test_real_verifier_and_rsa_backend_load_before_paid_work(self):
        self.assertEqual(a.preflight_runtime(), {"verifier_imports": True, "rs256_backend": True})

    def test_missing_cryptography_fails_closed(self):
        with patch.object(a, "import_module", side_effect=ModuleNotFoundError("No module named 'cryptography'")):
            with self.assertRaises(ModuleNotFoundError):
                a.preflight_runtime()

    def test_incomplete_jwt_backend_fails_closed(self):
        jwt = Mock()
        jwt.algorithms.get_default_algorithms.return_value = {"HS256": object()}
        with patch.object(a, "import_module", return_value=jwt):
            with self.assertRaisesRegex(ValueError, "RSA verification backend"):
                a.preflight_runtime()

SUB, TENANT, PRINCIPAL = "a" * 8 + "-aaaa-aaaa-aaaa-" + "a" * 12, "b" * 8 + "-bbbb-bbbb-bbbb-" + "b" * 12, "c" * 8 + "-cccc-cccc-cccc-" + "c" * 12


def credential(expiry=1600, resource=a.ARM, **changes):
    claims = dict(aud=resource, tid=TENANT, oid=PRINCIPAL, iat=900, nbf=900,
                  exp=expiry, iss="https://sts.windows.net/" + TENANT + "/", scp="user_impersonation")
    claims.update(changes)
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return dict(accessToken="e30." + payload + ".synthetic", expires_on=expiry,
                subscription=SUB, tenant=TENANT, tokenType="Bearer")


class AuthFixture:
    def setUp(self):
        self.now = 1000
        self.acquire = Mock(return_value=credential())
        self.receipts = []
        self.tokens = a.CliTokens(SUB, TENANT, PRINCIPAL, acquire=self.acquire,
                                 clock=lambda: self.now, observe=self.receipts.append)


class AuthTests(AuthFixture, unittest.TestCase):
    def test_valid_token_cached_only_outside_expiry_margin(self):
        first = self.tokens.get(a.ARM)
        self.now = 1299
        self.assertEqual(self.tokens.get(a.ARM), first)
        self.acquire.assert_called_once()
        self.acquire.return_value = credential(2200)
        self.now = 1300
        self.assertNotEqual(self.tokens.get(a.ARM), first)
        self.assertEqual(self.acquire.call_count, 2)

    def test_exact_old_cache_failure_near_expiry_fails_before_any_http(self):
        self.acquire.return_value = credential(1050)
        with self.assertRaisesRegex(ValueError, "remaining lifetime"):
            self.tokens.get(a.ARM)
        self.assertEqual(self.tokens.cache, {})
        self.now = 1100
        self.acquire.return_value = credential(2200)
        self.tokens.get(a.ARM)
        self.assertEqual(self.acquire.call_count, 2)

    def test_earliest_cli_or_claim_expiry_controls_refresh(self):
        for cli_expiry, jwt_expiry in ((1100, 4000), (4000, 1100)):
            data = credential(cli_expiry, exp=jwt_expiry)
            self.acquire.return_value = data
            with self.assertRaises(ValueError): self.tokens.get(a.ARM)

    def test_missing_malformed_or_expired_metadata_fails_closed(self):
        for expiry in (None, True, False, "tomorrow", "1600.0", 0, -1, 1000):
            self.acquire.return_value = credential() | {"expires_on": expiry}
            with self.subTest(expiry=expiry), self.assertRaises(ValueError): self.tokens.get(a.ARM)

    def test_wrong_account_principal_issuer_or_audience_rejected(self):
        for changes in (dict(oid=TENANT), dict(tid=PRINCIPAL), dict(aud=a.STORAGE),
                        dict(iss="https://untrusted.example/"), dict(nbf=1001), dict(iat=1001),
                        dict(scp="User.Read")):
            self.acquire.return_value = credential(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.tokens.get(a.ARM)
        for changes in (dict(subscription=TENANT), dict(tenant=SUB), dict(tokenType="Basic")):
            self.acquire.return_value = credential() | changes
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.tokens.get(a.ARM)

    def test_exact_arm_alias_and_storage_audiences_stay_separate(self):
        self.acquire.return_value = credential(aud="https://management.core.windows.net/")
        self.tokens.get(a.ARM)
        with self.assertRaisesRegex(ValueError, "audience mismatch"): self.tokens.get(a.STORAGE)
        self.acquire.return_value = credential(resource=a.STORAGE)
        self.tokens.get(a.STORAGE)
        with self.assertRaisesRegex(ValueError, "unapproved token resource"):
            self.tokens.get("https://untrusted.example/")

    def test_acquisition_delay_and_clock_rollback_cannot_extend_token_lifetime(self):
        def slow(_):
            self.now = 1301
            return credential()
        self.acquire.side_effect = slow
        with self.assertRaises(ValueError): self.tokens.get(a.ARM)
        self.acquire.side_effect = None
        self.now = 1000
        self.tokens.get(a.ARM)
        self.now = 899
        with self.assertRaises(ValueError): self.tokens.get(a.ARM)

    def test_receipts_do_not_contain_credentials(self):
        token = self.tokens.get(a.ARM)
        encoded = json.dumps(self.receipts)
        self.assertNotIn(token, encoded)
        self.assertNotIn("accessToken", encoded)
        self.assertEqual(self.receipts[0]["expires_on"], 1600)

    def test_cli_pins_subscription_without_incompatible_tenant_argument(self):
        self.tokens.acquire = self.tokens._cli
        result = Mock(returncode=0, stdout=json.dumps(credential()).encode())
        with patch.object(a.subprocess, "run", return_value=result) as run:
            self.tokens.get(a.ARM)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--subscription") + 1], SUB)
        self.assertNotIn("--tenant", command)

    def test_acquisition_failure_does_not_reuse_stale_credential_or_leak_stderr(self):
        self.tokens.get(a.ARM)
        self.now = 1400
        self.tokens.acquire = self.tokens._cli
        with patch.object(a.subprocess, "run", return_value=Mock(returncode=1, stdout=b"secret")):
            with self.assertRaisesRegex(ValueError, "acquisition failed"):
                self.tokens.get(a.ARM)
        self.assertEqual(self.tokens.cache, {})


class ArmTests(AuthFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.url = a.ARM + "subscriptions/" + SUB + "/resourceGroups/fresh/providers/Microsoft.Resources/deployments/a35-pilot-vm?api-version=2022-09-01"
        self.opener = Mock()
        self.client = a.ArmClient(self.tokens, opener=self.opener, observe=self.receipts.append)

    def response(self, status=200, data=None):
        response = io.BytesIO(json.dumps(data or {"properties": {"provisioningState": "Succeeded"}}).encode())
        response.status = status
        response.headers = {"x-ms-request-id": "synthetic-request"}
        self.opener.open.return_value = response

    def test_deployment_status_read_uses_pinned_arm_token(self):
        self.response()
        status, _, raw = self.client.request("GET", self.url)
        self.assertEqual(status, 200)
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, self.url)
        self.assertEqual(request.get_header("Authorization"), "Bearer " + self.acquire.return_value["accessToken"])
        self.assertEqual(json.loads(raw)["properties"]["provisioningState"], "Succeeded")

    def test_401_and_403_classified_preserved_and_never_retried(self):
        for method in ("GET", "PUT", "POST", "DELETE"):
            for status, code, kind in ((401, "ExpiredAuthenticationToken", "authentication"),
                                       (403, "AuthorizationFailed", "authorization")):
                with self.subTest(method=method, status=status):
                    self.opener.reset_mock()
                    raw = json.dumps({"error": {"code": code, "message": "SENSITIVE-UNLOGGED"}}).encode()
                    self.opener.open.side_effect = HTTPError(self.url, status, "failed", {}, io.BytesIO(raw))
                    with self.assertRaises(a.AuthenticationRejected) as caught:
                        self.client.request(method, self.url, {} if method == "PUT" else None)
                    self.opener.open.assert_called_once()
                    self.assertEqual(caught.exception.receipt["failure_class"], kind)
                    self.assertEqual(caught.exception.receipt["error_code"], code)
                    self.assertNotIn("SENSITIVE-UNLOGGED", str(caught.exception) + json.dumps(self.receipts))
                    self.assertNotIn(a.ARM, self.tokens.cache)

    def test_cleanup_refreshes_before_request_without_replaying_delete(self):
        self.tokens.get(a.ARM)
        self.now = 1301
        self.acquire.return_value = credential(2200)
        self.response(204)
        self.client.request("DELETE", self.url)
        self.assertEqual(self.acquire.call_count, 2)
        self.opener.open.assert_called_once()

    def test_auth_failure_during_deallocation_does_not_suppress_group_cleanup(self):
        from training.a35_screen_control import cleanup
        calls = []
        def call(method, url):
            calls.append((method, url))
            if method == "POST":
                raise a.AuthenticationRejected({"failure_class": "authentication", "status": 401,
                                                "error_code": "ExpiredAuthenticationToken"})
            return (404, {}) if method == "GET" else (202, {})
        result = cleanup(SUB, "d" * 32, call, sleep=lambda _: None)
        self.assertTrue(result["vm_group_absent"] and result["control_group_absent"])
        self.assertEqual([method for method, _ in calls], ["POST", "DELETE", "GET", "DELETE", "GET"])

    def test_bad_destination_or_subscription_rejected_before_acquisition(self):
        for url in (self.url.replace(SUB, TENANT), self.url.replace(SUB, SUB + "1"),
                    self.url.replace("management.azure.com", "management.azure.com.evil.example"),
                    self.url.replace("https://", "http://"), self.url + "#fragment",
                    self.url.replace("resourceGroups", "../resourceGroups"),
                    self.url.replace("resourceGroups", "%2e%2e/resourceGroups")):
            with self.subTest(url=url), self.assertRaises(ValueError): self.client.request("GET", url)
        self.acquire.assert_not_called()
        self.opener.open.assert_not_called()

    def test_redirect_does_not_forward_credentials(self):
        self.assertIsNone(a.NoRedirect().redirect_request(None, None, 302, None, {}, "https://evil.example"))
        self.response(302)
        self.assertEqual(self.client.request("GET", self.url)[0], 302)
        self.opener.open.assert_called_once()

    def test_effective_permissions_require_each_operation_without_exclusions(self):
        action = "Microsoft.Resources/deployments/read"
        a.require_actions({"value": [{"actions": ["*"], "notActions": []}]}, [action])
        a.require_actions({"value": [{"actions": ["*"], "notActions": [action]},
                                    {"actions": [action], "notActions": []}]}, [action])
        for document in ({"value": []}, {"value": [{"actions": ["*"], "notActions": [action]}]},
                         {"value": [], "nextLink": "unread"}, {"value": [{"actions": "*"}]}):
            with self.assertRaises(ValueError): a.require_actions(document, [action])


if __name__ == "__main__":
    unittest.main()
