"""No credentials, Azure requests or compute: deterministic operator auth tests."""

import base64
import io
import json
from pathlib import Path
import tempfile
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


class PreflightRefreshTests(AuthFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.elapsed = 0
        self.waits = []
        def sleep(seconds):
            self.waits.append(seconds)
            self.now += seconds
            self.elapsed += seconds
        self.tokens.sleep = sleep
        self.tokens.monotonic = lambda: self.elapsed

    def test_forced_preflight_bypasses_even_valid_operator_cache(self):
        self.tokens.get(a.ARM)
        self.acquire.return_value = credential(2200)
        self.assertEqual(self.tokens.get(a.ARM, force=True), self.acquire.return_value['accessToken'])
        self.assertEqual(self.acquire.call_count, 2)
        self.assertEqual(self.waits, [])

    def test_actual_failure_lifetime_renews_once_without_arm_or_owner_login(self):
        old, fresh = credential(1077), credential(2200)
        self.acquire.side_effect = [old, fresh]
        result = self.tokens.get(a.ARM, force=True)
        self.assertEqual(result, fresh['accessToken'])
        self.assertEqual(self.acquire.call_count, 2)
        self.assertEqual(sum(self.waits), 78)
        self.assertTrue(all(0 < x <= 30 for x in self.waits))
        self.assertEqual(self.receipts[-1]['remaining_seconds'], 1122)
        self.assertNotIn(old['accessToken'], json.dumps(self.receipts))
        self.assertNotIn(fresh['accessToken'], json.dumps(self.receipts))

    def test_same_aging_replacement_fails_closed_without_http_or_cache(self):
        self.acquire.return_value = credential(1077)
        with self.assertRaisesRegex(ValueError, 'remaining lifetime'):
            self.tokens.get(a.ARM, force=True)
        self.assertEqual(self.acquire.call_count, 2)
        self.assertEqual(self.tokens.cache, {})
        self.assertFalse(self.receipts[-1]['http_request_sent'])

    def test_expired_token_is_discarded_before_single_noninteractive_acquisition(self):
        self.acquire.side_effect = [credential(999), credential(2200)]
        self.tokens.get(a.ARM, force=True)
        self.assertEqual(self.waits, [])
        self.assertEqual(self.acquire.call_count, 2)

    def test_exact_margin_wait_is_bounded_and_strict_guard_is_preserved(self):
        self.acquire.side_effect = [credential(1300), credential(2200)]
        self.tokens.get(a.ARM, force=True)
        self.assertEqual(sum(self.waits), 301)
        self.assertEqual(len(self.waits), 11)
        self.assertEqual(self.acquire.call_count, 2)

    def test_wrong_identity_audience_or_future_token_never_triggers_refresh(self):
        for changes in ({'oid':TENANT}, {'aud':a.STORAGE}, {'iat':1001}, {'nbf':1001}):
            self.acquire.reset_mock()
            self.acquire.return_value = credential(1077, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.tokens.get(a.ARM, force=True)
            self.acquire.assert_called_once()
            self.assertEqual(self.waits, [])
            self.assertEqual(self.tokens.cache, {})

    def test_replacement_identity_and_lifetime_are_fully_revalidated(self):
        for replacement in (credential(2200, oid=TENANT), credential(2200, aud=a.STORAGE),
                            credential(1350), credential() | {'expires_on':None}):
            self.setUp()
            self.acquire.side_effect = [credential(1077), replacement]
            with self.subTest(replacement=replacement['expires_on']), self.assertRaises(ValueError):
                self.tokens.get(a.ARM, force=True)
            self.assertEqual(self.acquire.call_count, 2)
            self.assertEqual(self.tokens.cache, {})

    def test_refresh_failure_never_falls_back_to_discarded_token(self):
        self.acquire.side_effect = [credential(1077), ValueError('acquisition failed')]
        with self.assertRaisesRegex(ValueError, 'acquisition failed'):
            self.tokens.get(a.ARM, force=True)
        self.assertEqual(self.tokens.cache, {})

    def test_frozen_clock_cannot_create_unbounded_wait_or_admit_stale_token(self):
        self.tokens.sleep = Mock()
        self.acquire.return_value = credential(1077)
        with self.assertRaisesRegex(ValueError, 'wait did not complete'):
            self.tokens.get(a.ARM, force=True)
        self.assertEqual(self.tokens.sleep.call_count, 11)
        self.acquire.assert_called_once()
        self.assertEqual(self.tokens.cache, {})

    def test_ordinary_arm_operation_never_waits_or_sends_near_expiry_token(self):
        self.acquire.return_value = credential(1077)
        opener = Mock()
        client = a.ArmClient(self.tokens, opener=opener)
        with self.assertRaises(a.CredentialUnavailable):
            client.request('PUT', a.ARM+'subscriptions/'+SUB+'/resourceGroups/fresh', {})
        opener.open.assert_not_called()
        self.acquire.assert_called_once()
        self.assertEqual(self.waits, [])

    def test_storage_force_rolls_over_once_without_affecting_arm(self):
        self.acquire.side_effect = [credential(1180, resource=a.STORAGE),
                                    credential(2200, resource=a.STORAGE)]
        self.tokens.get(a.STORAGE, force=True)
        self.assertEqual(self.acquire.call_count, 2)
        self.assertEqual(sum(self.waits), 181)
        self.assertNotIn(a.ARM, self.tokens.cache)


class RuntimeArmReadRefreshTests(AuthFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.waits = []
        def sleep(seconds):
            self.waits.append(seconds)
            self.now += seconds
        self.tokens.sleep = sleep
        self.tokens.monotonic = lambda: self.now
        self.opener = Mock()
        def response(*args, **kwargs):
            value = io.BytesIO(b'{"properties":{"provisioningState":"Succeeded"}}')
            value.status, value.headers = 200, {'x-ms-request-id':'synthetic'}
            return value
        self.opener.open.side_effect = response
        self.client = a.ArmClient(self.tokens, opener=self.opener, observe=self.receipts.append)
        self.url = a.ARM+'subscriptions/'+SUB+'/resourceGroups/fresh/providers/Microsoft.Resources/deployments/a35-pilot-vm?api-version=2022-09-01'

    def test_admitted_arm_cache_ages_during_vm_poll_then_rolls_before_get(self):
        old, fresh = credential(1600), credential(2600)
        self.acquire.return_value = old
        self.tokens.get(a.ARM, force=True)
        self.now = 1301
        self.acquire.side_effect = [old, fresh]
        self.assertEqual(self.client.request('GET', self.url)[0], 200)
        self.assertEqual(sum(self.waits), 300)
        self.opener.open.assert_called_once()
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'), 'Bearer '+fresh['accessToken'])
        self.assertNotIn(old['accessToken'], json.dumps(self.receipts))

    def test_arm_299_300_301_boundaries_preserve_strict_margin(self):
        for remaining in (299,300,301):
            self.setUp()
            self.acquire.side_effect = [credential(1000+remaining),credential(2600)]
            self.client.request('GET', self.url)
            self.assertEqual(sum(self.waits), remaining+1 if remaining<=300 else 0)
            self.assertEqual(self.acquire.call_count, 2 if remaining<=300 else 1)
            self.opener.open.assert_called_once()

    def test_stale_or_near_expiry_replacement_blocks_get_without_fallback(self):
        for fresh in (credential(1299),credential(1599),credential(1600)):
            self.setUp()
            self.acquire.side_effect = [credential(1299),fresh]
            with self.assertRaises(a.CredentialUnavailable):self.client.request('GET',self.url)
            self.opener.open.assert_not_called()
            self.assertEqual(self.tokens.cache,{})
            self.assertEqual(self.acquire.call_count,2)

    def test_invalid_identity_or_missing_expiry_never_waits_or_sends(self):
        for value in (credential(1299,oid=TENANT),credential(1299,aud=a.STORAGE),
                      credential(1299)|{'expires_on':None}):
            self.setUp();self.acquire.return_value=value
            with self.assertRaises(a.CredentialUnavailable):self.client.request('GET',self.url)
            self.assertEqual(self.waits,[])
            self.opener.open.assert_not_called()

    def test_valid_runtime_cache_is_reused_without_unnecessary_cli_call(self):
        self.tokens.get(a.ARM)
        self.client.request('GET',self.url)
        self.acquire.assert_called_once()
        self.assertEqual(self.waits,[])

    def test_mutations_never_wait_or_replay_with_aging_broker_credential(self):
        for method in ('PUT','POST','PATCH','DELETE'):
            self.setUp();self.acquire.return_value=credential(1299)
            with self.assertRaises(a.CredentialUnavailable):self.client.request(method,self.url,{})
            self.opener.open.assert_not_called()
            self.acquire.assert_called_once()
            self.assertEqual(self.waits,[])

    def test_arm_rollover_keeps_storage_cache_and_audience_independent(self):
        storage=credential(2600,resource=a.STORAGE)
        self.acquire.return_value=storage;self.tokens.get(a.STORAGE)
        self.acquire.side_effect=[credential(1299),credential(2600)]
        self.client.request('GET',self.url)
        self.assertEqual(self.tokens.get(a.STORAGE),storage['accessToken'])
        self.assertEqual(self.acquire.call_count,3)

    def test_refresh_acquisition_failure_blocks_get_without_http_retry(self):
        self.acquire.side_effect=[credential(1299),ValueError('acquisition failed')]
        with self.assertRaises(a.CredentialUnavailable):self.client.request('GET',self.url)
        self.opener.open.assert_not_called()
        self.assertEqual(self.tokens.cache,{})


class StorageRefreshTests(AuthFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.waits = []
        def sleep(seconds):
            self.waits.append(seconds)
            self.now += seconds
        self.tokens.sleep = sleep
        self.tokens.monotonic = lambda: self.now

    def test_actual_180_second_ordinary_writer_read_refreshes_before_use(self):
        old = credential(1180, resource=a.STORAGE)
        fresh = credential(1481, resource=a.STORAGE)
        self.acquire.side_effect = [old, fresh]
        self.assertEqual(self.tokens.get(a.STORAGE), fresh['accessToken'])
        self.assertEqual(sum(self.waits), 181)
        self.assertEqual(self.acquire.call_count, 2)
        self.assertEqual(self.receipts[-1]['remaining_seconds'], 300)
        self.assertTrue(all(0 < seconds <= 30 for seconds in self.waits))

    def test_storage_299_300_301_second_admission_boundaries(self):
        for seconds in (299, 300, 301):
            self.setUp()
            first = credential(1000 + seconds, resource=a.STORAGE)
            fresh = credential(2400, resource=a.STORAGE)
            self.acquire.side_effect = [first, fresh]
            with self.subTest(seconds=seconds):
                actual = self.tokens.get(a.STORAGE)
                self.assertEqual(actual, (fresh if seconds < 300 else first)['accessToken'])
                self.assertEqual(self.acquire.call_count, 2 if seconds < 300 else 1)

    def test_refreshed_299_rejects_300_and_301_pass(self):
        for seconds in (299, 300, 301):
            self.setUp()
            self.acquire.side_effect = [credential(1180, resource=a.STORAGE),
                                        credential(1181 + seconds, resource=a.STORAGE)]
            with self.subTest(seconds=seconds):
                if seconds < 300:
                    with self.assertRaisesRegex(ValueError, 'remaining lifetime'):
                        self.tokens.get(a.STORAGE)
                    self.assertEqual(self.tokens.cache, {})
                else:
                    self.tokens.get(a.STORAGE)
                self.assertEqual(self.acquire.call_count, 2)

    def test_aging_operator_cache_is_discarded_and_broker_token_rolls_once(self):
        old = credential(1600, resource=a.STORAGE)
        self.acquire.return_value = old
        self.tokens.get(a.STORAGE)
        self.now = 1420
        fresh = credential(2200, resource=a.STORAGE)
        self.acquire.side_effect = [old, fresh]
        self.assertEqual(self.tokens.get(a.STORAGE), fresh['accessToken'])
        self.assertEqual(self.acquire.call_count, 3)
        self.assertEqual(sum(self.waits), 181)

    def test_missing_malformed_expiry_never_waits_or_caches(self):
        for field in ('expires_on', 'exp'):
            for bad in (None, True, False, 0, -1, 'tomorrow', '1300.0'):
                self.setUp()
                value = credential(1180, resource=a.STORAGE)
                if field == 'exp': value = credential(1180, resource=a.STORAGE, exp=bad)
                else: value[field] = bad
                self.acquire.return_value = value
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    self.tokens.get(a.STORAGE)
                self.assertEqual(self.waits, [])
                self.assertEqual(self.tokens.cache, {})

    def test_same_stale_replacement_fails_closed_without_a_third_acquisition(self):
        self.acquire.return_value = credential(1180, resource=a.STORAGE)
        with self.assertRaisesRegex(ValueError, 'remaining lifetime'):
            self.tokens.get(a.STORAGE)
        self.assertEqual(self.acquire.call_count, 2)
        self.assertEqual(self.tokens.cache, {})
        self.assertFalse(self.receipts[-1]['http_request_sent'])

    def test_expired_storage_never_reused(self):
        self.acquire.side_effect = [credential(999, resource=a.STORAGE),
                                    credential(1600, resource=a.STORAGE)]
        self.tokens.get(a.STORAGE)
        self.assertEqual(self.acquire.call_count, 2)
        self.assertEqual(self.waits, [])

    def test_scope_identity_resource_and_subscription_validate_before_rollover(self):
        invalid = [credential(1180, resource=a.STORAGE, **change) for change in (
            {'aud': a.ARM}, {'aud': 'https://other.blob.core.windows.net/'},
            {'tid': PRINCIPAL}, {'oid': TENANT}, {'scp': 'User.Read'},
            {'iss': 'https://untrusted.example/'}, {'nbf': 1001})]
        invalid += [credential(1180, resource=a.STORAGE) | change for change in (
            {'subscription': TENANT}, {'tenant': SUB}, {'tokenType': 'Basic'})]
        for value in invalid:
            self.setUp(); self.acquire.return_value = value
            with self.assertRaises(ValueError): self.tokens.get(a.STORAGE)
            self.acquire.assert_called_once()
            self.assertEqual(self.waits, [])
            self.assertEqual(self.tokens.cache, {})

    def test_replacement_identity_and_expiry_are_revalidated(self):
        for value in (credential(2200, resource=a.STORAGE, aud=a.ARM),
                      credential(2200, resource=a.STORAGE, oid=TENANT),
                      credential(2200, resource=a.STORAGE) | {'expires_on': None}):
            self.setUp()
            self.acquire.side_effect = [credential(1180, resource=a.STORAGE), value]
            with self.assertRaises(ValueError): self.tokens.get(a.STORAGE)
            self.assertEqual(self.acquire.call_count, 2)
            self.assertEqual(self.tokens.cache, {})

    def test_rollover_failure_does_not_discard_or_substitute_arm_credential(self):
        arm = credential(2200)
        self.acquire.return_value = arm
        self.tokens.get(a.ARM)
        self.acquire.side_effect = [credential(1180, resource=a.STORAGE), ValueError('acquisition failed')]
        with self.assertRaisesRegex(ValueError, 'acquisition failed'): self.tokens.get(a.STORAGE)
        self.assertNotIn(a.STORAGE, self.tokens.cache)
        self.assertEqual(self.tokens.get(a.ARM), arm['accessToken'])

    def test_successful_storage_rollover_keeps_arm_cache_and_resource_separate(self):
        arm = credential(2200)
        self.acquire.return_value = arm
        self.tokens.get(a.ARM)
        self.acquire.side_effect = [credential(1180, resource=a.STORAGE), credential(2200, resource=a.STORAGE)]
        storage = self.tokens.get(a.STORAGE)
        self.assertNotEqual(storage, arm['accessToken'])
        self.assertEqual(self.tokens.get(a.ARM), arm['accessToken'])
        self.assertEqual([x.args[0] for x in self.acquire.call_args_list], [a.ARM, a.STORAGE, a.STORAGE])

    def test_frozen_clock_bounds_wait_and_never_admits(self):
        self.tokens.sleep = Mock()
        self.acquire.return_value = credential(1180, resource=a.STORAGE)
        with self.assertRaisesRegex(ValueError, 'wait did not complete'): self.tokens.get(a.STORAGE)
        self.assertEqual(self.tokens.sleep.call_count, 11)
        self.acquire.assert_called_once()

    def test_earliest_storage_cli_or_jwt_expiry_controls_rollover(self):
        for cli_expiry, claim_expiry in ((1180, 2200), (2200, 1180)):
            self.setUp()
            self.acquire.side_effect = [credential(cli_expiry, resource=a.STORAGE, exp=claim_expiry),
                                        credential(2200, resource=a.STORAGE)]
            self.tokens.get(a.STORAGE)
            self.assertEqual(sum(self.waits), 181)

    def test_exact_container_url_only(self):
        base = 'https://kova42c1a27.blob.core.windows.net/cosmo-adapters/'
        a.validate_storage_blob_url(base + 'a35-nova-screen/proof.json')
        for bad in ('http://kova42c1a27.blob.core.windows.net/cosmo-adapters/proof.json',
                    base.replace('kova42c1a27', 'anotheraccount') + 'proof.json',
                    base.replace('cosmo-adapters', 'other-container') + 'proof.json',
                    base + '../secret/proof.json', base + '%2e%2e/secret',
                    base + 'a//b', base + 'a\\b', base + 'proof.json?sig=secret',
                    base + 'proof.json#fragment', base):
            with self.subTest(url=bad), self.assertRaises(ValueError): a.validate_storage_blob_url(bad)

    def test_joint_preflight_requires_both_and_blocks_allocation_on_storage_failure(self):
        allocate = Mock()
        self.acquire.side_effect = [credential(2400), credential(1180, resource=a.STORAGE),
                                    credential(1180, resource=a.STORAGE)]
        with self.assertRaises(ValueError):
            a.preflight_credentials(self.tokens)
            allocate()
        allocate.assert_not_called()

    def test_joint_preflight_detects_arm_margin_consumed_by_storage_rollover(self):
        self.acquire.side_effect = [credential(1400), credential(1180, resource=a.STORAGE),
                                    credential(2200, resource=a.STORAGE)]
        with self.assertRaisesRegex(ValueError, 'joint credential readiness'):
            a.preflight_credentials(self.tokens)

    def test_joint_preflight_passes_without_returning_or_logging_secrets(self):
        self.acquire.side_effect = [credential(2400), credential(1180, resource=a.STORAGE),
                                    credential(2200, resource=a.STORAGE)]
        result = a.preflight_credentials(self.tokens)
        self.assertEqual(result, {'arm_ready': True, 'storage_ready': True})
        self.assertNotIn('accessToken', json.dumps(result) + json.dumps(self.receipts))


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

    def test_unavailable_credential_keeps_cleanup_active_without_sending_invalid_token(self):
        from training.a35_screen_control import cleanup
        self.acquire.side_effect = [credential(1100), credential(2200)]
        methods = []
        def open_request(request, timeout):
            methods.append(request.method)
            response = io.BytesIO(b"{}")
            response.status = 404 if request.method == "GET" else 202
            response.headers = {}
            return response
        self.opener.open.side_effect = open_request
        def call(method, url):
            status, _, raw = self.client.request(method, url)
            return status, json.loads(raw)
        result = cleanup(SUB, "d" * 32, call, sleep=lambda _: None)
        self.assertTrue(result["vm_group_absent"] and result["control_group_absent"])
        self.assertEqual(methods, ["DELETE", "GET", "DELETE", "GET"])
        self.assertTrue(any(r.get("http_request_sent") is False for r in self.receipts))

    def test_network_body_persisted_before_downstream_attestation_rejects_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.json"
            receipt_path = Path(directory) / "receipt.json"
            def preserve(receipt, raw):
                path.write_bytes(raw)
                receipt_path.write_text(json.dumps(receipt))
            self.client.preserve_response = preserve
            self.response(data={"properties": {"unexpected": "evidence must survive cleanup"}})
            with self.assertRaisesRegex(ValueError, "attestation mismatch"):
                self.client.request("GET", self.url)
                raise ValueError("attestation mismatch")
            self.assertEqual(json.loads(path.read_bytes())["properties"]["unexpected"],
                             "evidence must survive cleanup")
            self.assertNotIn("Authorization", receipt_path.read_text())
            self.assertNotIn(self.acquire.return_value["accessToken"], receipt_path.read_text())

    def test_failed_evidence_persistence_blocks_response_admission(self):
        self.response()
        self.client.preserve_response = Mock(side_effect=OSError("evidence unavailable"))
        with self.assertRaisesRegex(OSError, "evidence unavailable"):
            self.client.request("GET", self.url)
        self.opener.open.assert_called_once()

    def test_open_timeout_is_preserved_without_replay_or_invented_http_response(self):
        self.opener.open.side_effect = TimeoutError("SENSITIVE-EXCEPTION-TEXT")
        saved = []
        self.client.preserve_response = lambda receipt, raw: saved.append((receipt, raw))
        with self.assertRaises(a.TransportFailure) as caught:
            self.client.request("GET", self.url + "&private_query=DO-NOT-LOG", timeout=20)
        self.opener.open.assert_called_once()
        receipt, raw = saved[0]
        self.assertEqual((receipt["stage"], receipt["status"], raw), ("open", None, b""))
        self.assertEqual(receipt["timeout_seconds"], 20)
        self.assertFalse(receipt["response_complete"])
        self.assertFalse(receipt["request_replayed"])
        self.assertIsNone(receipt["partial_body_sha256"])
        output = str(caught.exception) + json.dumps(receipt) + repr(raw) + json.dumps(self.receipts)
        for secret in ("SENSITIVE-EXCEPTION-TEXT", "DO-NOT-LOG", self.acquire.return_value["accessToken"]):
            self.assertNotIn(secret, output)

    def test_body_timeout_preserves_known_status_and_request_id_but_never_admits(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.status, response.headers = 200, {"x-ms-request-id": "synthetic-timeout"}
        response.read.side_effect = TimeoutError("The read operation timed out")
        self.opener.open.return_value = response
        self.client.preserve_response = Mock()
        with self.assertRaises(a.TransportFailure) as caught:
            self.client.request("GET", self.url, timeout=20)
        self.opener.open.assert_called_once()
        receipt = caught.exception.receipt
        self.assertEqual((receipt["stage"], receipt["status"], receipt["request_id"]),
                         ("read", 200, "synthetic-timeout"))
        self.assertFalse(receipt["response_complete"])
        self.client.preserve_response.assert_called_once_with(receipt, b"")

    def test_incomplete_body_preserves_only_bounded_partial_evidence(self):
        from http.client import IncompleteRead
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.status, response.headers = 200, {}
        response.read.side_effect = IncompleteRead(b"partial-evidence", 100)
        self.opener.open.return_value = response
        saved = []
        self.client.preserve_response = lambda receipt, raw: saved.append((receipt, raw))
        with self.assertRaises(a.TransportFailure):
            self.client.request("GET", self.url, limit=7)
        self.assertEqual(saved[0][1], b"partial")
        self.assertEqual(saved[0][0]["partial_body_bytes"], 7)
        self.assertFalse(saved[0][0]["response_complete"])
        self.opener.open.assert_called_once()

    def test_transport_evidence_write_failure_still_blocks_admission(self):
        self.opener.open.side_effect = TimeoutError()
        self.client.preserve_response = Mock(side_effect=OSError("evidence unavailable"))
        with self.assertRaisesRegex(OSError, "evidence unavailable"):
            self.client.request("PUT", self.url, {"resources": []})
        self.opener.open.assert_called_once()

    def test_transport_failure_does_not_suppress_independent_group_cleanup(self):
        from training.a35_screen_control import cleanup
        calls = []
        def call(method, url):
            calls.append(method)
            if method == "POST":
                raise a.TransportFailure({"stage": "open", "error_type": "TimeoutError",
                                          "method": method, "path": "/synthetic"})
            return (404, {}) if method == "GET" else (202, {})
        result = cleanup(SUB, "d" * 32, call, sleep=lambda _: None)
        self.assertTrue(result["vm_group_absent"] and result["control_group_absent"])
        self.assertEqual(calls, ["POST", "DELETE", "GET", "DELETE", "GET"])

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
