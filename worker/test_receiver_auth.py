"""Real local RSA signatures over synthetic claims; no Entra calls or real keys."""

import base64
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
import socket
from time import time
import unittest
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from worker import receiver_auth as auth


TENANT = "10000000-0000-4000-8000-000000000001"
CLIENT = "20000000-0000-4000-8000-000000000001"
OBJECT = "30000000-0000-4000-8000-000000000001"
OTHER_CLIENT = "20000000-0000-4000-8000-000000000002"
OTHER_OBJECT = "30000000-0000-4000-8000-000000000002"
API = "40000000-0000-4000-8000-000000000001"
ROLE = "Kova.Inference.Invoke"


def b64(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


class ReceiverAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def setUp(self):
        self.now = int(time())
        self.policy, self.snapshot = self.config()
        self.verifier = auth.ReceiverAuthenticator(self.policy, self.snapshot)

    def public_jwk(self, key=None, kid="fixture-key"):
        value = jwt.algorithms.RSAAlgorithm.to_jwk((key or self.key).public_key(), as_dict=True)
        return {"kty": "RSA", "use": "sig", "alg": "RS256", "kid": kid,
                "n": value["n"], "e": value["e"]}

    def config(self, version="2.0", *, keys=None, **changes):
        issuer = (f"https://sts.windows.net/{TENANT}/" if version == "1.0"
                  else f"https://login.microsoftonline.com/{TENANT}/v2.0")
        snapshot = json.dumps({"schema_version": 1, "issuer": issuer,
                               "keys": keys if keys is not None else [self.public_jwk()]}).encode()
        policy = auth.ReceiverPolicy(
            tenant_id=TENANT, audience="api://" + API if version == "1.0" else API,
            token_version=version, allowed_services=(auth.AllowedService(CLIENT, OBJECT),),
            required_roles=(ROLE,), keyset_sha256=hashlib.sha256(snapshot).hexdigest(),
            keyset_valid_until_epoch=self.now + 3600, max_token_lifetime_seconds=3600,
            clock_skew_seconds=0, enabled=True,
        )
        return replace(policy, **changes), snapshot

    def claims(self, policy=None, **changes):
        policy = policy or self.policy
        return {
            "iss": policy.issuer, "aud": policy.audience, "ver": policy.token_version,
            "tid": TENANT, "oid": OBJECT, "sub": "synthetic-service-subject", "idtyp": "app",
            "appid" if policy.token_version == "1.0" else "azp": CLIENT,
            "roles": [ROLE], "iat": self.now - 10, "nbf": self.now - 10,
            "exp": self.now + 600, **changes,
        }

    def token(self, claims=None, *, key=None, headers=None, algorithm="RS256"):
        return jwt.encode(self.claims() if claims is None else claims, key or self.key,
                          algorithm=algorithm, headers={"kid": "fixture-key", **(headers or {})})

    def raw_token(self, header_bytes, payload_bytes):
        signed = (b64(header_bytes) + "." + b64(payload_bytes)).encode("ascii")
        signature = self.key.sign(signed, padding.PKCS1v15(), hashes.SHA256())
        return signed.decode("ascii") + "." + b64(signature)

    def assert_rejected(self, token, verifier=None):
        with self.assertRaises(auth.ReceiverAuthError) as caught:
            (verifier or self.verifier)(token)
        if isinstance(token, str) and token:
            self.assertNotIn(token, str(caught.exception))
        self.assertNotIn("synthetic-service-subject", str(caught.exception))

    def test_real_signature_accepts_only_the_configured_app_service(self):
        result = self.verifier(self.token())
        self.assertEqual(result, auth.VerifiedServiceIdentity(TENANT, CLIENT, OBJECT,
                                                             self.now + 600, (ROLE,)))
        self.assertFalse(hasattr(result, "owner_id"))
        self.assertFalse(hasattr(result, "tier"))
        self.assertFalse(hasattr(result, "token"))
        with self.assertRaises(FrozenInstanceError):
            result.client_id = OTHER_CLIENT

    def test_v1_and_v2_use_separate_exact_issuer_audience_and_client_claims(self):
        for version in ("1.0", "2.0"):
            policy, snapshot = self.config(version)
            verifier = auth.ReceiverAuthenticator(policy, snapshot)
            result = verifier(self.token(self.claims(policy)))
            self.assertEqual(result.client_id, CLIENT)
            other = "azp" if version == "1.0" else "appid"
            self.assert_rejected(self.token(self.claims(policy, **{other: CLIENT})), verifier)
            self.assert_rejected(self.token(self.claims(policy, ver="1.0" if version == "2.0" else "2.0")), verifier)

    def test_unsigned_wrong_algorithm_and_wrong_key_tokens_are_rejected(self):
        self.assert_rejected(jwt.encode(self.claims(), "", algorithm="none", headers={"kid": "fixture-key"}))
        self.assert_rejected(self.token(key=self.other_key))
        self.assert_rejected(self.token(algorithm="RS512"))
        self.assert_rejected(jwt.encode(self.claims(), "synthetic-secret" * 4,
                                       algorithm="HS256", headers={"kid": "fixture-key"}))

    def test_mutated_signature_and_payload_are_rejected_after_valid_shape_checks(self):
        parts = self.token().split(".")
        signature = bytearray(base64.urlsafe_b64decode(parts[2] + "=" * (-len(parts[2]) % 4)))
        signature[0] ^= 1
        self.assert_rejected(".".join([*parts[:2], b64(signature)]))
        claims = self.claims(sub="another-service-subject")
        self.assert_rejected(".".join([parts[0], b64(json.dumps(claims).encode()), parts[2]]))

    def test_unknown_kid_does_not_fetch_keys_or_try_every_key(self):
        with patch.object(socket, "socket", side_effect=AssertionError("network attempted")), \
                patch("jwt.decode", side_effect=AssertionError("unknown key reached decoder")):
            self.assert_rejected(self.token(headers={"kid": "unapproved-key"}))

    def test_token_embedded_keys_urls_and_critical_header_extensions_are_rejected(self):
        for key, value in (("jku", "https://attacker.invalid/keys"), ("x5u", "https://attacker.invalid/key"),
                           ("jwk", self.public_jwk(self.other_key)), ("crit", ["b64"]),
                           ("b64", False), ("cty", "JWT")):
            header = {"alg": "RS256", "typ": "JWT", "kid": "fixture-key", key: value}
            self.assert_rejected(self.raw_token(json.dumps(header).encode(), json.dumps(self.claims()).encode()))

    def test_duplicate_and_escaped_duplicate_claims_are_rejected_even_when_signed(self):
        header = b'{"alg":"RS256","typ":"JWT","kid":"fixture-key"}'
        original = json.dumps(self.claims()).encode()
        for suffix in (b',"aud":"' + API.encode() + b'"}', b',"a\\u0075d":"' + API.encode() + b'"}'):
            self.assert_rejected(self.raw_token(header, original[:-1] + suffix))
        duplicate_header = b'{"alg":"RS256","alg":"RS256","typ":"JWT","kid":"fixture-key"}'
        self.assert_rejected(self.raw_token(duplicate_header, original))

    def test_invalid_json_numbers_utf8_and_excess_structure_are_rejected(self):
        header = b'{"alg":"RS256","typ":"JWT","kid":"fixture-key"}'
        for payload in (b"[]", b"\xff", b'{"value":NaN}', b'{"value":1e400}',
                        b'{"value":"\\ud800"}', b'{"x":' + b"[" * 32 + b"0" + b"]" * 32 + b"}"):
            self.assert_rejected(self.raw_token(header, payload))

    def test_compact_serialization_is_bounded_and_canonical(self):
        valid = self.token()
        for token in (None, b"bytes", "", "x" * 16385, valid + ".extra", " " + valid,
                      valid + "=", valid.replace(".", "=."), valid + "\n"):
            with self.subTest(shape=str(token)[:30]):
                self.assert_rejected(token)

    def test_wrong_issuer_tenant_or_audience_cannot_authenticate(self):
        for changes in ({"iss": self.policy.issuer + "/"}, {"iss": "https://attacker.invalid/issuer"},
                        {"aud": "api://" + API}, {"aud": [API]}, {"tid": OTHER_CLIENT},
                        {"aud": "https://cognitiveservices.azure.com/"}):
            self.assert_rejected(self.token(self.claims(**changes)))

    def test_app_id_and_object_id_must_match_the_same_allowlisted_pair(self):
        policy = replace(self.policy, allowed_services=(auth.AllowedService(CLIENT, OBJECT),
                                                        auth.AllowedService(OTHER_CLIENT, OTHER_OBJECT)))
        verifier = auth.ReceiverAuthenticator(policy, self.snapshot)
        self.assertEqual(verifier(self.token(self.claims(azp=OTHER_CLIENT, oid=OTHER_OBJECT))).object_id, OTHER_OBJECT)
        for changes in ({"azp": CLIENT, "oid": OTHER_OBJECT}, {"azp": OTHER_CLIENT, "oid": OBJECT}):
            self.assert_rejected(self.token(self.claims(**changes)), verifier)

    def test_delegated_and_id_token_shapes_cannot_be_used_as_app_tokens(self):
        for changes in ({"idtyp": "user"}, {"idtyp": None}, {"scp": "Kova.Inference.Invoke"},
                        {"scp": None}, {"nonce": "id-token-nonce"}, {"act": {}}, {"may_act": {}}):
            self.assert_rejected(self.token(self.claims(**changes)))

    def test_required_roles_cannot_be_missing_coerced_or_duplicated(self):
        for roles in ([], ROLE, [ROLE, ROLE], ["Different.Role"], [ROLE, {}], None):
            self.assert_rejected(self.token(self.claims(roles=roles)))
        result = self.verifier(self.token(self.claims(roles=[ROLE, "Unrelated.Role"])))
        self.assertEqual(result.required_roles, (ROLE,))

    def test_all_configured_roles_are_required_not_merely_one(self):
        policy = replace(self.policy, required_roles=(ROLE, "Kova.Second.Permission"))
        verifier = auth.ReceiverAuthenticator(policy, self.snapshot)
        self.assert_rejected(self.token(), verifier)
        self.assertEqual(verifier(self.token(self.claims(roles=list(policy.required_roles)))).required_roles,
                         policy.required_roles)

    def test_required_claims_are_not_optional(self):
        for field in ("iss", "aud", "exp", "iat", "nbf", "ver", "tid", "oid", "sub", "idtyp", "roles", "azp"):
            claims = self.claims()
            del claims[field]
            with self.subTest(field=field):
                self.assert_rejected(self.token(claims))

    def test_temporal_claims_reject_numeric_coercion_and_invalid_order(self):
        for field in ("exp", "iat", "nbf"):
            for value in (True, False, str(self.now), float(self.now), None):
                self.assert_rejected(self.token(self.claims(**{field: value})))
        for changes in ({"iat": self.now + 700}, {"nbf": self.now + 700},
                        {"exp": self.now - 11}, {"exp": self.now + 10000}):
            self.assert_rejected(self.token(self.claims(**changes)))

    def test_backend_enforces_expiry_and_future_issued_or_not_before_times(self):
        for changes in ({"iat": self.now - 1000, "nbf": self.now - 1000, "exp": self.now - 10},
                        {"iat": self.now + 300}, {"nbf": self.now + 300}):
            self.assert_rejected(self.token(self.claims(**changes)))

    def test_explicit_clock_skew_does_not_increase_token_lifetime_limit(self):
        policy = replace(self.policy, clock_skew_seconds=120)
        verifier = auth.ReceiverAuthenticator(policy, self.snapshot)
        verifier(self.token(self.claims(iat=self.now + 30, nbf=self.now + 30)))
        self.assert_rejected(self.token(self.claims(exp=self.now + 10000)), verifier)

    def test_expired_or_invalid_key_snapshot_clock_fails_closed(self):
        for now in (self.policy.keyset_valid_until_epoch, math.inf, math.nan, True, -1):
            with patch.object(auth, "time", return_value=now):
                with self.assertRaises(auth.ReceiverAuthUnavailable):
                    self.verifier(self.token())

    def test_snapshot_expiry_is_rechecked_after_signature_verification(self):
        original = jwt.decode
        def expire(*args, **kwargs):
            result = original(*args, **kwargs)
            times[0] = self.policy.keyset_valid_until_epoch
            return result
        times = [self.now]
        with patch.object(auth, "time", side_effect=lambda: times[0]), patch("jwt.decode", side_effect=expire):
            with self.assertRaises(auth.ReceiverAuthUnavailable):
                self.verifier(self.token())

    def test_default_disabled_policy_blocks_without_reading_keys_or_decoding(self):
        disabled = replace(self.policy, enabled=False)
        with patch.object(auth, "_load_keys", side_effect=AssertionError("keys read")), \
                patch("jwt.decode", side_effect=AssertionError("token verified")):
            verifier = auth.ReceiverAuthenticator(disabled, object())
            with self.assertRaises(auth.ReceiverAuthUnavailable):
                verifier("not-even-a-token")
            with self.assertRaises(auth.ReceiverAuthUnavailable):
                verifier.from_headers([])

    def test_policy_fields_cannot_be_inferred_or_relaxed(self):
        changes = ({"tenant_id": "common"}, {"token_version": "3.0"}, {"audience": "not-a-guid"},
                   {"allowed_services": ()}, {"allowed_services": [auth.AllowedService(CLIENT, OBJECT)]},
                   {"required_roles": ()}, {"required_roles": (ROLE, ROLE)}, {"required_roles": ("*",)},
                   {"keyset_sha256": ""}, {"keyset_valid_until_epoch": True},
                   {"max_token_lifetime_seconds": 0}, {"max_token_lifetime_seconds": math.inf},
                   {"clock_skew_seconds": True}, {"clock_skew_seconds": 301}, {"enabled": "true"})
        for values in changes:
            with self.subTest(values=values), self.assertRaises(auth.ReceiverAuthError):
                replace(self.policy, **values)

    def test_trusted_snapshot_hash_and_issuer_are_required(self):
        with self.assertRaises(auth.ReceiverAuthError):
            auth.ReceiverAuthenticator(replace(self.policy, keyset_sha256="0" * 64), self.snapshot)
        value = json.loads(self.snapshot)
        value["issuer"] = "https://attacker.invalid/issuer"
        snapshot = json.dumps(value).encode()
        policy = replace(self.policy, keyset_sha256=hashlib.sha256(snapshot).hexdigest())
        with self.assertRaises(auth.ReceiverAuthError):
            auth.ReceiverAuthenticator(policy, snapshot)

    def test_duplicate_keys_and_private_or_unapproved_jwk_fields_are_rejected(self):
        good = self.public_jwk()
        for keys in ([good, good], [{**good, "d": "private-key-must-not-be-loaded"}],
                     [{**good, "alg": "HS256"}], [{**good, "use": "enc"}],
                     [{**good, "kty": "EC"}], [{**good, "x5u": "https://attacker.invalid"}],
                     [{**good, "e": b64(b"\x03")}], []):
            policy, snapshot = self.config(keys=keys)
            with self.subTest(keys=str(keys)[:60]), self.assertRaises(auth.ReceiverAuthError):
                auth.ReceiverAuthenticator(policy, snapshot)

    def test_small_rsa_keys_are_not_accepted(self):
        weak = rsa.generate_private_key(public_exponent=65537, key_size=1024)
        policy, snapshot = self.config(keys=[self.public_jwk(weak)])
        with self.assertRaises(auth.ReceiverAuthError):
            auth.ReceiverAuthenticator(policy, snapshot)

    def test_reviewed_key_rotation_replaces_snapshot_without_request_driven_network(self):
        key = self.public_jwk(self.other_key, "rotated-key")
        policy, snapshot = self.config(keys=[key])
        token = self.token(key=self.other_key, headers={"kid": "rotated-key"})
        with patch.object(socket, "socket", side_effect=AssertionError("network called")):
            self.assert_rejected(token)
            rotated = auth.ReceiverAuthenticator(policy, snapshot)
            self.assertEqual(rotated(token).client_id, CLIENT)
            self.assert_rejected(self.token(), rotated)

    def test_only_one_raw_authorization_header_is_accepted(self):
        token = self.token()
        self.assertEqual(self.verifier.from_headers([("authorization", "Bearer " + token)]).object_id, OBJECT)
        for headers in ([], [("Authorization", "Bearer " + token), ("authorization", "Bearer " + token)],
                        [("Authorization", "Basic " + token)], [("Authorization", "Bearer " + token + ",other")],
                        [("Authorization", "Bearer " + token + "\r\n")], {"Authorization": "Bearer " + token}):
            with self.subTest(headers=str(headers)[:30]), self.assertRaises(auth.ReceiverAuthError):
                self.verifier.from_headers(headers)

    def test_forwarded_identity_owner_and_tier_headers_are_not_credentials(self):
        for header in ("X-MS-CLIENT-PRINCIPAL", "X-Owner-Id", "X-User-Id", "X-Plan", "Cookie"):
            with self.assertRaises(auth.ReceiverAuthError):
                self.verifier.from_headers([(header, "forged-admin")])
        identity = self.verifier.from_headers([("Authorization", "Bearer " + self.token()),
                                               ("X-Owner-Id", "other-user"), ("X-Plan", "pro")])
        self.assertFalse(hasattr(identity, "owner_id"))
        self.assertFalse(hasattr(identity, "tier"))

    def test_backend_is_always_given_fixed_algorithm_and_required_checks(self):
        original = jwt.decode
        with patch("jwt.decode", wraps=original) as decode:
            self.verifier(self.token())
        options = decode.call_args.kwargs
        self.assertEqual(options["algorithms"], ["RS256"])
        self.assertEqual(options["issuer"], self.policy.issuer)
        self.assertEqual(options["audience"], self.policy.audience)
        for name in ("verify_signature", "verify_exp", "verify_nbf", "verify_iat", "verify_iss", "verify_aud", "strict_aud"):
            self.assertIs(options["options"][name], True)

    def test_backend_errors_never_expose_token_or_provider_details(self):
        token = self.token()
        with patch("jwt.decode", side_effect=RuntimeError("SECRET " + token)):
            with self.assertRaises(auth.ReceiverAuthError) as caught:
                self.verifier(token)
        self.assertEqual(str(caught.exception), "receiver authentication rejected")
        self.assertTrue(caught.exception.__suppress_context__)

    def test_snapshot_and_policy_are_defensive_and_do_not_install_runtime(self):
        snapshot = bytearray(self.snapshot)
        policy = self.policy
        verifier = auth.ReceiverAuthenticator(policy, bytes(snapshot))
        snapshot[:] = b"changed"
        self.assertEqual(verifier(self.token()).client_id, CLIENT)
        with self.assertRaises(FrozenInstanceError):
            verifier.policy = replace(policy, enabled=False)
        with self.assertRaises(TypeError):
            verifier._keys["injected"] = self.other_key.public_key()

    def test_public_token_verification_does_not_contact_any_network_service(self):
        with patch.object(socket, "socket", side_effect=AssertionError("network called")):
            verifier = auth.ReceiverAuthenticator(self.policy, self.snapshot)
            self.assertEqual(verifier.from_headers([("Authorization", "Bearer " + self.token())]).client_id, CLIENT)


if __name__ == "__main__":
    unittest.main()
