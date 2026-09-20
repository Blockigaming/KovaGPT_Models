"""Adversarial tests for the independent paid-phase lifecycle authority."""
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from training import cosmo_lifecycle_authority as authority
from training.cosmo_generation_attestation import public_key_hex


NOW = datetime(2026, 9, 19, 19, 0, tzinfo=timezone.utc)


class CosmoLifecycleAuthorityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "repo"
        (self.root / "config").mkdir(parents=True)
        self.key = Ed25519PrivateKey.from_private_bytes(b"l" * 32)
        public = public_key_hex(self.key)
        self.trust = {
            "schema_version": 1,
            "status": "authority_pinned",
            "issuer": authority.ISSUER,
            "algorithm": "ed25519",
            "endpoint": "https://authority.example/v1/pilot/grants",
            "public_key_hex": public,
            "public_key_sha256": hashlib.sha256(bytes.fromhex(public)).hexdigest(),
            "bearer_token_file_environment_variable": authority.TOKEN_ENV,
            "azure_managed_identity_token_audience":
                "api://kova-cosmo-lifecycle-authority",
            "append_only_remote_ledger_required": True,
            "independent_azure_reader_required": True,
            "runner_ledger_mutation_allowed": False,
            "checked_in_private_key_allowed": False,
        }
        self.write_trust(self.trust)
        self.token = self.directory / "authority-token"
        self.token.write_text("t" * 64, encoding="ascii")
        self.token.chmod(0o600)
        self.azure_token_value = ".".join(("a" * 32, "b" * 32, "c" * 64))
        self.azure_instance = {
            "resource_id": (
                "/subscriptions/11111111-2222-3333-4444-555555555555/"
                "resourceGroups/kova-cosmo-pilot/providers/"
                "Microsoft.Compute/virtualMachines/kova-cosmo-t4"
            ),
            "vm_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "system_assigned_identity_principal_id":
                "99999999-8888-7777-6666-555555555555",
        }

    def imds_transport(self, url: str) -> dict:
        if url == authority.AZURE_COMPUTE_IMDS_URL:
            return {
                "resourceId": self.azure_instance["resource_id"],
                "vmId": self.azure_instance["vm_id"],
            }
        self.assertTrue(url.startswith(authority.AZURE_IDENTITY_IMDS_URL))
        return {
            "access_token": self.azure_token_value,
            "expires_on": str(int(datetime(
                2026, 9, 19, 20, 0, tzinfo=timezone.utc
            ).timestamp())),
            "resource": self.trust[
                "azure_managed_identity_token_audience"
            ],
            "token_type": "Bearer",
        }

    def write_trust(self, value: dict) -> None:
        (self.root / authority.TRUST_PATH).write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )

    def response_payload(self, request: dict, **changes) -> dict:
        counts = {phase: 0 for phase in authority.PHASES}
        counts[request["phase"]] = 1
        reserved_seconds = authority.PHASE_RESERVED_SECONDS[request["phase"]]
        value = {
            "schema_version": 1,
            "kind": "kova_cosmo_paid_phase_grant",
            "issuer": authority.ISSUER,
            "pilot_id": authority.PILOT_ID,
            "lifecycle_id": "lifecycle-001",
            "preflight_ledger_sequence": request[
                "preflight_ledger_sequence"
            ],
            "ledger_sequence": 2,
            "ledger_commit_id": "append-only-ledger-commit-002",
            "ledger_append_only": True,
            "ledger_status": "grant_committed_before_response",
            "grant_id": "grant-" + request["phase"],
            "phase": request["phase"],
            "source_commit": request["source_commit"],
            "runtime_evidence_sha256": request["runtime_evidence_sha256"],
            "context_sha256": request["context_sha256"],
            "request_nonce": request["request_nonce"],
            "runtime_deadline_utc": request["runtime_deadline_utc"],
            "azure_instance": request["azure_instance"],
            "azure_instance_identity": {
                "verification_method":
                    "microsoft_entra_system_assigned_managed_identity_token",
                "token_sha256": hashlib.sha256(
                    request["azure_instance_identity_token"].encode("ascii")
                ).hexdigest(),
                "token_audience": request[
                    "azure_instance_identity_token_audience"
                ],
                "verified_at_utc": "2026-09-19T19:00:00Z",
                "token_expires_at_utc": request[
                    "azure_instance_identity_token_expires_at_utc"
                ],
                "verified": True,
            },
            "issued_at_utc": "2026-09-19T19:00:00Z",
            "expires_at_utc": request["runtime_deadline_utc"],
            "grant_reserved_seconds": reserved_seconds,
            "grant_reserved_cost_usd": "0.5000",
            "phase_grants_committed": counts,
            "training_runs_consumed": counts["training"],
            "aggregate_reserved_seconds": reserved_seconds,
            "aggregate_reserved_cost_usd": "0.5000",
            "approved_budget_usd": "2.0000",
            "deployment_authorized": False,
        }
        value.update(changes)
        return value

    def signed(self, payload: dict,
               key: Ed25519PrivateKey | None = None) -> dict:
        signer = key or self.key
        return {
            "payload": payload,
            "signature": signer.sign(authority.canonical(payload)).hex(),
        }

    def acquire(self, transport) -> dict:
        with patch.dict(
            os.environ, {authority.TOKEN_ENV: str(self.token)}, clear=True
        ):
            return authority.acquire_phase_grant(
                phase="training",
                source_commit="a" * 40,
                runtime_evidence_sha256="b" * 64,
                lifecycle_id="lifecycle-001",
                preflight_ledger_sequence=1,
                runtime_deadline_utc="2026-09-19T19:40:00Z",
                context={"operation": "single_lora_sft_run"},
                azure_instance=self.azure_instance,
                root=self.root,
                now=NOW,
                transport=transport,
                instance_transport=self.imds_transport,
            )

    def test_each_paid_phase_requires_a_nonce_bound_committed_grant(self):
        for phase in authority.PHASES:
            deadline = {
                "runtime_probe": "2026-09-19T19:10:00Z",
                "training": "2026-09-19T19:40:00Z",
                "evaluation": "2026-09-19T19:10:00Z",
            }[phase]
            def transport(endpoint, token, request, phase=phase):
                self.assertEqual(endpoint, self.trust["endpoint"])
                self.assertEqual(token, "t" * 64)
                self.assertEqual(request["phase"], phase)
                return self.signed(self.response_payload(request))

            with self.subTest(phase=phase), patch.dict(
                os.environ, {authority.TOKEN_ENV: str(self.token)}, clear=True
            ):
                report = authority.acquire_phase_grant(
                    phase=phase,
                    source_commit="a" * 40,
                    runtime_evidence_sha256="b" * 64,
                    lifecycle_id="lifecycle-001",
                    preflight_ledger_sequence=1,
                    runtime_deadline_utc=deadline,
                    context={"operation": phase},
                    azure_instance=self.azure_instance,
                    root=self.root,
                    now=NOW,
                    transport=transport,
                    instance_transport=self.imds_transport,
                )
            self.assertEqual(report["phase"], phase)
            self.assertEqual(
                report["status"],
                "paid_phase_reserved_in_append_only_ledger",
            )
            self.assertFalse(report["deployment_authorized"])

    def test_replayed_response_with_different_nonce_is_rejected(self):
        captured = None

        def first_transport(_endpoint, _token, request):
            nonlocal captured
            captured = self.signed(self.response_payload(request))
            return captured

        self.acquire(first_transport)
        with patch.object(authority.secrets, "token_hex", return_value="f" * 64):
            with self.assertRaises(authority.AuthorityError):
                self.acquire(lambda *_arguments: captured)

    def test_fake_signer_and_tampered_grant_are_rejected(self):
        fake_key = Ed25519PrivateKey.from_private_bytes(b"f" * 32)

        def fake_transport(_endpoint, _token, request):
            return self.signed(self.response_payload(request), fake_key)

        def tampered_transport(_endpoint, _token, request):
            envelope = self.signed(self.response_payload(request))
            envelope["payload"]["aggregate_reserved_cost_usd"] = "2.0001"
            return envelope

        for transport in (fake_transport, tampered_transport):
            with self.subTest(transport=transport), self.assertRaises(
                authority.AuthorityError
            ):
                self.acquire(transport)

    def test_nonappend_only_duplicate_or_over_budget_claims_are_rejected(self):
        def invalid_transport(field, wrong):
            def transport(_endpoint, _token, request):
                payload = self.response_payload(request)
                if field == "training_count":
                    payload["phase_grants_committed"]["training"] = wrong
                    payload["training_runs_consumed"] = wrong
                else:
                    payload[field] = wrong
                return self.signed(payload)
            return transport

        invalid = (
            ("lifecycle_id", "different-lifecycle"),
            ("preflight_ledger_sequence", 0),
            ("ledger_sequence", 1),
            ("ledger_append_only", False),
            ("ledger_status", "pending_commit"),
            ("training_count", 2),
            ("aggregate_reserved_seconds", 3601),
            ("aggregate_reserved_cost_usd", "0.5261"),
            ("runtime_deadline_utc", "2026-09-19T19:39:59Z"),
            ("deployment_authorized", True),
        )
        for field, wrong in invalid:
            with self.subTest(field=field), self.assertRaises(
                authority.AuthorityError
            ):
                self.acquire(invalid_transport(field, wrong))

    def test_grant_is_bound_to_exact_vm_and_identity_token(self):
        def wrong_vm(_endpoint, _token, request):
            payload = self.response_payload(request)
            payload["azure_instance"] = dict(payload["azure_instance"])
            payload["azure_instance"]["vm_id"] = (
                "00000000-1111-2222-3333-444444444444"
            )
            return self.signed(payload)

        def wrong_token(_endpoint, _token, request):
            payload = self.response_payload(request)
            payload["azure_instance_identity"]["token_sha256"] = "0" * 64
            return self.signed(payload)

        def identity_verified_after_grant(_endpoint, _token, request):
            payload = self.response_payload(request)
            payload["azure_instance_identity"]["verified_at_utc"] = (
                "2026-09-19T19:00:01Z"
            )
            return self.signed(payload)

        for transport in (
            wrong_vm, wrong_token, identity_verified_after_grant,
        ):
            with self.subTest(transport=transport), self.assertRaises(
                authority.AuthorityError
            ):
                self.acquire(transport)

    def test_token_must_be_protected_external_file(self):
        inside = self.root / "token"
        inside.write_text("t" * 64, encoding="ascii")
        inside.chmod(0o600)
        permissive = self.directory / "permissive-token"
        permissive.write_text("t" * 64, encoding="ascii")
        permissive.chmod(0o644)
        for token in (inside, permissive):
            with self.subTest(token=token), patch.dict(
                os.environ, {authority.TOKEN_ENV: str(token)}, clear=True
            ), self.assertRaises(authority.AuthorityError):
                authority.acquire_phase_grant(
                    phase="training",
                    source_commit="a" * 40,
                    runtime_evidence_sha256="b" * 64,
                    lifecycle_id="lifecycle-001",
                    preflight_ledger_sequence=1,
                    runtime_deadline_utc="2026-09-19T19:40:00Z",
                    context={"operation": "training"},
                    azure_instance=self.azure_instance,
                    root=self.root,
                    now=NOW,
                    transport=lambda *_arguments: {},
                    instance_transport=self.imds_transport,
                )

    def test_imds_identity_and_exact_vm_are_required(self):
        def invalid_imds(variant):
            def fetch(url):
                value = self.imds_transport(url)
                if url == authority.AZURE_COMPUTE_IMDS_URL:
                    if variant == "resource_id":
                        value["resourceId"] += "-other"
                    elif variant == "vm_id":
                        value["vmId"] = (
                            "00000000-1111-2222-3333-444444444444"
                        )
                elif variant == "token":
                    value["access_token"] = "not-a-jwt"
                elif variant == "audience":
                    value["resource"] = "api://other"
                elif variant == "expired":
                    value["expires_on"] = "1"
                return value
            return fetch

        for variant in (
            "resource_id", "vm_id", "token", "audience", "expired",
        ):
            with self.subTest(variant=variant), patch.dict(
                os.environ, {authority.TOKEN_ENV: str(self.token)}, clear=True
            ), self.assertRaises(authority.AuthorityError):
                authority.acquire_phase_grant(
                    phase="training",
                    source_commit="a" * 40,
                    runtime_evidence_sha256="b" * 64,
                    lifecycle_id="lifecycle-001",
                    preflight_ledger_sequence=1,
                    runtime_deadline_utc="2026-09-19T19:40:00Z",
                    context={"operation": "training"},
                    azure_instance=self.azure_instance,
                    root=self.root,
                    now=NOW,
                    transport=lambda *_arguments: {},
                    instance_transport=invalid_imds(variant),
                )

    def test_real_imds_transport_bypasses_proxies_and_redirects(self):
        class Response(BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_arguments):
                self.close()

            def geturl(self):
                return authority.AZURE_COMPUTE_IMDS_URL

        opener = Mock()
        opener.open.return_value = Response(
            json.dumps({"resourceId": self.azure_instance["resource_id"]})
            .encode("utf-8")
        )
        with patch.object(
            authority.urllib.request, "build_opener", return_value=opener
        ) as build:
            value = authority._imds_transport(
                authority.AZURE_COMPUTE_IMDS_URL
            )
        self.assertEqual(value["resourceId"], self.azure_instance["resource_id"])
        handlers = build.call_args.args
        proxies = [item for item in handlers if isinstance(
            item, authority.urllib.request.ProxyHandler
        )]
        self.assertEqual(len(proxies), 1)
        self.assertEqual(proxies[0].proxies, {})
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Metadata"), "true")

    def test_unprovisioned_source_policy_makes_no_provider_call(self):
        unpinned = dict(self.trust)
        unpinned.update(
            status="authority_provisioning_required",
            endpoint=None,
            public_key_hex=None,
            public_key_sha256=None,
            azure_managed_identity_token_audience=None,
        )
        self.write_trust(unpinned)
        report = authority.dry_run(self.root)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["authority_pinned"])
        self.assertEqual(report["provider_calls_made"], 0)
        with self.assertRaises(authority.AuthorityError):
            self.acquire(lambda *_arguments: self.fail("transport called"))

    def test_authority_endpoint_must_be_exact_https_without_redirect_query(self):
        for endpoint in (
            "http://authority.example/v1/pilot/grants",
            "https://authority.example/v1/pilot/grants?next=evil",
            "https://user@authority.example/v1/pilot/grants",
            "https://authority.example/other",
        ):
            value = dict(self.trust)
            value["endpoint"] = endpoint
            self.write_trust(value)
            with self.subTest(endpoint=endpoint), self.assertRaises(
                authority.AuthorityError
            ):
                authority.load_trust_policy(self.root)


if __name__ == "__main__":
    unittest.main()
