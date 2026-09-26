"""Exercise the trainer-to-controller durable adapter boundary without Azure calls."""
import base64
from copy import deepcopy
from datetime import timedelta
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from training import cosmo_adapter_preservation as preservation
from training import cosmo_lifecycle_authority as authority
from training import cosmo_qlora_grant as grant_client
from training import cosmo_qlora_launch as launch
from training import test_cosmo_controller_grants as fixtures
from training.cosmo_controller_http import GrantApplication
from training.cosmo_controller_ledger import LedgerRejected, Response


class ProtectedAdapterBlob:
    def __init__(self):
        self.body = None
        self.corrupt_readback = False
        self.put_count = 0
        self.metadata = {}
        self.etag = '"object-1"'
        self.legal_hold = False

    def __call__(self, method, url, headers, body=b""):
        if "management.azure.com" in url:
            value = ({"state": "Locked", "immutabilityPeriodSinceCreationInDays": 1}
                     if "immutabilityPolicies" in url else
                     {"publicAccess": "None", "hasImmutabilityPolicy": True,
                      "hasLegalHold": self.legal_hold,
                      "immutableStorageWithVersioning": {"enabled": False}})
            return Response(200, {}, json.dumps({"properties": value}).encode())
        if method == "PUT":
            if headers.get("If-None-Match") != "*" or self.body is not None:
                return Response(412, {})
            self.body = body
            self.metadata = {k: v for k, v in headers.items() if k.startswith("x-ms-meta-")}
            self.put_count += 1
            return Response(201, {"etag": self.etag})
        if self.body is None:
            return Response(404, {})
        if headers.get("If-Match") not in (None, self.etag):
            return Response(412, {})
        return Response(200, {"etag": self.etag, "x-ms-blob-type": "BlockBlob", **self.metadata},
                        self.body + (b"corrupt" if self.corrupt_readback else b""))


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.GrantIssuerTests("runTest")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        compute = {"resourceId": self.f.vm["resource_id"], "vmId": self.f.vm["vm_id"],
            "location": "eastus", "vmSize": launch.SKU, "storageProfile": {
                "imageReference": {"publisher": "Canonical", "offer": "ubuntu-24_04-lts",
                    "sku": "server", "exactVersion": "24.04.202609040"}}}
        with patch.object(authority, "_executing_azure_identity", return_value=(
                self.f.token, self.f.token_digest, "2026-09-24T14:30:00Z")), \
             patch.object(authority, "_load_bearer_token", return_value="synthetic"):
            self.committed_grant = grant_client.acquire_training_grant(
                quote=self.f.quote, source_commit=self.f.f.context["source_commit"],
                subscription_id=self.f.quote_fixture.subscription,
                lifecycle_id=self.f.request["lifecycle_id"], preflight_ledger_sequence=2,
                azure_instance=self.f.vm, runtime_evidence=self.f.runtime_path,
                now=self.f.f.now, root=self.f.quote_fixture.root,
                instance_transport=lambda _: compute,
                transport=lambda _endpoint, _bearer, request: self.f.issuer.issue(request))
        self.issued = self.f.ledger.current_state()["events"][-1]["response_envelope"]["payload"]
        self.assertEqual(self.committed_grant["lifecycle_id"], self.issued["lifecycle_id"])
        self.secret = "a" * 64
        self.storage = ProtectedAdapterBlob()
        self.preserver = preservation.AdapterPreserver(ledger=self.f.ledger,
            artifact_container="cosmo-adapters", signing_key=self.f.f.verifier,
            token_for=lambda _: "x" * 64)
        self.preserver.io = self.storage
        self.app = GrantApplication(issuer=self.f.issuer, preserver=self.preserver,
            endpoint="https://example.test/v1/pilot/grants", bearer_token=self.secret)
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.adapter = Path(self.folder.name) / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        (self.adapter / "adapter_model.safetensors").write_bytes(b"synthetic-weights")

    def submit_http(self, request):
        raw = preservation._encode_request(request)
        env = {"wsgi.url_scheme": "https", "HTTP_HOST": "example.test",
               "PATH_INFO": "/v1/pilot/preservation", "REQUEST_METHOD": "POST",
               "HTTP_AUTHORIZATION": "Bearer " + self.secret,
               "CONTENT_TYPE": "application/json", "CONTENT_LENGTH": str(len(raw)),
               "wsgi.input": BytesIO(raw)}
        status = []
        response = b"".join(self.app(env, lambda value, _: status.append(value)))
        return status[0], json.loads(response)

    def test_readback_and_signed_ledger_commit_precede_guest_success(self):
        # The real adapter is larger than the ledger's 1 MiB envelope limit.
        (self.adapter / "adapter_model.safetensors").write_bytes(b"w" * (1024 * 1024 + 1))
        with patch.object(authority, "_load_bearer_token", return_value=self.secret):
            result = preservation.preserve_adapter(adapter=self.adapter,
                grant_payload=self.committed_grant, root=self.f.quote_fixture.root,
                transport=lambda _, token, request: self.submit_http(request)[1])
        self.assertEqual(result["status"], "adapter_preserved_and_committed")
        self.assertEqual(result["artifact_sha256"], hashlib.sha256(self.storage.body).hexdigest())
        self.assertEqual(self.storage.put_count, 1)

        state = self.f.ledger.replay(self.f.f.io.body)[0]
        self.assertEqual(state["sequence"], 4)
        self.assertEqual(state["events"][-1]["kind"], "family_preserved")
        with patch.object(authority, "_load_bearer_token", return_value=self.secret):
            retry = preservation.preserve_adapter(adapter=self.adapter,
                grant_payload=self.committed_grant, root=self.f.quote_fixture.root,
                transport=lambda _, token, request: self.submit_http(request)[1])
        self.assertEqual(retry, result)
        self.assertEqual(self.storage.put_count, 1)

    def test_preserver_requires_quoted_artifact_container(self):
        with self.assertRaisesRegex(LedgerRejected, "distinct containers"):
            preservation.AdapterPreserver(ledger=self.f.ledger,
                artifact_container="other-adapters", signing_key=self.f.f.verifier,
                token_for=lambda _: "x" * 64)

    def test_unchanged_adapter_bundle_is_identical_across_zip_clock_changes(self):
        with patch("zipfile.time.localtime", return_value=(2026, 9, 24, 12, 1, 0, 0, 0, 0)):
            first = preservation.bundle_adapter(self.adapter)
        with patch("zipfile.time.localtime", return_value=(2026, 9, 25, 15, 30, 0, 0, 0, 0)):
            later = preservation.bundle_adapter(self.adapter)
        self.assertEqual(first, later)
        self.assertEqual(hashlib.sha256(first).hexdigest(), hashlib.sha256(later).hexdigest())

    def test_retry_after_uncertain_ledger_append_recovers_existing_blob(self):
        raw = preservation.bundle_adapter(self.adapter)
        request = {"schema_version": 1, "kind": "kova_cosmo_preserve_adapter_v1",
                   "lifecycle_id": self.issued["lifecycle_id"],
                   "grant_id": self.issued["grant_id"],
                   "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                   "bundle_b64": base64.b64encode(raw).decode()}
        self.f.f.io.lose_append_response = True
        self.assertEqual(self.submit_http(request)[0], "403 Forbidden")
        self.assertEqual(self.f.ledger.current_state()["sequence"], 4)
        # A committed read-back can be confirmed even after new uploads close.
        self.f.f.now += timedelta(minutes=75)
        status, recovered = self.submit_http(request)
        self.assertEqual(status, "200 OK")
        self.assertEqual(recovered["payload"]["event"]["artifact_sha256"], request["artifact_sha256"])
        self.assertEqual(self.storage.put_count, 1)

    def test_retry_after_uncommitted_append_checks_exact_existing_blob(self):
        raw = preservation.bundle_adapter(self.adapter)
        request = {"schema_version": 1, "kind": "kova_cosmo_preserve_adapter_v1",
                   "lifecycle_id": self.issued["lifecycle_id"],
                   "grant_id": self.issued["grant_id"],
                   "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                   "bundle_b64": base64.b64encode(raw).decode()}
        with patch.object(self.f.ledger, "append", side_effect=LedgerRejected("temporary")):
            self.assertEqual(self.submit_http(request)[0], "403 Forbidden")
        self.assertEqual(self.f.ledger.current_state()["sequence"], 3)
        self.assertEqual(self.submit_http(request)[0], "200 OK")
        self.assertEqual(self.storage.put_count, 1)

        # Another bundle, altered metadata, or legal hold cannot reuse this commit.
        self.storage.metadata["x-ms-meta-grant-id"] = "different"
        self.assertEqual(self.submit_http(request)[0], "403 Forbidden")
        self.storage.metadata["x-ms-meta-grant-id"] = self.issued["grant_id"]
        self.storage.body += b"changed"
        self.assertEqual(self.submit_http(request)[0], "403 Forbidden")
        self.storage.body = raw
        self.storage.legal_hold = True
        self.assertEqual(self.submit_http(request)[0], "403 Forbidden")

    def test_corrupt_readback_cannot_report_success_or_commit_preservation(self):
        self.storage.corrupt_readback = True
        request = {"schema_version": 1, "kind": "kova_cosmo_preserve_adapter_v1",
                   "lifecycle_id": self.issued["lifecycle_id"],
                   "grant_id": self.issued["grant_id"]}
        raw = preservation.bundle_adapter(self.adapter)
        request.update(artifact_sha256=hashlib.sha256(raw).hexdigest(),
                       bundle_b64=base64.b64encode(raw).decode())
        self.assertEqual(self.submit_http(request)[0], "403 Forbidden")
        self.assertEqual(self.f.ledger.replay(self.f.f.io.body)[0]["sequence"], 3)

    def test_wrong_lifecycle_and_symlink_adapter_reject_before_blob_write(self):
        raw = preservation.bundle_adapter(self.adapter)
        request = {"schema_version": 1, "kind": "kova_cosmo_preserve_adapter_v1",
                   "lifecycle_id": "another-run", "grant_id": self.issued["grant_id"],
                   "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                   "bundle_b64": base64.b64encode(raw).decode()}
        self.assertEqual(self.submit_http(request)[0], "403 Forbidden")
        self.assertEqual(self.storage.put_count, 0)
        (self.adapter / "adapter_model.safetensors").unlink()
        (self.adapter / "adapter_model.safetensors").symlink_to(self.adapter / "adapter_config.json")
        with self.assertRaises(OSError):
            preservation.bundle_adapter(self.adapter)


if __name__ == "__main__":
    unittest.main()
