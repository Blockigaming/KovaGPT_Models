"""Exercise the trainer-to-controller durable adapter boundary without Azure calls."""
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from training import cosmo_adapter_preservation as preservation
from training import cosmo_lifecycle_authority as authority
from training import test_cosmo_controller_grants as fixtures
from training.cosmo_controller_http import GrantApplication
from training.cosmo_controller_ledger import LedgerRejected, Response


class ProtectedAdapterBlob:
    def __init__(self):
        self.body = None
        self.corrupt_readback = False
        self.put_count = 0

    def __call__(self, method, url, headers, body=b""):
        if "management.azure.com" in url:
            value = ({"state": "Locked", "immutabilityPeriodSinceCreationInDays": 1}
                     if "immutabilityPolicies" in url else
                     {"publicAccess": "None", "hasImmutabilityPolicy": True,
                      "immutableStorageWithVersioning": {"enabled": False}})
            return Response(200, {}, json.dumps({"properties": value}).encode())
        if method == "PUT":
            if headers.get("If-None-Match") != "*" or self.body is not None:
                return Response(412, {})
            self.body = body
            self.put_count += 1
            return Response(201, {"etag": '"object-1"'})
        if self.body is None:
            return Response(404, {})
        return Response(200, {"etag": '"object-1"', "x-ms-blob-type": "BlockBlob"},
                        self.body + (b"corrupt" if self.corrupt_readback else b""))


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.GrantIssuerTests("runTest")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.issued = self.f.issuer.issue(self.f.request)["payload"]
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
                grant_payload=self.issued, root=self.f.quote_fixture.root,
                transport=lambda _, token, request: self.submit_http(request)[1])
        self.assertEqual(result["status"], "adapter_preserved_and_committed")
        self.assertEqual(result["artifact_sha256"], hashlib.sha256(self.storage.body).hexdigest())
        self.assertEqual(self.storage.put_count, 1)
        state = self.f.ledger.replay(self.f.f.io.body)[0]
        self.assertEqual(state["sequence"], 4)
        self.assertEqual(state["events"][-1]["kind"], "family_preserved")
        with patch.object(authority, "_load_bearer_token", return_value=self.secret):
            with self.assertRaises(LedgerRejected):
                preservation.preserve_adapter(adapter=self.adapter,
                    grant_payload=self.issued, root=self.f.quote_fixture.root,
                    transport=lambda _, token, request: self.submit_http(request)[1])
        self.assertEqual(self.storage.put_count, 1)

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
