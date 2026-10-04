"""Synthetic acquisition checks; no external request, weights or model calls."""
from contextlib import ExitStack
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.private_provenance import PrivateCatalogError
from training import private_snapshot_download as d


class PrivateSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.destination = self.root / "snapshot"
        self.plan = {"base_model": "internal-nova-base-v1", "base_revision": "a" * 40,
                     "private_catalog_sha256": "b" * 64, "private_source_manifest_sha256": "c" * 64}
        self.body = b"synthetic license and weight fixture"
        self.manifest = {"files": [{"path": "LICENSE", "bytes": len(self.body),
                                  "sha256": hashlib.sha256(self.body).hexdigest()}]}
        self.source = {"download_url_template": "https://models.example.invalid/pinned/{file}"}
        self.stack.enter_context(patch.object(d, "load_plan", return_value=self.plan))
        self.catalog = self.stack.enter_context(patch.object(d, "load_catalog", return_value={
            "sources": {self.plan["base_model"]: self.source}}))
        self.stack.enter_context(patch.object(d, "source_manifest", return_value=self.manifest))
        self.verify = self.stack.enter_context(patch.object(d, "verify_snapshot"))
        self.request = self.stack.enter_context(patch.object(d, "urlopen", side_effect=lambda *_a, **_k: io.BytesIO(self.body)))

    def test_verified_files_and_license_are_preserved(self):
        d.download(self.destination)
        self.assertEqual((self.destination / "LICENSE").read_bytes(), self.body)
        self.catalog.assert_called_once_with(self.plan["private_catalog_sha256"])
        self.assertEqual(self.request.call_args.kwargs["timeout"], 30)
        self.verify.assert_called_once()

    def test_catalog_failure_makes_no_request(self):
        self.catalog.side_effect = PrivateCatalogError("Private model policy unavailable")
        with self.assertRaises(PrivateCatalogError): d.download(self.destination)
        self.request.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_changed_source_digest_makes_no_request(self):
        with patch.object(d, "source_manifest", side_effect=PrivateCatalogError("binding rejected")):
            with self.assertRaises(PrivateCatalogError): d.download(self.destination)
        self.request.assert_not_called()

    def test_untrusted_origin_shapes_rejected(self):
        for url in ("http://models.example.invalid/{file}", "https://user:secret@models.example.invalid/{file}",
                    "https://models.example.invalid/{file}?credential=secret", "https://models.example.invalid/fixed"):
            with self.subTest(url=url):
                self.source["download_url_template"] = url
                with self.assertRaises(ValueError): d.download(self.destination)
        self.request.assert_not_called()

    def test_path_escape_rejected_before_request(self):
        self.manifest["files"][0]["path"] = "../secret"
        with self.assertRaises(ValueError): d.download(self.destination)
        self.request.assert_not_called()
        self.assertFalse(self.destination.with_name("snapshot.partial").exists())

    def test_wrong_hash_removes_partial_without_retry(self):
        self.manifest["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): d.download(self.destination)
        self.assertEqual(self.request.call_count, 1)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.destination.with_name("snapshot.partial").exists())

    def test_oversize_removes_partial_without_retry(self):
        self.manifest["files"][0]["bytes"] -= 1
        with self.assertRaises(ValueError): d.download(self.destination)
        self.assertEqual(self.request.call_count, 1)
        self.assertFalse(self.destination.with_name("snapshot.partial").exists())

    def test_existing_destination_cannot_be_reused(self):
        self.destination.mkdir()
        with self.assertRaises(ValueError): d.download(self.destination)
        self.request.assert_not_called()


if __name__ == "__main__": unittest.main()
