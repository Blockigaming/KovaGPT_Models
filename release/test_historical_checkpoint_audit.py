"""Reject untrusted archives and account for saved-only source without running it."""

import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from release.historical_checkpoint_audit import CheckpointRejected, reconcile


class HistoricalCheckpointAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()

    def archive(self, files, manifest=None):
        archive = self.root / "checkpoint.zip"
        if manifest is None:
            manifest = "".join(
                f"{hashlib.sha256(content).hexdigest()}  {name}\n"
                for name, content in files
            ).encode()
        with zipfile.ZipFile(archive, "w") as output:
            for name, content in files:
                output.writestr(name, content)
            output.writestr("SHA256SUMS", manifest)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        return archive, digest

    def audit(self, archive, digest, *, source_paths, manifest_paths):
        return reconcile(archive, self.source, expected_digest=digest,
                         expected_source_paths=source_paths,
                         expected_manifest_paths=manifest_paths)

    def test_accounts_for_every_source_path_without_importing_saved_code(self):
        (self.source / "same.py").write_bytes(b"same")
        (self.source / "changed.py").write_bytes(b"new")
        files = [("Model_Source/same.py", b"same"),
                 ("Model_Source/changed.py", b"old"),
                 ("Model_Source/missing.py", b"raise RuntimeError('never import')"),
                 ("SOURCE_PROVENANCE.json", b"{}")]
        archive, digest = self.archive(files)
        report = self.audit(archive, digest, source_paths=3, manifest_paths=4)
        self.assertEqual(report["comparison"],
                         {"identical": 1, "changed": 1, "historical_only": 1})
        self.assertEqual(report["historical_only_paths"], ["missing.py"])
        self.assertFalse(report["a38_reconciled"])

    def test_manifest_mismatch_is_rejected_even_when_zip_hash_is_current(self):
        files = [("Model_Source/same.py", b"different")]
        manifest = f"{hashlib.sha256(b'original').hexdigest()}  Model_Source/same.py\n".encode()
        archive, digest = self.archive(files, manifest)
        with self.assertRaises(CheckpointRejected):
            self.audit(archive, digest, source_paths=1, manifest_paths=1)

    def test_path_traversal_is_rejected_without_extraction(self):
        archive, digest = self.archive([("Model_Source/../outside.py", b"never")])
        with self.assertRaises(CheckpointRejected):
            self.audit(archive, digest, source_paths=1, manifest_paths=1)
        self.assertFalse((self.root / "outside.py").exists())

    def test_duplicate_members_are_rejected(self):
        archive, digest = self.archive([("Model_Source/same.py", b"first"),
                                        ("Model_Source/same.py", b"second")])
        with self.assertRaises(CheckpointRejected):
            self.audit(archive, digest, source_paths=2, manifest_paths=2)


if __name__ == "__main__":
    unittest.main()
