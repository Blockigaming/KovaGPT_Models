"""A snapshot becomes visible only after exact pinned bytes pass verification."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from training import pinned_snapshot_download as download


class PinnedDownloadTests(unittest.TestCase):
    def test_verified_publish_and_tampered_download_cleanup(self):
        data = b"pinned-test-weights"
        manifest = {"model": "Qwen/Qwen3-0.6B", "revision": "a" * 40,
                    "files": [{"path": "model.safetensors", "bytes": len(data),
                               "sha256": hashlib.sha256(data).hexdigest()}]}
        with TemporaryDirectory() as folder, \
                patch.object(download.contract, "validate"), \
                patch.object(download, "_manifest", return_value=manifest):
            output = Path(folder) / "snapshot"
            with patch.object(download, "urlopen", return_value=BytesIO(data)) as request:
                report = download.download_snapshot("kova-cosmo", output)
            self.assertEqual((report["verified"], report["files"]), (True, 1))
            self.assertEqual((output / "model.safetensors").read_bytes(), data)
            self.assertEqual(request.call_args.args[0].full_url,
                             "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/" +
                             "a" * 40 + "/model.safetensors")
            failed = Path(folder) / "corrupted"
            with patch.object(download, "urlopen", return_value=BytesIO(b"x" * len(data))):
                with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
                    download.download_snapshot("kova-cosmo", failed)
            self.assertFalse(failed.exists())
            self.assertFalse(failed.with_name("corrupted.partial").exists())


if __name__ == "__main__":
    unittest.main()
