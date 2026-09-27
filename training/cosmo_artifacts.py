"""Immutable artifact verification for the pinned Qwen3-0.6B checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = "config/qwen3-0.6b-download-manifest.v1.json"
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
HEX64 = re.compile(r"[0-9a-f]{64}")


class ArtifactError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise ArtifactError("kova cosmo artifact rejected")


def load_manifest(root: Path = ROOT) -> dict:
    """Load the reviewed, exact-revision inventory without network access."""
    try:
        value = json.loads(
            (root / MANIFEST_PATH).read_text(encoding="utf-8")
        )
        need(type(value) is dict and list(value) == [
            "schema_version", "model", "revision", "files",
        ])
        need(value["schema_version"] == 1)
        need(value["model"] == MODEL)
        need(value["revision"] == REVISION)
        files = value["files"]
        need(type(files) is list and len(files) == 9)
        expected_names = [
            "LICENSE", "README.md", "config.json", "generation_config.json",
            "merges.txt", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "vocab.json",
        ]
        need([item.get("path") for item in files] == expected_names)
        for item in files:
            need(type(item) is dict and list(item) == [
                "path", "bytes", "sha256",
            ])
            need(type(item["bytes"]) is int and item["bytes"] > 0)
            need(type(item["sha256"]) is str and
                 HEX64.fullmatch(item["sha256"]) is not None)
        return value
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, RecursionError, json.JSONDecodeError):
        raise ArtifactError("kova cosmo artifact rejected") from None


DOWNLOAD_MANIFEST = load_manifest()
REQUIRED_ASSETS = tuple(item["path"] for item in DOWNLOAD_MANIFEST["files"])
EXPECTED_SHA256 = {
    item["path"]: item["sha256"] for item in DOWNLOAD_MANIFEST["files"]
}
EXPECTED_BYTES = {
    item["path"]: item["bytes"] for item in DOWNLOAD_MANIFEST["files"]
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(directory: Path) -> dict[str, dict]:
    try:
        need(directory.is_absolute() and directory.is_dir())
        need({item.name for item in directory.iterdir()} ==
             set(REQUIRED_ASSETS))
        inventory = {}
        for name in REQUIRED_ASSETS:
            path = directory / name
            need(path.is_file())
            digest = file_sha256(path)
            need(path.stat().st_size == EXPECTED_BYTES[name])
            need(digest == EXPECTED_SHA256[name])
            inventory[name] = {"bytes": path.stat().st_size, "sha256": digest}

        config = json.loads(
            (directory / "config.json").read_text(encoding="utf-8")
        )
        license_text = (directory / "LICENSE").read_text(encoding="utf-8")
        need(type(config) is dict)
        need(config.get("architectures") == ["Qwen3ForCausalLM"])
        need(config.get("model_type") == "qwen3")
        need(config.get("num_hidden_layers") == 28)
        need(config.get("hidden_size") == 1024)
        need(config.get("vocab_size") == 151936)
        need("Apache License" in license_text)
        need("Version 2.0, January 2004" in license_text)
        return inventory
    except (OSError, UnicodeError, ValueError, TypeError, KeyError,
            json.JSONDecodeError):
        raise ArtifactError("kova cosmo artifact rejected") from None


def dry_run() -> dict:
    return {
        "status": "download_manifest_verified_no_download_performed",
        "model": DOWNLOAD_MANIFEST["model"],
        "revision": DOWNLOAD_MANIFEST["revision"],
        "files": len(REQUIRED_ASSETS),
        "total_bytes": sum(EXPECTED_BYTES.values()),
        "all_files_sha256_pinned": (
            set(REQUIRED_ASSETS) == set(EXPECTED_SHA256)
        ),
        "model_weights_downloaded": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", nargs="?", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = dry_run() if arguments.snapshot is None else {
            "status": "snapshot_verified",
            "model": MODEL,
            "revision": REVISION,
            "inventory": verify_snapshot(arguments.snapshot),
        }
        print(json.dumps(report, sort_keys=True))
        return 0
    except ArtifactError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
