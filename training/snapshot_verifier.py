"""Offline, exact-inventory verification for a previously downloaded snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.three_family_contract import load_json

MANIFESTS = {
    "kova-cosmo": "qwen3-0.6b-download-manifest.v1.json",
    "kova-orion": "qwen3-1.7b-download-manifest.v1.json",
    "kova-nova": "qwen3-4b-download-manifest.v1.json",
}


def verify_snapshot(snapshot: Path, manifest: dict) -> dict:
    expected = {entry["path"]: entry for entry in manifest["files"]}
    actual = {
        str(path.relative_to(snapshot)).replace("\\", "/")
        for path in snapshot.rglob("*") if path.is_file()
    }
    if actual != set(expected):
        raise ValueError("snapshot_inventory_mismatch")
    for relative, entry in expected.items():
        path = snapshot / relative
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"snapshot_size_mismatch:{relative}")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"snapshot_hash_mismatch:{relative}")
    return {"verified": True, "offline_load_required": True, "files": len(expected)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=MANIFESTS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = load_json(Path(__file__).parents[1] / "config" / MANIFESTS[args.family])
    print(json.dumps(verify_snapshot(args.root, manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
