"""Offline, exact-inventory verification for previously downloaded snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from training import three_family_contract as contract

MANIFESTS = {
    "kova-cosmo": "qwen3-0.6b-download-manifest.v1.json",
    "kova-orion": "qwen3-1.7b-download-manifest.v1.json",
    "kova-nova": "qwen3-4b-download-manifest.v1.json",
}


def _identity(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode,
            metadata.st_nlink, metadata.st_size, metadata.st_mtime_ns,
            metadata.st_ctime_ns)


def _protected(directory: Path, entries: list[Path]) -> bool:
    # Only a read-only mount or a different owner with read-only permissions
    # can protect the tree from the unprivileged training process. A process
    # running as root cannot claim ownership-based protection.
    paths = [directory, *directory.rglob("*"), *entries]
    try:
        mounted_read_only = bool(os.statvfs(directory).f_flag & os.ST_RDONLY)
        return mounted_read_only or (os.geteuid() != 0 and all(
            path.lstat().st_uid != os.geteuid() and
            not path.lstat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            for path in paths))
    except OSError:
        return False


def verify_snapshot(snapshot: Path, manifest: dict, *, require_protected: bool = False) -> dict:
    expected = {entry["path"]: entry for entry in manifest["files"]}
    if not stat.S_ISDIR(snapshot.lstat().st_mode):
        raise ValueError("snapshot_root_not_directory")
    actual = set()
    entries = []
    for path in snapshot.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode) or path.lstat().st_nlink != 1:
            raise ValueError("snapshot_nonregular_or_linked_entry")
        actual.add(path.relative_to(snapshot).as_posix())
        entries.append(path)
    if actual != set(expected):
        raise ValueError("snapshot_inventory_mismatch")
    for relative, entry in expected.items():
        path = snapshot / relative
        hasher = hashlib.sha256()
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with os.fdopen(os.open(path, flags), "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError(f"snapshot_nonregular_or_linked_entry:{relative}")
            if metadata.st_size != entry["bytes"]:
                raise ValueError(f"snapshot_size_mismatch:{relative}")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
            if (_identity(os.fstat(stream.fileno())) != _identity(metadata) or
                    _identity(path.lstat()) != _identity(metadata)):
                raise ValueError(f"snapshot_identity_changed:{relative}")
        digest = hasher.hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"snapshot_hash_mismatch:{relative}")
    if {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if not p.is_dir()} != actual:
        raise ValueError("snapshot_inventory_changed")
    protected = _protected(snapshot, entries)
    if require_protected and not protected:
        raise ValueError("snapshot_must_be_protected_from_training_identity")
    return {"verified": True, "protected_for_loading": protected,
            "offline_load_required": True, "files": len(expected)}


def _manifest(family: str) -> dict:
    lineage = contract.load_json(contract.ROOT / "config/kova-private-lineage.v1.json")
    return contract._pinned_manifest(family, lineage["families"][family])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    family = parser.add_mutually_exclusive_group(required=True)
    family.add_argument("--family", choices=MANIFESTS)
    family.add_argument("--all-families", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.all_families:
        results = {
            name: verify_snapshot(args.root / name, _manifest(name))
            for name in MANIFESTS
        }
        report = {"verified": all(item["verified"] for item in results.values()), "families": results}
    else:
        report = verify_snapshot(args.root, _manifest(args.family))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
