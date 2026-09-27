"""Download only a hash-pinned Qwen snapshot, without touching Azure resources.

Use on an already admitted VM or a free temporary shell. The caller is
responsible for the network, disk and independent resource deadline. A finished
directory appears only after every file has passed the pinned manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote
from urllib.request import Request, urlopen

from training import three_family_contract as contract
from training.snapshot_verifier import MANIFESTS, _manifest, verify_snapshot


def download_snapshot(family: str, destination: Path) -> dict:
    contract.validate()
    if family not in MANIFESTS:
        raise ValueError("unsupported family")
    manifest = _manifest(family)
    if (not destination.is_absolute() or destination.exists() or
            destination.is_symlink() or not destination.parent.is_dir() or
            destination.parent.is_symlink()):
        raise ValueError("destination must be a new absolute directory")
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists() or staging.is_symlink():
        raise ValueError("staging directory already exists")
    staging.mkdir(mode=0o700)
    try:
        for entry in manifest["files"]:
            name = entry["path"]
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("manifest contains a nonflat file path")
            url = ("https://huggingface.co/" + manifest["model"] + "/resolve/" +
                   manifest["revision"] + "/" + quote(name, safe=""))
            digest = hashlib.sha256()
            size = 0
            request = Request(url, headers={"User-Agent": "KovaGPT-Pinned-Snapshot/1"})
            with urlopen(request, timeout=30) as source, (staging / name).open("xb") as target:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > entry["bytes"]:
                        raise ValueError("snapshot size mismatch: " + name)
                    digest.update(block)
                    target.write(block)
            if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError("snapshot hash or size mismatch: " + name)
        report = verify_snapshot(staging, manifest)
        staging.rename(destination)
        return {"family": family, "destination": str(destination), **report}
    except BaseException:
        shutil.rmtree(staging)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=MANIFESTS, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    manifest = _manifest(args.family)
    if not args.execute:
        print(json.dumps({"status": "download_plan_only", "family": args.family,
                          "model": manifest["model"], "revision": manifest["revision"],
                          "bytes": sum(item["bytes"] for item in manifest["files"]),
                          "files": len(manifest["files"])}))
        return 0
    print(json.dumps(download_snapshot(args.family, args.destination), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
