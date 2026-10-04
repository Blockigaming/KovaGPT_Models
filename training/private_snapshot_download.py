"""Private-catalog acquisition for the admitted Nova bootstrap; no public origins."""
import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from core.private_provenance import load_catalog, source_manifest
from training.a35_nova_screen import load_plan
from training.snapshot_verifier import verify_snapshot


def download(destination):
    plan = load_plan()
    catalog = load_catalog(plan["private_catalog_sha256"])
    manifest = source_manifest(plan["base_model"], plan["base_revision"],
                               plan["private_source_manifest_sha256"])
    template = catalog["sources"][plan["base_model"]]["download_url_template"]
    origin = urlsplit(template)
    if (origin.scheme != "https" or not origin.hostname or origin.username or origin.password
            or origin.query or origin.fragment or template.count("{file}") != 1):
        raise ValueError("Private acquisition origin rejected")
    if (not destination.is_absolute() or destination.exists() or destination.is_symlink()
            or not destination.parent.is_dir() or destination.parent.is_symlink()):
        raise ValueError("New snapshot directory required")
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists() or staging.is_symlink():
        raise ValueError("Snapshot staging already exists")
    staging.mkdir(mode=0o700)
    try:
        for entry in manifest["files"]:
            name = entry["path"]
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Private manifest path rejected")
            digest, size = hashlib.sha256(), 0
            request = Request(template.replace("{file}", quote(name, safe="")),
                              headers={"User-Agent": "KovaGPT-Private-Snapshot/1"})
            with urlopen(request, timeout=30) as source, (staging / name).open("xb") as target:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > entry["bytes"]:
                        raise ValueError("Private snapshot size rejected")
                    digest.update(block)
                    target.write(block)
            if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError("Private snapshot integrity rejected")
        verify_snapshot(staging, manifest)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    download(args.destination)
    print('{"status":"verified_private_snapshot","family":"kova-nova"}')
