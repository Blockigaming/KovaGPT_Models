"""Read an operator-supplied private catalog; never serialize its values.

The catalog is not distributed with public runtime source. Its path and digest
are trusted deployment inputs, not request parameters. Missing, modified or
malformed catalogs fail closed. Tests supply an explicitly synthetic catalog.
"""
import hashlib
import json
import os
from pathlib import Path
import re


class PrivateCatalogError(ValueError):
    pass


def load_catalog(expected_digest=None):
    try:
        path = Path(os.environ["KOVA_PRIVATE_CATALOG"])
        digest = os.environ["KOVA_PRIVATE_CATALOG_SHA256"]
        if expected_digest is not None and digest != expected_digest:
            raise ValueError()
        if not path.is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError()
        if path.is_symlink() or path.stat().st_size > 1048576:
            raise ValueError()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError()
        catalog = json.loads(raw)
        aliases = catalog["aliases"]
        if (catalog["schema_version"] != 1 or type(aliases) is not list
                or not aliases or len(aliases) > 1024
                or any(type(a) is not str or not 3 <= len(a) <= 256 for a in aliases)
                or type(catalog["sources"]) is not dict):
            raise ValueError()
        return catalog
    except (KeyError, OSError, ValueError, TypeError):
        raise PrivateCatalogError("Private model policy unavailable") from None


def source_manifest(source_id, expected_revision, expected_digest):
    """Bind real acquisition to a private manifest, not a public placeholder."""
    try:
        source = load_catalog()["sources"][source_id]
        manifest = source["manifest"]
        raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if (source["revision"] != expected_revision
                or hashlib.sha256(raw).hexdigest() != expected_digest):
            raise ValueError()
        return manifest
    except (KeyError, ValueError, TypeError):
        raise PrivateCatalogError("Private model source binding rejected") from None
