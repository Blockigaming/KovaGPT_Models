"""Read-only, byte-pinned inventory of the September 16 local checkpoint.

This does not restore the archived implementation, execute its code, or decide
that a missing feature has been accepted into the current Models source.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


ARCHIVE_SHA256 = "ab454f7131a74619e0ba36d521ac73942643fdf252b118d685a822058b1fad33"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 4_000_000
MAX_MEMBER_BYTES = 1_000_000
MAX_TOTAL_BYTES = 5_000_000
EXPECTED_SOURCE_PATHS = 119
EXPECTED_MANIFEST_PATHS = 152
SHA_LINE = re.compile(r"([0-9a-f]{64})  (.+)")


class CheckpointRejected(ValueError):
    pass


def _require(condition):
    if not condition:
        raise CheckpointRejected("historical checkpoint rejected")


def _safe_path(name):
    path = PurePosixPath(name)
    _require(name == path.as_posix() and not path.is_absolute()
             and all(part not in ("", ".", "..") for part in name.split("/"))
             and not any(ord(char) < 32 or char == "\\" for char in name))
    return path


def reconcile(archive_path, source_root=SOURCE_ROOT, *, expected_digest=ARCHIVE_SHA256,
              expected_source_paths=EXPECTED_SOURCE_PATHS,
              expected_manifest_paths=EXPECTED_MANIFEST_PATHS):
    """Compare every archived source byte with this checkout, without extraction."""
    with Path(archive_path).open("rb") as stream:
        raw = stream.read(MAX_ARCHIVE_BYTES + 1)
    _require(len(raw) <= MAX_ARCHIVE_BYTES
             and hashlib.sha256(raw).hexdigest() == expected_digest)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        _require(len(names) == len(set(names)) == expected_manifest_paths + 1
                 and "SHA256SUMS" in names)
        for info in infos:
            _safe_path(info.filename)
            mode = info.external_attr >> 16
            _require(not info.is_dir() and info.file_size <= MAX_MEMBER_BYTES
                     and (not mode or stat.S_IFMT(mode) in (0, stat.S_IFREG)))
        _require(sum(info.file_size for info in infos) <= MAX_TOTAL_BYTES)
        manifest = archive.read("SHA256SUMS").decode("utf-8").splitlines()
        _require(len(manifest) == expected_manifest_paths)
        hashes = {}
        for line in manifest:
            match = SHA_LINE.fullmatch(line)
            _require(match is not None)
            digest, name = match.groups()
            _safe_path(name)
            _require(name != "SHA256SUMS" and name in names and name not in hashes)
            _require(hashlib.sha256(archive.read(name)).hexdigest() == digest)
            hashes[name] = digest
        _require(set(hashes) == set(names) - {"SHA256SUMS"})

        sources = sorted(name for name in hashes if name.startswith("Model_Source/"))
        _require(len(sources) == expected_source_paths)
        counts = {"identical": 0, "changed": 0, "historical_only": 0}
        historical_only = []
        for name in sources:
            relative = name.removeprefix("Model_Source/")
            _require(bool(relative))
            current = Path(source_root).joinpath(*PurePosixPath(relative).parts)
            # Never follow a local symlink outside the source tree.
            if current.is_symlink() or not current.resolve().is_relative_to(Path(source_root).resolve()):
                raise CheckpointRejected("historical checkpoint rejected")
            if not current.exists():
                counts["historical_only"] += 1
                historical_only.append(relative)
            else:
                _require(current.is_file())
                current_hash = hashlib.sha256(current.read_bytes()).hexdigest()
                counts["identical" if current_hash == hashes[name] else "changed"] += 1

    return {
        "archive_sha256": expected_digest,
        "verified_archive_members": expected_manifest_paths,
        "historical_source_paths": expected_source_paths,
        "comparison": counts,
        "historical_only_paths": historical_only,
        "a38_reconciled": False,
        "phase_b_ready": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="original September 16 checkpoint ZIP")
    args = parser.parse_args()
    try:
        print(json.dumps(reconcile(args.archive), sort_keys=True))
    except (CheckpointRejected, OSError, UnicodeError, zipfile.BadZipFile):
        parser.exit(1, "historical checkpoint rejected\n")


if __name__ == "__main__":
    main()
