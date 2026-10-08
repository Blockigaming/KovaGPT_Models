"""Scan public identity surfaces and complete outgoing files using private terms.

No recognizer strings or matched content are emitted. Pre-existing historical
engineering evidence is not rewritten or claimed to have disappeared from Git.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

from core.private_provenance import load_catalog

ROOT = Path(__file__).resolve().parents[1]
SURFACES = (
    "core/public_identity.py", "core/private_provenance.py",
    "core/identity.py", "core/current_candidates.py", "core/test_public_identity.py",
    "README.md", "config/evaluation-gates.v2.json",
    "worker/handler.py", "execution/store.py", "execution/workers.py",
    "router/application.py", "docs/application-bridge.md",
    "evaluations/offline-suite.v1.json",
    "training/a35_identity_inputs.py", "training/test_a35_identity_inputs.py",
    "training/private_snapshot_download.py", "training/test_private_snapshot_download.py",
    "training/a35_nova_screen_bootstrap.sh",
    "config/a35-nova-screen.v1.json", "config/identity.v2.json",
    "data/a35-kovagpt-identity-overrides.v1.jsonl",
    "data/a35-kovagpt-identity-review.v1.json", "prompts/kova-identity.v5.txt",
    "docs/kovagpt-identity-v5.md", "docs/current-model-topology.md",
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, timeout=20, text=True)


def scan(base):
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("Exact publication parent required")
    terms = load_catalog().get("public_disclosure_terms")
    if not isinstance(terms, list) or not terms or any(not isinstance(t, str) or len(t) < 3 for t in terms):
        raise ValueError("Private disclosure rules required")
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.I)
    violations = []
    for name in SURFACES:
        if pattern.search((ROOT / name).read_text()):
            violations.append({"path": name, "scope": "active_public_surface"})
    changed = set(git("diff", "--name-only", base).splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    for name in sorted(changed):
        path = ROOT / name
        if not path.is_file():
            continue
        if pattern.search(path.read_text()):
            violations.append({"path": name, "scope": "complete_outgoing_file_identifier"})
    if violations:
        raise ValueError(json.dumps({"status": "FAIL", "violations": violations}))
    return {"status": "PASS", "parent": base, "active_surfaces": len(SURFACES),
            "changed_files": len(changed), "private_values_printed": False,
            "complete_outgoing_file_contents_scanned": True,
            "historical_engineering_evidence_rewritten": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    print(json.dumps(scan(parser.parse_args().base), sort_keys=True))
