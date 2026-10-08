"""Explicit source-test harness; never imported by runtime or paid launchers."""
import hashlib
import os
from pathlib import Path
import sys

path = Path(__file__).resolve().parents[1] / "tests/fixtures/synthetic-private-catalog.json"
env = dict(os.environ, KOVA_PRIVATE_CATALOG=str(path),
           KOVA_PRIVATE_CATALOG_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
os.execvpe(sys.argv[1], sys.argv[1:], env)
