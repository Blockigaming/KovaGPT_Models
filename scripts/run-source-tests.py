"""Discover all first-party source regression modules, including new guards."""
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
modules = sorted(".".join(path.relative_to(root).with_suffix("").parts)
                 for path in root.glob("*/test_*.py") if path.name != "test_support.py")
unittest.main(module=None, argv=[sys.argv[0], *modules])
