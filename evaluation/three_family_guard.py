"""Cryptographic evidence envelopes for three-family adapter evaluation."""

from __future__ import annotations

import base64
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

FAMILIES = {"kova-cosmo", "kova-orion", "kova-nova"}
SCHEMA = "kova-three-family-evaluation-evidence.v1"


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8", "strict")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_evidence(*, family: str, base_manifest_sha256: str, adapter_sha256: str,
                    runner_sha256: str, attempt_id: str, prompt: str, answer: str,
                    dimensions: list[str], private_key: Ed25519PrivateKey,
                    created_at: str | None = None) -> dict:
    if family not in FAMILIES:
        raise ValueError("unknown_family")
    for name, value in (("base_manifest", base_manifest_sha256), ("adapter", adapter_sha256),
                        ("runner", runner_sha256)):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid_{name}_sha256")
    if not attempt_id or not prompt or not answer or not dimensions:
        raise ValueError("empty_evaluation_binding")
    payload = {
        "schema": SCHEMA,
        "family": family,
        "base_manifest_sha256": base_manifest_sha256,
        "adapter_sha256": adapter_sha256,
        "runner_sha256": runner_sha256,
        "attempt_id": attempt_id,
        "prompt": prompt,
        "answer": answer,
        "dimensions": dimensions,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    signature = private_key.sign(_canonical(payload))
    public = private_key.public_key().public_bytes_raw()
    return {
        "payload": payload,
        "signature_ed25519_b64": base64.b64encode(signature).decode("ascii"),
        "public_key_ed25519_b64": base64.b64encode(public).decode("ascii"),
    }


def verify_evidence(evidence: dict, *, expected_family: str, expected_base_manifest_sha256: str,
                    expected_adapter_sha256: str, expected_runner_sha256: str) -> dict:
    if set(evidence) != {"payload", "signature_ed25519_b64", "public_key_ed25519_b64"}:
        raise ValueError("invalid_evidence_shape")
    payload = evidence["payload"]
    expected = {
        "family": expected_family,
        "base_manifest_sha256": expected_base_manifest_sha256,
        "adapter_sha256": expected_adapter_sha256,
        "runner_sha256": expected_runner_sha256,
    }
    if payload.get("schema") != SCHEMA:
        raise ValueError("invalid_evidence_schema")
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"evidence_binding_mismatch:{field}")
    try:
        public = Ed25519PublicKey.from_public_bytes(base64.b64decode(
            evidence["public_key_ed25519_b64"], validate=True))
        signature = base64.b64decode(evidence["signature_ed25519_b64"], validate=True)
        public.verify(signature, _canonical(payload))
    except Exception as exc:
        raise ValueError("invalid_evidence_signature") from exc
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("evaluation execution blocked until a signed adapter and authority archive exist")
    print(json.dumps({"family": args.family, "dry_run": True, "generated_answers": 0,
                      "provider_calls_made": 0, "signature_scheme": "Ed25519"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
