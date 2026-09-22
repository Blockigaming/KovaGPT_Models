"""Authenticate measured generation with runner-only Ed25519 authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

ROOT = Path(__file__).resolve().parents[1]
TRUST_POLICY_PATH = "config/kova-cosmo-generation-trust.v1.json"
MAX_CANONICAL_BYTES = 16 * 1024 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX128 = re.compile(r"[0-9a-f]{128}")
ALGORITHM = "ed25519"
PRIVATE_KEY_ENV = "KOVA_COSMO_GENERATION_SIGNING_KEY_FILE"


class AttestationError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise AttestationError("kova cosmo generation attestation rejected")


def canonical(value: object) -> bytes:
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
        need(0 < len(raw) <= MAX_CANONICAL_BYTES)
        return raw
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise AttestationError(
            "kova cosmo generation attestation rejected"
        ) from None


def load_trust_policy(root: Path = ROOT) -> dict:
    """Load the reviewed public verification key; private keys never enter."""
    try:
        value = json.loads(
            (root / TRUST_POLICY_PATH).read_text(encoding="utf-8")
        )
        need(type(value) is dict and list(value) == [
            "schema_version", "status", "algorithm", "public_key_hex",
            "public_key_sha256", "runner_private_key_environment_variable",
            "verifier_private_key_access_allowed",
            "checked_in_private_key_allowed",
        ])
        need(value["schema_version"] == 1)
        need(value["algorithm"] == ALGORITHM)
        need(value["runner_private_key_environment_variable"] ==
             PRIVATE_KEY_ENV)
        need(value["verifier_private_key_access_allowed"] is False)
        need(value["checked_in_private_key_allowed"] is False)
        public = value["public_key_hex"]
        fingerprint = value["public_key_sha256"]
        if value["status"] == "signing_public_key_provisioning_required":
            need(public is None and fingerprint is None)
        else:
            need(value["status"] == "runner_signing_public_key_pinned")
            need(type(public) is str and HEX64.fullmatch(public) is not None)
            need(type(fingerprint) is str and
                 HEX64.fullmatch(fingerprint) is not None)
            need(hashlib.sha256(bytes.fromhex(public)).hexdigest() ==
                 fingerprint)
        return value
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, RecursionError, json.JSONDecodeError):
        raise AttestationError(
            "kova cosmo generation attestation rejected"
        ) from None


def public_key_hex(private_key: Ed25519PrivateKey) -> str:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def load_signing_key(path: Path, *, expected_public_key_hex: str,
                     repository_root: Path = ROOT) -> Ed25519PrivateKey:
    """Load the runner-only seed and prove it matches reviewed public trust."""
    try:
        need(path.is_absolute() and path.is_file() and not path.is_symlink())
        resolved = path.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
        need(repository != resolved and repository not in resolved.parents)
        need(resolved.stat().st_mode & 0o077 == 0)
        with resolved.open("rb") as stream:
            raw = stream.read(66)
        encoded = raw.rstrip(b"\n")
        need(len(encoded) == 64 and raw in (encoded, encoded + b"\n"))
        text = encoded.decode("ascii")
        need(HEX64.fullmatch(text) is not None)
        need(type(expected_public_key_hex) is str and
             HEX64.fullmatch(expected_public_key_hex) is not None)
        key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(text))
        need(public_key_hex(key) == expected_public_key_hex)
        return key
    except (OSError, ValueError, TypeError, UnicodeError):
        raise AttestationError(
            "kova cosmo generation attestation rejected"
        ) from None


def measurement_payload(bundle: dict) -> dict:
    """Return exactly the runner-produced fields protected by the signature."""
    need(type(bundle) is dict)
    expected = [
        "schema_version", "kind", "plan_sha256", "source_commit",
        "adapter_sha256", "adapter_receipt_sha256", "evaluation_grant",
        "runner_attestation", "attempts",
    ]
    need(list(bundle) == expected)
    attempts = bundle["attempts"]
    need(type(attempts) is list)
    protected_attempts = []
    for row in attempts:
        need(type(row) is dict and "scores" in row)
        protected_attempts.append({
            key: value for key, value in row.items() if key != "scores"
        })
    return {
        "schema_version": bundle["schema_version"],
        "kind": bundle["kind"],
        "plan_sha256": bundle["plan_sha256"],
        "source_commit": bundle["source_commit"],
        "adapter_sha256": bundle["adapter_sha256"],
        "adapter_receipt_sha256": bundle["adapter_receipt_sha256"],
        "evaluation_grant": bundle["evaluation_grant"],
        "attempts": protected_attempts,
    }


def create_attestation(bundle: dict,
                       key: bytes | Ed25519PrivateKey) -> dict:
    if type(key) is bytes:
        need(len(key) == 32)
        key = Ed25519PrivateKey.from_private_bytes(key)
    need(isinstance(key, Ed25519PrivateKey))
    need(bundle.get("kind") == "measured")
    need(bundle.get("runner_attestation") is None)
    raw = canonical(measurement_payload(bundle))
    public = public_key_hex(key)
    return {
        "schema_version": 1,
        "kind": "kova_cosmo_guarded_runner_attestation",
        "algorithm": ALGORITHM,
        "public_key_sha256": hashlib.sha256(
            bytes.fromhex(public)
        ).hexdigest(),
        "measurement_sha256": hashlib.sha256(raw).hexdigest(),
        "signature": key.sign(raw).hex(),
    }


def verify_attestation(bundle: dict, *, public_key_hex_value: str) -> dict:
    try:
        need(type(public_key_hex_value) is str and
             HEX64.fullmatch(public_key_hex_value) is not None)
        value = bundle.get("runner_attestation")
        need(type(value) is dict and list(value) == [
            "schema_version", "kind", "algorithm", "public_key_sha256",
            "measurement_sha256", "signature",
        ])
        need(value["schema_version"] == 1)
        need(value["kind"] == "kova_cosmo_guarded_runner_attestation")
        need(value["algorithm"] == ALGORITHM)
        need(type(value["public_key_sha256"]) is str and
             HEX64.fullmatch(value["public_key_sha256"]) is not None)
        need(type(value["measurement_sha256"]) is str and
             HEX64.fullmatch(value["measurement_sha256"]) is not None)
        need(type(value["signature"]) is str and
             HEX128.fullmatch(value["signature"]) is not None)
        raw = canonical(measurement_payload(bundle))
        expected_public_fingerprint = hashlib.sha256(
            bytes.fromhex(public_key_hex_value)
        ).hexdigest()
        expected_measurement = hashlib.sha256(raw).hexdigest()
        need(value["public_key_sha256"] == expected_public_fingerprint)
        need(value["measurement_sha256"] == expected_measurement)
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(public_key_hex_value)
        ).verify(bytes.fromhex(value["signature"]), raw)
        return {
            "status": "guarded_runner_signature_verified",
            "algorithm": ALGORITHM,
            "public_key_sha256": expected_public_fingerprint,
            "measurement_sha256": expected_measurement,
        }
    except (InvalidSignature, OSError, ValueError, TypeError, KeyError,
            AttributeError, UnicodeError, RecursionError):
        raise AttestationError(
            "kova cosmo generation attestation rejected"
        ) from None
