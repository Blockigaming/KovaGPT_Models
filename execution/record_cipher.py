"""Authenticated record encryption; no key service, network, files or defaults.

Uses maintained cryptography AES-SIV rather than custom cryptography. Each seal
uses a fresh 128-bit nonce as associated data; SIV remains misuse resistant if a
random nonce ever repeats. Keys stay in the caller's protected key service.
"""

import base64
from dataclasses import dataclass, field
import json
import secrets

from execution.contracts import ExecutionError, canonical, identifier


MAX_RECORD_BYTES = 8 * 1024 * 1024


class RecordProtectionError(ExecutionError):
    """Ciphertext, scope, retention metadata or key availability failed validation."""


def need(condition):
    if not condition:
        raise RecordProtectionError("protected record unavailable")


@dataclass(frozen=True)
class KeyMaterial:
    key_id: str
    key: bytes = field(repr=False)

    def __post_init__(self):
        identifier(self.key_id, "encryption key ID")
        need(type(self.key) is bytes and len(self.key) == 64)


class RecordCipher:
    """active_key(owner) / lookup_key(owner,key_id) must use trusted key storage.

    A key ID is never a file path or URL. A server adapter must enforce key/owner
    isolation and approved rotation/revocation. No environment/CLI/key fallback
    or key generation occurs in this class; tests inject ephemeral keys.
    """
    def __init__(self, active_key, lookup_key):
        need(callable(active_key) and callable(lookup_key))
        self._active, self._lookup = active_key, lookup_key

    @staticmethod
    def _aad(owner, job_id, slot, retain_until_ms):
        for value, name in ((owner, "owner"), (job_id, "job"), (slot, "record slot")):
            identifier(value, name)
        need(type(retain_until_ms) is int and 0 < retain_until_ms < 2**53)
        return canonical({"domain": "kova.private-record.v1", "owner": owner,
                          "job": job_id, "slot": slot, "retain_until_ms": retain_until_ms})

    def seal(self, owner, job_id, slot, retain_until_ms, plaintext):
        need(type(plaintext) is bytes and 0 < len(plaintext) <= MAX_RECORD_BYTES)
        aad = self._aad(owner, job_id, slot, retain_until_ms)
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESSIV
            key = self._active(owner)
            need(type(key) is KeyMaterial)
            nonce = secrets.token_bytes(16)
            ciphertext = AESSIV(key.key).encrypt(plaintext, [aad, nonce])
            return canonical({"version": 1, "algorithm": "AES-256-SIV", "key_id": key.key_id,
                              "nonce": base64.b64encode(nonce).decode(),
                              "ciphertext": base64.b64encode(ciphertext).decode()})
        except Exception:
            raise RecordProtectionError("protected record unavailable") from None

    def open(self, owner, job_id, slot, retain_until_ms, envelope):
        need(type(envelope) is bytes and 0 < len(envelope) <= 12 * 1024 * 1024)
        aad = self._aad(owner, job_id, slot, retain_until_ms)
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESSIV
            value = json.loads(envelope)
            need(canonical(value) == envelope and type(value) is dict
                 and set(value) == {"version", "algorithm", "key_id", "nonce", "ciphertext"}
                 and type(value["version"]) is int and value["version"] == 1
                 and value["algorithm"] == "AES-256-SIV")
            identifier(value["key_id"], "encryption key ID")
            need(type(value["nonce"]) is str and type(value["ciphertext"]) is str)
            nonce = base64.b64decode(value["nonce"], validate=True)
            ciphertext = base64.b64decode(value["ciphertext"], validate=True)
            need(len(nonce) == 16 and 16 < len(ciphertext) <= MAX_RECORD_BYTES + 16)
            need(base64.b64encode(nonce).decode() == value["nonce"]
                 and base64.b64encode(ciphertext).decode() == value["ciphertext"])
            key = self._lookup(owner, value["key_id"])
            need(type(key) is KeyMaterial and key.key_id == value["key_id"])
            return AESSIV(key.key).decrypt(ciphertext, [aad, nonce])
        except Exception:
            raise RecordProtectionError("protected record unavailable") from None
