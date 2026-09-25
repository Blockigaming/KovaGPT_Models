"""One Cosmo QLoRA adapter transfer to independently protected Azure storage.

The guest sends a bounded adapter bundle to the controller. Only the controller
has Blob credentials and preservation signing key. A successful response is a
signed ledger record committed after the controller reads the immutable blob
back; no guest-side filesystem path counts as preservation.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import stat
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import zipfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from training import cosmo_lifecycle_authority as authority
from training.cosmo_controller_ledger import AzureBlobIO, LedgerRejected, need, parse_json

MAX_BUNDLE = 64 * 1024 * 1024
MAX_REQUEST = 90 * 1024 * 1024
FILES = ("adapter_config.json", "adapter_model.safetensors")


def _inspect_bundle(bundle: bytes):
    need(type(bundle) is bytes and 0 < len(bundle) <= MAX_BUNDLE,
         "bounded adapter bundle required")
    try:
        with zipfile.ZipFile(BytesIO(bundle)) as archive:
            names = archive.namelist()
            need(sorted(names) == sorted(FILES), "adapter bundle members changed")
            for name in FILES:
                info = archive.getinfo(name)
                need(info.file_size > 0 and info.file_size <= MAX_BUNDLE and
                     info.compress_size == info.file_size and
                     info.compress_type == zipfile.ZIP_STORED,
                     "adapter bundle entry invalid")
                with archive.open(info) as stream:
                    need(len(stream.read(MAX_BUNDLE + 1)) == info.file_size,
                         "adapter bundle truncated")
            config = json.loads(archive.read(FILES[0]))
            need(type(config) is dict and config.get("peft_type") == "LORA",
                 "adapter configuration invalid")
    except (ValueError, KeyError, OSError, zipfile.BadZipFile, UnicodeError,
            json.JSONDecodeError, NotImplementedError) as exc:
        raise LedgerRejected("invalid Cosmo adapter bundle") from exc


def bundle_adapter(adapter: Path) -> bytes:
    """Read the two PEFT artifacts without following symlinks or a replaced file."""
    need(adapter.is_dir() and not adapter.is_symlink(), "adapter directory missing")
    need(set(p.name for p in adapter.iterdir()) >= set(FILES),
         "trained adapter files missing")
    data = {}
    for name in FILES:
        fd = os.open(adapter / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            need(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_BUNDLE,
                 "invalid adapter file")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(MAX_BUNDLE + 1)
            need(len(raw) == info.st_size and os.fstat(fd).st_size == info.st_size,
                 "adapter file changed while reading")
            data[name] = raw
        finally:
            os.close(fd)
    result = BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in FILES:
            # ZIP's default writestr timestamp is local wall time, which would
            # make a retry differ from the immutable Blob after a lost reply.
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_STORED
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, data[name])
    raw = result.getvalue()
    _inspect_bundle(raw)
    return raw


def preservation_endpoint(grant_endpoint: str) -> str:
    parts = urlsplit(grant_endpoint)
    need(parts.scheme == "https" and parts.hostname and parts.port in (None, 443) and
         parts.path == "/v1/pilot/grants" and not parts.query and not parts.fragment and
         not parts.username and not parts.password, "invalid controller endpoint")
    return grant_endpoint[: -len("grants")] + "preservation"


def _encode_request(request: dict) -> bytes:
    # Ledger envelopes remain under 1 MiB; the artifact itself may be much
    # larger, so authority.canonical() must not be used for this HTTP body.
    raw = json.dumps(request, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("ascii")
    need(0 < len(raw) <= MAX_REQUEST, "adapter transfer exceeds the approved bound")
    return raw


def _post(endpoint, token, request):
    raw = _encode_request(request)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    http = urllib.request.Request(endpoint, data=raw, method="POST", headers={
        "Authorization": "Bearer " + token, "Content-Type": "application/json",
        "Accept": "application/json"})
    try:
        with opener.open(http, timeout=120) as response:
            need(response.status == 200 and response.geturl() == endpoint,
                 "adapter preservation request rejected")
            body = response.read(65537)
        need(0 < len(body) <= 65536, "invalid preservation response")
        return parse_json(body)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise LedgerRejected("adapter preservation uncertain; inspect remote ledger") from exc


def preserve_adapter(*, adapter: Path, grant_payload: dict, root: Path,
                     transport=None) -> dict:
    """Never return success until the controller's signed append is checked."""
    trust = authority.load_trust_policy(root)
    need(trust["status"] == "authority_pinned", "independent controller not released")
    bundle = bundle_adapter(adapter)
    digest = hashlib.sha256(bundle).hexdigest()
    request = {"schema_version": 1, "kind": "kova_cosmo_preserve_adapter_v1",
               "lifecycle_id": grant_payload["lifecycle_id"],
               "grant_id": grant_payload["grant_id"],
               "artifact_sha256": digest,
               "bundle_b64": base64.b64encode(bundle).decode("ascii")}
    _encode_request(request)
    token = authority._load_bearer_token(Path(os.environ.get(authority.TOKEN_ENV, "")),
                                         repository_root=root)
    endpoint = preservation_endpoint(trust["endpoint"])
    record = (transport or _post)(endpoint, token, request)
    need(type(record) is dict and set(record) == {"payload", "signature"},
         "signed preservation commit required")
    payload = record["payload"]
    need(type(payload) is dict and payload.get("kind") ==
         "kova_cosmo_controller_ledger_record" and
         type(payload.get("event")) is dict and
         payload["event"].get("kind") == "family_preserved" and
         payload["event"].get("artifact_sha256") == digest and
         payload["event"].get("preservation_receipt", {}).get("payload", {}).get("grant_id") ==
         grant_payload["grant_id"] and
         payload["event"]["preservation_receipt"]["payload"].get("lifecycle_id") ==
         grant_payload["lifecycle_id"] and
         type(record["signature"]) is str and re.fullmatch(r"[0-9a-f]{128}", record["signature"]),
         "preservation commit differs from the trained adapter")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(trust["public_key_hex"])).verify(
            bytes.fromhex(record["signature"]), authority.canonical(payload))
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise LedgerRejected("untrusted adapter preservation commit") from exc
    return {"status": "adapter_preserved_and_committed", "artifact_sha256": digest,
            "grant_id": grant_payload["grant_id"], "ledger_sequence": payload["sequence"],
            "destination_uri": payload["event"]["preservation_receipt"]["payload"]["destination_uri"]}


class AdapterPreserver:
    """Controller-side read-back and independent preservation attestation."""
    def __init__(self, *, ledger, artifact_container: str, signing_key: Ed25519PrivateKey,
                 token_for):
        need(type(artifact_container) is str and
             re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,61})[a-z0-9]", artifact_container),
             "private adapter container required")
        need(isinstance(signing_key, Ed25519PrivateKey) and
             signing_key.public_key().public_bytes_raw() == ledger.preservation_key,
             "independent preservation signing key mismatch")
        need(artifact_container != ledger.context["container"],
             "adapter and append ledger require distinct containers")
        self.ledger, self.container, self.key = ledger, artifact_container, signing_key
        self.io = AzureBlobIO(account=ledger.context["storage_account"], token_for=token_for,
                              maximum=MAX_BUNDLE, maximum_request=MAX_BUNDLE)

    def submit(self, request):
        need(type(request) is dict and set(request) == {
            "schema_version", "kind", "lifecycle_id", "grant_id", "artifact_sha256",
            "bundle_b64"} and request["schema_version"] == 1 and
            request["kind"] == "kova_cosmo_preserve_adapter_v1", "preservation request invalid")
        need(type(request["artifact_sha256"]) is str and
             re.fullmatch(r"[0-9a-f]{64}", request["artifact_sha256"]),
             "artifact digest invalid")
        try:
            bundle = base64.b64decode(request["bundle_b64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise LedgerRejected("adapter transfer corrupt") from exc
        _inspect_bundle(bundle)
        digest = hashlib.sha256(bundle).hexdigest()
        need(digest == request["artifact_sha256"], "adapter transfer digest mismatch")
        state, last_record = self.ledger.current_state_with_last_record()
        grants = [event for event in state["events"] if event.get("kind") == "training_grant"
                  and event.get("family") == "kova-cosmo"]
        need(len(grants) == 1, "one Cosmo grant required")
        grant = grants[0]
        envelope = grant["response_envelope"]
        issued = envelope["payload"]
        context = self.ledger.context
        need(request["lifecycle_id"] == state["lifecycle_id"] == issued["lifecycle_id"] and
             request["grant_id"] == issued["grant_id"] and
             context["lifecycle"]["pilot_resource_group_id"] == state["pilot_resource_group_id"],
             "artifact transfer belongs to another grant")
        now = self.ledger._time()
        group = context["storage_resource_group_id"]
        base = ("https://management.azure.com" + group +
                "/providers/Microsoft.Storage/storageAccounts/" + context["storage_account"] +
                "/blobServices/default/containers/" + self.container)
        container = self.io("GET", base + "?api-version=2023-05-01", {})
        policy = self.io("GET", base +
                         "/immutabilityPolicies/default?api-version=2023-05-01", {})
        need(container.status == policy.status == 200, "adapter storage policy missing")
        properties = parse_json(container.body)["properties"]
        locked = parse_json(policy.body)["properties"]
        need(properties.get("publicAccess", "None") == "None" and
             properties.get("hasImmutabilityPolicy") is True and
             properties.get("hasLegalHold", False) is False and
             properties.get("immutableStorageWithVersioning", {}).get("enabled", False) is False and
             locked.get("state") == "Locked" and
             type(locked.get("immutabilityPeriodSinceCreationInDays")) is int and
             locked["immutabilityPeriodSinceCreationInDays"] == context["retention_days"],
             "protected bounded adapter storage required")
        url = ("https://" + context["storage_account"] + ".blob.core.windows.net/" +
               self.container + "/cosmo-" + issued["grant_id"] + ".zip")
        metadata = {"x-ms-meta-artifact-sha256": digest,
                    "x-ms-meta-grant-id": issued["grant_id"],
                    "x-ms-meta-lifecycle-sha256": hashlib.sha256(
                        state["lifecycle_id"].encode("utf-8")).hexdigest(),
                    "x-ms-meta-ledger-sha256": hashlib.sha256(
                        state["ledger_id"].encode("utf-8")).hexdigest(),
                    "x-ms-meta-source-commit": issued["source_commit"]}
        preserved = [event for event in state["events"] if event.get("kind") == "family_preserved"]
        need(len(preserved) <= 1 and (not preserved or
             last_record["payload"]["event"] ==
             {key: value for key, value in preserved[0].items() if key != "sequence"}),
             "preservation state cannot be recovered")
        need(bool(preserved) or now < authority.timestamp(issued["watchdog_cleanup_trigger_utc"]),
             "adapter upload after watchdog trigger")
        if preserved:
            existing = self.io("GET", url, {})
            need(existing.status == 200 and existing.headers.get("etag"),
                 "committed adapter missing")
            etag = existing.headers["etag"]
        else:
            created = self.io("PUT", url, {"If-None-Match": "*", "x-ms-blob-type": "BlockBlob",
                                           "Content-Type": "application/zip", **metadata}, bundle)
            need(created.status in (201, 409, 412),
                 "protected adapter creation uncertain; do not overwrite")
            if created.status == 201:
                etag = created.headers.get("etag")
            else:
                existing = self.io("GET", url, {})
                need(existing.status == 200, "existing adapter unavailable")
                etag = existing.headers.get("etag")
            need(etag, "protected adapter ETag required")
        observed = self.io("GET", url, {"If-Match": etag})
        need(observed.status == 200 and observed.headers.get("etag") == etag and
             observed.headers.get("x-ms-blob-type") == "BlockBlob" and
             all(observed.headers.get(key) == value for key, value in metadata.items()) and
             observed.body == bundle and hashlib.sha256(observed.body).hexdigest() == digest,
             "independent adapter read-back differs")
        if preserved:
            receipt = preserved[0]["preservation_receipt"]["payload"]
            need(preserved[0]["artifact_sha256"] == digest and
                 receipt["destination_uri"] == url and
                 receipt["immutable_version"] == "etag:" + etag and
                 receipt["grant_id"] == issued["grant_id"] and
                 receipt["lifecycle_id"] == state["lifecycle_id"] and
                 receipt["ledger_id"] == state["ledger_id"] and
                 receipt["source_commit"] == issued["source_commit"],
                 "committed preservation differs from this adapter")
            return last_record
        receipt = {"schema_version": 1, "family": "kova-cosmo",
                   "grant_sequence": grant["sequence"], "artifact_sha256": digest,
                   "verified_sha256": digest, "destination_uri": url,
                   "immutable_version": "etag:" + etag,
                   "protected_destination": True, "outside_pilot_group": True,
                   "verification_succeeded": True,
                   "verified_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "lifecycle_id": state["lifecycle_id"], "ledger_id": state["ledger_id"],
                   "source_commit": issued["source_commit"], "grant_id": issued["grant_id"],
                   "grant_envelope_sha256": hashlib.sha256(authority.canonical(envelope)).hexdigest()}
        event = {"kind": "family_preserved", "family": "kova-cosmo",
                 "artifact_sha256": digest,
                 "preservation_receipt": {"payload": receipt,
                     "signature_ed25519_hex": self.key.sign(authority.canonical(receipt)).hex()}}
        return self.ledger.append(event, expected_sequence=state["sequence"])
