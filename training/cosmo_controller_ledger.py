"""Independent Cosmo controller ledger backed by a protected Azure append blob.

This is the persistence component, not a deployed authority or a paid release.
Only the trusted controller may possess its signing key and storage credential.
The caller must independently obtain price, watchdog and VM evidence before
submitting events. Importing this module and its default CLI make no requests.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import hashlib
import json
import re
import urllib.error
import urllib.request
import uuid

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from training import cosmo_lifecycle_authority as authority
from training import three_family_contract as contract

MAX_BYTES = 1024 * 1024
MAX_RECORD_BYTES = 65536
MAX_EVENTS = 64
KIND = "kova_cosmo_controller_ledger_record"


class LedgerRejected(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise LedgerRejected(reason)


def digest(value):
    return hashlib.sha256(authority.canonical(value)).hexdigest()


def utc_now():
    return datetime.now(timezone.utc)


def parse_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              LedgerRejected("non-finite JSON value")))
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise LedgerRejected("invalid ledger JSON") from exc


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict
    body: bytes = b""


class AzureBlobIO:
    """Bounded HTTPS transport; credentials come only from the control host.

    token_for receives one of the two fixed Azure resource audiences below.
    No automatic retries: a timed-out append may already have committed.
    """
    def __init__(self, *, account: str, token_for):
        need(bool(re.fullmatch(r"[a-z0-9]{3,24}", account)), "invalid storage account")
        self.host = account + ".blob.core.windows.net"
        self.token_for = token_for

    def __call__(self, method, url, headers, body=b""):
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        need(parts.scheme == "https" and parts.hostname in (
            "management.azure.com", self.host) and parts.port in (None, 443)
            and parts.username is None and parts.password is None and not parts.fragment,
            "untrusted ledger endpoint")
        need(method in ("GET", "PUT"), "unsupported ledger operation")
        need(parts.hostname != "management.azure.com" or method == "GET",
             "ledger may only read the management plane")
        resource = ("https://management.azure.com/" if parts.hostname ==
                    "management.azure.com" else "https://storage.azure.com/")
        token = self.token_for(resource)
        need(type(token) is str and re.fullmatch(r"[A-Za-z0-9._~+/=-]{32,16384}", token),
             "invalid control-host credential")
        need(len(body) <= MAX_RECORD_BYTES, "ledger request too large")

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, hdrs, newurl):
                return None

        request = urllib.request.Request(url, method=method,
            data=body if method == "PUT" else None, headers={
                **headers, "Authorization": "Bearer " + token,
                "x-ms-version": "2023-11-03",
                "x-ms-date": format_datetime(utc_now(), usegmt=True),
                "Content-Type": "application/octet-stream",
            })
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            try:
                stream = opener.open(request, timeout=10)
            except urllib.error.HTTPError as exc:
                stream = exc
            with stream:
                raw = stream.read(MAX_BYTES + 1)
                need(len(raw) <= MAX_BYTES, "ledger response too large")
                return Response(stream.code, {k.lower(): v for k, v in stream.headers.items()}, raw)
        except (OSError, urllib.error.URLError) as exc:
            raise LedgerRejected("ledger transport failed; outcome may be committed") from exc


class ControllerLedger:
    """Lease, replay, validate, sign, append conditionally, then read back.

    Context and verifier keys are provisioned on the independent controller,
    never taken from a guest request. Missing history is never recreated by
    append(). initialize() is a distinct, create-only operation.
    """
    def __init__(self, *, context: dict, signing_key: Ed25519PrivateKey,
                 transport, preservation_public_key: bytes, cleanup_public_key: bytes,
                 clock=utc_now):
        self.context = deepcopy(context)
        need(set(context) == {"source_commit", "lifecycle", "storage_resource_group_id",
            "storage_account", "container", "blob", "retention_days",
            "not_before_utc", "grant_deadline_utc"}, "controller context shape mismatch")
        need(type(context["source_commit"]) is str and
             re.fullmatch(r"[0-9a-f]{40}", context["source_commit"]), "source commit required")
        lifecycle = context["lifecycle"]
        contract._trusted_lifecycle(lifecycle, lifecycle)
        need(lifecycle["admission_scope"] == "cosmo-only", "only Cosmo is admitted")
        group = context["storage_resource_group_id"]
        pilot, watchdog = (lifecycle[k] for k in (
            "pilot_resource_group_id", "watchdog_resource_group_id"))
        need(type(group) is str and re.fullmatch(
            r"/subscriptions/[0-9a-f-]{36}/resourceGroups/[A-Za-z0-9_.()-]{1,90}", group)
             and group.casefold() not in (pilot.casefold(), watchdog.casefold())
             and group.split("/")[2].casefold() == pilot.split("/")[2].casefold(),
             "ledger must be outside both cleanup groups in the same subscription")
        for field, pattern in (("storage_account", r"[a-z0-9]{3,24}"),
                               ("container", r"[a-z0-9](?:[a-z0-9-]{1,61})[a-z0-9]"),
                               ("blob", r"[a-zA-Z0-9_-]{1,100}\.jsonl")):
            need(type(context[field]) is str and re.fullmatch(pattern, context[field]),
                 "invalid ledger " + field)
        need("--" not in context["container"], "invalid container name")
        need(type(context["retention_days"]) is int and 1 <= context["retention_days"] <= 90,
             "explicit bounded ledger retention required")
        self.starts = authority.timestamp(context["not_before_utc"])
        self.deadline = authority.timestamp(context["grant_deadline_utc"])
        need(0 < (self.deadline - self.starts).total_seconds() <= 7200,
             "controller grant window exceeds two hours")
        need(isinstance(signing_key, Ed25519PrivateKey), "controller signing key required")
        for key in (preservation_public_key, cleanup_public_key):
            need(type(key) is bytes and len(key) == 32, "independent verifier key required")
            need(key != signing_key.public_key().public_bytes_raw(),
                 "ledger signer cannot verify its own cleanup or preservation")
        self.signing_key = signing_key
        self.public_key = signing_key.public_key()
        self.preservation_key, self.cleanup_key = preservation_public_key, cleanup_public_key
        self.transport, self.clock = transport, clock
        self.context_sha256 = digest(context)
        account, container = context["storage_account"], context["container"]
        self.blob_url = f"https://{account}.blob.core.windows.net/{container}/{context['blob']}"
        self.container_url = ("https://management.azure.com" + group +
            f"/providers/Microsoft.Storage/storageAccounts/{account}/blobServices/default/containers/{container}")
        self.initial = {"sequence": 0, "terminal": False, "family_order": [],
                        "events": [], **deepcopy(lifecycle)}

    def _time(self):
        now = self.clock()
        need(isinstance(now, datetime) and now.tzinfo is not None, "UTC clock required")
        return now.astimezone(timezone.utc)

    def _policy(self):
        container = self.transport("GET", self.container_url + "?api-version=2023-05-01", {})
        policy = self.transport("GET", self.container_url +
            "/immutabilityPolicies/default?api-version=2023-05-01", {})
        need(container.status == policy.status == 200, "cannot verify ledger protection")
        props = parse_json(container.body).get("properties", {})
        lock = parse_json(policy.body).get("properties", {})
        need(props.get("publicAccess", "None") == "None" and
             props.get("hasImmutabilityPolicy") is True and
             props.get("hasLegalHold", False) is False and
             props.get("immutableStorageWithVersioning", {}).get("enabled", False) is False,
             "private container-level immutable storage required")
        need(lock.get("state") == "Locked" and
             type(lock.get("immutabilityPeriodSinceCreationInDays")) is int and
             lock["immutabilityPeriodSinceCreationInDays"] == self.context["retention_days"] and
             lock.get("allowProtectedAppendWrites") is True and
             lock.get("allowProtectedAppendWritesAll", False) is False,
             "locked retention and protected append writes required")

    def _lease(self):
        proposed = str(uuid.uuid4())
        result = self.transport("PUT", self.blob_url + "?comp=lease", {
            "x-ms-lease-action": "acquire", "x-ms-lease-duration": "60",
            "x-ms-proposed-lease-id": proposed})
        need(result.status == 201 and result.headers.get("x-ms-lease-id") == proposed,
             "exclusive controller lease unavailable")
        return proposed

    def _release(self, lease):
        result = self.transport("PUT", self.blob_url + "?comp=lease", {
            "x-ms-lease-action": "release", "x-ms-lease-id": lease})
        need(result.status == 200, "lease release failed; never retry the event automatically")

    def _read(self, lease):
        result = self.transport("GET", self.blob_url, {"x-ms-lease-id": lease})
        need(result.status == 200 and result.headers.get("x-ms-blob-type") == "AppendBlob"
             and result.headers.get("etag") and len(result.body) <= MAX_BYTES,
             "missing or invalid append ledger")
        return result

    def _transition(self, state, event, now):
        if event.get("kind") == "training_grant":
            need(set(event) == {"kind", "family", "quote_sha256", "request_sha256",
                                "response_envelope"}, "issuer-generated grant envelope required")
            events = state.get("events", [])
            need(events and event.get("quote_sha256") == events[-1].get("quote_sha256"),
                 "grant quote differs from the committed cost admission")
            envelope = event["response_envelope"]
            need(type(envelope) is dict and set(envelope) == {"payload", "signature"},
                 "signed grant response required")
            payload = envelope["payload"]
            need(type(payload) is dict and set(payload) == {
                "schema_version", "kind", "issuer", "source_commit", "subscription_id",
                "quote_sha256", "lifecycle_id", "preflight_ledger_sequence",
                "network_evidence_sha256", "ledger_sequence", "ledger_commit_id",
                "ledger_append_only", "ledger_status", "grant_id", "azure_instance",
                "azure_identity_token_sha256", "request_nonce", "issued_at_utc",
                "expires_at_utc", "allocation_deadline_utc", "watchdog_cleanup_trigger_utc",
                "training_runs_consumed", "all_in_reserved_usd", "all_in_ceiling_usd",
                "watchdog_healthy", "cleanup_scope_verified", "deployment_authorized"},
                "grant response shape mismatch")
            try:
                need(type(envelope["signature"]) is str and
                     re.fullmatch(r"[0-9a-f]{128}", envelope["signature"]),
                     "invalid grant signature")
                self.public_key.verify(bytes.fromhex(envelope["signature"]),
                                       authority.canonical(payload))
            except (InvalidSignature, ValueError, TypeError) as exc:
                raise LedgerRejected("untrusted grant response") from exc
            group = self.context["lifecycle"]["pilot_resource_group_id"]
            instance = authority.validate_azure_instance(payload["azure_instance"])
            need(payload["source_commit"] == self.context["source_commit"] and
                 payload["subscription_id"] == group.split("/")[2].lower() and
                 payload["lifecycle_id"] == self.context["lifecycle"]["lifecycle_id"] and
                 instance["resource_id"].casefold().startswith(group.casefold() + "/providers/microsoft.compute/virtualmachines/") and
                 payload["quote_sha256"] == event["quote_sha256"] and
                 payload["issuer"] == authority.ISSUER and
                 payload["kind"] == "kova_cosmo_qlora_training_grant" and
                 type(payload["schema_version"]) is int and payload["schema_version"] == 1,
                 "grant context mismatch")
            for key, expected in (("preflight_ledger_sequence", state["sequence"]),
                                  ("ledger_sequence", state["sequence"] + 1),
                                  ("training_runs_consumed", 1)):
                need(type(payload[key]) is int and payload[key] == expected,
                     "grant sequence or run limit mismatch")
            for key in ("quote_sha256", "network_evidence_sha256",
                        "azure_identity_token_sha256", "request_nonce"):
                need(type(payload[key]) is str and re.fullmatch(r"[0-9a-f]{64}", payload[key]),
                     "grant digest or nonce missing")
            need(type(event["request_sha256"]) is str and
                 re.fullmatch(r"[0-9a-f]{64}", event["request_sha256"]), "request digest missing")
            for key in ("ledger_commit_id", "grant_id"):
                need(type(payload[key]) is str and str(uuid.UUID(payload[key])) == payload[key],
                     "grant identifier missing")
            need(payload["ledger_append_only"] is True and
                 payload["ledger_status"] == "grant_committed_before_response" and
                 payload["watchdog_healthy"] is True and payload["cleanup_scope_verified"] is True and
                 payload["deployment_authorized"] is False and
                 payload["all_in_ceiling_usd"] == "3.3000" and
                 0 < authority.money(payload["all_in_reserved_usd"]) <= authority.money("3.3000"),
                 "grant controls or reservation invalid")
            issued = authority.timestamp(payload["issued_at_utc"])
            expires = authority.timestamp(payload["expires_at_utc"])
            allocation = authority.timestamp(payload["allocation_deadline_utc"])
            need(now - timedelta(seconds=60) <= issued <= now and
                 now + timedelta(minutes=45) <= expires == self.deadline and
                 payload["watchdog_cleanup_trigger_utc"] == self.context["grant_deadline_utc"] and
                 expires + timedelta(minutes=15) <= allocation <= self.starts + timedelta(hours=2),
                 "grant response deadline mismatch")
        return contract.append_ledger_event(state, event,
            expected_sequence=state["sequence"], now=now,
            trusted_lifecycle=self.context["lifecycle"],
            preservation_public_key=self.preservation_key, cleanup_public_key=self.cleanup_key)

    def _record(self, event, state, previous, now):
        payload = {"schema_version": 1, "kind": KIND,
            "context_sha256": self.context_sha256, "sequence": state["sequence"],
            "previous_sha256": previous, "committed_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": event, "state_sha256": digest(state)}
        record = {"payload": payload,
                  "signature": self.signing_key.sign(authority.canonical(payload)).hex()}
        raw = authority.canonical(record) + b"\n"
        need(len(raw) <= MAX_RECORD_BYTES, "ledger event too large")
        return record, raw

    def replay(self, raw):
        need(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES and raw.endswith(b"\n"),
             "missing or truncated ledger history")
        lines = raw.splitlines()
        need(1 <= len(lines) <= MAX_EVENTS + 1, "ledger record limit exceeded")
        state, previous, last_time = deepcopy(self.initial), None, self.starts
        for number, line in enumerate(lines):
            need(len(line) <= MAX_RECORD_BYTES, "ledger record too large")
            record = parse_json(line)
            need(type(record) is dict and set(record) == {"payload", "signature"},
                 "invalid signed ledger record")
            payload = record["payload"]
            need(type(payload) is dict and set(payload) == {"schema_version", "kind",
                "context_sha256", "sequence", "previous_sha256", "committed_at_utc",
                "event", "state_sha256"} and type(payload["schema_version"]) is int and
                payload["schema_version"] == 1 and
                payload["kind"] == KIND and payload["context_sha256"] == self.context_sha256 and
                type(payload["sequence"]) is int and payload["sequence"] == number and
                payload["previous_sha256"] == previous, "ledger chain or context mismatch")
            try:
                self.public_key.verify(bytes.fromhex(record["signature"]), authority.canonical(payload))
            except (InvalidSignature, TypeError, ValueError) as exc:
                raise LedgerRejected("untrusted ledger record") from exc
            when = authority.timestamp(payload["committed_at_utc"])
            need(last_time <= when <= self._time(), "invalid ledger chronology")
            if number == 0:
                need(payload["event"] is None and when < self.deadline, "invalid ledger genesis")
            else:
                need(type(payload["event"]) is dict, "invalid ledger event")
                if payload["event"].get("kind") in ("watchdog_health", "cost_admission", "training_grant"):
                    need(when < self.deadline, "admission after grant deadline")
                state = self._transition(state, payload["event"], when)
            need(payload["state_sha256"] == digest(state), "ledger replay mismatch")
            previous, last_time = digest(record), when
        return state, previous, last_time

    def _append_record(self, current, lease, event, state, previous, now):
        record, raw = self._record(event, state, previous, now)
        need(len(current.body) + len(raw) <= MAX_BYTES, "ledger size limit exceeded")
        admission = event is None or event.get("kind") in (
            "watchdog_health", "cost_admission", "training_grant")
        if admission:
            need(now <= self._time() < self.deadline, "grant deadline reached before commit")
        result = self.transport("PUT", self.blob_url + "?comp=appendblock", {
            "x-ms-lease-id": lease, "If-Match": current.headers["etag"],
            "x-ms-blob-condition-appendpos": str(len(current.body)),
            "x-ms-blob-condition-maxsize": str(MAX_BYTES)}, raw)
        need(result.status == 201 and result.headers.get("x-ms-blob-append-offset") ==
             str(len(current.body)), "append failed or uncertain; do not retry automatically")
        persisted = self._read(lease)
        need(persisted.body == current.body + raw, "ledger read-back failed after append")
        if admission:
            need(now <= self._time() < self.deadline,
                 "grant deadline reached after commit; reservation remains consumed")
            # A slow commit must not return a grant after either admission expires.
            for item in state["events"][-3:]:
                if item.get("kind") in ("watchdog_health", "cost_admission"):
                    contract._fresh_admission(item, now=self._time())
        return record

    def initialize(self):
        now = self._time()
        need(self.starts <= now < self.deadline, "initialization outside the approved window")
        self._policy()
        created = self.transport("PUT", self.blob_url,
            {"If-None-Match": "*", "x-ms-blob-type": "AppendBlob"})
        need(created.status == 201, "ledger already exists or creation failed")
        lease = self._lease()
        try:
            current = self._read(lease)
            need(current.body == b"", "ledger genesis already exists")
            return self._append_record(current, lease, None, self.initial, None, now)
        finally:
            self._release(lease)

    def append(self, event, *, expected_sequence):
        # Clone caller data before any I/O so concurrent request mutation cannot
        # change the event after its transition was validated.
        event = parse_json(authority.canonical(event))
        need(type(event) is dict and type(expected_sequence) is int and
             0 <= expected_sequence < MAX_EVENTS, "invalid append request")
        need(event.get("family") in (None, "kova-cosmo"), "only Cosmo events are allowed")
        self._policy()
        lease = self._lease()
        try:
            current = self._read(lease)
            state, previous, last_time = self.replay(current.body)
            now = self._time()
            need(state["sequence"] == expected_sequence and last_time <= now,
                 "stale sequence or clock")
            if event.get("kind") in ("watchdog_health", "cost_admission", "training_grant"):
                need(self.starts <= now < self.deadline, "admission after grant deadline")
            updated = self._transition(state, event, now)
            return self._append_record(current, lease, event, updated, previous, now)
        finally:
            self._release(lease)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("controller deployment and paid execution require a separate reviewed release")
    print(json.dumps({"status": "controller_ledger_source_only", "provider_calls_made": 0,
        "live_storage_verified": False, "authority_service_provisioned": False,
        "paid_actions_enabled": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
