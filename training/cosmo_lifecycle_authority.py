"""Verify the independent, append-only authority for every paid pilot phase."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT = Path(__file__).resolve().parents[1]
TRUST_PATH = "config/kova-cosmo-lifecycle-trust.v1.json"
TOKEN_ENV = "KOVA_COSMO_LIFECYCLE_TOKEN_FILE"
PILOT_ID = "kova-cosmo-qwen3-0.6b-eastus-t4-v1"
ISSUER = "kova-cosmo-lifecycle-authority-v1"
PHASES = ("runtime_probe", "training", "evaluation")
PHASE_RESERVED_SECONDS = {
    "runtime_probe": 600,
    "training": 2400,
    "evaluation": 600,
}
MAX_BYTES = 1024 * 1024
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX128 = re.compile(r"[0-9a-f]{128}")
UTC_TIMESTAMP = re.compile(
    r"20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)
MONEY = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{4}")
TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]{32,16384}")
JWT = re.compile(
    r"[A-Za-z0-9_-]{16,8192}\.[A-Za-z0-9_-]{16,8192}\."
    r"[A-Za-z0-9_-]{16,8192}"
)
AZURE_UUID = re.compile(
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
AZURE_VM_RESOURCE_ID = re.compile(
    r"/subscriptions/[0-9A-Fa-f-]{36}/resourceGroups/"
    r"[A-Za-z0-9_.()-]{1,90}/providers/Microsoft\.Compute/"
    r"virtualMachines/[A-Za-z0-9_.()-]{1,64}"
)
AZURE_COMPUTE_IMDS_URL = (
    "http://169.254.169.254/metadata/instance/compute"
    "?api-version=2021-02-01"
)
AZURE_IDENTITY_IMDS_URL = (
    "http://169.254.169.254/metadata/identity/oauth2/token"
    "?api-version=2018-02-01&resource="
)
MAX_IMDS_BYTES = 65536


class AuthorityError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise AuthorityError("kova cosmo lifecycle authority rejected")


def canonical(value: object) -> bytes:
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
        need(0 < len(raw) <= MAX_BYTES)
        return raw
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def timestamp(value: object) -> datetime:
    need(type(value) is str and UTC_TIMESTAMP.fullmatch(value) is not None)
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def money(value: object) -> Decimal:
    need(type(value) is str and MONEY.fullmatch(value) is not None)
    try:
        return Decimal(value)
    except InvalidOperation:
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def load_trust_policy(root: Path = ROOT) -> dict:
    try:
        value = json.loads(
            (root / TRUST_PATH).read_text(encoding="utf-8")
        )
        need(type(value) is dict and list(value) == [
            "schema_version", "status", "issuer", "algorithm", "endpoint",
            "public_key_hex", "public_key_sha256",
            "bearer_token_file_environment_variable",
            "azure_managed_identity_token_audience",
            "append_only_remote_ledger_required",
            "independent_azure_reader_required",
            "runner_ledger_mutation_allowed",
            "checked_in_private_key_allowed",
        ])
        need(value["schema_version"] == 1)
        need(value["issuer"] == ISSUER)
        need(value["algorithm"] == "ed25519")
        need(value["bearer_token_file_environment_variable"] == TOKEN_ENV)
        need(value["append_only_remote_ledger_required"] is True)
        need(value["independent_azure_reader_required"] is True)
        need(value["runner_ledger_mutation_allowed"] is False)
        need(value["checked_in_private_key_allowed"] is False)
        if value["status"] == "authority_provisioning_required":
            need(value["endpoint"] is None)
            need(value["public_key_hex"] is None)
            need(value["public_key_sha256"] is None)
            need(value["azure_managed_identity_token_audience"] is None)
        else:
            need(value["status"] == "authority_pinned")
            endpoint = value["endpoint"]
            public = value["public_key_hex"]
            fingerprint = value["public_key_sha256"]
            audience = value["azure_managed_identity_token_audience"]
            need(type(endpoint) is str and len(endpoint) <= 2048)
            parsed = urllib.parse.urlsplit(endpoint)
            need(parsed.scheme == "https" and parsed.hostname is not None)
            need(parsed.username is None and parsed.password is None)
            need(parsed.path == "/v1/pilot/grants")
            need(parsed.query == "" and parsed.fragment == "")
            need(type(public) is str and HEX64.fullmatch(public) is not None)
            need(type(fingerprint) is str and
                 HEX64.fullmatch(fingerprint) is not None)
            need(hashlib.sha256(bytes.fromhex(public)).hexdigest() ==
                 fingerprint)
            need(type(audience) is str and 8 <= len(audience) <= 2048)
            need(audience.startswith("api://") or
                 audience.startswith("https://"))
        return value
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, RecursionError, json.JSONDecodeError):
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def verify_envelope(value: object, *, expected_kind: str,
                    root: Path = ROOT) -> tuple[dict, str]:
    try:
        trust = load_trust_policy(root)
        need(trust["status"] == "authority_pinned")
        need(type(value) is dict and list(value) == ["payload", "signature"])
        payload = value["payload"]
        signature = value["signature"]
        need(type(payload) is dict)
        need(payload.get("kind") == expected_kind)
        need(payload.get("issuer") == trust["issuer"])
        need(type(signature) is str and
             HEX128.fullmatch(signature) is not None)
        raw = canonical(payload)
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(trust["public_key_hex"])
        ).verify(bytes.fromhex(signature), raw)
        return payload, hashlib.sha256(canonical(value)).hexdigest()
    except (InvalidSignature, OSError, ValueError, TypeError, KeyError,
            AttributeError, UnicodeError, RecursionError):
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def read_signed_record(path: Path, *, expected_kind: str,
                       repository_root: Path = ROOT) -> tuple[dict, str]:
    try:
        need(path.is_absolute() and path.is_file() and not path.is_symlink())
        resolved = path.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
        need(repository != resolved and repository not in resolved.parents)
        with resolved.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        need(0 < len(raw) <= MAX_BYTES)
        value = json.loads(raw.decode("utf-8"))
        payload, _ = verify_envelope(
            value, expected_kind=expected_kind, root=repository_root
        )
        return payload, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, KeyError, UnicodeError,
            RecursionError, json.JSONDecodeError):
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def _load_bearer_token(path: Path, *, repository_root: Path) -> str:
    try:
        need(path.is_absolute() and path.is_file() and not path.is_symlink())
        resolved = path.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
        need(repository != resolved and repository not in resolved.parents)
        need(resolved.stat().st_mode & 0o077 == 0)
        with resolved.open("r", encoding="ascii") as stream:
            token = stream.read(16385).strip()
        need(TOKEN.fullmatch(token) is not None)
        return token
    except (OSError, ValueError, TypeError, UnicodeError):
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def validate_azure_instance(value: object) -> dict:
    """Validate the exact VM identity observed by the independent preflight."""
    need(type(value) is dict and set(value) == {
        "resource_id", "vm_id", "system_assigned_identity_principal_id",
    })
    need(type(value["resource_id"]) is str and
         AZURE_VM_RESOURCE_ID.fullmatch(value["resource_id"]) is not None)
    need(type(value["vm_id"]) is str and
         AZURE_UUID.fullmatch(value["vm_id"]) is not None)
    need(type(value["system_assigned_identity_principal_id"]) is str and
         AZURE_UUID.fullmatch(
             value["system_assigned_identity_principal_id"]
         ) is not None)
    return dict(value)


def _imds_transport(url: str) -> dict:
    """Query only Azure's link-local IMDS with proxies and redirects disabled."""
    try:
        parsed = urllib.parse.urlsplit(url)
        need(parsed.scheme == "http" and parsed.hostname == "169.254.169.254")
        need(parsed.port is None and parsed.username is None and
             parsed.password is None and parsed.fragment == "")
        request = urllib.request.Request(
            url, method="GET", headers={
                "Metadata": "true",
                "Accept": "application/json",
            },
        )

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, request, file_pointer, code, message,
                                 headers, new_url):
                raise AuthorityError(
                    "kova cosmo lifecycle authority rejected"
                )

        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect
        )
        with opener.open(request, timeout=2) as response:
            need(response.status == 200 and response.geturl() == url)
            body = response.read(MAX_IMDS_BYTES + 1)
        need(0 < len(body) <= MAX_IMDS_BYTES)
        value = json.loads(body.decode("utf-8"))
        need(type(value) is dict)
        return value
    except (OSError, ValueError, TypeError, UnicodeError,
            urllib.error.URLError, json.JSONDecodeError):
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def _executing_azure_identity(
    expected_instance: dict, audience: str, current: datetime,
    *, transport=None,
) -> tuple[str, str, str]:
    """Bind the caller to exact IMDS metadata and its managed identity token."""
    try:
        get = transport or _imds_transport
        compute = get(AZURE_COMPUTE_IMDS_URL)
        need(type(compute) is dict)
        observed_resource_id = compute.get("resourceId")
        observed_vm_id = compute.get("vmId")
        need(type(observed_resource_id) is str and
             AZURE_VM_RESOURCE_ID.fullmatch(observed_resource_id) is not None)
        need(type(observed_vm_id) is str and
             AZURE_UUID.fullmatch(observed_vm_id) is not None)
        need(observed_resource_id.casefold() ==
             expected_instance["resource_id"].casefold())
        need(observed_vm_id.casefold() == expected_instance["vm_id"].casefold())

        identity_url = AZURE_IDENTITY_IMDS_URL + urllib.parse.quote(
            audience, safe=""
        )
        identity = get(identity_url)
        need(type(identity) is dict)
        token = identity.get("access_token")
        need(type(token) is str and JWT.fullmatch(token) is not None)
        need(identity.get("token_type") == "Bearer")
        need(identity.get("resource") == audience)
        expires_on = identity.get("expires_on")
        need(type(expires_on) is str and expires_on.isascii() and
             expires_on.isdigit() and 1 <= len(expires_on) <= 12)
        expiry = datetime.fromtimestamp(int(expires_on), timezone.utc)
        need(current < expiry <= current + timedelta(days=2))
        expiry_text = expiry.strftime("%Y-%m-%dT%H:%M:%SZ")
        return (
            token,
            hashlib.sha256(token.encode("ascii")).hexdigest(),
            expiry_text,
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, OverflowError):
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def _https_transport(endpoint: str, token: str, request_value: dict) -> dict:
    raw = canonical(request_value)
    request = urllib.request.Request(
        endpoint, data=raw, method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, request, file_pointer, code, message,
                                 headers, new_url):
                raise AuthorityError(
                    "kova cosmo lifecycle authority rejected"
                )

        opener = urllib.request.build_opener(NoRedirect)
        with opener.open(request, timeout=15) as response:
            need(response.status == 200)
            need(response.geturl() == endpoint)
            body = response.read(MAX_BYTES + 1)
        need(0 < len(body) <= MAX_BYTES)
        value = json.loads(body.decode("utf-8"))
        need(type(value) is dict)
        return value
    except (OSError, ValueError, TypeError, UnicodeError,
            urllib.error.URLError, json.JSONDecodeError):
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def acquire_phase_grant(
    *, phase: str, source_commit: str, runtime_evidence_sha256: str,
    lifecycle_id: str, preflight_ledger_sequence: int,
    context: dict, azure_instance: dict, runtime_deadline_utc: str,
    root: Path = ROOT,
    now: datetime | None = None, transport=None, instance_transport=None,
) -> dict:
    """Reserve one paid phase in the remote append-only budget ledger."""
    try:
        trust = load_trust_policy(root)
        need(trust["status"] == "authority_pinned")
        need(phase in PHASES)
        need(type(source_commit) is str and
             HEX40.fullmatch(source_commit) is not None)
        need(type(runtime_evidence_sha256) is str and
             HEX64.fullmatch(runtime_evidence_sha256) is not None)
        need(type(lifecycle_id) is str and 0 < len(lifecycle_id) <= 256)
        need(type(preflight_ledger_sequence) is int and
             0 < preflight_ledger_sequence < 2**63)
        need(type(context) is dict and 0 < len(context) <= 32)
        azure_instance = validate_azure_instance(azure_instance)
        current = now or datetime.now(timezone.utc)
        need(current.tzinfo is not None and
             current.utcoffset() == timedelta(0))
        deadline = timestamp(runtime_deadline_utc)
        need(current < deadline <= current + timedelta(
            seconds=PHASE_RESERVED_SECONDS[phase]
        ))
        nonce = secrets.token_hex(32)
        context_sha256 = hashlib.sha256(canonical(context)).hexdigest()
        (
            azure_identity_token,
            azure_identity_token_sha256,
            azure_identity_token_expires_at_utc,
        ) = _executing_azure_identity(
            azure_instance,
            trust["azure_managed_identity_token_audience"],
            current,
            transport=instance_transport,
        )
        request_value = {
            "schema_version": 1,
            "kind": "kova_cosmo_paid_phase_grant_request",
            "pilot_id": PILOT_ID,
            "lifecycle_id": lifecycle_id,
            "preflight_ledger_sequence": preflight_ledger_sequence,
            "phase": phase,
            "source_commit": source_commit,
            "runtime_evidence_sha256": runtime_evidence_sha256,
            "context_sha256": context_sha256,
            "azure_instance": azure_instance,
            "azure_instance_identity_token": azure_identity_token,
            "azure_instance_identity_token_audience": trust[
                "azure_managed_identity_token_audience"
            ],
            "azure_instance_identity_token_expires_at_utc":
                azure_identity_token_expires_at_utc,
            "runtime_deadline_utc": runtime_deadline_utc,
            "requested_at_utc": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "request_nonce": nonce,
        }
        token_path = Path(os.environ.get(TOKEN_ENV, ""))
        token = _load_bearer_token(token_path, repository_root=root)
        response = (transport or _https_transport)(
            trust["endpoint"], token, request_value
        )
        payload, envelope_sha256 = verify_envelope(
            response, expected_kind="kova_cosmo_paid_phase_grant", root=root
        )
        need(list(payload) == [
            "schema_version", "kind", "issuer", "pilot_id", "lifecycle_id",
            "preflight_ledger_sequence", "ledger_sequence",
            "ledger_commit_id", "ledger_append_only",
            "ledger_status", "grant_id", "phase", "source_commit",
            "runtime_evidence_sha256", "context_sha256", "request_nonce",
            "runtime_deadline_utc",
            "azure_instance", "azure_instance_identity",
            "issued_at_utc", "expires_at_utc", "grant_reserved_seconds",
            "grant_reserved_cost_usd", "phase_grants_committed",
            "training_runs_consumed", "aggregate_reserved_seconds",
            "aggregate_reserved_cost_usd", "approved_budget_usd",
            "deployment_authorized",
        ])
        need(payload["schema_version"] == 1)
        need(payload["pilot_id"] == PILOT_ID)
        need(payload["lifecycle_id"] == lifecycle_id)
        need(payload["preflight_ledger_sequence"] ==
             preflight_ledger_sequence)
        need(type(payload["ledger_sequence"]) is int and
             preflight_ledger_sequence < payload["ledger_sequence"] < 2**63)
        for field in ("ledger_commit_id", "grant_id"):
            need(type(payload[field]) is str and 0 < len(payload[field]) <= 256)
        need(payload["ledger_append_only"] is True)
        need(payload["ledger_status"] == "grant_committed_before_response")
        need(payload["phase"] == phase)
        need(payload["source_commit"] == source_commit)
        need(payload["runtime_evidence_sha256"] == runtime_evidence_sha256)
        need(payload["context_sha256"] == context_sha256)
        need(payload["request_nonce"] == nonce)
        need(payload["runtime_deadline_utc"] == runtime_deadline_utc)
        need(payload["azure_instance"] == azure_instance)
        identity = payload["azure_instance_identity"]
        need(type(identity) is dict and list(identity) == [
            "verification_method", "token_sha256", "token_audience",
            "verified_at_utc", "token_expires_at_utc", "verified",
        ])
        need(identity["verification_method"] ==
             "microsoft_entra_system_assigned_managed_identity_token")
        need(identity["token_sha256"] == azure_identity_token_sha256)
        need(identity["token_audience"] ==
             trust["azure_managed_identity_token_audience"])
        need(identity["verified"] is True)
        issued = timestamp(payload["issued_at_utc"])
        expires = timestamp(payload["expires_at_utc"])
        need(expires == timestamp(payload["runtime_deadline_utc"]))
        identity_verified = timestamp(identity["verified_at_utc"])
        token_expires = timestamp(identity["token_expires_at_utc"])
        need(identity["token_expires_at_utc"] ==
             azure_identity_token_expires_at_utc)
        clock_skew = timedelta(minutes=5)
        need(current - clock_skew <= issued <= current + clock_skew)
        need(issued < expires == deadline)
        need(current - clock_skew <= identity_verified <= current + clock_skew)
        need(issued - clock_skew <= identity_verified <= issued)
        need(token_expires >= expires)
        reserved_seconds = payload["grant_reserved_seconds"]
        aggregate_seconds = payload["aggregate_reserved_seconds"]
        need(type(reserved_seconds) is int and
             reserved_seconds == PHASE_RESERVED_SECONDS[phase])
        need(type(aggregate_seconds) is int and
             reserved_seconds <= aggregate_seconds <= 3600)
        reserved_cost = money(payload["grant_reserved_cost_usd"])
        aggregate_cost = money(payload["aggregate_reserved_cost_usd"])
        need(Decimal("0") < reserved_cost <= aggregate_cost <=
             Decimal("0.5260"))
        hourly_compute = Decimal("0.5260")
        need(reserved_cost >= (
            hourly_compute * Decimal(reserved_seconds) / Decimal(3600)
        ))
        need(aggregate_cost >= (
            hourly_compute * Decimal(aggregate_seconds) / Decimal(3600)
        ))
        need(money(payload["approved_budget_usd"]) == Decimal("2.0000"))
        counts = payload["phase_grants_committed"]
        need(type(counts) is dict and list(counts) == list(PHASES))
        need(all(type(counts[item]) is int and 0 <= counts[item] <= 1
                 for item in PHASES))
        need(counts[phase] == 1)
        need(sum(counts.values()) <= 3)
        need(aggregate_seconds == sum(
            PHASE_RESERVED_SECONDS[item] * counts[item] for item in PHASES
        ))
        need(type(payload["training_runs_consumed"]) is int and
             payload["training_runs_consumed"] == counts["training"])
        need(payload["deployment_authorized"] is False)
        return {
            "status": "paid_phase_reserved_in_append_only_ledger",
            "pilot_id": PILOT_ID,
            "lifecycle_id": payload["lifecycle_id"],
            "phase": phase,
            "grant_id": payload["grant_id"],
            "ledger_sequence": payload["ledger_sequence"],
            "ledger_commit_id": payload["ledger_commit_id"],
            "phase_grant_sha256": envelope_sha256,
            "phase_grant_envelope": response,
            "phase_grant_context": context,
            "azure_instance": azure_instance,
            "azure_instance_identity_verified": True,
            "aggregate_reserved_seconds": aggregate_seconds,
            "aggregate_reserved_cost_usd":
                payload["aggregate_reserved_cost_usd"],
            "training_runs_consumed": payload["training_runs_consumed"],
            "deployment_authorized": False,
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, RecursionError, InvalidSignature):
        raise AuthorityError(
            "kova cosmo lifecycle authority rejected"
        ) from None


def dry_run(root: Path = ROOT) -> dict:
    trust = load_trust_policy(root)
    return {
        "status": "blocked" if trust["status"] != "authority_pinned"
                  else "authority_pinned_paid_phases_still_source_gated",
        "authority_pinned": trust["status"] == "authority_pinned",
        "append_only_remote_ledger_required": True,
        "independent_azure_reader_required": True,
        "exact_azure_vm_identity_grant_binding_required": True,
        "direct_proxy_disabled_azure_imds_required": True,
        "paid_phase_grants": list(PHASES),
        "paid_phase_reserved_seconds": dict(PHASE_RESERVED_SECONDS),
        "maximum_grants_per_phase": 1,
        "maximum_training_runs": 1,
        "approved_budget_usd": "2.0000",
        "provider_calls_made": 0,
        "deployment_authorized": False,
        "phase_b_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        print(json.dumps(dry_run(), sort_keys=True))
        return 0
    except AuthorityError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
