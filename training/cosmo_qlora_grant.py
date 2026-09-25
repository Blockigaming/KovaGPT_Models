"""Single-use, instance-bound Cosmo training grant from an independent ledger.

The remote signer must atomically commit the only training run before replying.
This client cannot create an authority or make the guest its own cost watchdog.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import os
from pathlib import Path
import re
import secrets
import time

from training import cosmo_lifecycle_authority as authority
from training import cosmo_qlora_launch as launch


class GrantRejected(ValueError):
    pass


MIN_GRANT_LEAD = timedelta(minutes=45)
MAX_GRANT_RESPONSE_DELAY = timedelta(seconds=60)
MAX_GRANT_RECOVERY_DELAY = timedelta(seconds=165)
MAX_CLOCK_SKEW = timedelta(seconds=2)


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise GrantRejected(reason)


def verified_network(payload: dict, instance: dict, *, current: datetime) -> dict:
    """Validate the authority's fresh postdeployment Azure control-plane read."""
    need(type(payload) is dict and set(payload) == {
        "vm_nic_ids", "vm_nic_id", "nic_public_ip_id", "subnet_id",
        "subnet_default_outbound_access", "nat_gateway_id",
        "nat_gateway_public_ip_id", "nat_gateway_sku", "nat_public_ip_sku",
        "network_security_group_id", "inbound_deny_rule",
        "allowed_outbound_tcp_ports", "other_outbound_denied",
        "verified_from_azure_control_plane", "observed_at_utc",
    }, "signed live network evidence missing")
    match = re.fullmatch(
        r"(/subscriptions/[0-9a-f-]{36}/resourceGroups/[a-zA-Z0-9_.()\-]{1,90})"
        r"/providers/Microsoft\.Compute/virtualMachines/kova-t4-([a-z0-9]{3,16})",
        instance["resource_id"], re.IGNORECASE)
    need(match is not None, "VM ID is not a scoped pilot instance")
    prefix = match.group(1) + "/providers/Microsoft.Network/"
    suffix = match.group(2)
    nic = prefix + "networkInterfaces/kova-t4-nic-" + suffix
    expected = {
        "vm_nic_ids": [nic], "vm_nic_id": nic, "nic_public_ip_id": None,
        "subnet_id": prefix + "virtualNetworks/kova-t4-vnet-" + suffix + "/subnets/pilot",
        "subnet_default_outbound_access": False,
        "nat_gateway_id": prefix + "natGateways/kova-t4-egress-nat-" + suffix,
        "nat_gateway_public_ip_id": prefix + "publicIPAddresses/kova-t4-egress-ip-" + suffix,
        "nat_gateway_sku": "Standard", "nat_public_ip_sku": "Standard",
        "network_security_group_id": prefix + "networkSecurityGroups/kova-t4-egress-nsg-" + suffix,
        "inbound_deny_rule": "deny-all-inbound", "allowed_outbound_tcp_ports": [80, 443],
        "other_outbound_denied": True, "verified_from_azure_control_plane": True,
    }
    need(all(payload[key] == value for key, value in expected.items()),
         "live VM NIC, NAT or NSG differs from approved network")
    observed = authority.timestamp(payload["observed_at_utc"])
    need(observed <= current < observed + timedelta(minutes=5),
         "live network observation is stale")
    return payload


def read_runtime_preflight(path: Path, *, quote_sha256: str, source_commit: str,
                           subscription_id: str, deadline_utc: str,
                           cleanup_trigger_utc: str,
                           now: datetime | None = None,
                           root: Path = launch.ROOT) -> dict:
    """Accept lifecycle, ledger and instance values only from the authority."""
    try:
        payload, _ = authority.read_signed_record(
            path, expected_kind="kova_cosmo_qlora_runtime_preflight",
            repository_root=root)
        need(set(payload) == {
            "schema_version", "kind", "issuer", "quote_sha256", "source_commit",
            "subscription_id", "lifecycle_id", "preflight_ledger_sequence",
            "azure_instance", "allocation_deadline_utc",
            "watchdog_cleanup_trigger_utc", "observed_at_utc",
            "watchdog_healthy", "cleanup_scope_verified", "azure_network",
        } and payload["schema_version"] == 1,
             "signed runtime evidence shape mismatch")
        need(payload["quote_sha256"] == quote_sha256 and
             payload["source_commit"] == source_commit and
             payload["subscription_id"] == subscription_id and
             payload["allocation_deadline_utc"] == deadline_utc and
             payload["watchdog_cleanup_trigger_utc"] == cleanup_trigger_utc and
             payload["watchdog_healthy"] is True and
             payload["cleanup_scope_verified"] is True,
             "runtime preflight changed quote or safety controls")
        instance = authority.validate_azure_instance(payload["azure_instance"])
        need(instance["resource_id"].casefold().startswith(
             "/subscriptions/" + subscription_id + "/"),
             "runtime preflight VM subscription mismatch")
        observed = authority.timestamp(payload["observed_at_utc"])
        current = now or datetime.now(timezone.utc)
        need(current.tzinfo is not None and
             observed <= current.astimezone(timezone.utc) <
             observed + timedelta(minutes=5) and
             current.astimezone(timezone.utc) <
             authority.timestamp(cleanup_trigger_utc),
             "runtime preflight is stale or cleanup has begun")
        need(type(payload["preflight_ledger_sequence"]) is int and
             0 < payload["preflight_ledger_sequence"] < 2**63 and
             type(payload["lifecycle_id"]) is str and
             0 < len(payload["lifecycle_id"]) <= 256,
             "missing authoritative lifecycle")
        verified_network(payload["azure_network"], instance,
                         current=current.astimezone(timezone.utc))
        return payload
    except (authority.AuthorityError, KeyError, TypeError, AttributeError,
            OSError, ValueError) as exc:
        raise GrantRejected("independent signed runtime preflight rejected") from exc


def acquire_training_grant(*, quote: Path, source_commit: str,
                           subscription_id: str, lifecycle_id: str,
                           preflight_ledger_sequence: int, azure_instance: dict,
                           runtime_evidence: Path,
                           now: datetime | None = None,
                           response_now: datetime | None = None,
                           root: Path = launch.ROOT,
                           transport=None, instance_transport=None) -> dict:
    """Bind a signed quote to an atomic one-run ledger commit and real Azure VM."""
    admission = launch.assess_signed_quote(
        quote, source_commit=source_commit, subscription_id=subscription_id,
        now=now, root=root)
    current = now or datetime.now(timezone.utc)
    need(current.tzinfo is not None, "UTC grant clock required")
    current = current.astimezone(timezone.utc)
    need(type(lifecycle_id) is str and 0 < len(lifecycle_id) <= 256 and
         type(preflight_ledger_sequence) is int and
         0 < preflight_ledger_sequence < 2**63,
         "lifecycle and ledger sequence required")
    try:
        preflight = read_runtime_preflight(
            runtime_evidence, quote_sha256=admission["quote_sha256"],
            source_commit=source_commit, subscription_id=subscription_id,
            deadline_utc=admission["allocation_deadline_utc"], now=current,
            cleanup_trigger_utc=admission["watchdog_cleanup_trigger_utc"],
            root=root)
        need(preflight["lifecycle_id"] == lifecycle_id and
             preflight["preflight_ledger_sequence"] == preflight_ledger_sequence and
             preflight["azure_instance"] == azure_instance,
             "grant is not bound to the signed live network preflight")
        network_sha256 = hashlib.sha256(authority.canonical(
            preflight["azure_network"])).hexdigest()
        trust = authority.load_trust_policy(root)
        need(trust["status"] == "authority_pinned", "independent authority is absent")
        instance = authority.validate_azure_instance(azure_instance)
        need(instance["resource_id"].casefold().startswith(
             "/subscriptions/" + subscription_id + "/"),
             "VM is in a different subscription")
        deadline = authority.timestamp(admission["allocation_deadline_utc"])
        cleanup_trigger = authority.timestamp(admission["watchdog_cleanup_trigger_utc"])
        need(current + MIN_GRANT_LEAD + MAX_GRANT_RECOVERY_DELAY <=
             cleanup_trigger <= deadline <=
             current + timedelta(seconds=launch.MAX_ALLOCATION_SECONDS),
             "insufficient time for one training grant before watchdog cleanup")
        compute = (instance_transport or authority._imds_transport)(
            authority.AZURE_COMPUTE_IMDS_URL)
        image = compute["storageProfile"]["imageReference"]
        need(compute["resourceId"].casefold() == instance["resource_id"].casefold()
             and compute["vmId"].casefold() == instance["vm_id"].casefold()
             and compute["location"].casefold() == "eastus"
             and compute["vmSize"] == launch.SKU
             and image["publisher"] == "Canonical"
             and image["offer"] == "ubuntu-24_04-lts"
             and image["sku"] == "server"
             and image.get("exactVersion") == "24.04.202609040",
             "executing VM or exact Ubuntu image does not match the quote")
        identity_token, token_sha256, token_expiry = authority._executing_azure_identity(
            instance, trust["azure_managed_identity_token_audience"], current,
            transport=instance_transport)
        nonce = secrets.token_hex(32)
        request = {
            "schema_version": 1, "kind": "kova_cosmo_qlora_training_grant_request",
            "source_commit": source_commit, "subscription_id": subscription_id,
            "quote_sha256": admission["quote_sha256"],
            "network_evidence_sha256": network_sha256,
            "lifecycle_id": lifecycle_id,
            "preflight_ledger_sequence": preflight_ledger_sequence,
            "azure_instance": instance,
            "azure_instance_identity_token": identity_token,
            "azure_instance_identity_token_audience": trust[
                "azure_managed_identity_token_audience"],
            "azure_instance_identity_token_expires_at_utc": token_expiry,
            "allocation_deadline_utc": admission["allocation_deadline_utc"],
            "watchdog_cleanup_trigger_utc": admission["watchdog_cleanup_trigger_utc"],
            "training_runs_limit": 1, "all_in_ceiling_usd": "3.3000",
            "request_nonce": nonce,
            "requested_at_utc": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        token_path = Path(os.environ.get(authority.TOKEN_ENV, ""))
        token = authority._load_bearer_token(token_path, repository_root=root)
        def send(timeout):
            if transport is None:
                return authority._https_transport(
                    trust["endpoint"], token, request, timeout=timeout)
            return transport(trust["endpoint"], token, request)

        recovered = False
        started = time.monotonic()
        try:
            response = send(MAX_GRANT_RESPONSE_DELAY.total_seconds())
        except authority.AuthorityError:
            # Retry the exact nonce and signed request only. The controller
            # can replay the original envelope but cannot grant a second run.
            recovered = True
            limit = started + MAX_GRANT_RECOVERY_DELAY.total_seconds()
            while True:
                remaining = limit - time.monotonic()
                if remaining <= 0:
                    raise
                try:
                    response = send(min(10, remaining))
                    break
                except authority.AuthorityError:
                    if time.monotonic() + 5 >= limit:
                        raise
                    time.sleep(5)
        payload, digest = authority.verify_envelope(
            response, expected_kind="kova_cosmo_qlora_training_grant", root=root)
        received = response_now or (now if now is not None else datetime.now(timezone.utc))
        need(received.tzinfo is not None and
             current <= received.astimezone(timezone.utc) <=
             current + (MAX_GRANT_RECOVERY_DELAY if recovered else MAX_GRANT_RESPONSE_DELAY),
             "grant response clock or elapsed time invalid")
        received = received.astimezone(timezone.utc)
        need(received + MIN_GRANT_LEAD <= cleanup_trigger,
             "insufficient training time after committed grant")
        need(set(payload) == {
            "schema_version", "kind", "issuer", "source_commit", "subscription_id",
            "quote_sha256", "lifecycle_id", "preflight_ledger_sequence",
            "network_evidence_sha256",
            "ledger_sequence", "ledger_commit_id", "ledger_append_only",
            "ledger_status", "grant_id", "azure_instance",
            "azure_identity_token_sha256", "request_nonce",
            "issued_at_utc", "expires_at_utc", "allocation_deadline_utc",
            "watchdog_cleanup_trigger_utc",
            "training_runs_consumed", "all_in_reserved_usd",
            "all_in_ceiling_usd", "watchdog_healthy", "cleanup_scope_verified",
            "deployment_authorized",
        } and payload["schema_version"] == 1,
             "training grant shape mismatch")
        need(all(payload[k] == request[k] for k in (
            "source_commit", "subscription_id", "quote_sha256", "lifecycle_id",
            "preflight_ledger_sequence", "network_evidence_sha256",
            "azure_instance", "request_nonce",
            "allocation_deadline_utc", "watchdog_cleanup_trigger_utc",
            "all_in_ceiling_usd")),
             "training grant request binding mismatch")
        need(payload["ledger_append_only"] is True and
             payload["ledger_status"] == "grant_committed_before_response" and
             type(payload["ledger_sequence"]) is int and
             preflight_ledger_sequence < payload["ledger_sequence"] < 2**63 and
             type(payload["ledger_commit_id"]) is str and payload["ledger_commit_id"] and
             type(payload["grant_id"]) is str and payload["grant_id"] and
             payload["training_runs_consumed"] == 1 and
             type(payload["training_runs_consumed"]) is int and
             payload["azure_identity_token_sha256"] == token_sha256 and
             payload["watchdog_healthy"] is True and
             payload["cleanup_scope_verified"] is True and
             payload["deployment_authorized"] is False,
             "atomic one-run grant or independent cleanup control absent")
        reserved = authority.money(payload["all_in_reserved_usd"])
        need(Decimal(admission["worst_case_all_in_usd"]) <= reserved <=
             launch.CEILING, "training grant does not cover the worst-case bound")
        issued = authority.timestamp(payload["issued_at_utc"])
        expires = authority.timestamp(payload["expires_at_utc"])
        need(current - timedelta(minutes=5) <= issued <=
             received + MAX_CLOCK_SKEW and
             received < expires <= cleanup_trigger and
             authority.timestamp(token_expiry) >= expires,
             "training grant is stale or exceeds the deadline")
        return {"status": "one_training_run_committed", "grant_id": payload["grant_id"],
                "lifecycle_id": payload["lifecycle_id"],
                "ledger_sequence": payload["ledger_sequence"], "grant_sha256": digest,
                "allocation_deadline_utc": payload["allocation_deadline_utc"],
                "watchdog_cleanup_trigger_utc": payload["watchdog_cleanup_trigger_utc"],
                "azure_instance": instance, "training_runs_consumed": 1}
    except (authority.AuthorityError, KeyError, TypeError, AttributeError, OSError,
            ValueError) as exc:
        raise GrantRejected("independent single-use training grant rejected") from exc
