"""Independent controller grant issuer; Azure and HTTP adapters are separate.

The guest supplies a request, never the controller's quote, preflight, keys,
ledger context, or verifier. A separately configured control-host verifier
must validate the Entra signature and read live Azure identity/network/watchdog
state. cosmo_controller_http.build_application wires the production Azure reader.
"""
from datetime import timedelta
import hashlib
from pathlib import Path
import re
import uuid

from training import cosmo_lifecycle_authority as authority
from training import cosmo_qlora_launch as launch
from training import cosmo_qlora_grant as client
from training.cosmo_controller_ledger import LedgerRejected, digest, need, parse_json


class GrantIssuer:
    def __init__(self, *, ledger, quote: Path, runtime_evidence: Path,
                 verify_live_request, root: Path = launch.ROOT):
        self.ledger, self.quote, self.runtime_evidence = ledger, quote, runtime_evidence
        self.root, self.verify_live_request = root, verify_live_request
        need(callable(verify_live_request), "independent Azure request verifier required")
        trust = authority.load_trust_policy(root)
        need(trust["status"] == "authority_pinned" and
             ledger.public_key.public_bytes_raw().hex() == trust["public_key_hex"],
             "controller signing key must match the independently pinned authority")
        self.audience = trust["azure_managed_identity_token_audience"]

    def issue(self, request):
        """Return the signed response only after its exact bytes are durable.

        verify_live_request is trusted deployment code, not a guest-supplied
        callback. It must fetch/validate current Entra keys and Azure resources
        independently. The returned observation below is never accepted from
        an HTTP body. cosmo_controller_azure supplies the production read adapter.
        """
        request = parse_json(authority.canonical(request))
        need(type(request) is dict and set(request) == {
            "schema_version", "kind", "source_commit", "subscription_id",
            "quote_sha256", "network_evidence_sha256", "lifecycle_id",
            "preflight_ledger_sequence", "azure_instance",
            "azure_instance_identity_token", "azure_instance_identity_token_audience",
            "azure_instance_identity_token_expires_at_utc", "allocation_deadline_utc",
            "watchdog_cleanup_trigger_utc", "training_runs_limit", "all_in_ceiling_usd",
            "request_nonce", "requested_at_utc"}, "training request shape mismatch")
        need(type(request["schema_version"]) is int and request["schema_version"] == 1 and
             request["kind"] == "kova_cosmo_qlora_training_grant_request" and
             type(request["training_runs_limit"]) is int and request["training_runs_limit"] == 1 and
             request["all_in_ceiling_usd"] == "3.3000" and
             type(request["request_nonce"]) is str and
             re.fullmatch(r"[0-9a-f]{64}", request["request_nonce"]), "invalid grant bounds or nonce")
        context = self.ledger.context
        lifecycle = context["lifecycle"]
        commit = launch.clean_source_commit(self.root)
        subscription = lifecycle["pilot_resource_group_id"].split("/")[2].lower()
        need(commit == context["source_commit"] == request["source_commit"] and
             request["subscription_id"] == subscription and
             request["lifecycle_id"] == lifecycle["lifecycle_id"], "request source/lifecycle mismatch")
        now = self.ledger._time()
        requested = authority.timestamp(request["requested_at_utc"])
        need(requested <= now <= requested + client.MAX_GRANT_RESPONSE_DELAY,
             "stale or future grant request")
        admission = launch.assess_signed_quote(self.quote, source_commit=commit,
            subscription_id=subscription, now=now, root=self.root)
        preflight = client.read_runtime_preflight(self.runtime_evidence,
            quote_sha256=admission["quote_sha256"], source_commit=commit,
            subscription_id=subscription, deadline_utc=admission["allocation_deadline_utc"],
            cleanup_trigger_utc=admission["watchdog_cleanup_trigger_utc"], now=now, root=self.root)
        network_digest = digest(preflight["azure_network"])
        bound = {"quote_sha256": admission["quote_sha256"],
            "network_evidence_sha256": network_digest,
            "lifecycle_id": preflight["lifecycle_id"],
            "preflight_ledger_sequence": preflight["preflight_ledger_sequence"],
            "azure_instance": preflight["azure_instance"],
            "allocation_deadline_utc": admission["allocation_deadline_utc"],
            "watchdog_cleanup_trigger_utc": admission["watchdog_cleanup_trigger_utc"]}
        need(all(request[k] == v for k, v in bound.items()) and
             type(request["preflight_ledger_sequence"]) is int and
             context["grant_deadline_utc"] == admission["watchdog_cleanup_trigger_utc"],
             "request changed the independently signed runtime or deadline")
        instance = authority.validate_azure_instance(request["azure_instance"])
        need(instance["resource_id"].casefold().startswith(
            lifecycle["pilot_resource_group_id"].casefold() + "/providers/microsoft.compute/virtualmachines/"),
            "request VM is outside the exclusive pilot group")
        token = request["azure_instance_identity_token"]
        need(type(token) is str and authority.JWT.fullmatch(token) and
             request["azure_instance_identity_token_audience"] == self.audience,
             "invalid managed-identity token or audience")
        token_digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        # This callable is held only by the independent controller. Raw tokens
        # never enter the persistent ledger, error text, or response envelope.
        observed = self.verify_live_request(token=token, audience=self.audience,
            instance=instance, network=preflight["azure_network"],
            cleanup_trigger_utc=admission["watchdog_cleanup_trigger_utc"])
        now = self.ledger._time()
        need(type(observed) is dict and set(observed) == {
            "token_sha256", "audience", "azure_instance", "token_expires_at_utc",
            "network_evidence_sha256", "watchdog_cleanup_trigger_utc", "observed_at_utc",
            "entra_signature_verified", "azure_control_plane_verified", "watchdog_healthy",
            "cleanup_scope_verified"}, "independent request verification missing")
        need(observed["token_sha256"] == token_digest and observed["audience"] == self.audience and
             observed["azure_instance"] == instance and
             observed["network_evidence_sha256"] == network_digest and
             observed["watchdog_cleanup_trigger_utc"] == admission["watchdog_cleanup_trigger_utc"] and
             observed["token_expires_at_utc"] == request["azure_instance_identity_token_expires_at_utc"] and
             all(observed[k] is True for k in ("entra_signature_verified", "azure_control_plane_verified",
                 "watchdog_healthy", "cleanup_scope_verified")), "identity or live Azure control mismatch")
        verified_at = authority.timestamp(observed["observed_at_utc"])
        trigger = authority.timestamp(admission["watchdog_cleanup_trigger_utc"])
        token_expiry = authority.timestamp(observed["token_expires_at_utc"])
        need(requested <= verified_at <= now <= requested + client.MAX_GRANT_RESPONSE_DELAY and
             now + client.MIN_GRANT_LEAD <= trigger <= token_expiry,
             "verification expired or insufficient training time")
        # Recheck source inputs after the external verification call, which may
        # be slow. Expiry must be tested using the post-verification clock.
        refreshed = launch.assess_signed_quote(self.quote, source_commit=commit,
            subscription_id=subscription, now=now, root=self.root)
        need(refreshed == admission, "account quote changed during independent verification")
        need(launch.clean_source_commit(self.root) == commit,
             "source changed during independent verification")
        need(now < authority.timestamp(preflight["observed_at_utc"]) + timedelta(minutes=5),
             "runtime evidence expired during independent verification")
        payload = {"schema_version": 1, "kind": "kova_cosmo_qlora_training_grant",
            "issuer": authority.ISSUER, "source_commit": commit, "subscription_id": subscription,
            **bound, "all_in_ceiling_usd": "3.3000", "request_nonce": request["request_nonce"],
            "ledger_sequence": request["preflight_ledger_sequence"] + 1,
            "ledger_commit_id": str(uuid.uuid4()), "ledger_append_only": True,
            "ledger_status": "grant_committed_before_response", "grant_id": str(uuid.uuid4()),
            "azure_identity_token_sha256": token_digest,
            "issued_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "expires_at_utc": admission["watchdog_cleanup_trigger_utc"],
            "training_runs_consumed": 1, "all_in_reserved_usd": admission["worst_case_all_in_usd"],
            "watchdog_healthy": True, "cleanup_scope_verified": True, "deployment_authorized": False}
        envelope = {"payload": payload,
            "signature": self.ledger.signing_key.sign(authority.canonical(payload)).hex()}
        committed = self.ledger.append({"kind": "training_grant", "family": "kova-cosmo",
            "quote_sha256": admission["quote_sha256"], "request_sha256": digest(request),
            "response_envelope": envelope}, expected_sequence=request["preflight_ledger_sequence"])
        after = self.ledger._time()
        need(committed["payload"]["sequence"] == payload["ledger_sequence"] and
             now <= after <= requested + client.MAX_GRANT_RESPONSE_DELAY and
             after + client.MIN_GRANT_LEAD <= trigger,
             "grant committed but response window expired; the run remains consumed")
        return envelope
