import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

export function validateReceiverAuth(value) {
  assert.deepEqual(Object.keys(value ?? {}).sort(), [
    "schema_version", "status", "implementation", "token_contract", "selection", "safety", "boundaries",
  ].sort());
  assert.equal(value.schema_version, 1);
  assert.equal(value.status, "source_only_disabled");
  assert.equal(value.implementation, "worker/receiver_auth.py");
  assert.equal(value.token_contract, "single_tenant_entra_app_only_RS256");
  assert.deepEqual(value.selection, {
    tenant_id: null, audience: null, token_version: null,
    allowed_services: [], required_roles: [], keyset_sha256: null,
    keyset_valid_until_epoch: null, max_token_lifetime_seconds: null, clock_skew_seconds: null,
  });
  assert.deepEqual(value.safety, {
    enabled: false, public_listener_installed: false, application_startup_wired: false,
    credential_acquisition_authorized: false, model_execution_authorized: false,
    gpu_execution_authorized: false, deployment_authorized: false, production_routing_authorized: false,
  });
  assert.deepEqual(value.boundaries, {
    service_identity_is_user_identity: false, token_header_key_urls_allowed: false,
    request_driven_key_refresh_allowed: false, unsigned_tokens_allowed: false,
    delegated_tokens_allowed: false, client_and_object_ids_must_match_one_pair: true,
    all_required_roles_must_match: true, key_snapshot_expiry_fails_closed: true,
    live_identity_verified: false,
  });
  return { status: "receiver_auth_source_valid_live_binding_disabled", phase_b_ready: false };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const config = JSON.parse(readFileSync(new URL("../config/receiver-auth.v1.json", import.meta.url), "utf8"));
  console.log(JSON.stringify(validateReceiverAuth(config)));
}
