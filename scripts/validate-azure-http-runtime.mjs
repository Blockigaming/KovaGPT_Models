import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const selection = ["origin", "destination_address", "resource", "identity_client_id", "http_hop_timeout_seconds", "tenant_id", "required_api_role"];
const safety = ["network_execution_authorized", "credential_acquisition_authorized", "paid_execution_authorized", "deployment_authorized", "production_routing_authorized"];
const verification = {
  cpu_protocol_verified: true,
  loopback_tls_verified: true,
  azure_network_verified: false,
  azure_identity_verified: false,
  target_authorization_verified: false,
  address_ownership_and_freshness_verified: false,
  model_runtime_verified: false,
};
const contract = {
  tls_hostname_verification: true,
  minimum_tls_version: "1.2",
  numeric_destination_required: true,
  dns_resolution_in_request_path: false,
  environment_proxies_allowed: false,
  redirects_allowed: false,
  automatic_retries_allowed: false,
  deadline_covers_connect_tls_headers_and_body: true,
  cancellation_interrupts_blocked_io: true,
  close_aborts_socket: true,
  identity_header_sent_to_inference: false,
  credential_chain_fallback_allowed: false,
  token_cache_enabled: false,
  token_resource_server_controlled: true,
};

export function validateAzureHTTPRuntime(value) {
  assert.equal(value?.schema_version, 1);
  assert.equal(value.status, "source_only_live_blocked");
  assert.deepEqual(Object.keys(value).sort(), ["schema_version", "status", "implementation", "selection", "verification", "safety", "transport_contract"].sort());
  assert.deepEqual(value.implementation, {
    transport: "worker/bounded_http.py",
    binding: "worker/azure_http_runtime.py",
    authentication: "container_apps_managed_identity_rest",
    standard_library_only: true,
    identity_api_version: "2019-08-01",
  });
  assert.deepEqual(value.selection, Object.fromEntries(selection.map((key) => [key, null])));
  assert.deepEqual(value.safety, Object.fromEntries(safety.map((key) => [key, false])));
  assert.deepEqual(value.verification, verification);
  assert.deepEqual(value.transport_contract, contract);
  return { status: "http_runtime_source_verified_live_blocked", network_calls_made: 0 };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const config = JSON.parse(readFileSync(new URL("../config/azure-http-runtime.v1.json", import.meta.url), "utf8"));
  console.log(JSON.stringify(validateAzureHTTPRuntime(config)));
}
