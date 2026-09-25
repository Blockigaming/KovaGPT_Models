import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { MODE_IDS_BY_TIER } from "../tests/fixtures/app-mode-entitlements.mjs";

export function validateApplicationBridge(value) {
  assert.deepEqual(Object.keys(value ?? {}).sort(), [
    "schema_version", "status", "selection_schema", "application_source", "application_chat_modes",
    "legacy", "rules", "safety", "unresolved", "superseded_by", "must_not_drive_current_routing",
  ].sort());
  assert.equal(value.schema_version, 1);
  assert.equal(value.status, "superseded_non_authoritative_history");
  assert.equal(value.selection_schema, "kova-models.v1");
  assert.equal(value.superseded_by, "config/current-product-policy.v3.json");
  assert.equal(value.must_not_drive_current_routing, true);
  assert.deepEqual(value.application_source, {
    repository: "Blockigaming/KovaGPT",
    commit: "b964d1bc9fcd81fcfc70e97a22cbf4a516605c94",
    path: "src/lib/mode-entitlements.mjs",
    git_blob_sha: "e9ce5faeeb855e824b83855a58f8924183eea30b",
    fixture: "tests/fixtures/app-mode-entitlements.mjs",
  });
  const fixture = readFileSync(new URL("../tests/fixtures/app-mode-entitlements.mjs", import.meta.url));
  const blobHash = createHash("sha1").update(`blob ${fixture.length}\0`).update(fixture).digest("hex");
  assert.equal(blobHash, value.application_source.git_blob_sha, "fixture must match exact app source bytes");
  assert.deepEqual(value.application_chat_modes, MODE_IDS_BY_TIER);
  assert.deepEqual(value.legacy, {
    unversioned_selection: "existing_application_handler",
    thinking: "unmapped_existing_application_handler",
    legacy_auto_alias_changed: false,
    cached_history_rewritten: false,
  });
  assert.deepEqual(value.rules, {
    server_entitlement_required: true,
    explicit_server_auto_enablement_required: true,
    auto_final_route_must_be_allowed: true,
    work_route_allowlist_required: true,
    extra_high_route_id: "extra-high",
    extra_high_application_id: "extra_high",
    client_provider_overrides_allowed: false,
    client_permission_overrides_allowed: false,
  });
  assert.deepEqual(value.safety, {
    application_startup_wired: false,
    production_routing_authorized: false,
    paid_execution_authorized: false,
    legacy_history_migration_authorized: false,
  });
  assert.deepEqual(value.unresolved, [
    "free_thinking_custom_model_mapping", "complete_work_plan_matrix",
    "application_startup_provider_cutover", "older_application_identity_prompt_reconciliation",
  ]);
  return { status: "historical_application_bridge_checked_current_policy_authoritative", provider_calls: 0 };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const config = JSON.parse(readFileSync(new URL("../config/application-bridge.v1.json", import.meta.url), "utf8"));
  console.log(JSON.stringify(validateApplicationBridge(config)));
}
