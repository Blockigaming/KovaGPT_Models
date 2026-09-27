import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { validateApplicationBridge } from "../scripts/validate-application-bridge.mjs";

const load = () => JSON.parse(readFileSync(new URL("../config/application-bridge.v1.json", import.meta.url), "utf8"));

test("application bridge CLI verifies exact pinned app source without provider calls", () => {
  assert.equal(validateApplicationBridge(load()).provider_calls, 0);
  const result = spawnSync(process.execPath, [fileURLToPath(new URL("../scripts/validate-application-bridge.mjs", import.meta.url))], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
});

test("bridge cannot imply live routing or rewrite legacy history", () => {
  for (const key of Object.keys(load().safety)) {
    for (const replacement of [undefined, true, null, "false"]) {
      const value = load();
      if (replacement === undefined) delete value.safety[key];
      else value.safety[key] = replacement;
      assert.throws(() => validateApplicationBridge(value), key);
    }
  }
  for (const key of ["legacy_auto_alias_changed", "cached_history_rewritten"]) {
    const value = load(); value.legacy[key] = true;
    assert.throws(() => validateApplicationBridge(value), key);
  }
});

test("source revision and exact tier entitlements cannot drift silently", () => {
  for (const key of ["commit", "git_blob_sha", "path", "fixture"]) {
    const value = load(); value.application_source[key] = "unverified";
    assert.throws(() => validateApplicationBridge(value), key);
  }
  const value = load(); value.application_chat_modes.plus.push("max");
  assert.throws(() => validateApplicationBridge(value));
});

test("client permission overrides and guessed thinking mappings fail closed", () => {
  for (const key of ["client_provider_overrides_allowed", "client_permission_overrides_allowed"]) {
    const value = load(); value.rules[key] = true;
    assert.throws(() => validateApplicationBridge(value), key);
  }
  const value = load(); value.legacy.thinking = "high";
  assert.throws(() => validateApplicationBridge(value));
});
