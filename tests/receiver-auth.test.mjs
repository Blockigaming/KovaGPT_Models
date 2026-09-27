import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { validateReceiverAuth } from "../scripts/validate-receiver-auth.mjs";

const load = () => JSON.parse(readFileSync(new URL("../config/receiver-auth.v1.json", import.meta.url), "utf8"));

test("receiver source preflight stays disabled and makes no network requests", () => {
  assert.equal(validateReceiverAuth(load()).phase_b_ready, false);
  const result = spawnSync(process.execPath, [fileURLToPath(new URL("../scripts/validate-receiver-auth.mjs", import.meta.url))], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).phase_b_ready, false);
});

test("every receiver execution and deployment guard rejects removal or promotion", () => {
  for (const field of Object.keys(load().safety)) {
    for (const changed of [undefined, true, null, "false"]) {
      const value = load();
      if (changed === undefined) delete value.safety[field];
      else value.safety[field] = changed;
      assert.throws(() => validateReceiverAuth(value), field);
    }
  }
});

test("no real tenant, audience, identity, role, signing key or lifetime is selected", () => {
  for (const field of Object.keys(load().selection)) {
    const value = load();
    value.selection[field] = "guessed-selection";
    assert.throws(() => validateReceiverAuth(value), field);
  }
});

test("receiver claims cannot become user identity or bypass role and key rules", () => {
  for (const [field, current] of Object.entries(load().boundaries)) {
    const value = load(); value.boundaries[field] = !current;
    assert.throws(() => validateReceiverAuth(value), field);
  }
});

test("receiver verification dependency lock has exact versions and hashes only", () => {
  const source = readFileSync(new URL("../requirements/receiver-auth-py312-linux.lock", import.meta.url), "utf8");
  const records = source.split("\n").filter((line) => line && !line.startsWith("#"));
  assert.equal(records.length, 4);
  assert.ok(records.every((line) => /^[A-Za-z]+==\d+\.\d+(?:\.\d+)? --hash=sha256:[a-f0-9]{64}$/.test(line)));
  assert.ok(records.some((line) => line.startsWith("PyJWT==2.14.0 ")));
  assert.ok(records.some((line) => line.startsWith("cryptography==50.0.1 ")));
});
