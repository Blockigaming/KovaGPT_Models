import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { realizedMargin, requiredPrice } from "../scripts/price-floor.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
for (const script of ["validate-data.mjs", "validate-stack.mjs"]) {
  test(`${script} passes`, () => {
    const result = spawnSync(process.execPath, [join(root, "scripts", script)], { encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
  });
}

test("stack validation rejects a changed approved identity prompt", (t) => {
  const output = mkdtempSync(join(tmpdir(), "kova-identity-validation-"));
  t.after(() => rmSync(output, { recursive: true, force: true }));
  mkdirSync(join(output, "scripts"));
  mkdirSync(join(output, "prompts"));
  cpSync(join(root, "config"), join(output, "config"), { recursive: true });
  copyFileSync(join(root, "scripts/validate-stack.mjs"), join(output, "scripts/validate-stack.mjs"));
  writeFileSync(
    join(output, "prompts/kova-identity.v3.txt"),
    `${readFileSync(join(root, "prompts/kova-identity.v3.txt"), "utf8")}\n`,
  );
  const result = spawnSync(process.execPath, [join(output, "scripts/validate-stack.mjs")], {
    encoding: "utf8",
  });
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /truthful_kova_identity_required/u);
});

test("dataset compiler creates hashed isolated splits", () => {
  const output = mkdtempSync(join(tmpdir(), "kova-data-"));
  const result = spawnSync(
    process.execPath,
    [join(root, "scripts/compile-dataset.mjs"), `--output=${output}`],
    { cwd: root, encoding: "utf8" },
  );
  assert.equal(result.status, 0, result.stderr);
  const manifest = JSON.parse(readFileSync(join(output, "manifest.json"), "utf8"));
  assert.equal(manifest.source_records, 4);
  assert.equal(manifest.files.train.records, 3);
  assert.equal(manifest.files.validation.records, 1);
  assert.match(manifest.files.train.sha256, /^[a-f0-9]{64}$/u);
});

test("training defaults to dry-run and paid execution stays blocked", () => {
  const script = join(root, "scripts/training-command.mjs");
  const dryRun = spawnSync(process.execPath, [script], { cwd: root, encoding: "utf8" });
  assert.equal(dryRun.status, 0, dryRun.stderr);
  assert.equal(JSON.parse(dryRun.stdout).paid_resources_started, false);
  const execute = spawnSync(process.execPath, [script, "--execute"], {
    cwd: root,
    encoding: "utf8",
  });
  assert.equal(execute.status, 1);
  assert.match(execute.stderr, /paid training blocked/i);
});

test("Runpod plan scales to zero and stays blocked from paid execution", () => {
  const config = JSON.parse(readFileSync(join(root, "config/runpod-serverless.v1.json"), "utf8"));
  assert.equal(config.worker_type, "flex");
  assert.equal(config.active_workers, 0);
  assert.equal(config.billing.usage_model, "metered_pay_per_second");
  assert.equal(config.billing.flat_rate_plan, false);
  assert.equal(config.billing.auto_pay_enabled, false);
  assert.equal(config.billing.automatic_credit_reload_allowed, false);
  assert.equal(config.model_storage.runtime_model_download_allowed, false);
  assert.equal(config.model_storage.network_volume_enabled, false);
  assert.equal(config.safety.paid_execution_authorized, false);
  assert.equal(config.safety.deployment_authorized, false);
  assert.deepEqual(config.endpoints.map((endpoint) => endpoint.id), ["kova-core", "kova-ultra"]);
  assert.ok(config.endpoints.every((endpoint) =>
    endpoint.deployed === false &&
    endpoint.active_workers === 0 &&
    endpoint.worker_type === "flex" &&
    endpoint.flashboot_required === true &&
    endpoint.cached_model_required === true
  ));
});

test("model catalog keeps both unselected engines and all public profiles blocked", () => {
  const catalog = JSON.parse(readFileSync(join(root, "config/model-catalog.v1.json"), "utf8"));
  assert.deepEqual(catalog.physical_engines.map((engine) => engine.id), ["kova-core", "kova-ultra"]);
  for (const engine of catalog.physical_engines) {
    assert.equal(engine.upstream_model, null, engine.id);
    assert.equal(engine.deployment_ready, false, engine.id);
  }
  for (const profile of catalog.public_profiles) {
    assert.equal(profile.separate_foundation_weights, false, profile.id);
    assert.equal(profile.deployment_ready, false, profile.id);
  }
  assert.deepEqual(catalog.user_facing_hierarchy.map((mode) => mode.display_name), [
    "Kova Auto", "Kova Cosmo", "Kova Orion", "Kova Nova",
    "Nova Extra High", "Nova Max", "Kova Ultra",
  ]);
});

test("price floor targets a 42.6% gross margin before rounding", () => {
  const price = requiredPrice(0.574);
  assert.ok(Math.abs(price - 1) < 1e-12);
  assert.ok(Math.abs(realizedMargin(0.574, price) - 0.426) < 1e-12);
});

test("product target contains six chat modes and eighteen Work combinations", () => {
  const surface = JSON.parse(readFileSync(join(root, "config/product-surface.v1.json"), "utf8"));
  assert.deepEqual(surface.chat_modes.map((mode) => mode.display_name), [
    "Kova Cosmo", "Kova Orion", "Kova Nova",
    "Nova Extra High", "Nova Max", "Kova Ultra",
  ]);
  assert.equal(surface.work_families.length * surface.work_efforts.length, 18);
  assert.equal(surface.chat_modes.find((mode) => mode.id === "instant").activity_updates, false);
  assert.equal(surface.chat_modes.find((mode) => mode.id === "ultra").activity_updates, true);
  assert.ok(surface.chat_modes.slice(0, 5).every((mode) => mode.engine === "kova-core"));
  assert.equal(surface.chat_modes.at(-1).engine, "kova-ultra");
});

test("provider architecture keeps Cloudflare at the edge and both inference engines on Runpod", () => {
  const config = JSON.parse(readFileSync(join(root, "config/provider-architecture.v1.json"), "utf8"));
  assert.equal(config.application_plane.provider, "azure_container_apps");
  assert.equal(config.application_plane.delete_existing_azure_deployments, false);
  assert.equal(config.edge_plane.provider, "cloudflare");
  assert.equal(config.edge_plane.inference_allowed, false);
  assert.equal(config.inference_decision.provider, "runpod_serverless");
  const core = config.engines.find((engine) => engine.id === "kova-core");
  assert.equal(core.provider, "runpod_serverless");
  assert.equal(core.endpoint_name_reserved, "kova-core");
  assert.equal(core.endpoint_deployed, false);
  assert.equal(core.active_workers, 0);
  assert.equal(core.selected_model, null);
  const ultra = config.engines.find((engine) => engine.id === "kova-ultra");
  assert.equal(ultra.provider, "runpod_serverless");
  assert.equal(ultra.endpoint_name_reserved, "kova-ultra");
  assert.equal(ultra.active_workers, 0);
});

test("Core serving keeps model, quantization, GPU, server, and image selection blocked", () => {
  const config = JSON.parse(readFileSync(join(root, "config/core-serving.v1.json"), "utf8"));
  assert.equal(config.endpoint_deployed, false);
  assert.equal(config.selected_candidate_id, null);
  assert.equal(config.selected_gpu, null);
  assert.equal(config.selected_serving_engine, null);
  assert.equal(config.container_image_digest, null);
  assert.deepEqual(config.reasoning_efforts, ["low", "medium", "xhigh"]);
  assert.deepEqual(config.candidates.map((candidate) => candidate.model), [
    "Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8",
  ]);
  assert.ok(config.candidates.every((candidate) =>
    candidate.context_tokens === 262144 &&
    candidate.compatibility_verified === false &&
    candidate.benchmark_complete === false
  ));
});

test("Core container pins upstream source while every build and deployment action stays blocked", () => {
  const config = JSON.parse(readFileSync(join(root, "config/core-container.v1.json"), "utf8"));
  assert.equal(config.status, "source_only_build_blocked");
  assert.equal(config.upstream_worker.repository, "runpod-workers/worker-vllm");
  assert.equal(config.upstream_worker.release_tag, "v2.27.0");
  assert.equal(config.upstream_worker.source_commit, "76054c22c79c515f07065f523598d8efb2f9b682");
  assert.equal(config.upstream_worker.bundled_vllm_version, "0.29.0");
  assert.equal(config.upstream_worker.resolved_image_digest, null);
  assert.equal(config.integration.selected, false);
  assert.equal(config.integration.queue_protocol_adapter, "worker/runpod_vllm.py");
  assert.equal(config.integration.queue_protocol_adapter_implemented, true);
  assert.equal(config.integration.cpu_protocol_fixture_verified, true);
  assert.equal(config.integration.live_provider_envelope_verified, false);
  assert.equal(config.transport.target, "runpod_serverless_queue");
  assert.equal(config.transport.queue_input_shape, "openai_passthrough");
  assert.equal(config.transport.openai_route, "/v1/chat/completions");
  assert.equal(config.transport.stream_worker_output, "raw_openai_sse");
  assert.equal(config.transport.full_runpod_http_envelope_supported, false);
  assert.equal(config.transport.network_client_implemented, false);
  assert.equal(config.transport.live_transport_verified, false);
  assert.equal(config.candidate_runtime_configuration.client_overrides_allowed, false);
  assert.deepEqual(config.candidate_runtime_configuration.profiles.map((profile) => profile.candidate_id), [
    "qwen3.8-27b-bf16", "qwen3.8-27b-fp8",
  ]);
  assert.equal(config.context.selected_max_model_len, null);
  assert.equal(config.weights.runtime_model_download_allowed, false);
  assert.equal(config.weights.network_volume_enabled, false);
  assert.equal(config.weights.selected_packaging_strategy, null);
  assert.equal(config.endpoint.selected_type, null);
  assert.deepEqual(Object.keys(config.safety).sort(), [
    "compatibility_tested", "deployment_authorized", "endpoint_creation_authorized",
    "image_built", "image_pulled", "paid_benchmark_authorized",
    "production_routing_authorized", "runtime_package_installs_allowed",
  ]);
  assert.ok(Object.values(config.safety).every((value) => value === false));
});

test("hardware planning covers both Core candidates without stale provider inventory claims", () => {
  const config = JSON.parse(readFileSync(join(root, "config/hardware-benchmark.v1.json"), "utf8"));
  assert.equal(config.schema_version, 2);
  assert.equal(config.selected_candidate_id, null);
  assert.equal(config.selected_provider_hardware_id, null);
  assert.equal(config.provider_inventory_snapshot, null);
  assert.equal(config.provider_price_snapshot, null);
  assert.equal(config.inventory_must_be_refreshed_at_benchmark_time, true);
  assert.deepEqual(config.candidate_matrices.map((matrix) => [
    matrix.candidate_id, matrix.minimum_benchmark_vram_gb,
  ]), [
    ["qwen3.8-27b-bf16", 80],
    ["qwen3.8-27b-fp8", 48],
  ]);
  assert.ok(config.candidate_matrices.every((matrix) =>
    matrix.selected_provider_hardware_id === null &&
    matrix.compatibility_verified === false &&
    matrix.benchmark_complete === false
  ));
  assert.equal(config.paid_benchmark_authorized, false);
  assert.equal(config.deployment_authorized, false);
  assert.equal(config.production_routing_authorized, false);
});
