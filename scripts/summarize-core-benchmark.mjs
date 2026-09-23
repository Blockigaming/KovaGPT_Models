import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";

const current = JSON.parse(execFileSync("python3", ["-E", "-S", "-B", "-c", `
import json
from core.current_candidates import CORE_SERVING
from router.policy import resolve_route
routes = [resolve_route({'surface':'chat','route_id':route}) for route in ('instant','medium','high','extra-high','max')]
routes += [resolve_route({'surface':surface,'family':family,'effort':effort}) for surface, families in (('chat',('cosmo','orion')),('work',('cosmo','orion','nova'))) for family in families for effort in ('Light','Medium','High','Extra High','Max')]
print(json.dumps({'serving':CORE_SERVING,'routes':routes}))
`], { cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8" }));
const serving = current.serving;
const servingCapabilities = JSON.parse(readFileSync(new URL("../config/core-serving.v1.json", import.meta.url), "utf8"));
const candidates = new Map(serving.candidates.map((candidate) => [candidate.model, candidate]));
const outcomes = new Set(["success", "failed", "quarantined"]);
const recordTypes = new Set(["attempt", "lifecycle_close"]);
const commonRequired = [
  "record_type", "worker_lifecycle_id", "model", "model_revision", "adapter_sha256", "gpu_type_id",
  "gpu_count", "serving_engine", "endpoint_type", "container_image_digest",
  "gpu_rate_per_second_usd", "measurement_source",
];
const attemptRequired = [
  "request_id", "correlation_id", "attempt_id", "outcome", "route_id", "stage_id", "public_response",
  "cold_start", "reasoning_effort", "worker_start_ms", "model_load_ms", "queue_ms", "inference_ms",
  "time_to_first_token_ms", "input_tokens", "output_tokens",
];
const closeRequired = ["close_event_id", "billed_lifecycle_ms", "attributed_idle_timeout_ms"];
const logicalRequestIdPattern = /^kova-exec-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/u;

const average = (records, key) => records.length
  ? records.reduce((total, record) => total + record[key], 0) / records.length
  : null;

const stageIds = (policy) => {
  const stages = [];
  for (const [phase, field] of [
    ["planning", "planning_passes"], ["answer", "answer_passes"],
    ["critic", "critic_passes"], ["verification", "verification_passes"],
  ]) {
    for (let index = 1; index <= (policy[field] ?? 0); index += 1) stages.push(`${phase}-${index}`);
  }
  return stages;
};

const routeSpecs = new Map();
for (const route of current.routes) {
  const [planning_passes, answer_passes, critic_passes, verification_passes] = route.passes;
  const stages = stageIds({planning_passes, answer_passes, critic_passes, verification_passes});
  routeSpecs.set(route.route_id, {stages, public_stage: stages.at(-1), reasoning_effort: route.reasoning_effort,
    model: `kova-${route.profile}`});
}

export const coreConfigurationKey = (record) => [
  `${record.model}@${record.model_revision}`,
  record.adapter_sha256,
  `${record.gpu_type_id}x${record.gpu_count}`,
  record.serving_engine,
  record.endpoint_type,
  record.container_image_digest,
].join("|");

const configurationFrom = (record, candidateCatalog) => ({
  model: record.model,
  model_revision: record.model_revision,
  adapter_sha256: record.adapter_sha256,
  quantization: candidateCatalog.get(record.model).quantization,
  gpu_type_id: record.gpu_type_id,
  gpu_count: record.gpu_count,
  serving_engine: record.serving_engine,
  endpoint_type: record.endpoint_type,
  container_image_digest: record.container_image_digest,
});

function summarizeGroup(records, routeId, allocations) {
  const spec = routeSpecs.get(routeId);
  const successfulAttempts = records.filter((record) => record.outcome === "success");
  const requestIds = new Set(records.map((record) => record.request_id));
  const requestTiming = [];
  const successfulRequests = new Set();
  for (const requestId of requestIds) {
    const requestAttempts = records.filter((record) => record.request_id === requestId);
    const successes = successfulAttempts.filter((record) => record.request_id === requestId);
    for (const stageId of spec.stages) {
      if (successes.filter((record) => record.stage_id === stageId).length > 1) {
        throw new Error(`request ${requestId} contains duplicate successful stage ${stageId}`);
      }
    }
    if (!spec.stages.every((stageId) => successes.some((record) => record.stage_id === stageId))) continue;
    const publicAttempt = successes.find((record) => record.stage_id === spec.public_stage);
    const requestTimeToFirstTokenMs = requestAttempts.reduce((total, record) => (
      total + record.worker_start_ms + record.model_load_ms + record.queue_ms +
      (record === publicAttempt ? record.time_to_first_token_ms : record.inference_ms)
    ), 0);
    successfulRequests.add(requestId);
    requestTiming.push({request_id: requestId, request_time_to_first_token_ms: requestTimeToFirstTokenMs});
  }
  const successfulPublicAttempts = successfulAttempts.filter((record) =>
    record.stage_id === spec.public_stage && successfulRequests.has(record.request_id),
  );
  const startupCost = allocations.reduce((total, item) => total + item.startup_cost_usd, 0);
  const activeCost = allocations.reduce((total, item) => total + item.active_cost_usd, 0);
  const idleCost = allocations.reduce((total, item) => total + item.idle_cost_usd, 0);
  const totalComputeCost = allocations.reduce((total, item) => total + item.total_lifecycle_cost_usd, 0);
  return {
    attempts: records.length,
    worker_lifecycles: new Set(records.map((record) => record.worker_lifecycle_id)).size,
    logical_requests: requestIds.size,
    successful_requests: successfulRequests.size,
    failed_attempts: records.filter((record) => record.outcome === "failed").length,
    quarantined_attempts: records.filter((record) => record.outcome === "quarantined").length,
    non_successful_attempts: records.length - successfulAttempts.length,
    total_attributable_compute_cost_usd: totalComputeCost,
    allocated_startup_cost_usd: startupCost,
    allocated_active_cost_usd: activeCost,
    allocated_idle_cost_usd: idleCost,
    total_observed_attempt_inference_ms: records.reduce((total, record) => total + record.inference_ms, 0),
    average_compute_cost_per_successful_request_usd:
      successfulRequests.size ? totalComputeCost / successfulRequests.size : null,
    startup_share_of_compute_percent: totalComputeCost > 0 ? startupCost / totalComputeCost * 100 : null,
    active_share_of_compute_percent: totalComputeCost > 0 ? activeCost / totalComputeCost * 100 : null,
    idle_share_of_compute_percent: totalComputeCost > 0 ? idleCost / totalComputeCost * 100 : null,
    average_success_stage_latency_ms: average(successfulAttempts, "inference_ms"),
    average_request_time_to_first_token_ms: average(requestTiming, "request_time_to_first_token_ms"),
    average_public_output_tokens: average(successfulPublicAttempts, "output_tokens"),
    successful_request_timing: requestTiming,
  };
}

// The CLI always uses the source candidate registry; synthetic catalogs are for offline unit tests.
export function summarizeCoreBenchmark(records, candidateCatalog = candidates) {
  if (!Array.isArray(records) || records.length === 0) throw new Error("Core benchmark requires records");
  const seenRecordIds = new Set();
  const requestIdentities = new Map();
  const requestRoutes = new Map();
  const lifecycles = new Map();
  const attempts = [];

  for (const [index, record] of records.entries()) {
    for (const field of commonRequired) if (!(field in record)) throw new Error(`record ${index} missing ${field}`);
    if (!recordTypes.has(record.record_type)) throw new Error(`record ${index} invalid record_type`);
    const allowedFields = new Set([
      ...commonRequired,
      ...(record.record_type === "attempt" ? attemptRequired : closeRequired),
    ]);
    const extraFields = Object.keys(record).filter((field) => !allowedFields.has(field));
    if (extraFields.length) throw new Error(`record ${index} unsupported fields:${extraFields.sort().join(",")}`);
    for (const field of ["worker_lifecycle_id", "model", "model_revision", "gpu_type_id", "serving_engine", "endpoint_type", "container_image_digest"]) {
      if (typeof record[field] !== "string" || !record[field].trim()) throw new Error(`record ${index} invalid ${field}`);
    }
    const candidate = candidateCatalog.get(record.model);
    if (!candidate || candidate.revision !== record.model_revision) throw new Error(`record ${index} unverified model revision`);
    if (!/^[a-f0-9]{64}$/u.test(record.adapter_sha256)
        || record.adapter_sha256 !== candidate.adapter_sha256) throw new Error(`record ${index} unverified adapter digest`);
    if (!Number.isInteger(record.gpu_count) || record.gpu_count < 1 || record.gpu_count > 8) {
      throw new Error(`record ${index} invalid gpu_count`);
    }
    if (!servingCapabilities.serving_engine_candidates.includes(record.serving_engine)) throw new Error(`record ${index} invalid serving_engine`);
    if (!servingCapabilities.endpoint_type_candidates.includes(record.endpoint_type)) throw new Error(`record ${index} invalid endpoint_type`);
    if (!/^sha256:[a-f0-9]{64}$/u.test(record.container_image_digest)) throw new Error(`record ${index} invalid container_image_digest`);
    if (!Number.isFinite(record.gpu_rate_per_second_usd) || record.gpu_rate_per_second_usd <= 0) {
      throw new Error(`record ${index} invalid gpu_rate_per_second_usd`);
    }
    if (record.measurement_source !== "server_provider_runtime") throw new Error(`record ${index} invalid measurement_source`);

    const configurationKey = coreConfigurationKey(record);
    const lifecycle = lifecycles.get(record.worker_lifecycle_id) ?? {
      configuration_key: configurationKey,
      configuration: configurationFrom(record, candidateCatalog),
      attempts: [],
      closes: [],
      rates: new Set(),
    };
    if (lifecycle.configuration_key !== configurationKey) {
      throw new Error(`lifecycle ${record.worker_lifecycle_id} mixes serving configurations`);
    }
    lifecycle.rates.add(record.gpu_rate_per_second_usd);

    if (record.record_type === "attempt") {
      for (const field of attemptRequired) if (!(field in record)) throw new Error(`record ${index} missing ${field}`);
      for (const field of ["request_id", "correlation_id", "attempt_id", "stage_id", "route_id"]) {
        if (typeof record[field] !== "string" || !record[field]) throw new Error(`record ${index} invalid ${field}`);
      }
      if (!logicalRequestIdPattern.test(record.request_id)) {
        throw new Error(`record ${index} request_id is not a server logical execution ID`);
      }
      if (seenRecordIds.has(`attempt:${record.attempt_id}`)) throw new Error(`record ${index} duplicate attempt_id`);
      seenRecordIds.add(`attempt:${record.attempt_id}`);
      const routeSpec = routeSpecs.get(record.route_id);
      if (!routeSpec) throw new Error(`record ${index} invalid route_id`);
      if (record.model !== routeSpec.model) throw new Error(`record ${index} model differs from route family`);
      if (!outcomes.has(record.outcome)) throw new Error(`record ${index} invalid outcome`);
      if (!["low", "medium", "xhigh"].includes(record.reasoning_effort)) throw new Error(`record ${index} invalid reasoning_effort`);
      if (record.reasoning_effort !== routeSpec.reasoning_effort) throw new Error(`record ${index} reasoning_effort does not match route`);
      for (const field of ["public_response", "cold_start"]) {
        if (typeof record[field] !== "boolean") throw new Error(`record ${index} invalid ${field}`);
      }
      if (!routeSpec.stages.includes(record.stage_id)) throw new Error(`record ${index} invalid stage_id for route`);
      if (record.public_response !== (record.stage_id === routeSpec.public_stage)) {
        throw new Error(`record ${index} public_response does not match route DAG`);
      }
      for (const field of ["worker_start_ms", "model_load_ms", "queue_ms", "inference_ms"]) {
        if (!Number.isFinite(record[field]) || record[field] < 0) throw new Error(`record ${index} invalid ${field}`);
      }
      for (const field of ["input_tokens", "output_tokens"]) {
        if (!Number.isInteger(record[field]) || record[field] < 0) throw new Error(`record ${index} invalid ${field}`);
      }
      if (record.outcome === "success" && record.input_tokens <= 0) {
        throw new Error(`record ${index} successful attempt missing input tokens`);
      }
      if (record.outcome === "success" && record.output_tokens <= 0) {
        throw new Error(`record ${index} successful attempt missing output tokens`);
      }
      const firstToken = record.time_to_first_token_ms;
      if (firstToken !== null && (!Number.isFinite(firstToken) || firstToken < 0)) {
        throw new Error(`record ${index} invalid time_to_first_token_ms`);
      }
      if (!record.public_response && firstToken !== null) {
        throw new Error(`record ${index} private stage must not claim first-token timing`);
      }
      if (record.public_response && record.outcome === "success" && firstToken === null) {
        throw new Error(`record ${index} successful public stage missing first-token timing`);
      }
      if (firstToken !== null && firstToken > record.inference_ms) throw new Error(`record ${index} first token exceeds inference`);
      const startupMs = record.worker_start_ms + record.model_load_ms;
      if (record.cold_start && startupMs <= 0) throw new Error(`record ${index} cold attempt missing measured startup`);
      if (!record.cold_start && startupMs !== 0) throw new Error(`record ${index} warm attempt contains startup attribution`);

      const requestIdentity = `${configurationKey}|${record.route_id}`;
      const priorRequestIdentity = requestIdentities.get(record.request_id);
      if (priorRequestIdentity && priorRequestIdentity !== requestIdentity) {
        throw new Error(`request ${record.request_id} mixes serving configuration or route`);
      }
      requestIdentities.set(record.request_id, requestIdentity);
      requestRoutes.set(record.request_id, record.route_id);
      lifecycle.attempts.push(record);
      attempts.push(record);
    } else {
      for (const field of closeRequired) {
        if (!(field in record)) throw new Error(`record ${index} missing ${field}`);
      }
      if (typeof record.close_event_id !== "string" || !record.close_event_id) throw new Error(`record ${index} invalid close_event_id`);
      if (seenRecordIds.has(`close:${record.close_event_id}`)) throw new Error(`record ${index} duplicate close_event_id`);
      seenRecordIds.add(`close:${record.close_event_id}`);
      if (!Number.isFinite(record.attributed_idle_timeout_ms) || record.attributed_idle_timeout_ms <= 0) {
        throw new Error(`record ${index} lifecycle close missing measured idle tail`);
      }
      if (!Number.isFinite(record.billed_lifecycle_ms) || record.billed_lifecycle_ms <= 0) {
        throw new Error(`record ${index} lifecycle close missing billed wall time`);
      }
      if (record.attributed_idle_timeout_ms > record.billed_lifecycle_ms) {
        throw new Error(`record ${index} idle tail exceeds billed lifecycle`);
      }
      lifecycle.closes.push(record);
    }
    lifecycles.set(record.worker_lifecycle_id, lifecycle);
  }

  for (const [lifecycleId, lifecycle] of lifecycles) {
    if (!lifecycle.attempts.length) throw new Error(`lifecycle ${lifecycleId} has no attempts`);
    if (lifecycle.attempts.filter((record) => record.cold_start).length !== 1) {
      throw new Error(`lifecycle ${lifecycleId} requires exactly one cold-start attribution`);
    }
    if (lifecycle.closes.length !== 1) throw new Error(`lifecycle ${lifecycleId} requires exactly one close event`);
    if (lifecycle.rates.size !== 1) throw new Error(`lifecycle ${lifecycleId} mixes GPU rates`);
    const cold = lifecycle.attempts.find((record) => record.cold_start);
    const startupMs = cold.worker_start_ms + cold.model_load_ms;
    const close = lifecycle.closes[0];
    if (startupMs + close.attributed_idle_timeout_ms > close.billed_lifecycle_ms) {
      throw new Error(`lifecycle ${lifecycleId} startup and idle exceed billed wall time`);
    }
    const activeMs = close.billed_lifecycle_ms - startupMs - close.attributed_idle_timeout_ms;
    const longestInferenceMs = Math.max(...lifecycle.attempts.map((record) => record.inference_ms));
    if (longestInferenceMs > activeMs) {
      throw new Error(`lifecycle ${lifecycleId} longest attempt exceeds billed active window`);
    }
    const requestIds = [...new Set(lifecycle.attempts.map((record) => record.request_id))];
    const longestSequentialRequestMs = Math.max(...requestIds.map((requestId) =>
      lifecycle.attempts
        .filter((record) => record.request_id === requestId)
        .reduce((total, record) => total + record.inference_ms, 0)
    ));
    if (longestSequentialRequestMs > activeMs) {
      throw new Error(`lifecycle ${lifecycleId} sequential request path exceeds billed active window`);
    }
  }

  const pricedAttempts = attempts.map((record) => {
    const startupMs = record.worker_start_ms + record.model_load_ms;
    return {
      ...record,
      startup_ms: startupMs,
      configuration_key: coreConfigurationKey(record),
    };
  });
  const pricedLifecycles = [];
  for (const [lifecycleId, lifecycle] of lifecycles) {
    const close = lifecycle.closes[0];
    const cold = lifecycle.attempts.find((record) => record.cold_start);
    const startupMs = cold.worker_start_ms + cold.model_load_ms;
    const idleMs = close.attributed_idle_timeout_ms;
    const activeMs = close.billed_lifecycle_ms - startupMs - idleMs;
    const rate = close.gpu_rate_per_second_usd;
    pricedLifecycles.push({
      ...close,
      configuration_key: lifecycle.configuration_key,
      startup_ms: startupMs,
      active_ms: activeMs,
      startup_cost_usd: startupMs / 1000 * rate,
      active_cost_usd: activeMs / 1000 * rate,
      idle_cost_usd: idleMs / 1000 * rate,
      billed_lifecycle_cost_usd: close.billed_lifecycle_ms / 1000 * rate,
    });
  }

  const lifecycleAllocations = [];
  for (const [lifecycleId, lifecycle] of lifecycles) {
    const lifecycleAttempts = pricedAttempts.filter((record) => record.worker_lifecycle_id === lifecycleId);
    const pricedLifecycle = pricedLifecycles.find((record) => record.worker_lifecycle_id === lifecycleId);
    const requestIds = [...new Set(lifecycleAttempts.map((record) => record.request_id))];
    const requestInferenceMs = new Map(requestIds.map((requestId) => [
      requestId,
      lifecycleAttempts
        .filter((record) => record.request_id === requestId)
        .reduce((total, record) => total + record.inference_ms, 0),
    ]));
    const totalInferenceMs = [...requestInferenceMs.values()].reduce((total, value) => total + value, 0);
    for (const requestId of requestIds) {
      const activeShare = totalInferenceMs > 0
        ? requestInferenceMs.get(requestId) / totalInferenceMs
        : 1 / requestIds.length;
      const startupCost = pricedLifecycle.startup_cost_usd / requestIds.length;
      const activeCost = pricedLifecycle.active_cost_usd * activeShare;
      const idleCost = pricedLifecycle.idle_cost_usd / requestIds.length;
      lifecycleAllocations.push({
        worker_lifecycle_id: lifecycleId,
        configuration_key: lifecycle.configuration_key,
        request_id: requestId,
        route_id: requestRoutes.get(requestId),
        allocation_policy: "startup_idle_equal_per_request_active_proportional_to_observed_attempt_inference",
        observed_attempt_inference_ms: requestInferenceMs.get(requestId),
        startup_cost_usd: startupCost,
        active_cost_usd: activeCost,
        idle_cost_usd: idleCost,
        total_lifecycle_cost_usd: startupCost + activeCost + idleCost,
      });
    }
  }

  const byConfiguration = {};
  for (const record of pricedAttempts) {
    const key = record.configuration_key;
    byConfiguration[key] ??= {configuration: configurationFrom(record, candidateCatalog), by_route: {}};
  }
  for (const [configurationKey, configurationGroup] of Object.entries(byConfiguration)) {
    for (const routeId of routeSpecs.keys()) {
      const group = pricedAttempts.filter((record) => record.configuration_key === configurationKey && record.route_id === routeId);
      if (group.length) {
        const allocations = lifecycleAllocations.filter((item) => item.configuration_key === configurationKey && item.route_id === routeId);
        configurationGroup.by_route[routeId] = summarizeGroup(group, routeId, allocations);
      }
    }
  }

  const totalStartupCost = pricedLifecycles.reduce((total, record) => total + record.startup_cost_usd, 0);
  const totalActiveCost = pricedLifecycles.reduce((total, record) => total + record.active_cost_usd, 0);
  const totalIdleCost = pricedLifecycles.reduce((total, record) => total + record.idle_cost_usd, 0);
  const totalBilledCost = pricedLifecycles.reduce((total, record) => total + record.billed_lifecycle_cost_usd, 0);
  const allocatedCost = lifecycleAllocations.reduce((total, record) => total + record.total_lifecycle_cost_usd, 0);
  const tolerance = Math.max(1, Math.abs(totalBilledCost)) * 1e-12;
  if (Math.abs(totalStartupCost + totalActiveCost + totalIdleCost - totalBilledCost) > tolerance) {
    throw new Error("lifecycle cost components do not conserve billed cost");
  }
  if (Math.abs(allocatedCost - totalBilledCost) > tolerance) {
    throw new Error("request allocations do not conserve billed lifecycle cost");
  }
  return {
    schema_version: 6,
    provider: "runpod_serverless",
    cost_scope: "runpod_compute_only_excludes_storage_app_tools_payment_and_taxes",
    billing_contract: "provider_measured_billed_lifecycle_wall_time_is_authoritative;attempt_durations_are_diagnostics_only",
    lifecycle_cost_allocation: "startup_and_idle_equal_per_request;active_proportional_to_observed_attempt_inference;all_components_conserved",
    timing_contract: "request_ttft_includes_startup_queue_failed_retries_private_dag_stages_and_final_public_first_visible_delta",
    attempts: pricedAttempts.length,
    worker_lifecycles: lifecycles.size,
    total_billed_lifecycle_ms: pricedLifecycles.reduce((total, record) => total + record.billed_lifecycle_ms, 0),
    total_observed_attempt_inference_ms: pricedAttempts.reduce((total, record) => total + record.inference_ms, 0),
    total_attributable_compute_cost_usd: totalBilledCost,
    total_startup_component_cost_usd: totalStartupCost,
    total_active_component_cost_usd: totalActiveCost,
    total_idle_component_cost_usd: totalIdleCost,
    by_configuration: byConfiguration,
    lifecycle_allocations: lifecycleAllocations,
    attempt_records: pricedAttempts,
    lifecycle_close_records: pricedLifecycles,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const input = process.argv[2];
  if (!input) throw new Error("usage: npm run benchmark:core:summarize -- benchmark.json");
  process.stdout.write(`${JSON.stringify(summarizeCoreBenchmark(JSON.parse(readFileSync(input, "utf8"))), null, 2)}\n`);
}
