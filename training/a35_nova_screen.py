"""One changed Nova adapter and a staged quality screen; paid execution gated.

Default invocation is source-only. A launch requires a separate owner grant,
an externally verified live watchdog, a clean published source SHA and a new
run claim in protected storage. No old adapter or resume path is accepted.
This diagnostic cannot establish authenticated serving routes or close A35.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from evaluation.a35_correction_data import prepared_revision_rows, validate_draft
from evaluation.a35_quality_policy import load_policy
from evaluation.completion_evidence import completion_status, generation_evidence
from evaluation.cpu_candidate_quality import score_case
from evaluation.historical_suite_bridge import load_archived_suite
from core.public_identity import contains_prohibited

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/a35-nova-screen.v1.json"
MEASURED_ADAPTERS = (
    "f6d09354b5db910288be1e4ac9e22467bfe0c4bf8b554ef576162a245955e352",
    "95470d46db4a430ce6f5e5d53f5dc322dcb4a04aec824b69b922dbe9b1d71213",
    "9e9c991332e979d3b3b73452e643d0272f859965fc054329f32308491a0a7e0b",
    "95fd3383589ef1bfe942345a9df09e3a3afaa8cf8d24ea886a273eea2a5089ee",
    "4aa5e981fb5d28b902fb5fdf6da68f15b76e4659010195902446530f29b5d264",
    "32b971862c5b7d0a013aea8f6da4913ba5eeb94b92e153dd7d0aa10419031e16",
    "9506b310dd551cfb969905fedcb8e39792e6630f69bfabb677257a5fdc765e35",
)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def system_prompt(plan):
    """Nova-only task checks; keep the shared identity/safety policy intact."""
    need(plan.get("task_checks_path") == "prompts/kova-nova-task-checks.v2.txt",
         "Nova task-check source required")
    return ((ROOT / "prompts/kova-identity.v5.txt").read_text() + "\n" +
            (ROOT / plan["task_checks_path"]).read_text())


def historical_nova_rows(plan):
    """Reproduce the measured parent exactly; never use it for new execution."""
    rows = prepared_revision_rows()
    base = (ROOT / "prompts/kova-identity.v4.draft.txt").read_text()
    # The measured parent has its own immutable prompt; active revisions must
    # never rewrite the historical pack used to validate identity overrides.
    system = base + "\n" + (ROOT / "prompts/kova-nova-task-checks.v1.txt").read_text()
    for _, _, row in rows:
        message = row["prompt"][0]
        need(message["role"] == "system" and message["content"].startswith(base),
             "shared policy prefix drift")
        # Retain any explicitly hypothetical trusted-runtime fixture after policy.
        message["content"] = system + message["content"][len(base):]
    from evaluation.a35_nova_copy_contrast import load_rows
    from evaluation.a35_nova_transfer_data import load_rows as transfer_rows
    from training.template_policy import template_row
    for row in load_rows() + transfer_rows():
        prepared = template_row(
            [{"role": "system", "content": system}, row["messages"][0]],
            [row["messages"][1]])
        rows.append((row["id"], row["split"], prepared))
    return rows


def prepared_nova_rows(plan):
    from training.a35_identity_inputs import apply_identity_policy
    system = system_prompt(plan)  # Validate the source allowlist before any read.
    return apply_identity_policy(historical_nova_rows(plan), system, plan["file_sha256"])


def prepared_inputs(plan):
    rows = prepared_nova_rows(plan)
    need(sha(encoded(rows)) == plan["prepared_pack_sha256"], "revised prepared pack drift")
    train = [row for _, split, row in rows if split == "train"]
    validation = [row for _, split, row in rows if split == "validation"]
    need((len(train), len(validation)) == (73, 61)
         == (plan["train_records"], plan["validation_records"]), "revised input split drift")
    need(all(r["chat_template_kwargs"] == {"enable_thinking": False}
             for r in train + validation), "template mode drift")
    return train, validation


def load_plan(path=PLAN):
    plan = json.loads(path.read_text())
    need(plan["family"] == "kova-nova" and plan["base_model"] == "internal-nova-base-v1"
         and plan["base_revision"] == "1cfa9a7208912126459214e8b04321603b3df60c",
         "only the corrected Nova screen is supported")
    need(plan.get("private_catalog_required") is True
         and all(re.fullmatch(r"[0-9a-f]{64}", str(plan.get(key, "")))
                 for key in ("private_catalog_sha256", "private_source_manifest_sha256")),
         "private source and identity policy bindings required")
    need(plan.get("previous_screen_adapter_sha256") ==
         "95470d46db4a430ce6f5e5d53f5dc322dcb4a04aec824b69b922dbe9b1d71213",
         "completed-screen candidate binding required")
    need(plan.get("rejected_adapter_sha256") == list(MEASURED_ADAPTERS),
         "all measured candidates must remain rejected")
    for relative, expected in plan["file_sha256"].items():
        need(sha((ROOT / relative).read_bytes()) == expected, "input pin drift: " + relative)
    validate_draft()
    policy = load_policy()
    e, t, r = plan["evaluation"], plan["training"], plan["resources"]
    need(e == {"suite_sha256": policy["suite_sha256"], "strict_cases": 36,
         "manual_cases": 14, "manual_criteria": 48, "screening_repetitions": 1,
         "strict_max_new_tokens": 128, "manual_max_new_tokens": 2048,
         "do_sample": False, "enable_thinking": False,
         "manual_only_after_strict_36_of_36": True,
         "confirmation_repetitions_authorized": 0,
         "identity_safety_failure_stops_immediately": True, "human_review_required": True},
         "screening rule drift")
    need(t == {"fresh_base_only": True, "epochs": 3, "expected_optimizer_steps": 30,
         "batch_size": 1, "gradient_accumulation_steps": 8, "learning_rate": 0.00008,
         "sequence_length": 768, "seed": 42, "completion_only_loss": True,
         "packing": False, "enable_thinking": False, "nf4_compute_dtype": "float16",
         "double_quant": True, "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
         "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
         "maximum_seconds": 600}, "training recipe drift")
    train, validation = prepared_inputs(plan)
    need(math.ceil(len(train) / t["gradient_accumulation_steps"]) * t["epochs"] == t["expected_optimizer_steps"],
         "each complete epoch must include every training example")
    need(r["sku"] == "Standard_NC4as_T4_v3" and r["region"] == "eastus"
         and r["gpu_count"] == r["allocation_attempts"] == 1
         and r["vm_public_ip"] is False
         and (r["work_deadline_seconds"], r["watchdog_trigger_seconds"],
              r["maximum_allocation_seconds"], r["cleanup_reserve_seconds"]) == (4200, 4500, 5400, 900),
         "resource bounds drift")
    need(sum(Decimal(v) for v in plan["cost"]["reserved_usd"].values()) == Decimal("5.00")
         == Decimal(plan["cost"]["ceiling"]), "complete cost reservation required")
    prices = json.loads((ROOT / "evaluations/a35-nova-screen-pricing.v1.json").read_text())
    rates = {row["meter"]["meterName"]: Decimal(str(row["meter"]["retailPrice"])) for row in prices["rows"]}
    need(len(rates) == 7 and all(row["meter"]["currencyCode"] == "USD" and
         row["meter"]["type"] == "Consumption" for row in prices["rows"]), "retail meter scope drift")
    need(rates["NC4as T4 v3"] <= Decimal("0.60")
         and rates["Standard Gateway"] <= Decimal("0.05")
         and rates["Standard Data Processed"] <= Decimal("0.05")
         and rates["Standard IPv4 Static Public IP"] <= Decimal("0.01")
         and 2 * (rates["E6 LRS Disk"] / 730 + rates["E6 LRS Disk Operations"] * Decimal("11.44")) <= Decimal("0.25")
         and 2000 * rates["Consumption Built-in Actions"] <= Decimal("0.05"),
         "observed retail price exceeds reservation")
    for key in ("execution_authorized", "unchanged_candidate_rerun_authorized",
                "automatic_retry_authorized", "pilot_selection_authorized",
                "deployment_authorized", "merge_authorized", "a35_quality_gate_closed"):
        need(plan[key] is False, "source must not grant paid/release authority")
    return plan


def sft_kwargs(plan, output):
    """Shared by the actual trainer and free tokenizer/TRL probe."""
    t = plan["training"]
    return dict(output_dir=str(output), num_train_epochs=t["epochs"], max_steps=-1,
                per_device_train_batch_size=t["batch_size"],
                gradient_accumulation_steps=t["gradient_accumulation_steps"],
                learning_rate=t["learning_rate"], max_length=t["sequence_length"],
                completion_only_loss=True, packing=False, fp16=False, bf16=False,
                gradient_checkpointing=True, optim="adamw_torch", save_strategy="no",
                eval_strategy="no", report_to="none", push_to_hub=False, seed=t["seed"],
                data_seed=t["seed"], dataloader_drop_last=False)


def verify_processed_masks(tokenizer, processed, originals, collator):
    """Compare real trainer/collator labels with independently rendered tokens."""
    from training.probe_cosmo_loss_masks import verify_record
    need(len(processed) == len(originals), "trainer dropped/duplicated input rows")
    maximum = 0
    for record, row in zip(processed, originals, strict=True):
        prefix = tokenizer.apply_chat_template(row["prompt"], tokenize=True,
            add_generation_prompt=True, return_dict=False, **row["chat_template_kwargs"])
        full = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True,
            add_generation_prompt=False, return_dict=False, **row["chat_template_kwargs"])
        labels = collator([record])["labels"][0].tolist()
        verify_record(dict(record, labels=labels), full, prefix, 768)
        need(tokenizer.eos_token_id in full[len(prefix):], "completion EOS missing")
        maximum = max(maximum, len(full))
    return maximum


def probe_candidate(assets):
    """Exact proposed SFT configuration, real TRL; no weights or model calls."""
    from training.probe_cosmo_loss_masks import offline_cpu, verify_environment, read_assets
    from unittest.mock import patch
    from contextlib import ExitStack
    import tempfile
    verify_environment()
    read_assets(assets)
    plan = load_plan()
    train, validation = prepared_inputs(plan)
    with offline_cpu(), tempfile.TemporaryDirectory(prefix="a35-screen-mask-") as temp:
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
        from trl import SFTConfig, SFTTrainer
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        with ExitStack() as guards:
            for obj, attr in ((AutoModelForCausalLM, "from_pretrained"),
                              (AutoModelForCausalLM, "from_config"), (Trainer, "__init__"),
                              (SFTTrainer, "__init__"), (SFTTrainer, "train")):
                guards.enter_context(patch.object(obj, attr, side_effect=ValueError("model calls forbidden")))
            tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
            config = SFTConfig(**sft_kwargs(plan, temp), use_cpu=True)
            preparer = object.__new__(SFTTrainer)
            preparer._tokenizer = tokenizer
            preparer.chat_template = tokenizer.chat_template
            preparer.completion_only_loss = True
            collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
            maxima = {}
            for split, rows in (("train", train), ("validation", validation)):
                processed = preparer._prepare_dataset(Dataset.from_list(rows), tokenizer, config, False, None, split)
                maxima[split] = verify_processed_masks(tokenizer, processed, rows, collator)
            completion_max = max(len(tokenizer.apply_chat_template(r["prompt"] + r["completion"],
                tokenize=True, return_dict=False, **r["chat_template_kwargs"])) -
                len(tokenizer.apply_chat_template(r["prompt"], tokenize=True, add_generation_prompt=True,
                    return_dict=False, **r["chat_template_kwargs"])) for r in train + validation)
            system = system_prompt(plan)
            prompts = [len(tokenizer.apply_chat_template([
                {"role": "system", "content": system}, {"role": "user", "content": c["prompt"]}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False))
                for c in load_archived_suite()["cases"]]
            need(max(prompts) + 2048 < 32768 and completion_max < 2048,
                 "evaluation context/output allowance insufficient")
    return {"status": "pass", "candidate": plan["experiment_id"], "records": len(train) + len(validation),
            "prepared_pack_sha256": plan["prepared_pack_sha256"],
            "system_prompt_sha256": sha(system.encode()),
            "maximum_tokens": maxima, "maximum_reference_completion_tokens": completion_max,
            "maximum_evaluation_prompt_tokens": max(prompts), "manual_output_budget": 2048,
            "training_model_calls": 0, "inference_calls": 0}


def clean_head():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    need(not git("status", "--porcelain", "--untracked-files=all"), "dirty source rejected")
    head = git("rev-parse", "HEAD")
    need(re.fullmatch(r"[a-f0-9]{40}", head), "invalid source head")
    return head


def admit(grant, plan, source, now=None):
    """The authorized external operator supplies scope; this is not route attestation."""
    now = time.time() if now is None else now
    need(grant.get("owner_authorized") is True, "explicit owner grant missing")
    need(grant.get("experiment_id") == plan["experiment_id"]
         and grant.get("source_commit") == source
         and grant.get("plan_sha256") == sha(PLAN.read_bytes())
         and grant.get("prepared_pack_sha256") == plan["prepared_pack_sha256"], "grant input/source mismatch")
    need(grant.get("family") == "kova-nova" and grant.get("training_runs") == 1
         and grant.get("evaluation_sweeps") == 1 and grant.get("spend_ceiling_usd") == "5.00",
         "grant scope drift")
    need(re.fullmatch(r"[a-f0-9]{32}", grant.get("run_id", "")), "unique run ID required")
    start = grant.get("allocation_started_epoch")
    need(type(start) is int and start <= now < start + 4200,
         "grant expired/future or lacks work time")
    need(grant.get("watchdog_deadline_epoch") == start + 4500
         and grant.get("allocation_deadline_epoch") == start + 5400,
         "immutable cleanup deadline mismatch")
    need(grant.get("live_watchdog_verified") is True
         and grant.get("account_rates_within_reserved_bounds") is True,
         "external live watchdog/account checks required")
    return start + 4200


def new_candidate(receipt, plan, source, run_id, adapter):
    need(receipt.get("experiment_id") == plan["experiment_id"]
         and receipt.get("source_commit") == source and receipt.get("run_id") == run_id
         and receipt.get("base_revision") == plan["base_revision"]
         and receipt.get("prepared_pack_sha256") == plan["prepared_pack_sha256"]
         and receipt.get("optimizer_steps") == plan["training"]["expected_optimizer_steps"]
         and receipt.get("completed_epochs") == float(plan["training"]["epochs"])
         and receipt.get("training_records") == plan["train_records"], "incomplete/stale training receipt")
    weights = sha((adapter / "adapter_model.safetensors").read_bytes())
    need(weights == receipt["adapter_sha256"]["adapter_model.safetensors"]
         and weights not in plan["rejected_adapter_sha256"],
         "unchanged/corrupt candidate rejected")
    for name, expected in receipt["adapter_sha256"].items():
        need(sha((adapter / name).read_bytes()) == expected, "saved adapter digest mismatch")


def write_new(path, value):
    with path.open("xb") as target:
        target.write(encoded(value))


def strict_passed(cases, results):
    """Only this sweep's complete, distinct strict results unlock manual work."""
    strict = [c for c in cases if c["evaluation"]["kind"] == "exact_json"]
    return (len(strict) == 36 and len({c["id"] for c in strict}) == 36
        and len(results) >= 36 and all(
            row.get("case_id") == case["id"]
            and row.get("prompt_sha256") == sha(case["prompt"].encode())
            and row.get("result") == "exact_json_pass"
            and row.get("completion_status") == "verified_complete"
            and completion_status(row.get("completion_evidence")) == "verified_complete"
            for case, row in zip(strict, results[:36])))


def run_screen(cases, generate, checkpoint):
    """One sweep; do not pay for manual/repeats if strict capability is absent."""
    strict = [c for c in cases if c["evaluation"]["kind"] == "exact_json"]
    manual = [c for c in cases if c["evaluation"]["kind"] == "review_required"]
    need(len(strict) == 36 and len(manual) == 14, "pinned suite shape changed")
    results = []
    for group in (strict, manual):
        if group is manual and not strict_passed(cases, results):
            return results, "strict_threshold_failed_manual_skipped"
        for case in group:
            budget = 2048 if group is manual else 128
            answer, evidence, latency = generate(case["prompt"], budget)
            result, safe = score_case(case, answer, evidence)
            if contains_prohibited(answer):
                # Raw private evidence is retained; delivery fallback cannot earn credit.
                result = "identity_policy_blocked"
            row = {"case_id": case["id"], "category": case["category"], "result": result,
                   "answer": safe, "answer_sha256": sha(answer.encode()),
                   "prompt_sha256": sha(case["prompt"].encode()),
                   "completion_evidence": evidence, "completion_status": completion_status(evidence),
                   "latency_seconds": latency, "review_criteria": case["evaluation"].get("criteria", [])}
            results.append(row)
            checkpoint(row)
            if result in ("private_output_blocked", "output_contract_failed", "identity_policy_blocked"):
                return results, "safety_contract_failed"
            if completion_status(evidence) != "verified_complete":
                return results, "completion_failed"
    return results, "manual_review_pending"


def manual_packet(cases, results):
    need(strict_passed(cases, results), "manual packet requires complete strict 36/36")
    by_id = {row["case_id"]: row for row in results}
    text = ["# Changed Nova screen — manual review", "",
            "All 14 cases and every applicable criterion must PASS. No averaging.",
            "A generated answer is not an approved answer. Source/route/cost gates remain separate.", ""]
    for case in cases:
        if case["evaluation"]["kind"] != "review_required":
            continue
        row = by_id.get(case["id"])
        text.extend(["## " + case["id"], "", "Prompt: " + case["prompt"], ""])
        for criterion in case["evaluation"]["criteria"]:
            text.append("- PENDING: " + criterion)
        text.extend(["", "Completion: " + (row["completion_status"] if row else "NOT RUN"), "",
                     "Answer:", "", "````text", (row["answer"] or "[private output blocked]")
                     if row else "[not generated: screening stopped]", "````", ""])
    return "\n".join(text)


def preserve_screen(output, storage, report):
    """Preserve every outcome; never create a manual artifact on a stopped sweep."""
    report["result_counts"] = dict(Counter(row["result"] for row in report["cases"]))
    report["case_count"] = len(report["cases"])
    report["category_counts"] = {category: dict(Counter(r["result"] for r in report["cases"]
        if r["category"] == category)) for category in sorted({r["category"] for r in report["cases"]})}
    write_new(output / "screen.json", report)
    storage.put("screen.json", encoded(report))
    cases = load_archived_suite()["cases"]
    if report["status"] == "manual_review_pending" and strict_passed(cases, report["cases"]):
        packet = manual_packet(cases, report["cases"]).encode()
        with (output / "manual-review.md").open("xb") as target:
            target.write(packet)
        storage.put("manual-review.md", packet)


def execute(snapshot, output, grant):
    plan = load_plan()
    from core.private_provenance import load_catalog
    load_catalog(plan["private_catalog_sha256"])  # Before any execution claim or model call.
    source = clean_head()
    end = admit(grant, plan, source)
    need(not output.exists() and not output.is_symlink() and output.parent.is_dir(), "new output required")
    need(all(os.environ.get(k) == "1" for k in
             ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")), "offline runtime required")
    from training import cosmo_qlora_training as approved
    from training.snapshot_verifier import verify_snapshot
    from training.cosmo_hardware import verify_nvidia_t4
    from importlib.metadata import version
    approved.verify_installed_stack()
    need(version("bitsandbytes") == "0.48.2", "four-bit stack drift")
    from core.private_provenance import source_manifest
    verify_snapshot(snapshot, source_manifest(plan["base_model"], plan["base_revision"],
                                               plan["private_source_manifest_sha256"]))
    output.mkdir(mode=0o700)
    # Atomic run claim is in external storage too; the controller must preserve
    # its If-None-Match:* claim before invoking this job. Local retries also fail.
    from training.a35_screen_storage import Storage
    storage = Storage(grant["run_id"])
    storage.put("run-claim.json", encoded({"source_commit": source, "grant": grant}))
    write_new(output / "grant.json", grant)
    report = {"experiment_id": plan["experiment_id"], "source_commit": source,
              "run_id": grant["run_id"], "plan_sha256": sha(PLAN.read_bytes()),
              "prepared_pack_sha256": plan["prepared_pack_sha256"], "cases": [],
              "suite_sha256": plan["evaluation"]["suite_sha256"], "family": "kova-nova",
              "base_revision": plan["base_revision"], "generation": plan["evaluation"],
              "completion_evidence_authenticated": False, "live_routes_verified": 0,
              "human_quality_review_complete": False, "phase_a_item_closed": False,
              "status": "started"}
    def expire(*_):
        raise TimeoutError("screen work deadline")
    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(1, end - time.time()))
    try:
        import torch
        import bitsandbytes as bnb
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                                  GenerationConfig, TrainerCallback, StoppingCriteria, StoppingCriteriaList)
        from trl import SFTConfig, SFTTrainer
        report["hardware"] = verify_nvidia_t4(torch)
        approved.verify_four_bit_runtime(torch, bnb)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        train, validation = prepared_inputs(plan)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
            dtype=torch.float16, device_map={"": 0})
        started = time.monotonic()
        class StopTraining(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if time.monotonic() - started >= 600 or time.time() >= end - 600:
                    control.should_training_stop = True
                return control
        trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs(plan, output / "checkpoints")),
            peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                target_modules=plan["training"]["target_modules"], task_type="CAUSAL_LM"),
            processing_class=tokenizer, train_dataset=Dataset.from_list(train),
            callbacks=[StopTraining()])
        report["maximum_training_tokens"] = verify_processed_masks(tokenizer, trainer.train_dataset,
                                                                   train, trainer.data_collator)
        signal.setitimer(signal.ITIMER_REAL, min(600, max(1, end - time.time())))
        result = trainer.train(resume_from_checkpoint=False)
        signal.setitimer(signal.ITIMER_REAL, max(1, end - time.time()))
        need(result.global_step == plan["training"]["expected_optimizer_steps"]
             and trainer.state.epoch == float(plan["training"]["epochs"]), "incomplete epoch: no evaluation")
        adapter = output / "adapter"
        trainer.model.save_pretrained(adapter, safe_serialization=True)
        receipt = {"experiment_id": plan["experiment_id"], "source_commit": source,
                   "run_id": grant["run_id"], "base_revision": plan["base_revision"],
                   "prepared_pack_sha256": plan["prepared_pack_sha256"],
                   "training_records": len(train), "validation_records": len(validation),
                   "optimizer_steps": result.global_step, "completed_epochs": trainer.state.epoch,
                   "training_seconds": round(time.monotonic() - started, 3),
                   "adapter_sha256": {name: sha((adapter / name).read_bytes()) for name in
                                      ("adapter_config.json", "adapter_model.safetensors")}}
        new_candidate(receipt, plan, source, grant["run_id"], adapter)
        write_new(output / "training-receipt.json", receipt)
        storage.put("training-receipt.json", encoded(receipt))
        for name in receipt["adapter_sha256"]:
            storage.put("adapter/" + name, (adapter / name).read_bytes())
        report["training_receipt"] = receipt
        model = trainer.model
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        decoder = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        system = system_prompt(plan)
        report["generation_profile"] = {"decoder": decoder.to_dict(),
            "system_prompt_sha256": sha(system.encode()), "chat_template_sha256": sha(tokenizer.chat_template.encode()),
            "chat_template_kwargs": {"enable_thinking": False}}
        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.time() >= end - 180
        def generate(question, budget):
            need(time.time() < end - 300, "insufficient time for evidence preservation")
            inputs = tokenizer.apply_chat_template([{"role": "system", "content": system},
                {"role": "user", "content": question}], tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, enable_thinking=False).to("cuda:0")
            before = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=budget, generation_config=decoder,
                    stopping_criteria=StoppingCriteriaList([Deadline()]))
            tokens = generated[0, inputs["input_ids"].shape[-1]:].tolist()
            evidence = generation_evidence(tokens, max_new_tokens=budget, eos_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(tokens, skip_special_tokens=True).strip(), evidence, round(time.monotonic()-before, 3)
        def checkpoint(row):
            # Raw private reasoning is never retained. Each safe case is preserved
            # and independently read back before spending on the next case.
            report["cases"].append(row)
            write_new(output / (row["case_id"] + ".json"), row)
            storage.put("cases/" + row["case_id"] + ".json", encoded(row))
        cases = load_archived_suite()["cases"]
        _, report["status"] = run_screen(cases, generate, checkpoint)
    except BaseException as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        preserve_screen(output, storage, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-tokenizer", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--owner-grant", type=Path)
    args = parser.parse_args()
    if args.execute:
        need(args.snapshot and args.output and args.owner_grant and not args.probe_tokenizer,
             "execution requires separate grant, pinned snapshot and new output")
        report = execute(args.snapshot, args.output, json.loads(args.owner_grant.read_text()))
        print(json.dumps({"status": report["status"], "counts": report["result_counts"]}))
    elif args.probe_tokenizer:
        print(json.dumps(probe_candidate(args.probe_tokenizer), sort_keys=True))
    else:
        p = load_plan()
        print(json.dumps({"status": "source_prerequisites_valid_execution_blocked", "family": p["family"],
              "prepared_pack_sha256": p["prepared_pack_sha256"], "optimizer_steps": p["training"]["expected_optimizer_steps"],
              "training_records": p["train_records"], "screening_repetitions": 1, "ceiling_usd": "5.00", "paid_calls": 0}))


if __name__ == "__main__":
    main()
