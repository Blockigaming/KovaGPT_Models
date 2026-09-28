"""Measure held-out completion loss for an experimental Azure T4 adapter."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

from training import cosmo_qlora_training as approved
from training import three_family_contract as contract
from training.cosmo_runtime_probe import completion_tokens
from training.snapshot_verifier import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
TRAINING_COMMIT = "240070e7db0cead3ce912780807f3091466bdf97"
LENGTHS = {"kova-cosmo": 1024, "kova-orion": 1024, "kova-nova": 768}
ACCOUNT = "kova42c1a27"
CONTAINER = "cosmo-adapters"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def upload_report(payload: bytes, family: str) -> dict:
    blob = f"gpu-experimental/2026-09-28/{family}/heldout-evaluation.json"
    checksum = hashlib.sha256(payload).hexdigest()
    metadata = ("http://169.254.169.254/metadata/identity/oauth2/token"
                "?api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F")
    no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with no_proxy.open(urllib.request.Request(metadata, headers={"Metadata": "true"}),
                       timeout=15) as response:
        token = json.load(response)["access_token"]
    url = (f"https://{ACCOUNT}.blob.core.windows.net/{CONTAINER}/"
           + urllib.parse.quote(blob, safe="/"))
    request = urllib.request.Request(url, data=payload, method="PUT", headers={
        "Authorization": "Bearer " + token,
        "x-ms-version": "2023-11-03", "x-ms-blob-type": "BlockBlob",
        "x-ms-blob-content-type": "application/json",
        "x-ms-meta-sha256": checksum, "If-None-Match": "*",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 201:
            raise ValueError("evaluation Blob upload failed")
        etag = response.headers["ETag"]
    with urllib.request.urlopen(urllib.request.Request(url, headers={
            "Authorization": "Bearer " + token, "x-ms-version": "2023-11-03",
            }), timeout=60) as response:
        if response.read() != payload:
            raise ValueError("evaluation Blob readback mismatch")
    return {"blob": blob, "sha256": checksum, "etag": etag,
            "independent_readback_verified": True}


def evaluate(family: str, snapshot: Path, candidate: Path, output: Path,
             deadline_utc: str) -> dict:
    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    if (family not in LENGTHS or deadline.tzinfo is None or
            datetime.now(timezone.utc) + timedelta(minutes=12) >= deadline or
            output.exists() or not output.parent.is_dir()):
        raise ValueError("bounded evaluation inputs invalid")
    approved.verify_installed_stack()
    from importlib.metadata import version
    if version("bitsandbytes") != "0.48.2":
        raise ValueError("four-bit dependency drift")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from training.cosmo_hardware import verify_nvidia_t4

    verify_nvidia_t4(torch)
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    pinned = lineage["families"][family]
    verify_snapshot(snapshot, contract._pinned_manifest(family, pinned))
    receipt = json.loads((candidate / "receipt.json").read_text())
    dataset_sha = contract.load_json(
        ROOT / "config/kova-three-family-dataset.v2.json")["dataset_sha256"]
    if (receipt.get("kind") !=
            f"{family.replace('-', '_')}_gpu_t4_nf4_lora_experiment" or
            receipt.get("source_commit") not in (
                TRAINING_COMMIT, "288b7dd08c076bdbeaf7f2114049ae9fa4c3f439") or
            receipt.get("status") != "complete" or
            receipt.get("optimizer_steps") != 7 or
            receipt.get("train_records") != 27 or
            receipt.get("validation_records") != 15 or
            receipt.get("dataset_sha256") != dataset_sha or
            receipt.get("base_revision") != pinned["immutable_revision"]):
        raise ValueError("unrecognized completed T4 adapter")
    hashes = receipt["adapter_sha256"]
    adapter = candidate / "adapter"
    if (set(hashes) != {"adapter_config.json", "adapter_model.safetensors"} or
            any(not (adapter / name).is_file() or digest(adapter / name) != expected
                for name, expected in hashes.items())):
        raise ValueError("trained adapter hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    _, validation = approved.prepared_rows()
    cases = [completion_tokens(tokenizer, row, LENGTHS[family])
             for row in validation]
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    base.eval()

    def score(model) -> tuple[float, int, list[float]]:
        total, count, individual = 0.0, 0, []
        with torch.inference_mode():
            for ids, labels in cases:
                source = torch.tensor([ids], dtype=torch.long, device="cuda:0")
                target = torch.tensor([labels], dtype=torch.long, device="cuda:0")
                loss = float(model(input_ids=source, labels=target,
                                   use_cache=False).loss)
                tokens = sum(value != -100 for value in labels[1:])
                if not (0 < loss < 100 and tokens > 0):
                    raise ValueError("invalid held-out completion loss")
                total += loss * tokens
                count += tokens
                individual.append(round(loss, 8))
        return total / count, count, individual

    base_loss, base_tokens, base_cases = score(base)
    adapted = PeftModel.from_pretrained(
        base, str(adapter), is_trainable=False, local_files_only=True)
    adapted.eval()
    adapter_loss, adapter_tokens, adapter_cases = score(adapted)
    if base_tokens != adapter_tokens:
        raise ValueError("held-out token mismatch")
    report = {
        "kind": "experimental_t4_nf4_heldout_completion_loss",
        "family": family, "training_commit": receipt["source_commit"],
        "base_revision": pinned["immutable_revision"],
        "dataset_sha256": dataset_sha, "adapter_sha256": hashes,
        "validation_records": len(validation), "completion_tokens": base_tokens,
        "base_mean_completion_loss": round(base_loss, 8),
        "adapter_mean_completion_loss": round(adapter_loss, 8),
        "adapter_minus_base_loss": round(adapter_loss - base_loss, 8),
        "base_case_losses": base_cases, "adapter_case_losses": adapter_cases,
        "human_quality_review_complete": False,
        "signed_controller_pilot": False, "production_routing_approved": False,
    }
    payload = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    output.write_bytes(payload)
    report["azure_report"] = upload_report(payload, family)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(LENGTHS), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.family, args.snapshot, args.candidate,
                              args.output, args.deadline_utc), sort_keys=True))


if __name__ == "__main__":
    main()
