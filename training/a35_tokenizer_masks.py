"""Verify complete revised input masks with pinned Qwen assets and real TRL.

Only tokenizer JSON assets may be fetched. No model construction, weights,
forward/backward pass, optimizer, inference, training or GPU is permitted.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

from evaluation.a35_correction_data import MANIFEST, ROOT, prepared_revision_rows
from training.probe_cosmo_loss_masks import (offline_cpu, verify_environment,
                                             verify_record, verify_batch)
from training.probe_cosmo_tokenizer import ASSETS, REVISION


def digest(body):
    return hashlib.sha256(body).hexdigest()


def family_bindings():
    result = []
    for family,size in (("kova-cosmo","0.6b"),("kova-orion","1.7b"),("kova-nova","4b")):
        manifest_path = ROOT/f"config/qwen3-{size}-download-manifest.v1.json"
        source = json.loads(manifest_path.read_text())
        assets = {f["path"]:f for f in source["files"] if f["path"] in ASSETS}
        if set(assets)!=set(ASSETS) or any(assets[name]["sha256"]!=expected for name,expected in ASSETS.items()):
            raise ValueError("family tokenizer source mismatch")
        recipe = json.loads((ROOT/f"config/{family}-qlora.v1.json").read_text())
        result.append({"family":family,"model":source["model"],"revision":source["revision"],
                       "download_manifest_sha256":digest(manifest_path.read_bytes()),
                       "sequence_budget":recipe["training"]["maximum_sequence_length"],"assets":assets})
    return result


def fetch_assets(directory):
    """Explicit free tokenizer-only fetch from an immutable official revision."""
    bindings = family_bindings()
    directory.mkdir(parents=True,exist_ok=True)
    for name,expected in ASSETS.items():
        target=directory/name
        if target.exists() and digest(target.read_bytes())==expected: continue
        url=f"https://huggingface.co/Qwen/Qwen3-0.6B/resolve/{REVISION}/{name}"
        with urlopen(url,timeout=30) as response:
            body=response.read(bindings[0]["assets"][name]["bytes"]+1)
        if len(body)!=bindings[0]["assets"][name]["bytes"] or digest(body)!=expected:
            raise ValueError("downloaded tokenizer asset rejected")
        target.write_bytes(body)


def probe(directory):
    installed=verify_environment()
    bindings=family_bindings()
    assets={}
    for name,expected in ASSETS.items():
        with (directory/name).open("rb") as stream: body=stream.read(20_000_001)
        if len(body)>20_000_000 or digest(body)!=expected: raise ValueError("tokenizer asset mismatch")
        assets[name]=body
    source=prepared_revision_rows()
    manifest=json.loads(MANIFEST.read_text())
    reports=[]
    with offline_cpu(), tempfile.TemporaryDirectory(prefix="a35-mask-only-") as temporary:
        local=Path(temporary)/"tokenizer"; local.mkdir()
        for name,body in assets.items(): (local/name).write_bytes(body)
        import torch
        from datasets import Dataset, disable_progress_bars
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, Trainer
        from trl import SFTConfig, SFTTrainer
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        from unittest.mock import patch
        if torch.version.cuda is not None: raise ValueError("CPU-only PyTorch required")
        disable_progress_bars()
        with ExitStack() as guard:
            for owner,attribute in ((AutoModel,"from_pretrained"),(AutoModel,"from_config"),
                                    (AutoModelForCausalLM,"from_pretrained"),(AutoModelForCausalLM,"from_config"),
                                    (Trainer,"__init__"),(SFTTrainer,"__init__"),(SFTTrainer,"train")):
                guard.enter_context(patch.object(owner,attribute,side_effect=ValueError("model/training forbidden")))
            tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
            for binding in bindings:
                settings=SFTConfig(output_dir=str(Path(temporary)/"unused"),use_cpu=True,
                                   fp16=False,bf16=False,max_length=binding["sequence_budget"],
                                   packing=False,completion_only_loss=True,report_to="none")
                preparer=object.__new__(SFTTrainer)
                preparer._tokenizer=tokenizer; preparer.chat_template=tokenizer.chat_template
                preparer.completion_only_loss=True
                collator=DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
                results=[]; padding=0; batches=0
                for split in ("train","validation"):
                    originals={identifier:row for identifier,s,row in source if s==split}
                    dataset=Dataset.from_list([dict(row,probe_id=identifier) for identifier,row in originals.items()])
                    processed=preparer._prepare_dataset(dataset,tokenizer,settings,False,None,split)
                    if len(processed)!=len(originals) or set(processed["probe_id"])!=set(originals):
                        raise ValueError("TRL record loss/duplication")
                    for record in processed:
                        row=originals[record["probe_id"]]
                        prompt_ids=tokenizer.apply_chat_template(row["prompt"],tokenize=True,
                                      add_generation_prompt=True,return_dict=False,**row["chat_template_kwargs"])
                        full_ids=tokenizer.apply_chat_template(row["prompt"]+row["completion"],tokenize=True,
                                      add_generation_prompt=False,return_dict=False,**row["chat_template_kwargs"])
                        # Derive boundary independently from TRL's completion_mask.
                        labels=[-100]*len(prompt_ids)+full_ids[len(prompt_ids):]
                        batch=collator([record])
                        independently_labeled=dict(record,labels=batch["labels"][0].tolist())
                        boundary=verify_record(independently_labeled,full_ids,prompt_ids,settings.max_length)
                        eos_positions=[index for index,token in enumerate(full_ids)
                                       if index>=boundary and token==tokenizer.eos_token_id]
                        if (independently_labeled["labels"]!=labels or not eos_positions or
                            tokenizer.decode(full_ids[eos_positions[-1]+1:],skip_special_tokens=False).strip()):
                            raise ValueError("complete assistant objective/EOS rejected")
                        results.append({"id":record["probe_id"],"split":split,
                                        "tokens":len(full_ids),"prompt_tokens":boundary,
                                        "completion_tokens":len(full_ids)-boundary,
                                        "completion_eos_labelled":True,
                                        "input_ids_sha256":digest(json.dumps(full_ids).encode()),
                                        "labels_sha256":digest(json.dumps(labels).encode())})
                    # Mixed-length, multirow collation checks attention and padding.
                    for start in range(0,len(processed),4):
                        records=[]
                        for index in range(start,min(start+4,len(processed))):
                            record=dict(processed[index])
                            record["labels"]=collator([record])["labels"][0].tolist()
                            records.append(record)
                        padding+=verify_batch(collator(records),records,tokenizer.pad_token_id); batches+=1
                if not padding: raise ValueError("padding not exercised")
                results.sort(key=lambda r:r["id"])
                reports.append({key:value for key,value in binding.items() if key!="assets"} |
                               {"records_verified":len(results),"maximum_tokens":max(r["tokens"] for r in results),
                                "maximum_training_tokens":max(r["tokens"] for r in results if r["split"]=="train"),
                                "batches_verified":batches,"padding_tokens_verified":padding,"records":results})
    return {"schema_version":1,"status":"pinned_tokenizer_real_trl_complete_masks_verified",
            **{field:manifest[field] for field in ("prompt_sha256","dataset_sha256","review_sha256","validation_overrides_sha256")},
            "approved_corpus_sha256":digest((ROOT/"data/kova-identity-shared.v2.jsonl").read_bytes()),
            "tokenizer_asset_sha256":ASSETS,"template_kwargs":{"enable_thinking":False},
            "software":installed,"cpu_lock_sha256":digest((ROOT/"requirements/cosmo-loss-mask-py312-linux-cpu.lock").read_bytes()),
            "probe_sha256":digest(Path(__file__).read_bytes()),"families":reports,
            "all_completion_masks_verified":True,"all_sequence_budgets_verified":True,
            "model_weights_loaded":False,"training_started":False,"model_calls_made":0,
            "gpu_used":False,"quality_gate_closed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets",type=Path)
    parser.add_argument("--fetch",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.fetch: fetch_assets(args.assets)
    result=probe(args.assets)
    body=(json.dumps(result,indent=2,sort_keys=True)+"\n").encode()
    manifest=json.loads(MANIFEST.read_text())
    if args.check and body!=(ROOT/manifest["tokenizer_report_path"]).read_bytes():
        raise ValueError("committed tokenizer proof differs from actual pinned TRL result")
    if args.output: args.output.write_bytes(body)
    print(json.dumps({"families":[{k:v for k,v in r.items() if k!="records"} for r in result["families"]],
                      "all_completion_masks_verified":True,"all_sequence_budgets_verified":True,
                      "model_weights_loaded":False,"training_started":False},sort_keys=True))


if __name__=="__main__": main()
