"""Actual pinned TRL input preparation/collation on CPU; no model or training."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from evaluation.a35_nova_behavior_data import load_manifest, load_rows, prepared_rows
from training.a35_nova_screen import (ROOT, encoded, load_plan, prepared_nova_rows,
                                     sft_kwargs, verify_processed_masks)
from training.probe_cosmo_loss_masks import (offline_cpu, read_assets,
                                            verify_batch, verify_environment,
                                            verify_record)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probe(assets):
    software = verify_environment()
    verified_assets = read_assets(assets)
    manifest = load_manifest()
    plan = load_plan()
    parent = prepared_nova_rows(plan)
    source = prepared_rows()
    changed, annotations = load_rows()
    changed_ids = {row['id'] for row in changed}
    require(len(changed_ids) == 36, 'changed-row inventory drift')
    parent_map = {rid: (split, row) for rid, split, row in parent}
    preserved = [(rid, split, row) for rid, split, row in source if rid not in changed_ids]
    require(len(preserved) == 98 and all(parent_map[rid] == (split, row)
            for rid, split, row in preserved), 'preserved inputs changed')
    identity_ids = {json.loads(line)['id'] for line in
        (ROOT/'data/a35-kovagpt-identity-overrides.v1.jsonl').read_text().splitlines()}
    require(len(identity_ids) == 18 and identity_ids <= {rid for rid, _, _ in preserved},
            'identity override changed or lost')
    require(Counter(split for _, split, _ in source) == {'train': 73, 'validation': 61},
            'split counts changed')
    require(all(row['chat_template_kwargs'] == {'enable_thinking': False}
                for _, _, row in source), 'thinking mode drift')
    reports = {}
    with offline_cpu(), tempfile.TemporaryDirectory(prefix='nova-behavior-cpu-') as temp:
        local = Path(temp)/'tokenizer'; local.mkdir()
        for name, body in verified_assets.items():
            (local/name).write_bytes(body)
        import torch
        from datasets import Dataset, disable_progress_bars
        from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, Trainer
        from trl import SFTConfig, SFTTrainer
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        require(torch.version.cuda is None, 'CPU-only torch required')
        torch.set_num_threads(2)
        disable_progress_bars()
        with ExitStack() as guards:
            for owner, attribute in ((AutoModel,'from_pretrained'), (AutoModel,'from_config'),
                    (AutoModelForCausalLM,'from_pretrained'), (AutoModelForCausalLM,'from_config'),
                    (Trainer,'__init__'), (SFTTrainer,'__init__'), (SFTTrainer,'train')):
                guards.enter_context(patch.object(owner, attribute,
                    side_effect=ValueError('model construction and training forbidden')))
            tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True,
                                                      trust_remote_code=False)
            expected_template = json.loads(verified_assets['tokenizer_config.json'])['chat_template']
            require(tokenizer.chat_template == expected_template, 'chat template changed')
            config = SFTConfig(**sft_kwargs(plan, Path(temp)/'unused'), use_cpu=True)
            require(config.max_length == 768 and not config.packing and config.completion_only_loss,
                    'actual training configuration drift')
            preparer = object.__new__(SFTTrainer)
            preparer._tokenizer = tokenizer
            preparer.chat_template = tokenizer.chat_template
            preparer.completion_only_loss = config.completion_only_loss
            collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
            for scope, selected in [('changed_36', [r for r in source if r[0] in changed_ids]),
                                     ('complete_134', source)]:
                results = []; padding = 0; batches = 0
                for split in ('train','validation'):
                    originals = {rid: row for rid, s, row in selected if s == split}
                    rows = [dict(deepcopy(row), probe_id=rid) for rid,row in originals.items()]
                    processed = preparer._prepare_dataset(Dataset.from_list(rows), tokenizer,
                                                           config, config.packing, None, split)
                    require(len(processed) == len(originals) and
                        set(processed['probe_id']) == set(originals), 'TRL dropped or duplicated rows')
                    # Exercise the same verifier called by the actual training launcher.
                    verify_processed_masks(tokenizer, processed, list(originals.values()), collator)
                    labeled = []
                    for record in processed:
                        rid = record['probe_id']; row = originals[rid]
                        prefix = tokenizer.apply_chat_template(row['prompt'], tokenize=True,
                            add_generation_prompt=True, return_dict=False, **row['chat_template_kwargs'])
                        full = tokenizer.apply_chat_template(row['prompt']+row['completion'],
                            tokenize=True, add_generation_prompt=False, return_dict=False,
                            **row['chat_template_kwargs'])
                        batch = collator([record])
                        labels = batch['labels'][0].tolist()
                        boundary = verify_record(dict(record, labels=labels), full, prefix, config.max_length)
                        target = row['completion'][0]['content']
                        suffix = tokenizer.decode(full[boundary:], skip_special_tokens=False)
                        require(suffix == target + tokenizer.eos_token + '\n',
                                'target answer/EOS/terminal newline changed')
                        eos = [i for i in range(boundary,len(full)) if full[i] == tokenizer.eos_token_id]
                        require(len(eos) == 1 and labels[eos[0]] == tokenizer.eos_token_id,
                                'assistant EOS lost or duplicated')
                        loss_tokens = sum(v != -100 for v in labels)
                        require(loss_tokens == len(full)-boundary > 0,
                                'empty or incorrect assistant supervision')
                        require(sum(v != -100 for v in labels[1:]) == loss_tokens,
                                'causal shift loses supervised target token')
                        labeled.append(dict(record, labels=labels))
                        results.append({'id':rid,'split':split,'changed':rid in changed_ids,
                            'tokens':len(full),'prompt_tokens':boundary,'loss_tokens':loss_tokens,
                            'assistant_eos_positions':eos,'assistant_eos_labelled':True,
                            'target_answer_complete':True,'target_answer_sha256':sha(target.encode()),
                            'input_ids_sha256':sha(json.dumps(full).encode()),
                            'labels_sha256':sha(json.dumps(labels).encode())})
                    # Actual batch size one plus mixed-length padding challenge batches.
                    for size in (plan['training']['batch_size'], 4):
                        for start in range(0,len(labeled),size):
                            records = labeled[start:start+size]
                            padding += verify_batch(collator(records), records, tokenizer.pad_token_id)
                            batches += 1
                require(padding > 0, 'padding not exercised')
                reports[scope] = {'records_verified':len(results),
                    'split_counts':dict(Counter(r['split'] for r in results)),
                    'maximum_tokens':max(r['tokens'] for r in results),
                    'maximum_training_tokens':max(r['tokens'] for r in results if r['split']=='train'),
                    'minimum_loss_tokens':min(r['loss_tokens'] for r in results),
                    'maximum_loss_tokens':max(r['loss_tokens'] for r in results),
                    'total_loss_tokens':sum(r['loss_tokens'] for r in results),
                    'batches_checked':batches,'padding_tokens_checked':padding,
                    'records':sorted(results,key=lambda r:r['id'])}
            subset = {r['id']:r for r in reports['changed_36']['records']}
            require(all(subset[r['id']] == r for r in reports['complete_134']['records']
                        if r['changed']), 'changed-row tokenization depends on pack membership')
            implementation = {'trl_preparer_sha256':sha(inspect.getsource(SFTTrainer._prepare_dataset).encode()),
                'trl_collator_sha256':sha(inspect.getsource(DataCollatorForLanguageModeling).encode()),
                'training_sft_kwargs_sha256':sha(inspect.getsource(sft_kwargs).encode()),
                'training_mask_verifier_sha256':sha(inspect.getsource(verify_processed_masks).encode())}
            special = {'eos_token_id':tokenizer.eos_token_id,'pad_token_id':tokenizer.pad_token_id,
                'chat_template_sha256':sha(tokenizer.chat_template.encode()),
                'assistant_suffix':'Exact target text, one EOS, terminal newline; all supervised'}
    return {'schema_version':1,'status':'PASS','candidate':manifest['candidate'],
        'quality_status':'UNMEASURED','prepared_pack_sha256':sha(encoded(source)),
        'template_kwargs':{'enable_thinking':False},'software':software,
        'cpu_lock_sha256':sha((ROOT/'requirements/cosmo-loss-mask-py312-linux-cpu.lock').read_bytes()),
        'tokenizer_asset_sha256':{k:sha(v) for k,v in verified_assets.items()},
        'probe_sha256':sha(Path(__file__).read_bytes()),'implementation':implementation,
        'special_tokens':special,'training_recipe':plan['training'],
        'unchanged_prepared_records':98,'unchanged_identity_overrides':18,'scopes':reports,
        'actual_trl_preparation':True,'actual_trl_collation':True,
        'prompt_padding_masks_correct':True,'targets_complete':True,'eos_handling_verified':True,
        'gpu_used':False,'model_weights_loaded':False,'model_calls':0,'training_started':False,
        'execution_authorized':False,'quality_improvement_proved':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.assets)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('software','scopes','training_recipe','tokenizer_asset_sha256')},sort_keys=True))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='records'}
                      for k,v in result['scopes'].items()},sort_keys=True))


if __name__ == '__main__':
    main()
