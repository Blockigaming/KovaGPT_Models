"""Inactive behavior-v10 screen binding; external fresh authority is required."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from evaluation.a35_nova_behavior_data import load_manifest, prepared_rows
from training import a35_nova_screen as parent

ROOT = parent.ROOT
PLAN = ROOT/'config/a35-nova-behavior-screen.v1.json'
PARENT_PLAN = ROOT/'config/a35-nova-screen.v1.json'
CONSUMED_ATTEMPT = '5d4f68aa6d0a4560b50873ca8beef278'
OLD_RESERVED_ATTEMPT = '0f3e4ca276d745f3bb4396c36bdda56c'
ORIGINAL_LOAD = parent.load_plan
ORIGINAL_ADMIT = parent.admit
system_prompt = parent.system_prompt
strict_passed = parent.strict_passed
load_archived_suite = parent.load_archived_suite
score_case = parent.score_case
new_candidate = parent.new_candidate


def prepared_inputs(plan):
    rows = prepared_rows()
    parent.need(parent.sha(parent.encoded(rows)) == plan['prepared_pack_sha256'],
                'behavior input pack drift')
    train = [row for _, split, row in rows if split == 'train']
    validation = [row for _, split, row in rows if split == 'validation']
    parent.need((len(train),len(validation)) == (73,61), 'behavior split drift')
    return train, validation


def load_plan():
    original = ORIGINAL_LOAD(PARENT_PLAN)
    plan = json.loads(PLAN.read_text())
    manifest = load_manifest()
    allowed = {'experiment_id','prepared_pack_sha256','file_sha256',
        'parent_plan_sha256','behavior_manifest_sha256','input_validation_path',
        'input_validation_sha256','consumed_attempt_replay_authorized'}
    parent.need(set(plan)-set(original) == allowed-set(original), 'unexpected behavior plan fields')
    parent.need(all(plan[k] == value for k,value in original.items() if k not in allowed),
                'frozen recipe, evaluation, identity, resources or cost changed')
    parent.need(plan['experiment_id'] == manifest['candidate'] == 'a35-nova-behavior-v10',
                'candidate binding drift')
    parent.need(plan['parent_plan_sha256'] == parent.sha(PARENT_PLAN.read_bytes()) and
        plan['behavior_manifest_sha256'] == parent.sha((ROOT/'config/a35-nova-behavior-correction.v1.json').read_bytes()),
        'parent/correction manifest drift')
    parent.need(plan['consumed_attempt_replay_authorized'] is False and
                plan['execution_authorized'] is False, 'source cannot grant execution')
    parent.need(all(plan['file_sha256'].get(k) == v for k,v in original['file_sha256'].items()),
                'historical input pin changed')
    for name, pin in plan['file_sha256'].items():
        parent.need(parent.sha((ROOT/name).read_bytes()) == pin, 'behavior source pin drift: '+name)
    path = ROOT/plan['input_validation_path']
    parent.need(plan['input_validation_path'] == 'evaluations/a35-nova-behavior-input-validation.v1.json'
        and parent.sha(path.read_bytes()) == plan['input_validation_sha256'], 'CPU proof drift')
    proof = json.loads(path.read_text())
    parent.need(proof['status'] == 'PASS' and proof['candidate'] == manifest['candidate'] and
        proof['prepared_pack_sha256'] == plan['prepared_pack_sha256'] and
        proof['scopes']['changed_36']['records_verified'] == 36 and
        proof['scopes']['complete_134']['records_verified'] == 134 and
        proof['scopes']['complete_134']['maximum_tokens'] <= plan['training']['sequence_length'] and
        proof['probe_sha256'] == parent.sha((ROOT/'training/probe_a35_nova_behavior_inputs.py').read_bytes()) and
        proof['unchanged_identity_overrides'] == 18 and proof['unchanged_prepared_records'] == 98 and
        proof['training_recipe'] == plan['training'] and proof['template_kwargs'] == {'enable_thinking':False} and
        all(proof[k] is True for k in ('actual_trl_preparation','actual_trl_collation',
            'prompt_padding_masks_correct','targets_complete','eos_handling_verified')) and
        proof['model_calls'] == 0 and proof['training_started'] is False,
        'complete actual CPU input proof required')
    prepared_inputs(plan)
    return plan


def admit(grant, plan, source, now=None):
    parent.need(grant.get('run_id') not in (CONSUMED_ATTEMPT, OLD_RESERVED_ATTEMPT),
                'consumed or differently bound prior attempt cannot be reused')
    parent.need(grant.get('prior_consumed_attempt_id') == CONSUMED_ATTEMPT and
        grant.get('prior_no_second_paid_run_explicitly_superseded') is True and
        grant.get('authorization_scope') == 'NEW_BEHAVIOR_V10_ATTEMPT_ONLY' and
        grant.get('authorization_id') not in (None,'nova-three-epoch-fc83f05-owner-20261005'),
        'separate explicit owner authority required; no second paid run is currently authorized')
    return ORIGINAL_ADMIT(grant, plan, source, now=now)


def execute(snapshot, output, grant):
    plan = load_plan()
    inputs = prepared_inputs(plan)
    def fixed_inputs(received):
        parent.need(received == plan, 'runtime plan drift')
        return deepcopy(inputs)
    # Reuse the unchanged, bounded trainer, raw strict grader, completion checks,
    # conditional manual path and protected-storage receipts with exact v10 inputs.
    with ExitStack() as context:
        context.enter_context(patch.object(parent,'PLAN',PLAN))
        context.enter_context(patch.object(parent,'load_plan',lambda:deepcopy(plan)))
        context.enter_context(patch.object(parent,'prepared_inputs',fixed_inputs))
        context.enter_context(patch.object(parent,'admit',admit))
        return parent.execute(snapshot, output, grant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--owner-grant',type=Path)
    args = parser.parse_args()
    if args.execute:
        parent.need(args.snapshot and args.output and args.owner_grant,
                    'separate explicit grant, snapshot and new output required')
        report = execute(args.snapshot,args.output,json.loads(args.owner_grant.read_text()))
    else:
        plan = load_plan()
        report = {'status':'INACTIVE_INPUT_VALIDATED','candidate':plan['experiment_id'],
            'plan_sha256':parent.sha(PLAN.read_bytes()),'prepared_pack_sha256':plan['prepared_pack_sha256'],
            'training_records':73,'validation_records':61,'model_calls':0,
            'quality_status':'UNMEASURED','execution_authorized':False}
    print(json.dumps(report,sort_keys=True))


if __name__ == '__main__':
    main()
