"""Validate/prepare synthetic Kova identity examples; never load or train a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from release.model_revisions import MODEL_SOURCE_REFERENCES

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = 'config/kova-cosmo-pilot.v1.json'
PROMPT_PATH = 'prompts/kova-identity.v2.txt'
DATA_PATH = 'data/kova-identity-pilot.v1.jsonl'
REVIEW_PATH = 'data/kova-identity-pilot-review.v1.json'
MAX_BYTES = 1024 * 1024


class PilotError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise PilotError('identity pilot source rejected')


def same(value: object, expected: object) -> None:
    need(type(value) is type(expected))
    if isinstance(expected, dict):
        need(value.keys() == expected.keys())
        for key in expected:
            same(value[key], expected[key])
    elif isinstance(expected, list):
        need(len(value) == len(expected))
        for a, b in zip(value, expected):
            same(a, b)
    else:
        need(value == expected)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def unique(pairs: list) -> dict:
    value = {}
    for key, item in pairs:
        need(key not in value)
        value[key] = item
    return value


def parse(text: str) -> object:
    def reject(_value):
        raise PilotError('identity pilot source rejected')
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=reject, parse_float=reject)


def read_asset(root: Path, relative: str) -> bytes:
    with (root / relative).open('rb') as stream:
        body = stream.read(MAX_BYTES + 1)
    need(0 < len(body) <= MAX_BYTES)
    return body


def load(root: Path = ROOT) -> tuple[dict, str, list]:
    """Check fixed source paths, fingerprints and training/validation separation."""
    try:
        prompt_bytes = read_asset(root, PROMPT_PATH)
        data_bytes = read_asset(root, DATA_PATH)
        review_bytes = read_asset(root, REVIEW_PATH)
        plan = parse(read_asset(root, PLAN_PATH).decode('utf-8'))
        ref = MODEL_SOURCE_REFERENCES['work-cosmo']
        same(plan, {
            'schema_version': 1, 'status': 'source_preparation_only',
            'display_name': 'Kova Cosmo', 'model_slot': 'work-cosmo', 'method': 'lora',
            'base_model': ref.model, 'base_revision': ref.revision,
            'prompt_path': PROMPT_PATH, 'dataset_path': DATA_PATH,
            'review_path': REVIEW_PATH,
            'prompt_sha256': digest(prompt_bytes), 'dataset_sha256': digest(data_bytes),
            'review_sha256': digest(review_bytes),
            'provenance': {'source': 'synthetic_examples_prepared_for_kova',
                           'private_customer_data': False, 'human_review_complete': False},
            'proposed_compute': 'Standard_NC4as_T4_v3',
            'hyperparameters': None, 'approved_budget_usd': None,
            'execution': {'model_download_authorized': False,
                          'training_authorized': False, 'deployment_authorized': False},
            'upstream_notices_must_be_preserved': True,
            'trained_adapter_sha256': None, 'phase_b_ready': False,
        })
        prompt = prompt_bytes.decode('utf-8')
        need(prompt.startswith('You are Kova,'))
        rows = []
        ids, questions = set(), set()
        counts = {'train': 0, 'validation': 0}
        for line in data_bytes.decode('utf-8').splitlines():
            need(bool(line.strip()))
            row = parse(line)
            need(type(row) is dict)
            fixture = 'runtime_fixture' in row
            need(set(row) == {'id', 'split', 'kind', 'messages'} |
                 ({'runtime_fixture'} if fixture else set()))
            need(type(row['id']) is str and bool(row['id']) and row['id'] not in ids)
            need(type(row['split']) is str and row['split'] in counts)
            need(type(row['kind']) is str and row['kind'] in
                 {'identity', 'task', 'truthfulness', 'privacy', 'provenance'})
            messages = row['messages']
            need(type(messages) is list and len(messages) == 2)
            for item, role in zip(messages, ('user', 'assistant')):
                need(type(item) is dict and set(item) == {'role', 'content'})
                need(item['role'] == role and type(item['content']) is str)
                need(0 < len(item['content'].strip()) <= 8192)
            question = ' '.join(messages[0]['content'].split()).casefold()
            need(question not in questions)
            if fixture:
                need(row['split'] == 'validation' and row['kind'] == 'provenance')
                same(row['runtime_fixture'], {
                    'slot': ref.slot, 'upstream_model': ref.model,
                    'upstream_revision': ref.revision})
            else:
                need(row['kind'] != 'provenance')
            ids.add(row['id'])
            questions.add(question)
            counts[row['split']] += 1
            rows.append(row)
        need(counts == {'train': 24, 'validation': 12})
        return plan, prompt, rows
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, RecursionError):
        raise PilotError('identity pilot source rejected') from None


def format_messages(prompt: str, row: dict) -> list[dict]:
    """Format a validated pilot row, preserving hypothetical provenance context."""
    system = prompt
    if 'runtime_fixture' in row:
        # A source-controlled hypothetical evaluation, NEVER a real attestation.
        system += ('\nOffline evaluation fixture only. For this hypothetical session, '
                   'trusted runtime metadata is: ' +
                   json.dumps(row['runtime_fixture'], sort_keys=True) + '\n')
    return [{'role': 'system', 'content': system},
            *(dict(message) for message in row['messages'])]


def prepare(root: Path = ROOT, output: Path | None = None) -> dict:
    plan, prompt, rows = load(root)
    parts = {'train': [], 'validation': []}
    for row in rows:
        parts[row['split']].append(json.dumps({
            'messages': format_messages(prompt, row)
        }, ensure_ascii=False, separators=(',', ':')))
    encoded = {key: ('\n'.join(lines) + '\n').encode('utf-8')
               for key, lines in parts.items()}
    report = {
        'status': 'source_prepared_not_trained',
        'display_name': plan['display_name'], 'model_slot': plan['model_slot'],
        'prompt_sha256': plan['prompt_sha256'], 'dataset_sha256': plan['dataset_sha256'],
        'review_sha256': plan['review_sha256'],
        'files': {key: {'filename': key + '.jsonl', 'records': len(parts[key]),
                        'sha256': digest(body)} for key, body in encoded.items()},
        'human_review_complete': False, 'model_outputs_evaluated': False,
        'model_weights_downloaded': False, 'training_started': False,
        'paid_resources_started': False, 'runtime_integrated': False,
        'phase_b_ready': False, 'closed_checklist_ids': [],
    }
    if output is not None:
        try:
            output.mkdir(parents=True, exist_ok=False)
            for split, body in encoded.items():
                with (output / (split + '.jsonl')).open('xb') as stream:
                    stream.write(body)
            # Write receipt last. Existing directories/files are never overwritten.
            with (output / 'manifest.json').open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(report, indent=2) + '\n')
        except OSError:
            raise PilotError('identity pilot output not written; use a new folder') from None
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New folder for data only; no training')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(output=args.output), sort_keys=True))
    except PilotError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
