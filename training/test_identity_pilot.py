"""Source/data checks, not measured behavior of a trained model."""
from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from training import identity_pilot as pilot


class IdentityPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'source'
        for relative in (pilot.PLAN_PATH, pilot.PROMPT_PATH,
                         pilot.DATA_PATH, pilot.REVIEW_PATH):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((pilot.ROOT / relative).read_bytes())

    def plan(self, edit):
        path = self.root / pilot.PLAN_PATH
        value = json.loads(path.read_text())
        edit(value)
        path.write_text(json.dumps(value))

    def rows(self, edit):
        path = self.root / pilot.DATA_PATH
        value = [json.loads(x) for x in path.read_text().splitlines()]
        edit(value)
        body = ''.join(json.dumps(x) + '\n' for x in value).encode()
        path.write_bytes(body)
        self.plan(lambda p: p.update(dataset_sha256=hashlib.sha256(body).hexdigest()))

    def rejected(self):
        with self.assertRaisesRegex(pilot.PilotError, '^identity pilot source rejected$'):
            pilot.prepare(self.root)

    def test_counts_and_no_runtime_claims(self):
        report = pilot.prepare(self.root)
        self.assertEqual(report['files']['train']['records'], 24)
        self.assertEqual(report['files']['validation']['records'], 12)
        for field in ('human_review_complete', 'model_outputs_evaluated',
                      'model_weights_downloaded', 'training_started',
                      'paid_resources_started', 'runtime_integrated', 'phase_b_ready'):
            self.assertIs(report[field], False)
        self.assertEqual(report['closed_checklist_ids'], [])

    def test_output_is_reproducible_and_hashed(self):
        a, b = Path(self.tmp.name) / 'a', Path(self.tmp.name) / 'b'
        report = pilot.prepare(self.root, a)
        self.assertEqual(report, pilot.prepare(self.root, b))
        for split in ('train', 'validation'):
            raw = (a / (split + '.jsonl')).read_bytes()
            self.assertEqual(raw, (b / (split + '.jsonl')).read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), report['files'][split]['sha256'])
            for line in raw.splitlines():
                self.assertEqual([m['role'] for m in json.loads(line)['messages']],
                                 ['system', 'user', 'assistant'])

    def test_existing_output_folder_is_preserved(self):
        out = Path(self.tmp.name) / 'existing'; out.mkdir()
        (out / 'sentinel').write_text('keep')
        with self.assertRaises(pilot.PilotError): pilot.prepare(self.root, out)
        self.assertEqual((out / 'sentinel').read_text(), 'keep')
        self.assertEqual(len(list(out.iterdir())), 1)

    def test_ordinary_identity_targets_only_use_kova_branding(self):
        _, _, rows = pilot.load(self.root)
        for row in rows:
            if row['kind'] in ('identity', 'task'):
                answer = row['messages'][1]['content']
                self.assertNotRegex(answer.lower(), r'qwen|meta|microsoft|alibaba|azure')
                if row['kind'] == 'identity': self.assertIn('Kova', answer)

    def test_hypothetical_origin_test_is_truthful_but_not_training_input(self):
        out = Path(self.tmp.name) / 'out'; pilot.prepare(self.root, out)
        train = (out / 'train.jsonl').read_text()
        validation = (out / 'validation.jsonl').read_text()
        self.assertNotIn('Offline evaluation fixture only', train)
        self.assertIn('Offline evaluation fixture only', validation)
        self.assertIn('Qwen/Qwen3-0.6B', validation)

    def test_prompt_preserves_truthful_disclosure_and_no_false_training_claim(self):
        _, text, _ = pilot.load(self.root)
        self.assertIn('Do not volunteer underlying-model', text)
        self.assertIn('using trusted runtime information', text)
        self.assertIn('Do not falsely deny known upstream', text)
        self.assertIn('not evidence of what is running', text)

    def test_no_network_or_subprocess_is_used(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('network')), \
             patch.object(subprocess, 'Popen', side_effect=AssertionError('process')):
            pilot.prepare(self.root, Path(self.tmp.name) / 'no-network')

    def test_training_or_download_authorization_cannot_be_promoted(self):
        for field in ('model_download_authorized', 'training_authorized', 'deployment_authorized'):
            path = self.root / pilot.PLAN_PATH
            original = path.read_bytes()
            with self.subTest(field=field):
                self.plan(lambda p: p['execution'].update({field: True})); self.rejected()
                path.write_bytes(original)

    def test_paid_budget_and_hyperparameters_remain_unselected(self):
        for field, value in [('approved_budget_usd', 10), ('hyperparameters', {'epochs': 1}),
                             ('trained_adapter_sha256', 'f' * 64), ('phase_b_ready', True)]:
            path = self.root / pilot.PLAN_PATH; original = path.read_bytes()
            with self.subTest(field=field):
                self.plan(lambda p: p.update({field: value})); self.rejected()
                path.write_bytes(original)

    def test_base_revision_and_notice_retention_cannot_be_removed(self):
        for field, value in [('base_revision', 'main'), ('upstream_notices_must_be_preserved', False),
                             ('base_model', 'unselected'), ('model_slot', 'work-nova')]:
            path = self.root / pilot.PLAN_PATH; original = path.read_bytes()
            with self.subTest(field=field):
                self.plan(lambda p: p.update({field: value})); self.rejected()
                path.write_bytes(original)

    def test_private_data_or_review_promotion_is_rejected(self):
        self.plan(lambda p: p['provenance'].update(human_review_complete=True)); self.rejected()

    def test_prompt_tampering_fails_before_output_creation(self):
        (self.root / pilot.PROMPT_PATH).write_text('replace me')
        out = Path(self.tmp.name) / 'invalid'
        with self.assertRaises(pilot.PilotError): pilot.prepare(self.root, out)
        self.assertFalse(out.exists())

    def test_dataset_tampering_is_rejected(self):
        path = self.root / pilot.DATA_PATH
        path.write_text(path.read_text() + '\n'); self.rejected()

    def test_normalized_duplicate_prompt_is_rejected_across_splits(self):
        self.rows(lambda rows: rows[-1]['messages'][0].update(
            content='  ' + rows[0]['messages'][0]['content'].upper() + '  '))
        self.rejected()

    def test_duplicate_record_id_is_rejected(self):
        self.rows(lambda rows: rows[-1].update(id=rows[0]['id'])); self.rejected()

    def test_wrong_roles_are_not_privileged_training_messages(self):
        self.rows(lambda rows: rows[0]['messages'][0].update(role='system')); self.rejected()

    def test_empty_answers_are_rejected(self):
        self.rows(lambda rows: rows[0]['messages'][1].update(content=' ')); self.rejected()

    def test_fixture_cannot_move_to_training(self):
        self.rows(lambda rows: next(r for r in rows if 'runtime_fixture' in r).update(split='train'))
        self.rejected()

    def test_unknown_record_field_is_rejected(self):
        self.rows(lambda rows: rows[0].update(execute=True)); self.rejected()

    def test_missing_asset_and_malformed_json_are_sanitized(self):
        path = self.root / pilot.PLAN_PATH
        for raw in (b'{', b'\xff', b'null', b'NaN', b'[' * 2000 + b']' * 2000):
            with self.subTest(raw=raw[:8]): path.write_bytes(raw); self.rejected()
        path.unlink(); self.rejected()

    def test_duplicate_keys_and_float_types_are_rejected(self):
        path = self.root / pilot.PLAN_PATH; original = path.read_text()
        for raw in (original.replace('"schema_version": 1', '"schema_version": 2, "schema_version": 1'),
                    original.replace('"schema_version": 1', '"schema_version": 1.0')):
            path.write_text(raw); self.rejected()

    def test_oversized_source_is_rejected(self):
        (self.root / pilot.DATA_PATH).write_bytes(b' ' * (pilot.MAX_BYTES + 1)); self.rejected()

    def test_cli_does_not_offer_execute_flag(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            pilot.main(['--execute'])
        self.assertEqual(error.exception.code, 2)

    def test_invalid_source_has_nonzero_cli_and_no_payload(self):
        with patch.object(pilot, 'prepare', side_effect=pilot.PilotError('identity pilot source rejected')):
            error = io.StringIO()
            with redirect_stderr(error): self.assertEqual(pilot.main([]), 1)
        self.assertEqual(error.getvalue(), 'identity pilot source rejected\n')


if __name__ == '__main__':
    unittest.main()
