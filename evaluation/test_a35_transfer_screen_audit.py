"""Public-contract synthetic fixtures; no private outputs or model calls."""
from collections import Counter
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from evaluation import a35_transfer_screen_audit as audit
from evaluation.completion_evidence import generation_evidence
from evaluation.cpu_candidate_quality import score_case
from evaluation.historical_suite_bridge import load_archived_suite
from training import a35_nova_screen as screen


def synthetic(module, records, steps):
    report = json.loads(audit.latest.previous.REPORT.read_bytes())
    report.update(source_commit=module.SOURCE, run_id=module.RUN,
                  prepared_pack_sha256=module.PACK, plan_sha256=module.PLAN)
    report['generation_profile']['system_prompt_sha256'] = audit.latest.PROMPT
    report['training_receipt'].update(source_commit=module.SOURCE, run_id=module.RUN,
        prepared_pack_sha256=module.PACK, optimizer_steps=steps,
        training_records=records[0], validation_records=records[1], completed_epochs=1.0,
        adapter_sha256={'adapter_model.safetensors': module.ADAPTER,
                       'adapter_config.json': module.ADAPTER_CONFIG})
    cases = {c['id']: c for c in load_archived_suite()['cases']}
    for index, row in enumerate(report['cases']):
        case = cases[row['case_id']]
        expected = case['evaluation']['expected']
        answer = json.dumps(expected)
        if row['case_id'] in module.FAILURES:
            kind = module.FAILURES[row['case_id']][0]
            answer = 'null' if kind == 'content' else json.dumps({'synthetic_wrapper': expected})
        row.update(answer=answer, answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                   latency_seconds=0.0, completion_evidence=generation_evidence(
                       [7] * (20 if index == 0 else 1) + [9], max_new_tokens=128, eos_token_id=9))
        row['result'] = score_case(case, answer, row['completion_evidence'])[0]
    report['result_counts'] = dict(Counter(r['result'] for r in report['cases']))
    report['category_counts'] = {cat: dict(Counter(r['result'] for r in report['cases']
                                  if r['category'] == cat)) for cat in module.COUNTS}
    return report


class TransferScreenAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path, self.latest_path = self.root/'current.json', self.root/'latest.json'
        self.report = synthetic(audit, (72, 60), 9)
        self.path.write_text(json.dumps(self.report))
        self.latest_path.write_text(json.dumps(synthetic(audit.latest, (61, 49), 8)))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module, path in ((audit, self.path), (audit.latest, self.latest_path)):
            self.stack.enter_context(patch.object(module, 'REPORT_SHA256',
                                     hashlib.sha256(path.read_bytes()).hexdigest()))

    def test_all_13_failures_and_every_case_compared_without_internal_cause_claim(self):
        result = audit.audit(self.path, self.latest_path)
        self.assertEqual(len(result['cases']), 36)
        self.assertEqual(len(result['failures']), 13)
        self.assertEqual(Counter(r['defect_kind'] for r in result['failures']),
                         {'content': 11, 'format_only': 2})
        self.assertTrue(all(r['internal_root_cause'] == 'not isolated by these screens'
                            for r in result['failures']))
        self.assertFalse(result['quality_gate_closed'])
        self.assertEqual(result['model_calls_made'], 0)

    def test_recoveries_and_historical_regression_are_separate(self):
        result = audit.audit(self.path, self.latest_path)
        self.assertEqual([r['case_id'] for r in result['cases']
                          if r['against_latest_21'] == 'recovered'], ['data-05', 'instructions-05'])
        self.assertFalse([r for r in result['cases'] if r['against_latest_21'] == 'newly_regressed'])
        self.assertEqual([r['case_id'] for r in result['cases']
                          if r['against_historical_23'] == 'newly_regressed'], ['code-06'])
        self.assertEqual(result['category_pass_counts'], {'math': '6/10', 'code_reading': '2/8',
                         'reasoning': '3/6', 'data_analysis': '6/6', 'instruction_following': '6/6'})

    def test_latest_measured_adapter_is_rejected_by_future_recipe(self):
        self.assertIn(audit.ADAPTER, screen.load_plan()['rejected_adapter_sha256'])
        self.assertEqual(len(screen.MEASURED_ADAPTERS), 6)
        self.assertIn("32b971862c5b7d0a013aea8f6da4913ba5eeb94b92e153dd7d0aa10419031e16",
                      screen.load_plan()['rejected_adapter_sha256'])

    def test_partial_wrong_or_inflated_report_fails_even_with_known_container_digest(self):
        for name in ('missing_case', 'wrong_source', 'wrong_steps', 'wrong_total',
                     'wrong_grade', 'wrong_answer_digest', 'incomplete_output'):
            changed = deepcopy(self.report)
            if name == 'missing_case': changed['cases'].pop()
            elif name == 'wrong_source': changed['source_commit'] = 'e' * 40
            elif name == 'wrong_steps': changed['training_receipt']['optimizer_steps'] = 18
            elif name == 'wrong_total': changed['result_counts']['exact_json_pass'] = 36
            elif name == 'wrong_grade': changed['cases'][2]['result'] = 'exact_json_pass'
            elif name == 'wrong_answer_digest': changed['cases'][0]['answer_sha256'] = '0' * 64
            elif name == 'incomplete_output': changed['cases'][0]['completion_evidence']['finish_reason'] = 'length'
            self.path.write_text(json.dumps(changed))
            with self.subTest(name=name), patch.object(audit, 'REPORT_SHA256',
                    hashlib.sha256(self.path.read_bytes()).hexdigest()), self.assertRaises(ValueError):
                audit.audit(self.path, self.latest_path)

    def test_report_tampering_is_rejected(self):
        self.path.write_text('{}')
        with self.assertRaises(ValueError): audit.audit(self.path, self.latest_path)

    def test_23_of_36_has_no_manual_generation_file_or_upload(self):
        cases = load_archived_suite()['cases']
        case_by_id = {c['id']: c for c in cases}
        by_prompt = {case_by_id[r['case_id']]['prompt']: r for r in self.report['cases']}
        calls = []
        def replay(prompt, budget):
            self.assertIn(prompt, by_prompt, 'manual generation forbidden')
            self.assertEqual(budget, 128)
            row = by_prompt[prompt]
            calls.append(row['case_id'])
            return row['answer'], row['completion_evidence'], row['latency_seconds']
        rows, status = screen.run_screen(cases, replay, lambda _: None)
        self.assertEqual(len(calls), 36)
        self.assertEqual(status, 'strict_threshold_failed_manual_skipped')
        output = self.root/'output'; output.mkdir()
        with patch.object(screen, 'manual_packet') as manual:
            storage = Mock()
            screen.preserve_screen(output, storage, {'status': status, 'cases': rows})
            manual.assert_not_called()
            self.assertFalse((output/'manual-review.md').exists())
            self.assertEqual([p.name for p in output.iterdir()], ['screen.json'])
            self.assertEqual([c.args[0] for c in storage.put.call_args_list], ['screen.json'])


if __name__ == '__main__':
    unittest.main()
