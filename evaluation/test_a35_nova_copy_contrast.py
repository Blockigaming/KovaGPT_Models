"""Offline label, isolation, schema and leakage checks for the additive pair."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import a35_nova_copy_contrast as data
from evaluation.historical_suite_bridge import load_archived_suite
from evaluation.quality_evidence import EvidenceRejected


class NovaCopyContrastTests(unittest.TestCase):
    @contextmanager
    def changed(self, mutate_rows=None, mutate_review=None, *, repin=True):
        rows = [json.loads(line) for line in (data.ROOT / data.DATA_PATH).read_text().splitlines()]
        review = json.loads((data.ROOT / data.REVIEW_PATH).read_text())
        if mutate_rows:
            mutate_rows(rows)
        if mutate_review:
            mutate_review(review)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = "".join(json.dumps(r) + "\n" for r in rows)
            reviewed = json.dumps(review)
            for name, body in ((data.DATA_PATH, raw), (data.REVIEW_PATH, reviewed)):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
            with patch.object(data, "ROOT", root), \
                 patch.object(data, "DATA_SHA256", hashlib.sha256(raw.encode()).hexdigest()
                              if repin else data.DATA_SHA256), \
                 patch.object(data, "REVIEW_SHA256", hashlib.sha256(reviewed.encode()).hexdigest()
                              if repin else data.REVIEW_SHA256):
                yield

    def test_independent_labels_preserve_outer_isolation_and_nested_sharing(self):
        rows = data.load_rows()
        self.assertEqual(json.loads(rows[0]["messages"][1]["content"]), [[6, 7, 9], [4, 2, 5]])
        nested = json.loads(rows[1]["messages"][1]["content"])
        self.assertEqual(nested, [["oak", "elm"], True, False])
        self.assertIs(type(nested[1]), bool)
        self.assertIs(type(nested[2]), bool)

    def test_full_pack_leakage_checks_preserve_110_existing_records(self):
        report = data.validate()
        self.assertEqual(report["existing_records_unchanged"], 110)
        self.assertEqual(report["new_training_records"], 1)
        self.assertEqual(report["new_validation_records"], 1)
        self.assertEqual(report["structural_similarity"]["issues"], [])
        self.assertEqual(report["model_calls_made"], 0)
        self.assertFalse(report["quality_improvement_proved"])

    def test_model_rows_never_contain_review_or_reference_metadata(self):
        for row in data.load_rows():
            self.assertEqual(set(row), {"id", "split", "messages"})
            for message in row["messages"]:
                self.assertEqual(set(message), {"role", "content"})
                self.assertNotIn("policy_review", message["content"])
                self.assertNotIn("code-06", message["content"])

    def test_changed_bytes_fail_the_immutable_pin(self):
        with self.changed(repin=False), self.assertRaises(EvidenceRejected):
            data.load_rows()

    def test_target_mutation_fails_even_if_file_is_repinned(self):
        def mutate(rows):
            rows[0]["messages"][1]["content"] = "[[6,2,5],[4,2,5]]"
        with self.changed(mutate), self.assertRaises(EvidenceRejected):
            data.load_rows()

    def test_wrong_json_root_and_boolean_coercion_fail(self):
        for index, target in ((0, '{"value":[[6,7,9],[4,2,5]]}'),
                              (1, '[["oak","elm"],1,0]')):
            def mutate(rows):
                rows[index]["messages"][1]["content"] = target
            with self.subTest(target=target), self.changed(mutate), self.assertRaises(EvidenceRejected):
                data.load_rows()

    def test_missing_duplicate_or_wrong_split_records_fail(self):
        mutations = [lambda rows: rows.pop(),
                     lambda rows: rows.__setitem__(1, rows[0]),
                     lambda rows: rows[1].update(split="train")]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.changed(mutate), self.assertRaises(EvidenceRejected):
                data.load_rows()

    def test_extra_model_metadata_and_bad_message_roles_fail(self):
        mutations = [lambda rows: rows[0].update(reference="must not enter training"),
                     lambda rows: rows[0]["messages"][0].update(role="system")]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.changed(mutate), self.assertRaises(EvidenceRejected):
                data.load_rows()

    def test_reference_cannot_execute_external_code(self):
        def mutate(review):
            review["records"][0]["reference"]["inputs"]["source"] = "import os\nresult=[]"
        with self.changed(mutate_review=mutate), self.assertRaises(EvidenceRejected):
            data.load_rows()

    def test_exact_benchmark_prompt_is_rejected_before_model_use(self):
        prompt = next(c["prompt"] for c in load_archived_suite()["cases"] if c["id"] == "code-06")
        def mutate(rows):
            rows[0]["messages"][0]["content"] = prompt
        with self.changed(mutate), self.assertRaises(EvidenceRejected):
            data.load_rows()

    def test_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(data, "ROOT", Path(temporary)):
            with self.assertRaises(FileNotFoundError):
                data.load_rows()


if __name__ == "__main__":
    unittest.main()
