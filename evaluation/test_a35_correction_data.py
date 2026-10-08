"""Check draft-label integrity, distinct held-out inputs and no benchmark repair."""

import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import a35_correction_data as data
from evaluation.quality_evidence import EvidenceRejected


class CorrectionDraftTests(unittest.TestCase):
    def test_checked_in_draft_targets_and_pins_are_validated_without_model_calls(self):
        result = data.validate_draft()
        self.assertGreater(result["reference_labels_verified"], 0)
        self.assertFalse(result["active_training_dataset_changed"])
        self.assertTrue(result["semantic_overlap_review_complete"])
        self.assertEqual(result["manual_targets_require_review"], 0)
        self.assertEqual(result["prompt_requirements_incorporated"], 34)
        self.assertEqual(result["groups_validated"], 33)
        self.assertEqual(result["structural_similarity"]["issues"], [])
        self.assertEqual(result["model_calls_made"], 0)

    def test_reference_filter_preserves_order_duplicates_negatives_and_zero(self):
        self.assertEqual(data.reference_answer("filter_map", {
            "values": [-8, 5, -8, 0, 6, 3], "divisor": 2, "factor": 3}),
            [-24, -24, 0, 18])
        self.assertEqual(data.reference_answer("copy_or_alias", {
            "values": [4], "append": 9, "copy": True}), [4])
        self.assertEqual(data.reference_answer("copy_or_alias", {
            "values": [4], "append": 9, "copy": False}), [4, 9])

    def test_owner_delegated_inputs_cannot_promote_themselves_to_execution(self):
        original = json.loads(data.MANIFEST.read_text())
        for flag in ("training_authorized", "paid_execution_authorized", "integration_into_training_authorized",
                     "pilot_selection_authorized", "deployment_authorized", "merge_authorized"):
            changed = dict(original, **{flag: True})
            with patch.object(type(data.MANIFEST), "read_text", return_value=json.dumps(changed)):
                with self.assertRaises(EvidenceRejected):
                    data.validate_draft()

    def test_probability_reference_matches_independent_enumeration(self):
        for green,other,draws,minimum in ((2,3,3,1),(3,4,3,2),(4,5,2,1)):
            outcomes=list(combinations(range(green+other),draws))
            success=sum(sum(i<green for i in draw)>=minimum for draw in outcomes)
            result=data.reference_answer("urn_probability",{"counts":{"g":green,"o":other},
                "color":"g","draws":draws,"minimum":minimum,"format":"string"})
            self.assertEqual(Fraction(result),Fraction(success,len(outcomes)))

    def test_geometry_reference_handles_orientation_and_concavity(self):
        vertices=[[0,0],[8,0],[8,3],[5,3],[5,6],[0,6]]
        expected=8*6-3*3
        for points in (vertices,list(reversed(vertices))):
            self.assertEqual(data.reference_answer("polygon_area",{"vertices":points}),{"area":expected})

    def test_dependency_reference_partitions_cycles_without_silent_schedule(self):
        tasks={"b":["a"],"a":[],"c":["d"],"d":["c"]}
        self.assertEqual(data.reference_answer("dependency_schedule",{"tasks":tasks,"partition":True}),
                         {"runnable":["a","b"],"blocked":["c","d"]})
        with self.assertRaises(EvidenceRejected):
            data.reference_answer("dependency_schedule",{"tasks":tasks,"partition":False})

    def test_authored_trace_respects_nested_copy_and_alias_lifetimes(self):
        source="a=[[4]]\nb=a.copy()\nb[0].append(7)\nb.append([9])\nresult=a"
        self.assertEqual(data.authored_python(source),[[4,7]])
        self.assertEqual(data.authored_python("a=[2,5,8]\nb=a\na[:2]=[1]\nresult=b"),[1,8])

    def test_authored_fixture_rejects_external_and_dynamic_execution(self):
        for source in ("import os", "while True: pass", "result=().__class__", "result=__import__('os')"):
            with self.subTest(source=source), self.assertRaises(EvidenceRejected):
                data.authored_python(source)

    def test_preserved_privacy_and_provenance_guards_cannot_be_removed(self):
        manifest=json.loads(data.MANIFEST.read_text())
        text=(data.ROOT/manifest["prompt_path"]).read_text()
        for clause in ("payment data","secret keys","legal entities","data access",
                       "disclose known upstream origin","planned selections","no unrequested"):
            with self.subTest(clause=clause), self.assertRaises(EvidenceRejected):
                data.validate_prompt(text.replace(clause,""))

    def test_prepared_model_messages_exclude_review_metadata(self):
        rows=data.prepared_revision_rows()
        self.assertEqual(len(rows),110)
        for _,_,row in rows:
            self.assertEqual(set(row),{"prompt","completion","chat_template_kwargs"})
            self.assertEqual(row["chat_template_kwargs"],{"enable_thinking":False})
            for message in row["prompt"]+row["completion"]:
                self.assertEqual(set(message),{"role","content"})
                self.assertNotIn('"policy_review"',message["content"])

    def test_input_schema_rejects_review_metadata_even_with_recomputed_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest=json.loads(data.MANIFEST.read_text())
            paths=[manifest["dataset_path"],manifest["prompt_path"],manifest["review_path"],
                   manifest["validation_overrides_path"],
                   "data/kova-identity-shared.v2.jsonl","prompts/kova-identity.v3.txt"]
            for path in paths:
                target=root/path; target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((data.ROOT/path).read_bytes())
            rows=[json.loads(x) for x in (root/manifest["dataset_path"]).read_text().splitlines()]
            rows[0]["review_complete"]=True
            body="".join(json.dumps(r)+"\n" for r in rows)
            (root/manifest["dataset_path"]).write_text(body)
            manifest["dataset_sha256"]=data.hashlib.sha256(body.encode()).hexdigest()
            manifest_path=root/"manifest.json"; manifest_path.write_text(json.dumps(manifest))
            with patch.object(data,"ROOT",root),patch.object(data,"MANIFEST",manifest_path):
                with self.assertRaises(EvidenceRejected): data.validate_draft(require_mask_report=False)


if __name__ == "__main__":
    unittest.main()
