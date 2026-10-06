"""Source/data invariants, independent synthetic-label checks; no models."""
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import permutations
import json
from math import ceil
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluation import a35_nova_behavior_data as data
from evaluation.quality_evidence import canonical, strict_json


class BehaviorInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.review = data.load_rows()

    def test_every_synthetic_label_matches_separate_reference(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                expected = data.reference_answer(self.review[row["id"]]["reference"])
                self.assertEqual(canonical(strict_json(row["messages"][1]["content"])), canonical(expected))

    def test_probability_labels_by_physical_ordered_outcome_enumeration(self):
        for row in self.rows:
            ref = self.review[row["id"]]["reference"]
            if ref["operation"] != "ordered_probability":
                continue
            p = ref["inputs"]
            labels = [color for color, count in p["counts"].items() for _ in range(count)]
            total = favorable = 0
            for draw in permutations(range(len(labels)), len(p["sequence"])):
                total += 1
                favorable += [labels[i] for i in draw] == p["sequence"]
            expected = Fraction(favorable, total)
            answer = strict_json(row["messages"][1]["content"])
            with self.subTest(row=row["id"]):
                self.assertEqual(answer["fraction"], [expected.numerator, expected.denominator])
                if "population" in answer:
                    self.assertEqual(answer["population"], list(range(len(labels),len(labels)-len(p["sequence"]),-1)))
                    counts = dict(p["counts"])
                    for color, eligible in zip(p["sequence"], answer["eligible"], strict=True):
                        self.assertEqual(eligible, counts[color]); counts[color] -= 1

    def test_geometry_labels_by_polygon_area_and_boundary(self):
        for row in self.rows:
            ref = self.review[row["id"]]["reference"]
            if ref["operation"] != "geometry":
                continue
            p = ref["inputs"]; perimeter = p["fence"]+p["gate"]
            # Enumerate integral rectangles, independently of the oracle formula.
            candidates = [(x,y) for x in range(1, perimeter) for y in range(1, perimeter)
                          if 2*x+2*y == perimeter and
                          (x == p["length"] if "length" in p else x*p["ratio"][1] == y*p["ratio"][0])]
            self.assertEqual(len(candidates), 1)
            x,y = candidates[0]; points = [(0,0),(x,0),(x,y),(0,y)]
            cross = sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(points,points[1:]+points[:1]))
            expected = abs(cross)//2-p["cutout"][0]*p["cutout"][1]
            answer = strict_json(row["messages"][1]["content"])
            self.assertEqual(answer["usable"], expected)
            if p["checked"]:
                self.assertEqual(answer["sides"], [x,y]); self.assertEqual(answer["perimeter_check"], perimeter)

    def test_signed_linear_equations_by_substitution_and_perturbation(self):
        checked = 0
        for a in (-7,-2,1,5):
            for c in (-4,0,3):
                for root in (-13,-1,0,8):
                    b = 11; d = a*root+b-c*root
                    result = data.reference_answer({"operation":"equation","inputs":
                        {"a":a,"b":b,"c":c,"d":d,"key":"z","checked":True}})
                    self.assertEqual(result["z"],root); self.assertEqual(result["left"],result["right"])
                    self.assertNotEqual(a*(root+1)+b, c*(root+1)+d)
                    checked += 1
        self.assertEqual(checked,48)

    def test_ledger_labels_by_unitwise_expansion(self):
        for row in self.rows:
            ref = self.review[row["id"]]["reference"]
            if ref["operation"] != "ledger":
                continue
            p=ref["inputs"]; stock=[1]*p["opening"]
            for count, base, extra in p["groups"]:
                for _ in range(count):
                    stock.extend([1]*base); stock.extend([1]*extra)
            for removal in p["removals"]: del stock[-removal:]
            answer=strict_json(row["messages"][1]["content"])
            self.assertEqual(answer["balance" if p["checked"] else p["key"]],len(stock))

    def test_predicate_copy_increment_and_median_witnesses(self):
        by_id={r["id"]:strict_json(r["messages"][1]["content"]) for r in self.rows}
        self.assertEqual(by_id["a35-nova-behavior-predicate-0-train"],{"indices":[0,2,3],"outputs":[-1,3,-1]})
        self.assertEqual(by_id["a35-nova-behavior-predicate-1-train"],[34,-2,79,34])
        self.assertEqual(by_id["a35-nova-behavior-predicate-0-validation"],[[0,4],[3,4]])
        self.assertEqual(by_id["a35-nova-behavior-predicate-1-validation"],[-2,-2,1])
        self.assertEqual(by_id["a35-nova-behavior-copy-1-train"],[[4,5,6],[8]])
        original=by_id["a35-nova-behavior-copy-0-validation"]["original"]
        replica=by_id["a35-nova-behavior-copy-0-validation"]["replica"]
        self.assertEqual(original["label"],"old"); self.assertEqual(replica["label"],"new")
        self.assertEqual(original["items"],replica["items"])
        self.assertEqual(by_id["a35-nova-behavior-copy-1-validation"],{"values":[3]})
        self.assertEqual(by_id["a35-nova-behavior-increment-1-train"],{"red":13,"blue":18})
        self.assertEqual(by_id["a35-nova-behavior-increment-0-validation"],{"oak":2,"pine":-2})
        self.assertEqual(by_id["a35-nova-behavior-increment-1-validation"],{"north":5,"south":10})
        self.assertEqual(by_id["a35-nova-behavior-median-0-train"]["median"],6.5)
        self.assertEqual(by_id["a35-nova-behavior-median-0-validation"]["median"],8)
        self.assertEqual(by_id["a35-nova-behavior-median-1-validation"],{"median":9})

    def test_requested_string_and_object_roots_are_not_interchangeable(self):
        for row in self.rows:
            group=self.review[row["id"]]["group"]; answer=row["messages"][1]["content"]
            if group == "string_root":
                parsed=strict_json(answer); self.assertIs(type(parsed),str)
                with self.assertRaises(ValueError): strict_json(parsed)
                self.assertNotEqual(canonical({"stage":parsed}),canonical(parsed))
            if group == "median":
                parsed=strict_json(answer); self.assertIs(type(parsed),dict)
                self.assertNotEqual(canonical([parsed["median"]]),canonical(parsed))

    def test_authority_budget_and_quality_mutants_are_rejected(self):
        original=json.loads(data.MANIFEST.read_text())
        mutations=[{"execution_authorized":True},{"training_authorized":True},
            {"paid_inference_authorized":True},{"consumed_grant_reuse_authorized":True},
            {"quality_status":"MEASURED"},{"quality_improvement_proved":True},
            {"new_allocations":1},{"new_training_runs":1},{"model_calls_made":1},
            {"preserved_recipe":dict(original["preserved_recipe"],optimizer_steps=36)},
            {"quality_gates":dict(original["quality_gates"],strict=35)}]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"proposal.json"
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    p.write_text(json.dumps(dict(original,**mutation)))
                    with patch.object(data,"MANIFEST",p), self.assertRaises(ValueError): data.load_manifest()

    def test_corrupt_labels_and_stale_pins_are_rejected(self):
        original=json.loads(data.MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in original["file_sha256"]:
                p=root/name; p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((data.ROOT/name).read_bytes())
            p=root/original["dataset_path"]
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            rows[0]["messages"][1]["content"]='{"incoming":0,"removed":0,"balance":0}'
            p.write_text("".join(json.dumps(row)+"\n" for row in rows))
            manifest=root/"manifest.json";manifest.write_text(json.dumps(original))
            with patch.object(data,"ROOT",root),patch.object(data,"MANIFEST",manifest),self.assertRaises(ValueError):data.load_rows()
            original["file_sha256"][original["dataset_path"]]=hashlib.sha256(p.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(original))
            with patch.object(data,"ROOT",root),patch.object(data,"MANIFEST",manifest),self.assertRaises(ValueError):data.load_rows()

    def test_full_pack_preserves_identity_and_98_original_rows_without_network(self):
        from training.a35_nova_screen import load_plan,prepared_nova_rows
        parent=prepared_nova_rows(load_plan()); originals={rid:(split,row) for rid,split,row in parent}
        with patch.object(socket,"create_connection",side_effect=AssertionError("network forbidden")), \
             patch.object(socket.socket,"connect",side_effect=AssertionError("network forbidden")):
            result=data.prepared_rows()
        retained={rid for rid,_,_ in result if rid in originals}
        self.assertEqual(len(retained),98)
        for rid,split,row in result:
            if rid in retained:self.assertEqual((split,row),originals[rid])
        identity={json.loads(line)["id"] for line in
            (data.ROOT/"data/a35-kovagpt-identity-overrides.v1.jsonl").read_text().splitlines()}
        self.assertTrue(identity<=retained)
        train=sum(s=="train" for _,s,_ in result);val=sum(s=="validation" for _,s,_ in result)
        self.assertEqual((train,val),(73,61));self.assertEqual(ceil(train/8)*3,30)

    def test_builder_rejects_execution_flag_without_side_effects(self):
        result=subprocess.run([sys.executable,str(data.ROOT/"scripts/build_a35_nova_behavior_inputs.py"),"--execute"],
                              capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn("unrecognized arguments",result.stderr)


if __name__ == "__main__":
    unittest.main()
