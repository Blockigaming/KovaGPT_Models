"""Independent labels and fail-closed input guards; no model inference."""

from contextlib import contextmanager
from fractions import Fraction
import hashlib
from itertools import permutations, combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import a35_nova_transfer_data as data
from evaluation import a35_latest_screen_audit as measured
from evaluation.quality_evidence import EvidenceRejected


class TransferDataTests(unittest.TestCase):
    @contextmanager
    def changed(self, mutate_rows=None, mutate_review=None, repin=True):
        rows=[json.loads(line) for line in (data.ROOT/data.DATA_PATH).read_text().splitlines()]
        review=json.loads((data.ROOT/data.REVIEW_PATH).read_text())
        if mutate_rows: mutate_rows(rows)
        if mutate_review: mutate_review(review)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            raw=''.join(json.dumps(r)+'\n' for r in rows); reviewed=json.dumps(review)
            for name,body in ((data.DATA_PATH,raw),(data.REVIEW_PATH,reviewed)):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body)
            with patch.object(data,'ROOT',root), patch.object(data,'DATA_SHA256',hashlib.sha256(raw.encode()).hexdigest() if repin else data.DATA_SHA256), patch.object(data,'REVIEW_SHA256',hashlib.sha256(reviewed.encode()).hexdigest() if repin else data.REVIEW_SHA256):
                yield

    def test_complete_failure_class_coverage_without_quality_credit(self):
        report=data.validate()
        self.assertEqual(set(measured.FAILURE_CLASSES.values()),data.GROUPS|{'copy'})
        self.assertEqual(report['records_validated'],20)
        self.assertEqual(report['reference_labels_verified'],20)
        self.assertEqual(report['structural_similarity']['issues'],[])
        self.assertFalse(report['quality_improvement_proved'])
        self.assertEqual(report['model_calls_made'],0)

    def test_ordered_probability_by_enumeration_of_distinct_counters(self):
        counters=[(color,i) for color,n in {'white':4,'black':3,'red':2}.items() for i in range(n)]
        outcomes=list(permutations(counters,3))
        hits=sum([c[0] for c in draw]==['white','black','white'] for draw in outcomes)
        expected=Fraction(hits,len(outcomes))
        actual=data.reference_answer({'operation':'ordered_draws','inputs':{'counts':{'white':4,'black':3,'red':2},'sequence':['white','black','white']}})
        self.assertEqual(actual,{'numerator':expected.numerator,'denominator':expected.denominator})
        self.assertEqual(expected,Fraction(1,14))

    def test_unordered_probability_by_combinations_not_same_formula(self):
        groups=list(combinations(range(7),4))
        p=Fraction(sum(sum(i<3 for i in group)==2 for group in groups),len(groups))
        self.assertEqual(data.reference_answer({'operation':'unordered_hits','inputs':{'eligible':3,'total':7,'draws':4,'hits':2}}),{'fraction':[p.numerator,p.denominator]})
        self.assertEqual(p,Fraction(18,35))

    def test_missing_duplicate_or_wrong_split_fails_closed(self):
        for mutate in (lambda rows:rows.pop(),lambda rows:rows.__setitem__(1,rows[0]),lambda rows:rows[0].update(split='validation')):
            with self.changed(mutate),self.assertRaises((ValueError,KeyError)):
                data.load_rows()

    def test_changed_bytes_fail_the_pin(self):
        with self.changed(lambda rows:rows[0]['messages'][1].update(content='0'),repin=False),self.assertRaises(EvidenceRejected):
            data.load_rows()

    def test_invalid_json_python_set_and_boolean_coercion_rejected(self):
        for group,value in [('array_root',"{'coral','mauve'}"),('boolean_root','0'),('string_root','{"value":"quiet"}')]:
            def mutate(rows):
                next(r for r in rows if r['id']==f'a35-nova-transfer-{group}-train')['messages'][1]['content']=value
            with self.changed(mutate),self.assertRaises((ValueError,json.JSONDecodeError)):
                data.load_rows()

    def test_no_reference_metadata_can_enter_model_messages(self):
        with self.changed(lambda rows:rows[0].update(reference='forbidden')),self.assertRaises(EvidenceRejected):
            data.load_rows()

    def test_oracle_rejects_unbounded_counts_and_external_execution(self):
        for reference in ({'operation':'ordered_draws','inputs':{'counts':{'x':1000},'sequence':['x']}}, {'operation':'python_trace','inputs':{'source':"import os\nresult=os.environ"}}, {'operation':'unknown','inputs':{}}):
            with self.assertRaises((ValueError,KeyError)):
                data.reference_answer(reference)

    def test_every_label_mutation_is_rejected_even_after_repinned_bytes(self):
        for index in range(20):
            with self.subTest(row=index),self.changed(lambda rows:rows[index]['messages'][1].update(content='null')),self.assertRaises(EvidenceRejected):
                data.load_rows()


EXPECTED={
 'arithmetic':({'tiles':1456},[34,30]),'probability':({'numerator':1,'denominator':14},{'fraction':[18,35]}),
 'geometry':({'garden_area':247},[9,90]),'predicate':(['m-2','m0','n-2'],[12,5,19,12]),
 'increment':({'east':15,'west':19},[5,11,16]),'task_ids':(['k','m','q','r','z'],{'ready':['emit','lint']}),
 'string_root':('quiet','cedar'),'integer_root':(5,7),'boolean_root':(False,True),
 'array_root':(['coral','mauve'],['elm','oak'])}


def class_test(group):
    def test(self):
        rows={r['id']:r for r in data.load_rows()}
        for split,expected in zip(('train','validation'),EXPECTED[group],strict=True):
            row=rows[f'a35-nova-transfer-{group}-{split}']
            actual=json.loads(row['messages'][1]['content'])
            self.assertIs(type(actual),type(expected))
            self.assertEqual(actual,expected)
        self.assertNotEqual(rows[f'a35-nova-transfer-{group}-train']['messages'][0],rows[f'a35-nova-transfer-{group}-validation']['messages'][0])
    return test

for group in sorted(EXPECTED):
    setattr(TransferDataTests,'test_independent_'+group+'_labels',class_test(group))

if __name__=='__main__':
    unittest.main()
