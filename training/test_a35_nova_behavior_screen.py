"""Offline binding tests; synthetic grants never authorize or measure a model."""
from copy import deepcopy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

from training import a35_nova_behavior_screen as screen


class BehaviorScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = screen.load_plan()
        cls.original = screen.ORIGINAL_LOAD(screen.PARENT_PLAN)
        cls.inputs = screen.prepared_inputs(cls.plan)

    def test_exact_new_pack_and_unchanged_recipe(self):
        self.assertEqual([len(v) for v in self.inputs],[73,61])
        for name in ('training','evaluation','resources','cost','rejected_adapter_sha256'):
            self.assertEqual(self.plan[name],self.original[name])
        self.assertEqual(self.plan['prepared_pack_sha256'],
            '9d7045a87fce0c2e2748d5c0b658e4f0e5cc598b82a90e02406edf2754f167a7')
        self.assertFalse(self.plan['execution_authorized'])

    def test_plan_mutations_fail_before_input_or_model_work(self):
        mutations = [('prepared_pack_sha256','0'*64),('input_validation_sha256','0'*64),
            ('experiment_id','old-candidate'),('execution_authorized',True),
            ('consumed_attempt_replay_authorized',True),('train_records',72)]
        for name,value in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                data=deepcopy(self.plan); data[name]=value
                path=Path(tmp)/'plan.json'; path.write_text(json.dumps(data))
                with (patch.object(screen,'PLAN',path), patch.object(screen,'ORIGINAL_LOAD',return_value=self.original),
                     patch.object(screen,'prepared_inputs',return_value=self.inputs)):
                    with self.assertRaises(ValueError): screen.load_plan()
        for name in ('training','evaluation','resources','cost'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                data=deepcopy(self.plan); data[name]={}
                path=Path(tmp)/'plan.json'; path.write_text(json.dumps(data))
                with patch.object(screen,'PLAN',path), patch.object(screen,'ORIGINAL_LOAD',return_value=self.original):
                    with self.assertRaises(ValueError): screen.load_plan()

    def grant(self):
        return {'owner_authorized':True,'experiment_id':self.plan['experiment_id'],
            'source_commit':'b'*40,'plan_sha256':screen.parent.sha(screen.PLAN.read_bytes()),
            'prepared_pack_sha256':self.plan['prepared_pack_sha256'],'family':'kova-nova',
            'training_runs':1,'evaluation_sweeps':1,'spend_ceiling_usd':'5.00','run_id':'a'*32,
            'allocation_started_epoch':1000,'watchdog_deadline_epoch':5500,
            'allocation_deadline_epoch':6400,'live_watchdog_verified':True,
            'account_rates_within_reserved_bounds':True,'prior_consumed_attempt_id':screen.CONSUMED_ATTEMPT,
            'prior_no_second_paid_run_explicitly_superseded':True,
            'authorization_scope':'NEW_BEHAVIOR_V10_ATTEMPT_ONLY','authorization_id':'SYNTHETIC_TEST_ONLY'}

    def test_consumed_and_differently_bound_grants_rejected(self):
        mutations=[('run_id',screen.CONSUMED_ATTEMPT),('run_id',screen.OLD_RESERVED_ATTEMPT),
            ('prior_no_second_paid_run_explicitly_superseded',False),('authorization_scope','old'),
            ('authorization_id','nova-three-epoch-fc83f05-owner-20261005'),('owner_authorized',False),
            ('training_runs',2),('evaluation_sweeps',2),('spend_ceiling_usd','6.00'),
            ('prepared_pack_sha256','0'*64),('watchdog_deadline_epoch',5600)]
        with patch.object(screen.parent,'PLAN',screen.PLAN):
            self.assertEqual(screen.admit(self.grant(),self.plan,'b'*40,now=1100),5200)
            for name,value in mutations:
                with self.subTest(name=name):
                    grant=self.grant(); grant[name]=value
                    with self.assertRaises(ValueError): screen.admit(grant,self.plan,'b'*40,now=1100)

    def test_integration_uses_exact_pack_and_restores_parent_globals(self):
        original_plan=screen.parent.PLAN
        original_load=screen.parent.load_plan
        original_inputs=screen.parent.prepared_inputs
        def observed(snapshot,output,grant):
            self.assertEqual(screen.parent.PLAN,screen.PLAN)
            self.assertEqual(screen.parent.load_plan(),self.plan)
            inputs=screen.parent.prepared_inputs(self.plan)
            self.assertEqual(inputs,self.inputs)
            self.assertIsNot(inputs,self.inputs)
            self.assertIs(screen.parent.admit,screen.admit)
            return {'synthetic_binding_check':True,'model_calls':0}
        with (patch.object(screen,'load_plan',return_value=self.plan),
             patch.object(screen,'prepared_inputs',return_value=self.inputs),
             patch.object(screen.parent,'execute',side_effect=observed)):
            self.assertEqual(screen.execute(Path('/unused'),Path('/unused'),{}),
                             {'synthetic_binding_check':True,'model_calls':0})
        self.assertIs(screen.parent.PLAN,original_plan)
        self.assertIs(screen.parent.load_plan,original_load)
        self.assertIs(screen.parent.prepared_inputs,original_inputs)


if __name__ == '__main__': unittest.main()
