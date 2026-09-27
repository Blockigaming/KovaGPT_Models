"""Synthetic Work ledger contracts, including independent SQLite connections."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

from execution.contracts import ALL_ROUTES, ExecutionBlocked, ExecutionGrant
from execution.runner import LocalRunner
from execution.store import LocalJobStore
from execution.test_support import ModelFixture, SyntheticAdapterTestCase, make_spec
from usage.work_week import MAX, WEEK_MS, WorkUsageError, WorkUsageLedger, WorkWeek, WorkQuote, usage_units


class WorkWeekTests(SyntheticAdapterTestCase):
    def setUp(self):
        SyntheticAdapterTestCase.setUp(self)
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / 'usage.sqlite')
        self.now = 1900000000000
        self.week = WorkWeek('usage-owner', 'week-1', self.now-1000, self.now-1000+WEEK_MS, 1000)
        self.g = ExecutionGrant('usage-owner', 'plus', ALL_ROUTES, True)
        self.receipts = {}
        self.ledger = self.open(initialize=True); self.addCleanup(self.ledger.close)
        self.ledger.configure_week(self.week)

    def open(self, initialize=False, **changes):
        options = {'authorization': lambda: self.g,
                   'verify_receipt': lambda owner,job,rid: self.receipts[rid],
                   'admin_authorized': lambda w: True, 'clock_ms': lambda: self.now,
                   'fast_available': lambda q: True, 'initialize': initialize}
        options.update(changes)
        return WorkUsageLedger(self.path, **options)

    def reserve(self, job='work-1', units=100, fast=False, ledger=None, spec=None):
        spec = spec or make_spec('work:cosmo:light', deadline=self.now+50000)
        quote = WorkQuote(spec.plan['route_id'], spec.fingerprint, units, fast, 'consent-1' if fast else None)
        return (ledger or self.ledger).reserve_for_job(job, spec, quote), spec, quote

    def receipt(self, spec, units=80, rid='receipt-1', **changes):
        self.receipts[rid] = {'spec_sha256': spec.fingerprint, 'standard_units': units,
                             'quiescent': True, 'outcome': 'succeeded', **changes}

    def test_reservations_and_settlement_drive_percentage_without_double_charge(self):
        result, spec, _ = self.reserve()
        self.assertEqual(result['work_units'], 100)
        self.assertEqual(self.ledger.snapshot()['remaining_percent'], 90)
        self.receipt(spec)
        self.ledger.settle('work-1','receipt-1')
        self.ledger.settle('work-1','receipt-1')
        snap=self.ledger.snapshot()
        self.assertEqual((snap['reserved_units'],snap['used_units'],snap['remaining_percent']), (0,80,92))
        self.assertEqual(snap['resets_at_ms'], self.week.ends_ms)

    def test_pro_is_exactly_five_times_the_same_plus_week_not_a_usage_reset(self):
        self.reserve(units=200)
        self.assertEqual(self.ledger.snapshot()['remaining_percent'],80)
        self.g=replace(self.g,tier='pro')
        snap=self.ledger.snapshot()
        self.assertEqual(snap['capacity_units'],5000)
        self.assertEqual((snap['reserved_units'],snap['remaining_percent']),(200,96))
        self.g=replace(self.g,tier='plus')
        self.assertEqual(self.ledger.snapshot()['remaining_percent'],80)

    def test_downgrade_below_consumption_shows_zero_and_cannot_create_more_work(self):
        self.g=replace(self.g,tier='pro'); self.reserve(units=2000)
        self.g=replace(self.g,tier='plus')
        self.assertEqual(self.ledger.snapshot()['remaining_percent'],0)
        with self.assertRaises(WorkUsageError): self.reserve('extra', units=1)

    def test_chat_executes_when_work_is_exhausted_without_any_work_debit(self):
        self.reserve(units=1000)
        before=self.ledger.snapshot()
        for route in ('instant','medium','high'):
            spec=make_spec(route,deadline=self.now+50000)
            result=self.ledger.reserve_for_job('chat-'+route,spec)
            self.assertFalse(result['metered'])
            store=LocalJobStore(clock_ms=lambda:self.now)
            try:
                job=store.create(self.g,route,spec)
                runner=LocalRunner(store,lambda:self.g,ModelFixture().worker(),clock_ms=lambda:self.now)
                self.assertEqual(runner.run(job)['state'],'succeeded')
            finally: store.close()
        self.assertEqual(self.ledger.snapshot(),before)

    def test_chat_has_no_work_week_dependency_even_when_missing_or_expired(self):
        self.now=self.week.ends_ms+1
        for tier in ('plus','pro'):
            self.g=replace(self.g,tier=tier)
            self.assertFalse(self.ledger.reserve_for_job('chat',make_spec('instant'))['metered'])
        self.g=replace(self.g,owner_id='no-work-policy')
        self.assertFalse(self.ledger.reserve_for_job('chat-other',make_spec('instant'))['metered'])
        self.assertIsNone(self.ledger.snapshot()['remaining_percent'])

    def test_fast_uses_three_halves_customer_units_not_internal_cost_target(self):
        result,spec,_=self.reserve(units=101,fast=True)
        self.assertEqual(result['work_units'],152)
        self.assertEqual(self.ledger.snapshot()['remaining_percent'],84.8)
        self.receipt(spec,units=11)
        self.ledger.settle('work-1','receipt-1')
        self.assertEqual(self.ledger.snapshot()['used_units'],17)
        self.assertNotEqual(usage_units(100,fast=True),125)

    def test_fast_requires_consent_and_verified_server_availability(self):
        spec=make_spec('work:cosmo:light')
        with self.assertRaises(ValueError): WorkQuote(spec.plan['route_id'],spec.fingerprint,100,True)
        blocked=self.open(fast_available=lambda q:False);self.addCleanup(blocked.close)
        with self.assertRaises(WorkUsageError):self.reserve(fast=True,ledger=blocked)
        self.assertEqual(self.ledger.snapshot()['remaining_percent'],100)

    def test_chat_cannot_smuggle_a_work_quote_or_extra_charge(self):
        _,_,quote=self.reserve()
        with self.assertRaises(WorkUsageError):self.ledger.reserve_for_job('chat',make_spec('instant'),quote)

    def test_uncertain_or_nonquiescent_receipt_cannot_release_held_allowance(self):
        _,spec,_=self.reserve()
        for change in ({'quiescent':False},{'quiescent':1},{'outcome':'uncertain'},{'standard_units':True},
                       {'standard_units':101},{'spec_sha256':'a'*64}):
            self.receipt(spec,**change)
            with self.subTest(change=change),self.assertRaises(WorkUsageError):self.ledger.settle('work-1','receipt-1')
            self.assertEqual(self.ledger.snapshot()['reserved_units'],100)

    def test_cancellation_is_not_itself_quiescence_or_a_refund(self):
        _,spec,_=self.reserve()
        self.receipt(spec,units=10,outcome='cancelled',quiescent=False)
        with self.assertRaises(WorkUsageError):self.ledger.settle('work-1','receipt-1')
        self.receipt(spec,units=10,outcome='cancelled',quiescent=True)
        self.ledger.settle('work-1','receipt-1')
        self.assertEqual(self.ledger.snapshot()['used_units'],10)

    def test_duplicate_job_is_idempotent_but_changed_quotes_are_rejected(self):
        _,spec,quote=self.reserve()
        self.ledger.reserve_for_job('work-1',spec,quote)
        self.assertEqual(self.ledger.snapshot()['reserved_units'],100)
        for q in (replace(quote,standard_units=99),replace(quote,fast=True,consent_id='new')):
            with self.assertRaises(WorkUsageError):self.ledger.reserve_for_job('work-1',spec,q)

    def test_receipt_cannot_be_reused_across_two_jobs(self):
        _,spec,_=self.reserve(); self.reserve('work-2',spec=spec)
        self.receipt(spec);self.ledger.settle('work-1','receipt-1')
        with self.assertRaises(WorkUsageError):self.ledger.settle('work-2','receipt-1')
        self.assertEqual(self.ledger.snapshot()['reserved_units'],100)

    def test_separate_connections_cannot_reserve_the_same_last_units(self):
        gate=Barrier(2)
        def attempt(index):
            ledger=self.open()
            try:
                gate.wait(timeout=2)
                try:self.reserve('race-'+str(index),units=1000,ledger=ledger);return True
                except WorkUsageError:return False
            finally:ledger.close()
        with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(attempt,range(2)))
        self.assertEqual(sum(results),1)
        self.assertEqual(self.ledger.snapshot()['remaining_percent'],0)

    def test_restart_preserves_usage_and_original_reset(self):
        self.reserve(units=123)
        second=self.open();self.addCleanup(second.close)
        self.assertEqual(second.snapshot(),self.ledger.snapshot())
        second.configure_week(self.week)
        self.assertEqual(second.snapshot()['remaining_units'],877)

    def test_fresh_week_needs_explicit_configuration_not_an_automatic_fake_reset(self):
        self.reserve(units=123)
        self.now=self.week.ends_ms
        self.assertIsNone(self.ledger.snapshot()['remaining_percent'])
        next_week=replace(self.week,period_id='week-2',starts_ms=self.now,ends_ms=self.now+WEEK_MS)
        self.ledger.configure_week(next_week)
        self.assertEqual(self.ledger.snapshot()['remaining_percent'],87.7)
        self.assertEqual(self.ledger.snapshot()['previous_period_reserved_units'],123)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM work_usage').fetchone()[0],1)

    def test_week_cannot_renew_early_change_anchor_or_edit_base_allowance(self):
        for change in ({'plus_units':2000},{'period_id':'new-id'},
                       {'starts_ms':self.week.starts_ms+1,'ends_ms':self.week.ends_ms+1}):
            with self.assertRaises(WorkUsageError):self.ledger.configure_week(replace(self.week,**change))
        with self.assertRaises(WorkUsageError):replace(self.week,ends_ms=self.week.ends_ms-1)

    def test_old_week_work_is_not_rebilled_in_new_week(self):
        _,spec,quote=self.reserve()
        self.now=self.week.ends_ms
        self.ledger.configure_week(replace(self.week,period_id='week-2',starts_ms=self.now,ends_ms=self.now+WEEK_MS))
        with self.assertRaises(WorkUsageError):self.ledger.reserve_for_job('work-1',spec,quote)
        self.assertEqual(self.ledger.snapshot()['remaining_percent'],90)
        self.assertEqual(self.ledger.snapshot()['previous_period_reserved_units'],100)

    def test_owner_switch_never_exposes_another_accounts_snapshot(self):
        self.reserve(units=321)
        self.g=replace(self.g,owner_id='other-owner')
        self.assertEqual(self.ledger.snapshot()['status'],'unavailable')
        with self.assertRaises(WorkUsageError):self.ledger.settle('work-1','receipt-1')

    def test_free_and_revoked_accounts_cannot_access_work_usage_or_routes(self):
        for g in (replace(self.g,tier='free'),replace(self.g,execution_authorized=False)):
            self.g=g
            with self.assertRaises(ValueError):self.ledger.snapshot()
            with self.assertRaises(ValueError):self.reserve()

    def test_invalid_inputs_cannot_become_unbounded_capacity_or_coerced_debits(self):
        for bad in (True,False,None,0,-1,'100',1.0,float('inf'),MAX):
            with self.subTest(bad=bad),self.assertRaises(ValueError):replace(self.week,plus_units=bad)
        for bad in (True,-1,None,'1',float('nan')):
            with self.assertRaises(ValueError):usage_units(bad,fast=True)
        with self.assertRaises(ValueError):usage_units(MAX,fast=True)

    def test_callback_errors_are_sanitized_and_reservations_stay_held(self):
        self.reserve()
        broken=self.open(verify_receipt=lambda *a:(_ for _ in ()).throw(RuntimeError('PRIVATE')))
        self.addCleanup(broken.close)
        with self.assertRaisesRegex(WorkUsageError,'^work usage unavailable$'):broken.settle('work-1','receipt-1')
        self.assertEqual(self.ledger.snapshot()['reserved_units'],100)

    def test_reservation_has_no_prompt_answer_secret_or_model_execution(self):
        with patch('socket.socket',side_effect=AssertionError('network')),patch('subprocess.Popen',side_effect=AssertionError('execution')):
            self.reserve()
            snap=self.ledger.snapshot()
        self.assertFalse(snap['measured_provider_cost'])
        raw='\n'.join(self.ledger.db.iterdump())
        self.assertNotIn('PRIVATE input text',raw)

    def test_expired_job_cannot_consume_new_reservation(self):
        spec=make_spec('work:cosmo:light',deadline=self.now)
        with self.assertRaises(WorkUsageError):self.reserve(spec=spec)
        self.assertEqual(self.ledger.snapshot()['remaining_percent'],100)

    def test_clock_errors_are_sanitized_without_resetting_usage(self):
        self.reserve()
        broken=self.open(clock_ms=lambda:(_ for _ in ()).throw(RuntimeError('PRIVATE-CLOCK')))
        self.addCleanup(broken.close)
        snap=broken.snapshot()
        self.assertEqual(snap['status'],'unavailable')
        self.assertIsNone(snap['remaining_percent'])
        self.assertNotIn('PRIVATE-CLOCK',str(snap))
        with self.assertRaisesRegex(WorkUsageError,'^work usage unavailable$'):self.reserve('new',ledger=broken)
        self.assertEqual(self.ledger.snapshot()['reserved_units'],100)

    def test_revocation_during_fast_availability_cannot_commit_reservation(self):
        def fast(_):
            self.g=replace(self.g,execution_authorized=False)
            return True
        second=self.open(fast_available=fast);self.addCleanup(second.close)
        with self.assertRaises(WorkUsageError):self.reserve(ledger=second,fast=True)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM work_usage').fetchone()[0],0)

    def test_owner_switch_during_receipt_lookup_does_not_settle_other_user(self):
        _,spec,_=self.reserve();self.receipt(spec)
        original=self.g
        def verify(*args):
            self.g=replace(self.g,owner_id='other-owner')
            return self.receipts['receipt-1']
        second=self.open(verify_receipt=verify);self.addCleanup(second.close)
        with self.assertRaises(WorkUsageError):second.settle('work-1','receipt-1')
        self.g=original
        self.assertEqual(self.ledger.snapshot()['reserved_units'],100)

    def test_reset_crossing_during_snapshot_is_unavailable_not_fresh_100_percent(self):
        ticks=iter((self.week.ends_ms-1,self.week.ends_ms))
        second=self.open(clock_ms=lambda:next(ticks));self.addCleanup(second.close)
        snap=second.snapshot()
        self.assertEqual(snap['status'],'unavailable')
        self.assertIsNone(snap['remaining_percent'])

if __name__=='__main__':unittest.main()
