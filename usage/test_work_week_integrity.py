"""Reset, receipt, and storage failure tests for the Work-only reference ledger."""
from dataclasses import replace
import sqlite3
import unittest
from unittest.mock import patch

from execution.test_support import SyntheticAdapterTestCase, make_spec
from usage import test_work_week as fixtures
from usage.work_week import MAX, WEEK_MS, WorkUsageError, usage_units


class WorkWeekIntegrityTests(SyntheticAdapterTestCase):
    setUp = fixtures.WorkWeekTests.setUp
    open = fixtures.WorkWeekTests.open
    reserve = fixtures.WorkWeekTests.reserve
    receipt = fixtures.WorkWeekTests.receipt

    def next_week(self):
        self.now = self.week.ends_ms
        self.ledger.configure_week(replace(self.week, period_id='week-2',
            starts_ms=self.now, ends_ms=self.now + WEEK_MS))

    def test_original_job_deadline_can_cross_week_without_shortening_or_renewal(self):
        deadline = self.week.ends_ms + 50000
        spec = make_spec('work:cosmo:light', deadline=deadline)
        before = spec.encoded
        result, _, quote = self.reserve(spec=spec)
        self.assertTrue(result['new_reservation'])
        self.next_week()
        repeated = self.ledger.reserve_for_job('work-1', spec, quote)
        self.assertFalse(repeated['new_reservation'])
        self.assertEqual(repeated['period_id'], 'week-1')
        self.assertEqual(self.ledger.snapshot()['previous_period_reserved_units'], 100)
        self.assertEqual(spec.encoded, before)
        self.assertEqual(spec.limits.deadline_unix_ms, deadline)

    def test_uncertain_previous_week_hold_stays_visible_and_restricts_capacity(self):
        self.reserve(units=1000)
        self.next_week()
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot['remaining_percent'], 0)
        self.assertEqual(snapshot['previous_period_reserved_units'], 1000)
        with self.assertRaises(WorkUsageError):
            self.reserve('new-work', units=1)

    def test_settling_old_hold_releases_it_without_rebilling_old_usage_in_new_week(self):
        _, spec, _ = self.reserve(units=250)
        self.next_week()
        self.assertEqual(self.ledger.snapshot()['remaining_percent'], 75)
        self.receipt(spec, units=200)
        self.ledger.settle('work-1', 'receipt-1')
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot['remaining_percent'], 100)
        self.assertEqual(snapshot['used_units'], 0)
        self.assertEqual(snapshot['previous_period_reserved_units'], 0)
        row = self.ledger.db.execute('SELECT period,settled FROM work_usage').fetchone()
        self.assertEqual(tuple(row), ('week-1', 200))

    def test_reservation_response_distinguishes_new_held_and_settled_without_dispatch_permission(self):
        first, spec, quote = self.reserve()
        self.assertTrue(first['new_reservation'])
        self.assertFalse(first['authorizes_dispatch'])
        repeated = self.ledger.reserve_for_job('work-1', spec, quote)
        self.assertFalse(repeated['new_reservation'])
        self.assertEqual(repeated['reservation_state'], 'held')
        self.receipt(spec)
        self.ledger.settle('work-1', 'receipt-1')
        settled = self.ledger.reserve_for_job('work-1', spec, quote)
        self.assertEqual(settled['reservation_state'], 'settled')
        self.assertFalse(settled['new_reservation'])
        self.assertFalse(settled['authorizes_dispatch'])

    def test_mismatched_persisted_receipt_and_settlement_cannot_create_spare_capacity(self):
        self.reserve()
        self.ledger.db.execute('UPDATE work_usage SET settled=0 WHERE job=?', ('work-1',))
        with self.assertRaisesRegex(WorkUsageError, '^work usage unavailable$'):
            self.ledger.snapshot()
        with self.assertRaises(WorkUsageError):
            self.reserve('next')

    def test_persisted_charge_exceeding_reserved_amount_is_rejected_on_repeated_reserve(self):
        _, spec, quote = self.reserve()
        self.ledger.db.execute('UPDATE work_usage SET settled=101,receipt=?', ('receipt',))
        with self.assertRaises(WorkUsageError):
            self.ledger.reserve_for_job('work-1', spec, quote)

    def test_quiescent_settlement_does_not_erase_mismatched_record_corruption(self):
        _, spec, _ = self.reserve()
        self.receipt(spec)
        self.ledger.db.execute('UPDATE work_usage SET reserved=1 WHERE job=?', ('work-1',))
        with self.assertRaises(WorkUsageError):
            self.ledger.settle('work-1', 'receipt-1')
        row = self.ledger.db.execute('SELECT receipt,settled FROM work_usage').fetchone()
        self.assertEqual(tuple(row), (None, None))

    def test_sql_storage_errors_are_sanitized_and_do_not_issue_admission(self):
        with patch.object(self.ledger, '_totals', side_effect=sqlite3.OperationalError('PRIVATE-PATH')):
            with self.assertRaisesRegex(WorkUsageError, '^work usage unavailable$'):
                self.reserve()
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM work_usage').fetchone()[0], 0)

    def test_grant_change_on_last_clock_read_rolls_back_reservation(self):
        calls = 0
        original = self.g
        def clock():
            nonlocal calls
            calls += 1
            if calls == 3:
                self.g = replace(self.g, execution_authorized=False)
            return self.now
        second = self.open(clock_ms=clock)
        self.addCleanup(second.close)
        with self.assertRaises(WorkUsageError):
            self.reserve(ledger=second)
        self.g = original
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM work_usage').fetchone()[0], 0)

    def test_active_previous_week_hold_stays_visible_after_database_reopen(self):
        self.reserve(units=125)
        self.next_week()
        second = self.open()
        self.addCleanup(second.close)
        snapshot = second.snapshot()
        self.assertEqual(snapshot['remaining_percent'], 87.5)
        self.assertEqual(snapshot['previous_period_reserved_units'], 125)
        self.assertEqual(snapshot['reserved_units'], 125)

    def test_chat_does_not_open_work_transaction_when_work_storage_is_unavailable(self):
        with patch.object(self.ledger, '_transaction', side_effect=sqlite3.OperationalError('offline')):
            result = self.ledger.reserve_for_job('chat', make_spec('instant'))
        self.assertFalse(result['metered'])
        self.assertEqual(result['work_units'], 0)
        self.assertFalse(result['authorizes_dispatch'])

    def test_impossible_fast_settlements_cannot_report_or_release_capacity(self):
        _, spec, quote = self.reserve(fast=True)
        for charged in (1, 4, 7, 148):
            with self.subTest(charged=charged):
                self.ledger.db.execute("DELETE FROM work_usage WHERE job<>'work-1'")
                self.ledger.db.execute('UPDATE work_usage SET settled=?,receipt=? WHERE job=?',
                    (charged, 'receipt-corrupt', 'work-1'))
                operations = (
                    self.ledger.snapshot,
                    lambda: self.ledger.reserve_for_job('work-1', spec, quote),
                    lambda: self.ledger.settle('work-1', 'receipt-corrupt'),
                    lambda: self.reserve('new-work', units=1),
                )
                for operation in operations:
                    with self.subTest(operation=operation), self.assertRaisesRegex(
                            WorkUsageError, '^work usage unavailable$'):
                        operation()
                rows = self.ledger.db.execute('SELECT job,settled,receipt FROM work_usage').fetchall()
                self.assertEqual([tuple(row) for row in rows], [('work-1', charged, 'receipt-corrupt')])

    def test_every_reachable_fast_settlement_preserves_round_once_and_zero_use(self):
        _, spec, quote = self.reserve(fast=True)
        for standard in range(101):
            with self.subTest(standard=standard):
                charged = usage_units(standard, fast=True)
                self.ledger.db.execute('UPDATE work_usage SET settled=?,receipt=?',
                    (charged, 'receipt-valid'))
                snapshot = self.ledger.snapshot()
                self.assertEqual(snapshot['used_units'], charged)
                self.assertEqual(snapshot['remaining_units'], 1000 - charged)
                self.assertEqual(snapshot['reserved_units'], 0)
                repeated = self.ledger.reserve_for_job('work-1', spec, quote)
                self.assertEqual(repeated['reservation_state'], 'settled')
                self.assertFalse(repeated['authorizes_dispatch'])
                self.ledger.settle('work-1', 'receipt-valid')

    def test_standard_settlements_are_not_subject_to_fast_rounding(self):
        _, spec, _ = self.reserve()
        for charged in (1, 4, 7, 100):
            with self.subTest(charged=charged):
                self.ledger.db.execute('UPDATE work_usage SET settled=?,receipt=?',
                    (charged, 'receipt-standard'))
                self.assertEqual(self.ledger.snapshot()['used_units'], charged)
                self.ledger.settle('work-1', 'receipt-standard')

    def test_impossible_fast_settlement_stays_blocked_after_reopen(self):
        self.reserve(fast=True)
        self.ledger.db.execute('UPDATE work_usage SET settled=1,receipt=?', ('receipt-corrupt',))
        second = self.open()
        self.addCleanup(second.close)
        with self.assertRaisesRegex(WorkUsageError, '^work usage unavailable$'):
            second.snapshot()
        with self.assertRaises(WorkUsageError):
            self.reserve('new-work', units=1, ledger=second)
        self.assertEqual(self.ledger.db.execute('SELECT count(*) FROM work_usage').fetchone()[0], 1)

    def test_fast_settlement_validation_uses_exact_integer_arithmetic_at_limits(self):
        self.reserve(fast=True)
        row = dict(self.ledger.db.execute('SELECT * FROM work_usage').fetchone())
        row['standard_units'] = MAX * 2 // 3
        row['reserved'] = usage_units(row['standard_units'], fast=True)
        row['receipt'] = 'receipt-large'
        for standard in (0, 1, row['standard_units'] - 1, row['standard_units']):
            with self.subTest(standard=standard):
                row['settled'] = usage_units(standard, fast=True)
                self.ledger._validate_record(row)
        impossible = row['reserved'] - (row['reserved'] - 1) % 3
        row['settled'] = impossible
        self.assertEqual(impossible % 3, 1)
        with self.assertRaisesRegex(WorkUsageError, '^work usage unavailable$'):
            self.ledger._validate_record(row)


if __name__ == '__main__':
    unittest.main()
