"""Work-only quota reservations; never money movement or model/job execution.

An internal source reference backed by SQLite, not a production billing service.
All callbacks and quotes must be built by authenticated server code, never from
request JSON. No Plus quota, weekly reset anchor or faster backend is selected.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import sqlite3
from threading import RLock

from execution.contracts import ExecutionGrant, ExecutionSpec, identifier
from router.entitlements import ALL_WORK_ROUTES

MAX = 2**53 - 1
WEEK_MS = 7 * 24 * 60 * 60 * 1000


class WorkUsageError(ValueError):
    pass


def need(value):
    if not value:
        raise WorkUsageError('work usage unavailable')


def number(value, minimum=0, maximum=MAX):
    need(type(value) is int and minimum <= value <= maximum)
    return value


def usage_units(standard_units, *, fast=False):
    """Customer quota debit, NOT provider cost. Round once per complete job."""
    number(standard_units)
    need(type(fast) is bool)
    value = (standard_units * 3 + 1) // 2 if fast else standard_units
    return number(value)


@dataclass(frozen=True)
class WorkWeek:
    owner_id: str
    period_id: str
    starts_ms: int
    ends_ms: int
    plus_units: int

    def __post_init__(self):
        identifier(self.owner_id, 'usage owner')
        identifier(self.period_id, 'usage period')
        number(self.starts_ms, 1)
        number(self.ends_ms, 1)
        need(self.ends_ms - self.starts_ms == WEEK_MS)
        number(self.plus_units, 1, MAX // 5)


@dataclass(frozen=True)
class WorkQuote:
    """Trusted standard-work estimate pinned before dispatch, not actual cost."""
    route_id: str
    spec_sha256: str
    standard_units: int
    fast: bool = False
    consent_id: str | None = None

    def __post_init__(self):
        need(type(self.route_id) is str and self.route_id in ALL_WORK_ROUTES)
        need(type(self.spec_sha256) is str and len(self.spec_sha256) == 64
             and all(c in '0123456789abcdef' for c in self.spec_sha256))
        number(self.standard_units, 1, MAX * 2 // 3)
        need(type(self.fast) is bool)
        if self.fast:
            identifier(self.consent_id, 'Fast consent')
        else:
            need(self.consent_id is None)


class WorkUsageLedger:
    """Reserve before dispatch; uncertain work retains its full reservation.

    authorization() supplies the CURRENT authenticated user, without taking a
    client owner parameter. verify_receipt(owner, job, receipt) returns reconciled
    standard units and explicit quiescence. admin_authorized(week) governs setup.
    fast_available(quote) checks a separately approved backend; defaults to false.
    These are trust boundaries, not proof supplied by a caller-controlled boolean.

    Jobs and this ledger are separate databases in this reference. Production
    integration must provide one atomic admission transaction/outbox or retain
    ambiguous reservations for reconciliation, never silently refund them. This
    class does not install an HTTP endpoint, submit/cancel jobs or run a model.
    """
    def __init__(self, path, *, authorization, verify_receipt, admin_authorized,
                 clock_ms, fast_available=lambda _: False, initialize=False):
        need(type(initialize) is bool)
        need(all(callable(x) for x in (authorization, verify_receipt, admin_authorized, clock_ms, fast_available)))
        self._grant_source, self._receipt, self._admin = authorization, verify_receipt, admin_authorized
        self._clock, self._fast = clock_ms, fast_available
        self._lock = RLock()
        self.db = sqlite3.connect(path, timeout=2, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            if initialize:
                self.db.executescript('''BEGIN IMMEDIATE;
                  CREATE TABLE work_weeks(owner TEXT PRIMARY KEY, period TEXT NOT NULL,
                    starts INTEGER NOT NULL, ends INTEGER NOT NULL, plus_units INTEGER NOT NULL);
                  CREATE TABLE work_usage(owner TEXT NOT NULL, job TEXT NOT NULL, period TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, route TEXT NOT NULL, standard_units INTEGER NOT NULL,
                    fast INTEGER NOT NULL CHECK(fast IN (0,1)), consent TEXT,
                    reserved INTEGER NOT NULL, settled INTEGER, receipt TEXT,
                    PRIMARY KEY(owner,job), UNIQUE(owner,receipt));
                  COMMIT;''')
            self.db.execute('SELECT owner FROM work_weeks LIMIT 0')
            self.db.execute('SELECT fingerprint FROM work_usage LIMIT 0')
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    @contextmanager
    def _transaction(self):
        with self._lock:
            try:
                self.db.execute('BEGIN IMMEDIATE')
                yield
                self.db.execute('COMMIT')
            except BaseException as error:
                try:
                    if self.db.in_transaction:
                        self.db.execute('ROLLBACK')
                except sqlite3.Error:
                    pass  # A failed rollback is not proof the earlier write vanished.
                if isinstance(error, sqlite3.Error):
                    raise WorkUsageError('work usage unavailable') from None
                raise

    def _trusted(self, callback, *args):
        try:
            return callback(*args)
        except Exception:
            raise WorkUsageError("work usage unavailable") from None

    def _grant(self):
        g = self._trusted(self._grant_source)
        need(type(g) is ExecutionGrant and g.execution_authorized)
        return g

    def _same_grant(self, expected):
        need(self._grant() == expected)

    def _now(self):
        return number(self._trusted(self._clock), 1)

    def _week(self, owner):
        row = self.db.execute('SELECT * FROM work_weeks WHERE owner=?', (owner,)).fetchone()
        need(row is not None)
        week = WorkWeek(row['owner'], row['period'], row['starts'], row['ends'], row['plus_units'])
        need(week.starts_ms <= self._now() < week.ends_ms)
        return week

    def configure_week(self, week):
        need(type(week) is WorkWeek and self._trusted(self._admin, week) is True)
        with self._transaction():
            now = self._now()
            need(week.starts_ms <= now < week.ends_ms)
            prior = self.db.execute('SELECT * FROM work_weeks WHERE owner=?', (week.owner_id,)).fetchone()
            if prior:
                if prior['period'] == week.period_id:
                    need((prior['starts'], prior['ends'], prior['plus_units']) ==
                         (week.starts_ms, week.ends_ms, week.plus_units))
                    return  # Reconfiguration is idempotent, not a usage reset.
                need(now >= prior['ends'] and week.starts_ms >= prior['ends'])
                need(self.db.execute('SELECT 1 FROM work_usage WHERE owner=? AND period=? LIMIT 1',
                    (week.owner_id, week.period_id)).fetchone() is None)
            need(self._trusted(self._admin, week) is True)
            self.db.execute('INSERT INTO work_weeks VALUES(?,?,?,?,?) ON CONFLICT(owner) DO UPDATE SET '
                'period=excluded.period,starts=excluded.starts,ends=excluded.ends,plus_units=excluded.plus_units',
                (week.owner_id, week.period_id, week.starts_ms, week.ends_ms, week.plus_units))

    def _validate_record(self, row):
        # Redundant persisted fields must agree. A partial/corrupt settlement
        # cannot be interpreted as spare capacity or repaired by a later receipt.
        try:
            identifier(row['owner'], 'usage owner')
            identifier(row['job'], 'usage job')
            identifier(row['period'], 'usage period')
            number(row['fast'], 0, 1)
            quote = WorkQuote(row['route'], row['fingerprint'], row['standard_units'],
                              row['fast'] == 1, row['consent'])
            need(number(row['reserved'], 1) == usage_units(quote.standard_units, fast=quote.fast))
            need((row['settled'] is None) == (row['receipt'] is None))
            if row['settled'] is not None:
                charged = number(row['settled'], 0, row['reserved'])
                if quote.fast:
                    # A persisted debit must be reachable by ceil(3*n/2) for
                    # whole standard units. A mere upper bound accepts corrupt
                    # charges such as 1 or 4 and can invent spare capacity.
                    standard = charged * 2 // 3
                    need(usage_units(standard, fast=True) == charged)
                identifier(row['receipt'], 'usage receipt')
        except (ValueError, TypeError, KeyError, IndexError):
            raise WorkUsageError('work usage unavailable') from None

    def _totals(self, owner, period):
        # Current-period consumption plus ALL outstanding owner reservations.
        # A reset cannot make uncertain previous-period work disappear. Settlement
        # remains attributed to its original period; it is never charged twice.
        rows = self.db.execute(
            'SELECT * FROM work_usage WHERE owner=? AND (period=? OR settled IS NULL OR receipt IS NULL)',
            (owner, period))
        held = used = previous = 0
        for row in rows:
            self._validate_record(row)
            if row['settled'] is None:
                held = number(held + row['reserved'])
                if row['period'] != period:
                    previous = number(previous + row['reserved'])
            else:
                used = number(used + row['settled'])
        return held, used, previous

    @staticmethod
    def _reservation_report(period, reserved, state, *, new):
        return {'metered': True, 'work_units': reserved, 'period_id': period,
                'reservation_state': state, 'new_reservation': new,
                'authorizes_dispatch': False, 'phase_b_ready': False}

    def reserve_for_job(self, job_id, spec, quote=None):
        identifier(job_id, 'usage job')
        need(type(spec) is ExecutionSpec)
        g = self._grant()
        route = spec.plan['route_id']
        g.authorize(g.owner_id, route)
        if not route.startswith('work:'):
            need(quote is None)
            # No Work SQL transaction or quota/expiry lookup for included Chat.
            # This still requires current real authorization and job safety checks.
            self._same_grant(g)
            return {'metered': False, 'work_units': 0, 'authorizes_dispatch': False,
                    'phase_b_ready': False}
        with self._transaction():
            self._same_grant(g)
            need(g.tier in ('plus', 'pro') and type(quote) is WorkQuote)
            need((quote.route_id, quote.spec_sha256) == (route, spec.fingerprint))
            need(not quote.fast or self._trusted(self._fast, quote) is True)
            week = self._week(g.owner_id)
            need(self._now() < spec.limits.deadline_unix_ms)
            existing = self.db.execute('SELECT * FROM work_usage WHERE owner=? AND job=?', (g.owner_id, job_id)).fetchone()
            expected = (spec.fingerprint, route, quote.standard_units, int(quote.fast), quote.consent_id)
            if existing:
                self._validate_record(existing)
                need(tuple(existing[k] for k in ('fingerprint', 'route', 'standard_units', 'fast', 'consent')) == expected)
                # A still-valid original job retains its original admission week.
                # An idempotent reply is not permission to dispatch or replay it.
                self._same_grant(g)
                return self._reservation_report(existing['period'], existing['reserved'],
                    'held' if existing['settled'] is None else 'settled', new=False)
            units = usage_units(quote.standard_units, fast=quote.fast)
            held, used, previous = self._totals(g.owner_id, week.period_id)
            cap = week.plus_units * (5 if g.tier == 'pro' else 1)
            need(held + used + units <= cap)
            now = self._now()
            need(week.starts_ms <= now < min(week.ends_ms, spec.limits.deadline_unix_ms))
            self._same_grant(g)
            self.db.execute('INSERT INTO work_usage VALUES(?,?,?,?,?,?,?,?,?,NULL,NULL)',
                (g.owner_id, job_id, week.period_id, *expected, units))
            return self._reservation_report(week.period_id, units, 'held', new=True)

    def settle(self, job_id, receipt_id):
        identifier(job_id, 'usage job'); identifier(receipt_id, 'usage receipt')
        with self._transaction():
            g = self._grant()
            row = self.db.execute('SELECT * FROM work_usage WHERE owner=? AND job=?', (g.owner_id, job_id)).fetchone()
            need(row is not None)
            self._validate_record(row)
            if row['receipt'] is not None:
                need(row['receipt'] == receipt_id)
                self._same_grant(g)
                return  # Never debit or refund the same receipt twice.
            value = self._trusted(self._receipt, g.owner_id, job_id, receipt_id)
            need(type(value) is dict and set(value) == {'spec_sha256', 'standard_units', 'quiescent', 'outcome'})
            need(value['spec_sha256'] == row['fingerprint'] and value['quiescent'] is True)
            need(value['outcome'] in ('succeeded', 'failed', 'cancelled', 'expired'))
            units = number(value['standard_units'], 0, row['standard_units'])
            need(self.db.execute('SELECT 1 FROM work_usage WHERE owner=? AND receipt=?',
                (g.owner_id, receipt_id)).fetchone() is None)
            charged = usage_units(units, fast=bool(row['fast']))
            self._same_grant(g)
            self.db.execute('UPDATE work_usage SET settled=?,receipt=? WHERE owner=? AND job=?',
                (charged, receipt_id, g.owner_id, job_id))

    def snapshot(self):
        with self._transaction():
            g = self._grant()
            need(g.tier in ('plus', 'pro'))
            unavailable = {'scope': 'work_week', 'status': 'unavailable', 'remaining_percent': None,
                           'resets_at_ms': None, 'chat_uses_work_allowance': False}
            try:
                week = self._week(g.owner_id)
            except WorkUsageError:
                self._same_grant(g)
                return unavailable
            held, used, previous = self._totals(g.owner_id, week.period_id)
            cap = week.plus_units * (5 if g.tier == 'pro' else 1)
            remaining = max(0, cap - held - used)
            # Floor at two decimal places; never round a small positive/zero
            # remainder up to a misleading available amount.
            percent = (remaining * 10000 // cap) / 100
            as_of = self._now()
            self._same_grant(g)
            if not week.starts_ms <= as_of < week.ends_ms:
                return unavailable
            return {'scope': 'work_week', 'status': 'available', 'remaining_percent': percent,
                    'remaining_units': remaining, 'reserved_units': held, 'used_units': used,
                    'current_period_reserved_units': held - previous,
                    'previous_period_reserved_units': previous,
                    'capacity_units': cap, 'period_id': week.period_id,
                    'resets_at_ms': week.ends_ms, 'as_of_ms': as_of,
                    'includes_reservations': True, 'chat_uses_work_allowance': False,
                    'measured_provider_cost': False, 'phase_b_ready': False}
