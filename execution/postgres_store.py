"""PostgreSQL adapter for the existing encrypted execution journal.

No database is contacted on import. Runtime construction requires an explicitly
enabled server-owned connection factory. Schema installation is a separate,
explicit administrative action; runtime opening does not run migrations.
All model execution still requires the independent model/entitlement guards.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import ipaddress
import os
from pathlib import Path
import re
from threading import RLock

from execution.contracts import ExecutionError, positive_integer, require
from execution.private_store import PrivateJobStore
from execution.record_cipher import RecordCipher, need
from execution.store import now_ms


TABLES = frozenset(("jobs", "stages", "events", "private_job_policy", "private_job_tombstones", "model_store_version"))
TABLE_PATTERN = re.compile(r"\b(" + "|".join(sorted(TABLES, key=len, reverse=True)) + r")\b")


class PostgresStoreError(ExecutionError):
    """An unavailable/invalid database operation did not produce trusted state."""


def _schema(value):
    require(isinstance(value, str) and re.fullmatch(r"kova_[a-z0-9_]{1,43}", value),
            "explicit private execution schema required")
    return value


@dataclass(frozen=True)
class PostgresConnectionConfig:
    host: str
    address: str
    port: int
    database: str
    user: str
    root_certificate: str
    connect_timeout_seconds: int
    network_authorized: bool = False

    def __post_init__(self):
        require(isinstance(self.host, str) and re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", self.host)
                and ".." not in self.host, "invalid trusted database host")
        try:
            address = ipaddress.ip_address(self.address)
        except ValueError:
            raise PostgresStoreError("invalid trusted database address") from None
        require(str(address) == self.address and not address.is_unspecified and not address.is_multicast,
                "invalid trusted database address")
        positive_integer(self.port, "database port", 65535)
        positive_integer(self.connect_timeout_seconds, "database connect timeout", 30)
        for value in (self.database, self.user):
            require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value), "invalid database identity")
        require(isinstance(self.root_certificate, str) and os.path.isabs(self.root_certificate)
                and type(self.network_authorized) is bool, "invalid database connection policy")


def connect_postgres(config, credential_provider):
    """Use explicit host/address and verified TLS, without DSN/env fallbacks.

    The credential provider and protected CA file belong to server configuration.
    This function does not request/refresh credentials or establish Azure roles.
    Tests of the store use a separate private Unix-socket cluster, not this network.
    """
    require(type(config) is PostgresConnectionConfig and callable(credential_provider), "trusted connection policy required")
    if not config.network_authorized:
        raise PostgresStoreError("database network access remains disabled")
    try:
        import psycopg
        password = credential_provider()
        require(type(password) is str and 0 < len(password) <= 16384, "database credential unavailable")
        return psycopg.connect(host=config.host, hostaddr=config.address, port=config.port,
            dbname=config.database, user=config.user, password=password,
            sslmode="verify-full", sslrootcert=config.root_certificate,
            ssl_min_protocol_version="TLSv1.2", gssencmode="disable",
            connect_timeout=config.connect_timeout_seconds, application_name="kova-model-execution",
            target_session_attrs="read-write", autocommit=True, prepare_threshold=None)
    except Exception:
        raise PostgresStoreError("database connection unavailable") from None


class _Row:
    def __init__(self, names, values):
        self._names, self._values = names, values
    def __getitem__(self, key):
        return self._values[key if isinstance(key, (int, slice)) else self._names.index(key)]
    def keys(self):
        return self._names
    def __iter__(self):
        return iter(self._values)


def _row_factory(cursor):
    names = tuple(column.name for column in cursor.description) if cursor.description else ()
    return lambda values: _Row(names, tuple(values))


class _Connection:
    """Translate only this journal's static parameterized SQL, not client queries."""
    def __init__(self, connection, schema):
        self.connection, self.schema = connection, _schema(schema)
    def execute(self, statement, parameters=()):
        from psycopg import sql
        require(type(statement) is str and ";" not in statement and "%" not in statement,
                "only single static journal statements are supported")
        require(statement.count("?") == len(parameters), "journal parameter count mismatch")
        qualified = TABLE_PATTERN.sub(lambda match: sql.Identifier(self.schema, match[0]).as_string(self.connection), statement)
        return self.connection.execute(qualified.replace("?", "%s"), parameters)
    def close(self):
        self.connection.close()


def install_postgres_schema(connection, schema, *, administration_authorized=False):
    """Apply the dedicated initial schema only when explicitly authorized.

    Never called by runtime opening. No DROP/repair/overwrite or production endpoint
    selection is performed. A production migration controller must review target,
    role and backup state independently before calling this administrative API.
    """
    require(administration_authorized is True, "schema administration is not authorized")
    schema = _schema(schema)
    import psycopg
    from psycopg import sql
    require(type(connection) is psycopg.Connection and connection.autocommit, "owned autocommit connection required")
    source = (Path(__file__).resolve().parents[1] / "migrations" / "001_private_execution_postgres.sql").read_text()
    with connection.transaction():
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        connection.execute(sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(schema)))
        adapter = _Connection(connection, schema)
        for statement in source.split(";"):
            if statement.strip():
                adapter.execute(statement.strip())
        connection.execute(sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM PUBLIC").format(sql.Identifier(schema)))


class PostgresPrivateJobStore(PrivateJobStore):
    """Same encrypted journal and fencing semantics on an external PostgreSQL store.

    Short state transactions serialize with a schema-scoped transaction advisory
    lock. Model/crypto handler work is not run under a long database transaction.
    This conservative baseline is correctness-first, not a throughput claim; load
    tuning requires dedicated measurements before changing the lock discipline.
    Separate processes MUST use this adapter rather than bypassing its fencing.
    """
    def __init__(self, connection_factory, *, schema, cipher, retention_for, deletion_authorized,
                 statement_timeout_ms, lock_timeout_ms, enabled=False, clock_ms=now_ms):
        need(type(enabled) is bool and enabled and callable(connection_factory)
             and type(cipher) is RecordCipher and callable(retention_for) and callable(deletion_authorized)
             and callable(clock_ms))
        schema = _schema(schema)
        positive_integer(statement_timeout_ms, "statement timeout", 60000)
        positive_integer(lock_timeout_ms, "lock timeout", 60000)
        require(lock_timeout_ms < statement_timeout_ms, "lock timeout must be below statement timeout")
        self._cipher, self._retention_for, self._deletion_authorized = cipher, retention_for, deletion_authorized
        self._clock, self._lock = clock_ms, RLock()
        self._statement_timeout, self._lock_timeout = statement_timeout_ms, lock_timeout_ms
        self._lock_id = int.from_bytes(hashlib.sha256(("kova-execution-v1:" + schema).encode()).digest()[:8], "big") & (2**63 - 1)
        connection = None
        try:
            import psycopg
            connection = connection_factory()
            require(type(connection) is psycopg.Connection and connection.autocommit and not connection.closed
                    and connection.info.transaction_status == psycopg.pq.TransactionStatus.IDLE,
                    "owned idle PostgreSQL connection required")
            require(connection.info.server_version >= 150000, "PostgreSQL 15 or newer is required")
            connection.row_factory = _row_factory
            self._db = _Connection(connection, schema)
            with self._transaction() as db:
                version = db.execute("SELECT version FROM model_store_version").fetchall()
                require(len(version) == 1 and version[0][0] == 1, "execution schema version mismatch")
                for flag in ("fsync", "full_page_writes", "synchronous_commit"):
                    require(connection.execute("SHOW " + flag).fetchone()[0] == "on", "database durability settings are not enabled")
                public_access = connection.execute(
                    "SELECT count(*) FROM pg_namespace n, LATERAL aclexplode(coalesce(n.nspacl,acldefault('n',n.nspowner))) a WHERE n.nspname=%s AND a.grantee=0",
                    (schema,)).fetchone()[0]
                require(public_access == 0, "execution schema must not grant PUBLIC access")
                require(db.execute("SELECT COUNT(*) FROM jobs j LEFT JOIN private_job_policy p ON p.job=j.id WHERE p.job IS NULL").fetchone()[0] == 0,
                        "private execution policy missing")
        except BaseException as error:
            if connection is not None and callable(getattr(connection, "close", None)):
                connection.close()
            if isinstance(error, (ExecutionError, KeyboardInterrupt, SystemExit)):
                raise
            raise PostgresStoreError("PostgreSQL execution store unavailable") from None

    @contextmanager
    def _transaction(self):
        with self._lock:
            try:
                with self._db.connection.transaction():
                    # A fresh READ COMMITTED statement snapshot is essential after
                    # waiting on the advisory lock; do not inherit REPEATABLE READ.
                    self._db.connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    self._db.connection.execute("SELECT set_config('statement_timeout', %s, true)", (str(self._statement_timeout),))
                    self._db.connection.execute("SELECT set_config('lock_timeout', %s, true)", (str(self._lock_timeout),))
                    self._db.connection.execute("SELECT set_config('synchronous_commit', 'on', true)")
                    self._db.connection.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_id,))
                    yield self._db
            except BaseException as error:
                if isinstance(error, (ExecutionError, KeyboardInterrupt, SystemExit)):
                    raise
                # A lost commit acknowledgement is not permission to replay work.
                # The caller must reopen and reconcile the durable fence/state.
                raise PostgresStoreError("PostgreSQL transaction outcome requires reconciliation") from None
