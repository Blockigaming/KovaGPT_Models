# PostgreSQL encrypted execution-store adapter

Source integration for A27. This does not select, connect to, migrate or deploy any
production database. Tests create a disposable PostgreSQL cluster under the CI
runner's private temporary directory using installed binaries, with TCP disabled.
No Docker image, real credentials, Supabase endpoint or Azure resource is used.

## Shared execution semantics

PostgresPrivateJobStore reuses the actual PrivateJobStore and LocalJobStore model
state machine, ownership checks, encrypted payloads, idempotency, stage dependency
checks, runner/epoch fencing, cancellation, retention and deletion rules. Static
SQL binds values as parameters and qualifies the known tables to a validated
private schema; callers cannot supply SQL or a search-path override. This small
compatibility layer is for the existing journal's static queries only, not a
universal SQLite-to-PostgreSQL translator. New SQL requires explicit tests.

Short metadata transactions use a schema-scoped PostgreSQL transaction advisory
lock and READ COMMITTED isolation. This preserves the serialized reference journal
semantics across separate processes/connections. Model requests run outside those
transactions. Lock and statement timeouts are explicit, no retry is automatic,
and successful commits require synchronous_commit on. Startup also checks fsync,
full_page_writes, schema version and absence of PUBLIC schema access.

This is a conservative synchronization baseline, not a claim of measured production
throughput. Applications must use the adapter instead of writing around its locks.
Database administrators capable of rewriting tables remain trusted infrastructure,
not adversaries this interface can defeat. A dedicated server database role,
private reachability, approved schema permissions and credential rotation must be
verified before deployment; browsers must never connect as the worker role.

## Schema and connection controls

The initial PostgreSQL schema lives in migrations/001_private_execution_postgres.sql.
Its installation function is separate from runtime construction and refuses to run
without explicit administrative authorization. It creates a new named schema,
revokes PUBLIC access and applies the initial tables in one transaction. It never
drops, repairs, extends or silently migrates an existing production schema. Opening
an absent or wrong-version schema fails rather than creating it during a request.

The optional concrete connection function requires an explicitly authorized server
configuration with separate hostname/numeric address, port, database, user, protected
CA file and connect timeout. It enforces TLS verify-full and TLS 1.2 or newer.
Credentials come from a server callback, are not logged, and are not supplied in
a client DSN. Actual network/TLS/identity configuration still needs live verification.
The test factory connects only through its protected local Unix socket.

## Failure and recovery

A failed transaction rolls back complete metadata/record writes. If a commit
acknowledgement is lost, the adapter returns an unavailable/reconciliation error,
not permission to replay provider work. Reopening reads the durable state and
existing runner fence. Unknown in-flight stages still require confirmed supervisor
recovery and remain interrupted rather than automatically rerunning.

Clean stage frontiers survive closing connections and restarting the isolated
PostgreSQL process. Encryption, retention, final-only results and idempotency
metadata survive that restart as well. This validates backend behavior on the
observed test PostgreSQL version, not a cloud availability SLA, failover setup,
backup restore, account billing service or a production migration.

## Verification

The full npm test command includes execution.test_postgres_store. It runs actual
24-route execution, multi-connection create/claim races, ownership, failure rollback,
connection loss before commit, database-process restart, unknown-attempt fencing,
timeouts, schema rejection, encrypted rows and deletion tombstones. Missing test
binaries is a failure, not a silently skipped backend gate.

The pure-Python psycopg and typing_extensions wheels are version/hash pinned in
requirements/postgres-store-py312.lock. The existing system libpq and PostgreSQL
binaries are observed from the CI image; production runtime dependencies must be
pinned in its own immutable image. No key bytes or database files are archived.

Primary references checked September 16, 2026:
- https://www.psycopg.org/psycopg3/docs/basic/transactions.html
- https://www.postgresql.org/docs/17/explicit-locking.html
- https://www.postgresql.org/docs/17/runtime-config-client.html
- https://pypi.org/project/psycopg/3.3.5/
- https://pypi.org/project/typing-extensions/4.16.0/

The application, account/job API, queue supervisor, browser and live release gates
remain separate. Previously blocked publication files are not included here.
