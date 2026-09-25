# Private execution records: encrypted payloads, retention and deletion

Source-only A28 integration. No production durable-store choice, key-service
connection, real customer record, cloud resource or deployment is introduced.
This implementation extends the existing local execution journal; A27 remains
responsible for its production durable-store adapter and isolated backend tests.

## Encryption before storage

RecordCipher uses the maintained cryptography library's AES-256-SIV AEAD with an
explicit 64-byte secret key and fresh 128-bit nonce per seal. It is not a custom
cipher and keys are not stored in the database. Authenticated associated data binds
every payload to its owner, job, record slot and immutable retention deadline.
Moving ciphertext to another user, job or stage, or changing the retention value,
therefore fails authentication. A reviewed server key service must implement the
owner-scoped active/lookup callbacks; no environment, CLI, URL or plaintext fallback
is present. The existing hash-pinned CPU cryptography dependency is reused.

PrivateJobStore encrypts canonical execution snapshots and stage results BEFORE
SQL receives them. This includes source references/text inside execution messages.
The existing LocalRunner and ModelStageWorker execute all Core/Ultra routes over
that store; they do not need a second provider implementation. IDs, lifecycle
states, counters and fingerprints remain plaintext operational metadata. This is
not full-database encryption or a claim to conceal access patterns.

Existing plaintext jobs cannot be silently opened as encrypted jobs. Startup
rejects that mismatch instead of performing an unreviewed migration. Rotation is
an explicit maintenance transaction, disallowed while a runner owns the job, and
preserves exact plan bytes, results and original retention. Failed rotations roll
back the whole transaction. Revoked/missing keys do not trigger decryption fallback.

## Retention and cleanup

Retention expiry is an explicit trusted server value, not a selected product
retention period or model work-time budget. It must cover the job deadline and is
never extended by retry or reopening. Expired payloads cannot be read, replayed or
used to produce a successful result. A limited owner-scoped operational status
remains available without decrypting route names or private artifacts. When
expiration happens during execution, failed stages are drained and the runner fence
can be released without permitting late success. Expiry does not abandon still
running callbacks or prove remote GPU work stopped.

Only a named terminal job with no active runner can be deleted. A separate trusted
server policy must authorize the authenticated owner's request or the elapsed
retention decision. Started failed/uncertain work additionally requires verified
supervisor/provider quiescence evidence. The caller's receipt string is not proof
by itself: the authorization callback must verify it. Deletion remains available
after loss of paid-mode permission; identity authorization still applies.

Deleting the row removes its stage ciphertext, private policy and events. A minimal
owner/idempotency tombstone remains so deletion cannot re-admit the same request
and repeat work. This is logical deletion, not proof of physical disk/backup
sanitization or billing refund. Backup lifecycle, key erasure, tombstone retention,
legal holds and production key/retention administration remain deployment policy.
No purge daemon or automated destructive sweep is installed.

## Checks and limits

Tests use ephemeral keys and synthetic local databases. They exercise all 24
explicit routes, encrypted SQL/disk payloads, persisted pause/restart, owner/slot
binding, tampering, key rotation/rollback, revocation, retention boundaries, deletion
isolation, uncertain-work cleanup and blocked replay after deletion. Network access
is not needed. These are source tests, not independent cryptographic review or
production key-management certification.

PrivateJobStore is opt-in and disabled by default. No application startup, Azure
Key Vault, production database, Supabase, Stripe or model deployment is wired by
this change. The previously blocked job-API and tool-execution files are not
republished here. Phase B stays blocked by issue #10's full integration criteria.

Primary implementation reference: https://cryptography.io/en/latest/hazmat/primitives/aead/#cryptography.hazmat.primitives.ciphers.aead.AESSIV
