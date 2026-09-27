CREATE TABLE model_store_version (version integer PRIMARY KEY CHECK(version = 1));
INSERT INTO model_store_version VALUES (1);
CREATE TABLE jobs (
  id text PRIMARY KEY, owner text NOT NULL, idem text NOT NULL,
  fingerprint text NOT NULL CHECK(length(fingerprint)=64), spec bytea NOT NULL,
  state text NOT NULL CHECK(state IN ('queued','running','paused','cancelling','succeeded','failed','cancelled','expired','interrupted','waiting_tools')),
  runner text, epoch bigint NOT NULL DEFAULT 0 CHECK(epoch>=0),
  cancel_requested integer NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
  tokens_reserved bigint NOT NULL DEFAULT 0 CHECK(tokens_reserved>=0),
  cost_reserved bigint NOT NULL DEFAULT 0 CHECK(cost_reserved>=0),
  sequence bigint NOT NULL DEFAULT 0 CHECK(sequence>=0), UNIQUE(owner,idem)
);
CREATE TABLE stages (
  job text NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, id text NOT NULL,
  ordinal integer NOT NULL CHECK(ordinal>=0),
  state text NOT NULL CHECK(state IN ('pending','running','completed','skipped','failed','cancelled','expired','uncertain','waiting_tools')),
  attempt text, epoch bigint, result bytea, checksum text,
  PRIMARY KEY(job,id), UNIQUE(job,ordinal), CHECK((result IS NULL)=(checksum IS NULL))
);
CREATE TABLE events (
  job text NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  sequence bigint NOT NULL CHECK(sequence>0), type text NOT NULL, stage text,
  occurred_ms bigint NOT NULL CHECK(occurred_ms>0), PRIMARY KEY(job,sequence)
);
CREATE TABLE private_job_policy (
  job text PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
  retain_until_ms bigint NOT NULL CHECK(retain_until_ms>0)
);
CREATE TABLE private_job_tombstones (
  owner text NOT NULL, idem text NOT NULL, deleted_ms bigint NOT NULL CHECK(deleted_ms>0),
  PRIMARY KEY(owner,idem)
);
CREATE INDEX jobs_owner_state ON jobs(owner,state);
