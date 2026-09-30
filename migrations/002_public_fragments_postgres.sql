CREATE TABLE answer_fragments (
  job text NOT NULL, sequence bigint NOT NULL CHECK(sequence>0),
  stage text NOT NULL, attempt text NOT NULL,
  payload bytea NOT NULL CHECK(octet_length(payload)>0),
  checksum text NOT NULL CHECK(length(checksum)=64),
  chars integer NOT NULL CHECK(chars>0 AND chars<=8192),
  PRIMARY KEY(job,sequence),
  FOREIGN KEY(job,sequence) REFERENCES events(job,sequence) ON DELETE CASCADE
);
ALTER TABLE model_store_version DROP CONSTRAINT model_store_version_version_check;
ALTER TABLE model_store_version ADD CONSTRAINT model_store_version_version_check CHECK(version IN (1,2));
UPDATE model_store_version SET version=2 WHERE version=1;
