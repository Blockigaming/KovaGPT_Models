# Qwen3-0.6B immutable download and verification

Status: prepared only. These commands were not executed by this source change,
and no model weights were downloaded.

The reviewed manifest is
`config/qwen3-0.6b-download-manifest.v1.json`. It binds all nine allowed files
for `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca` by exact byte count and SHA-256.
The expected total is 1,519,207,673 bytes. The runtime verifier rejects a
missing, additional required, resized, or byte-changed runtime asset, including
`config.json`, `generation_config.json`, tokenizer files, and weights.

## Source-only check now

This command validates only the checked-in manifest and performs no network
request or download:

```sh
python3 -m training.cosmo_artifacts
```

## Future authorized download

Run only after quota, capacity, source release, spending release, the external
watchdog, and the control-plane deallocation deadline have all passed. Use a new
absolute directory outside the repository:

```sh
export KOVA_COSMO_CACHE=/absolute/external/path/new-huggingface-cache
test ! -e "$KOVA_COSMO_CACHE"
mkdir -m 700 "$KOVA_COSMO_CACHE"

KOVA_COSMO_SNAPSHOT="$(python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    allow_patterns=[
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
    cache_dir=os.environ["KOVA_COSMO_CACHE"],
))
PY
)"
export KOVA_COSMO_SNAPSHOT

python3 -m training.cosmo_artifacts "$KOVA_COSMO_SNAPSHOT"
```

Do not use a floating revision, broaden the allowlist, continue after a hash
failure, upload the snapshot, or place it inside the repository.

## Runner-isolated generation signing

Before a future measured evaluation, create one Ed25519 signing seed on a
trusted runner-isolation system. Expose the private seed only to the guarded
runner and distribute only its public key to evaluators and reviewers:

```sh
umask 077
python3 - <<'PY'
import hashlib
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

path = Path("/absolute/protected/path/cosmo-generation-signing.key")
key = Ed25519PrivateKey.generate()
seed = key.private_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PrivateFormat.Raw,
    encryption_algorithm=serialization.NoEncryption(),
)
public = key.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
)
with path.open("x", encoding="ascii") as stream:
    stream.write(seed.hex() + "\n")
path.chmod(0o600)
print("public_key_hex=" + public.hex())
print("public_key_sha256=" + hashlib.sha256(public).hexdigest())
PY
export KOVA_COSMO_GENERATION_SIGNING_KEY_FILE=/absolute/protected/path/cosmo-generation-signing.key
```

Place only the printed `public_key_hex` and `public_key_sha256` in
`config/kova-cosmo-generation-trust.v1.json` and change its status to
`runner_signing_public_key_pinned` in a separately reviewed, exact-head-green
source commit. The current trust policy intentionally contains no public key,
so measured evaluation remains blocked. Never commit, copy to a verifier, or
place the private seed in the evaluation output.

The runner's Ed25519 signature authenticates every generated answer and its
case, variant, tokens, timing, runtime, lifecycle phase grant, adapter, source
commit and plan lineage. Human scores remain a separate hash-bound overlay.
The evaluator and reviewer expose no signing-key argument; they verify with the
source-pinned public key and reject absent, forged, wrong-key or post-run-
tampered measurements. Because verification material cannot create a valid
signature, an evaluator cannot forge runner-generated answers.
