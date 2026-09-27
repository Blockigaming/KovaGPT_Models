# Local model-artifact verification before Azure serving

Phase A source component only. Phase B is not ready. This module reads local
files and returns narrowly scoped byte-integrity evidence; it never downloads
weights, loads a model, invokes a GPU, starts a server or changes Azure resources.

## Approved manifest, not a self-attested model name

`verify_model_artifact` requires a separately supplied expected manifest SHA-256.
The deployment/release controller must authenticate that digest outside the model
artifact location. Accepting a digest from the same untrusted payload is not an
approval mechanism. No real digest, storage location, model winner, image or GPU
has been selected by this change.

The manifest binds one existing pinned Core candidate ID, model name and revision
to an exact list of filenames, byte lengths and SHA-256 hashes. Both BF16 and FP8
candidates are supported as separate explicit choices. The validator rejects a
changed revision even if the caller supplies a matching digest for that changed
manifest. It never edits the catalog or grants model execution.

Required files include model configuration, tokenizer files and the safetensors
weight index. Every referenced shard must exist in the manifest, and every packaged
shard must be referenced. Config/tokenizer remote Python-class maps, executable or
pickle-style files, hidden files, unsafe paths, duplicate entries and ambiguous
JSON are rejected. Supported packages are flat directories with explicit filenames.
Additional formats or nested layouts need a separately reviewed extension.

## Filesystem and resource boundaries

The Linux-targeted verifier uses descriptor-relative, no-symlink reads, rejects
hard links and special files, and verifies exact file sets, sizes and hashes. It
checks file identity and metadata before/after reads and again before returning.
The package and its files cannot be writable by another account. The parent/root
path is trusted server configuration, not a request parameter.

Hashing uses bounded read buffers. Total bytes and verification timeout are explicit
caller limits, distinct from model latency and active-work budgets. Cancellation
and deadline checks are cooperative around filesystem operations; this does not
claim to interrupt an arbitrary blocked filesystem. A production mount must have
its own bounded availability and supervision strategy.

This cannot prevent a privileged host changing files after verification. The
serving process must consume a protected read-only snapshot and reverify after
replacement/restart. Matching a hash alone authenticates neither the publisher
nor the reviewer. Actual safetensors structure/dtype/shape, tokenizer behavior,
serving compatibility, context/VRAM fit, startup time and output quality require
separate loader and GPU rehearsals. This verifier intentionally does not import
Transformers, PyTorch, vLLM or a model loader.

## Use from trusted server code

Call `verify_model_artifact(root, manifest_bytes, expected_manifest_sha256=...,
maximum_total_bytes=..., timeout_seconds=...)` with an authenticated manifest pin.
Failure raises a sanitized `ModelArtifactError` without echoing file contents.
A success report still sets `model_loaded`, `serving_compatibility_verified`,
`gpu_execution_authorized` and `production_routing_authorized` to false.

There is no execute switch, default model download, network fallback or mutation
path. Startup/container integration remains required; this source module is not a
working Azure deployment or production admission grant.

## Offline verification

`python3 -m unittest worker.test_model_artifact` creates small synthetic local
files, including intentionally non-loadable fake weight bytes. It tests both
catalog pins, incorrect hashes/identity, missing or extra files, path/link/special-
file rejection, remote class maps, index inconsistency, resource limits, mutation
detection, cancellation, cleanup and no network access. No real weights are copied
into tests or committed.

Primary implementation references checked September 16, 2026:
- Python descriptor-relative/no-follow APIs: https://docs.python.org/3/library/os.html
- Safetensors versus pickle: https://huggingface.co/docs/safetensors/en/index
- Approved candidate metadata: repository `config/core-serving.v1.json`

Those references do not establish that a future packaged model was built from the
claimed upstream revision; release provenance must establish that separately.
