# Await-safe serving lifecycle identity

The September 17 continuation reproduced a source race in the guarded loader:
stop/restart during an awaited health probe allowed an old start/identity task
to inspect the replacement backend. Old success could mark an unchecked new
backend ready or return its engine to an old handle; old failure/cancellation
could close the replacement. A failed shutdown could also be overwritten by a
late successful startup probe. None of these fixtures loaded a real model.

`worker/serving_runtime.py` now captures both lifecycle ID and backend object.
After each awaited health check, it requires that same object, lifecycle and
expected state before reading identity, returning a handle or marking ready.
Failure cleanup is similarly ownership-scoped. Stop/restart and shutdown-failure
states cannot be overwritten by stale work. Cancellation of the current owner
still cleans up and propagates `CancelledError`. Observed configuration fields
also require exact types, so `8192.0` and `0` cannot masquerade as integer context
size and a literal false remote-code permission.

The controller must be owned by one event-loop thread; this is not a cross-thread
or distributed synchronization primitive. No await is held across a global lock,
no stale task is rebound to a new lifecycle, and no health check is used as proof
that remote work or billing stopped. Native initialization and shutdown remain
synchronous and require a supervised process. `wait_for` cancellation remains
cooperative and may take longer than its nominal timeout; no product timing
requirement or hard interruption guarantee is introduced.

Reproduction:

```
python3 -m unittest worker.test_runtime_interleavings worker.test_serving_runtime worker.test_runtime_lifecycle_identity -v
```

Twelve new deterministic methods use event-controlled synthetic health probes,
including pending starts and engine handoffs, late errors, cancellation, failed
shutdown and exact-type observations. Existing real-artifact byte-verifier tests
remain in the focused suite. Tests use no vLLM/torch weights, GPU or network.
The full registered suite includes these tests without removing previous gates.

This is a regression repair to A32, not another completed checklist item, an
independent review, public inference endpoint or Phase B authorization. All
checked-in loader/production execution flags and model/Chat/Work policies remain
unchanged. Historical A22/A26 publication restrictions are not touched.

Primary reference checked September 17, 2026:
https://docs.python.org/3.12/library/asyncio-task.html#task-cancellation
