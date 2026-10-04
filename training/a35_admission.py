"""A35 operator policy: bounded reads; never replay a cloud mutation.

This layer changes orchestration only; it never changes model inputs or grants authority.
"""
import hashlib
import http.client
import json
import socket
import ssl
import time
import urllib.error
from urllib.parse import urlsplit

TRANSIENT_STATUS = frozenset((429, 500, 502, 503, 504))
TRANSIENT_TYPES = frozenset(("TimeoutError", "ConnectionResetError", "ConnectionAbortedError",
                             "RemoteDisconnected", "IncompleteRead", "BrokenPipeError"))
AZURE_READ_HOSTS = frozenset(("management.azure.com", "login.microsoftonline.com",
                            "kova42c1a27.blob.core.windows.net", "apim-ratecard-v1.azure-api.net"))


class ClassifyingOpener:
    def __init__(self, opener):
        self.opener = opener

    def open(self, request, **kwargs):
        try:
            return self.opener.open(request, **kwargs)
        except urllib.error.URLError as exc:
            # Preserve a typed transient reason before the pinned ARM transport
            # deliberately removes exception text. Never convert TLS/auth errors.
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("transport timeout") from None
            if isinstance(exc.reason, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
                raise ConnectionResetError("transport interruption") from None
            raise


def transient(error):
    # Credential errors, certificate errors and arbitrary OSErrors are NOT retries.
    receipt = getattr(error, "receipt", {})
    if receipt.get("failure_class") == "transport":
        return receipt.get("error_type") in TRANSIENT_TYPES
    if isinstance(error, ssl.SSLError):
        return False
    if isinstance(error, urllib.error.URLError):
        return transient(error.reason)
    return isinstance(error, (TimeoutError, ConnectionResetError, ConnectionAbortedError,
                              BrokenPipeError, http.client.RemoteDisconnected, http.client.IncompleteRead))


def bounded_request(call, method, url, body=None, *, observe, sleep=time.sleep,
                    clock=time.time, deadline=None, **kwargs):
    parts = urlsplit(url)
    eligible = method == "GET" and body is None and parts.scheme == "https" and parts.hostname in AZURE_READ_HOSTS
    for attempt in range(3 if eligible else 1):
        options = dict(kwargs)
        if deadline is not None:
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("read deadline exhausted")
            options["timeout"] = min(options.get("timeout", 20), remaining)
        error = None
        try:
            result = call(method, url, body, **options)
            retryable = result[0] in TRANSIENT_STATUS
            status, headers, raw = result
            if not retryable:
                return result
        except Exception as exc:
            error = exc
            retryable = transient(exc)
            status, headers, raw = getattr(exc, "receipt", {}).get("status"), {}, b""
        retry = bool(eligible and retryable and attempt < 2)
        observe({"event": "bounded_request_failure", "method": method,
                 "host": parts.hostname, "path_sha256": hashlib.sha256(parts.path.encode()).hexdigest(),
                 "request_url_sha256": hashlib.sha256(url.encode()).hexdigest(),
                 "status": status, "error_type": type(error).__name__ if error else None,
                 "response_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
                 "request_number": attempt + 1, "retry_scheduled": retry,
                 "maximum_retries": 2 if eligible else 0, "paid_authorization_consumed_by_read": False})
        if not retry:
            if error is not None:
                raise error
            return result
        delay = 2 * (attempt + 1)
        if deadline is not None and clock() + delay >= deadline:
            raise TimeoutError("read retry would exceed original deadline")
        sleep(delay)


class AmbiguousMutation(RuntimeError):
    pass


def deployment_matches(observed, *, resource_id, template, parameters, location):
    """Compare Azure's typed/default-expanded parameters to the pinned template.

    Azure returns {type,value} and materializes resourceGroup().location even
    when a PUT supplied only {value}. Raw dictionary equality rejects that
    legitimate readback. Resolve only this known default expression; require
    every parameter, declared type, value, resource identity and deployment mode.
    """
    if not isinstance(observed, dict) or location != "eastus":
        return False
    props = observed.get("properties", {})
    definitions = template.get("parameters", {})
    actual = props.get("parameters", {})
    if (observed.get("id", "").casefold() != resource_id.casefold()
            or props.get("mode") != "Incremental"
            or props.get("provisioningState") not in ("Accepted", "Running", "Succeeded", "Failed", "Canceled")
            or not isinstance(actual, dict) or not definitions
            or set(actual) != set(definitions) or not set(parameters) <= set(definitions)):
        return False
    supported = {"string": str, "bool": bool, "int": int, "array": list, "object": dict}
    for name, definition in definitions.items():
        kind = definition.get("type", "").lower()
        if kind not in supported:
            return False  # A masked secret cannot establish exact readback.
        if name in parameters:
            expected = parameters[name]
        elif "defaultValue" in definition:
            expected = definition["defaultValue"]
            if expected == "[resourceGroup().location]":
                expected = location
            elif isinstance(expected, str) and expected.startswith("["):
                return False  # Never guess other ARM expressions.
        else:
            return False
        value = actual[name]
        if (not isinstance(value, dict) or set(value) != {"type", "value"}
                or not isinstance(value["type"], str) or value["type"].lower() != kind
                or type(expected) is not supported[kind] or type(value["value"]) is not supported[kind]
                or json.dumps(value["value"], sort_keys=True) != json.dumps(expected, sort_keys=True)):
            return False
    return True


def mutate_once(write, read, *, expected, observe, polls=3, sleep=time.sleep):
    """Submit exactly once, including on timeout. Reconcile at the fixed identity.

    A 404 after a lost response is not proof that Azure did not accept the write.
    We conservatively stop rather than issue a second potentially duplicate write.
    """
    ambiguous = False
    try:
        status, _ = write()
        if status not in (200, 201, 202, 204):
            if status not in TRANSIENT_STATUS:
                raise RuntimeError("mutation rejected: HTTP " + str(status))
            ambiguous = True
    except Exception as exc:
        if not transient(exc):
            raise
        ambiguous = True
        observe({"event": "mutation_response_lost", "error_type": type(exc).__name__, "write_replayed": False})
    for number in range(polls):
        status, actual = read()
        if status == 200:
            if not expected(actual):
                raise AmbiguousMutation("mutation readback identity/configuration mismatch")
            observe({"event": "mutation_reconciled", "response_was_ambiguous": ambiguous, "write_replayed": False})
            return actual
        if status != 404:
            raise AmbiguousMutation("mutation readback rejected: HTTP " + str(status))
        if number + 1 < polls:
            sleep(2)
    raise AmbiguousMutation("mutation absent or not yet observable; no write replay")
