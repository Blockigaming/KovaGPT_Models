"""Expiry-aware credentials and fail-closed ARM transport for the A35 operator.

Token claims are checked for consistency with the pinned CLI account, not used
as signature verification. The authenticated ARM response proves service-side
acceptance. No token, Authorization header or credential response is logged.
The transport never replays requests. The existing independent cleanup loop
may issue fresh cleanup requests after a failure; allocation is never retried.
"""

import base64
from fnmatch import fnmatchcase
import hashlib
from http.client import HTTPException
from importlib import import_module
import json
import re
import subprocess
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ARM = "https://management.azure.com/"
STORAGE = "https://storage.azure.com/"
AUDIENCES = {ARM: {ARM, "https://management.core.windows.net/"}, STORAGE: {STORAGE}}
UUID = re.compile(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}")
MARGIN_SECONDS = 300


def sufficient_lifetime(resource, expiry, now):
    # Storage admits the owner's exact >=300 boundary; ARM retains its existing
    # strictly-greater margin. Never substitute credentials between audiences.
    return expiry >= now + MARGIN_SECONDS if resource == STORAGE else expiry > now + MARGIN_SECONDS


def validate_storage_blob_url(url):
    """The Storage audience is global; independently pin the actual destination."""
    base = 'https://kova42c1a27.blob.core.windows.net/cosmo-adapters/'
    parts = urlsplit(url)
    need(url.startswith(base) and not parts.query and not parts.fragment
         and re.fullmatch(r'[A-Za-z0-9_./-]+', url[len(base):])
         and all(part not in ('', '.', '..') for part in url[len(base):].split('/')),
         'Storage account/container/object destination rejected')


def preflight_credentials(tokens):
    """Fresh joint readiness before allocation; any failure stops the caller.

    A Storage rollover can consume an earlier ARM margin. Check both again
    without another acquisition/wait loop before returning readiness.
    """
    for resource in (ARM, STORAGE):
        tokens.get(resource, force=True)
    now = tokens.clock()
    need(all(resource in tokens.cache and
             sufficient_lifetime(resource, tokens.cache[resource][1], now) and
             now >= tokens.cache[resource][2] for resource in (ARM, STORAGE)),
         'joint credential readiness expired during rollover')
    return {'arm_ready': True, 'storage_ready': True}


def preflight_runtime():
    """Exercise the real verifier imports/crypto backend before any paid work.

    Cloud Shell's system Python can contain PyJWT without its cryptography
    dependency. Install the existing receiver-auth lock in an isolated venv;
    importing just this stdlib transport is not sufficient readiness evidence.
    """
    for name in ("training.a35_screen_control", "training.cosmo_controller_azure"):
        import_module(name)
    jwt = import_module("jwt")
    need("RS256" in jwt.algorithms.get_default_algorithms(),
         "operator RSA verification backend unavailable")
    return {"verifier_imports": True, "rs256_backend": True}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def epoch(value):
    need(type(value) is int or type(value) is str and value.isascii() and value.isdigit(),
         "missing or malformed token expiry")
    value = int(value)
    need(value > 0, "invalid token epoch")
    return value


def require_actions(document, actions):
    """Check effective ARM permissions; incomplete inventories never admit."""
    need(type(document) is dict and not document.get("nextLink")
         and type(document.get("value")) is list, "incomplete effective permissions")
    permissions = document["value"]
    need(all(type(p) is dict and all(type(p.get(k)) is list and
         all(type(x) is str for x in p[k]) for k in ("actions", "notActions"))
         for p in permissions), "malformed effective permissions")
    for action in actions:
        need(any(any(fnmatchcase(action.casefold(), x.casefold()) for x in p["actions"])
             and not any(fnmatchcase(action.casefold(), x.casefold()) for x in p["notActions"])
             for p in permissions), "required ARM permission missing: " + action)


class CliTokens:
    def __init__(self, subscription, tenant, principal, *, acquire=None, clock=time.time,
                 monotonic=time.monotonic, sleep=time.sleep, observe=lambda _: None):
        need(all(type(x) is str and UUID.fullmatch(x) for x in (subscription, tenant, principal)),
             "pinned subscription, tenant and signed-in principal required")
        self.subscription, self.tenant, self.principal = subscription, tenant, principal
        self.acquire = acquire or self._cli
        self.clock, self.observe, self.cache = clock, observe, {}
        self.monotonic, self.sleep = monotonic, sleep

    def _cli(self, resource):
        # Azure CLI accepts subscription OR tenant, not both. Validate the
        # returned tenant and principal independently below.
        result = subprocess.run(["az", "account", "get-access-token", "--resource", resource,
            "--subscription", self.subscription, "--output", "json", "--only-show-errors"],
            capture_output=True, timeout=30)
        need(result.returncode == 0 and len(result.stdout) <= 65536,
             "pinned CLI credential acquisition failed")
        return json.loads(result.stdout)

    def get(self, resource, *, force=False):
        need(resource in AUDIENCES, "unapproved token resource")
        now = self.clock()
        cached = self.cache.get(resource)
        if not force and cached and sufficient_lifetime(resource, cached[1], now) and now >= cached[2]:
            return cached[0]
        self.cache.pop(resource, None)
        data = self.acquire(resource)
        token, expiry, issued, not_before, claims, now = self._validate(data, resource)
        if (force or resource == STORAGE) and not sufficient_lifetime(resource, expiry, now):
            # force bypasses OUR cache, not Cloud Shell's upstream broker cache.
            # az has no force-refresh switch. ARM retains forced-preflight-only
            # rollover. Storage also needs it for reads after VM provisioning.
            # Discard the aging token, wait past expiry, then acquire/validate
            # once. No HTTP request replay, allocation retry, cache deletion or
            # interactive login occurs; the other audience cache is untouched.
            wait = max(0, expiry - now + 1)
            need(wait <= MARGIN_SECONDS + 1, "credential refresh wait exceeds bound")
            self.observe({"event": "credential_refresh_required", "resource": resource,
                "expires_on": expiry, "observed_at": now, "remaining_seconds": int(expiry - now),
                "wait_seconds": wait, "http_request_sent": False})
            del data, token, claims
            deadline = self.monotonic() + wait
            for _ in range(11):
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    break
                self.sleep(min(30, remaining))
            need(self.monotonic() >= deadline, "credential refresh wait did not complete")
            data = self.acquire(resource)
            token, expiry, issued, not_before, claims, now = self._validate(data, resource)
        if not sufficient_lifetime(resource, expiry, now):
            self.observe({"event": "credential_lifetime_rejected", "resource": resource,
                "expires_on": expiry, "observed_at": now, "remaining_seconds": int(expiry - now),
                "required_margin_seconds": MARGIN_SECONDS, "http_request_sent": False})
            raise ValueError("credential expired or insufficient remaining lifetime")
        self.cache[resource] = token, expiry, not_before
        self.observe({"event": "credential_validated", "resource": resource, "audience": claims["aud"],
            "subscription": self.subscription, "tenant": self.tenant, "principal": self.principal,
            "issued_at": issued, "not_before": not_before, "expires_on": expiry,
            "observed_at": now, "remaining_seconds": int(expiry - now)})
        return token

    def _validate(self, data, resource):
        try:
            token = data["accessToken"]
            need(type(token) is str and len(token) <= 65536
                 and re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token),
                 "malformed credential")
            payload = token.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            expiry = min(epoch(data.get("expires_on")), epoch(claims.get("exp")))
            issued, not_before = epoch(claims.get("iat")), epoch(claims.get("nbf"))
            now = self.clock()  # Acquisition itself can consume time.
            need(data.get("subscription") == self.subscription and data.get("tenant") == self.tenant
                 and data.get("tokenType") == "Bearer" and claims.get("tid") == self.tenant
                 and claims.get("oid") == self.principal, "credential account/identity mismatch")
            need(claims.get("aud") in AUDIENCES[resource], "credential audience mismatch")
            need(claims.get("iss") in ("https://sts.windows.net/" + self.tenant + "/",
                                      "https://login.microsoftonline.com/" + self.tenant + "/v2.0"),
                 "credential issuer mismatch")
            need(type(claims.get("scp")) is str and "user_impersonation" in claims["scp"].split(),
                 "signed-in user delegation missing")
            need(issued <= now and not_before <= now, "credential not yet valid")
        except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed credential metadata") from None
        return token, expiry, issued, not_before, claims, now


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


class AuthenticationRejected(OSError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("ARM " + receipt["failure_class"] + " rejected: HTTP " +
                         str(receipt["status"]) + " " + str(receipt["error_code"]))


class CredentialUnavailable(OSError):
    """Fail closed without suppressing the existing bounded cleanup loop."""


class TransportFailure(OSError):
    """An incomplete exchange is evidence of failure, never an ARM response."""

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("ARM transport failed during " + receipt["stage"] + ": " +
                         receipt["error_type"] + " " + receipt["method"] + " " + receipt["path"])


class ArmClient:
    def __init__(self, tokens, *, opener=None, observe=lambda _: None,
                 preserve_response=lambda receipt, raw: None):
        self.tokens, self.opener, self.observe = tokens, opener or build_opener(NoRedirect()), observe
        self.preserve_response = preserve_response

    def request(self, method, url, body=None, *, limit=16_000_000, timeout=30):
        parts = urlsplit(url)
        scope = "/subscriptions/" + self.tokens.subscription
        need(parts.scheme == "https" and parts.netloc == "management.azure.com"
             and not parts.fragment and not parts.username and not parts.password
             and (parts.path.casefold() == scope or parts.path.casefold().startswith(scope + "/"))
             and "%" not in parts.path and "\\" not in parts.path
             and all(p not in (".", "..") for p in parts.path.split("/")),
             "ARM credential destination/scope rejected")
        need(method in ("GET", "PUT", "POST", "DELETE", "PATCH"), "unapproved ARM method")
        try:
            credential = self.tokens.get(ARM)
        except (ValueError, subprocess.SubprocessError) as exc:
            self.observe({"event": "credential_unavailable", "method": method,
                          "path": parts.path, "failure_class": "credential",
                          "error_type": type(exc).__name__, "http_request_sent": False})
            raise CredentialUnavailable("validated ARM credential unavailable; no request sent") from None
        headers = {"Authorization": "Bearer " + credential}
        if body is not None:
            body = json.dumps(body, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = Request(url, method=method, data=body, headers=headers)
        stage, status, response_headers = "open", None, {}
        try:
            try:
                response = self.opener.open(request, timeout=timeout)
            except HTTPError as error:
                response = error
            with response as response:
                status = response.status
                response_headers = {k.lower(): v for k, v in response.headers.items()}
                stage = "read"
                raw = response.read(limit + 1)
                need(len(raw) <= limit, "ARM response exceeded evidence bound")
        except (OSError, HTTPException) as exc:
            partial = getattr(exc, "partial", b"")
            partial = partial[:limit] if type(partial) is bytes else b""
            receipt = {"event": "arm_transport_failure", "method": method, "path": parts.path,
                "request_url_sha256": hashlib.sha256(url.encode()).hexdigest(),
                "stage": stage, "status": status, "error_type": type(exc).__name__,
                "timeout_seconds": timeout, "response_complete": False,
                "partial_body_bytes": len(partial),
                "partial_body_sha256": hashlib.sha256(partial).hexdigest() if partial else None,
                "request_id": response_headers.get("x-ms-request-id"),
                "correlation_id": response_headers.get("x-ms-correlation-request-id"),
                "failure_class": "transport", "request_replayed": False}
            # Do not log exception text, query strings, credentials, or request
            # bodies. Preserve known status/partial bytes without claiming a
            # complete response. An evidence write failure also blocks admission.
            self.preserve_response(dict(receipt), partial)
            self.observe(dict(receipt))
            raise TransportFailure(receipt) from None
        try:
            value = json.loads(raw) if raw else {}
        except (ValueError, UnicodeError):
            value = {}
        error = value.get("error", {}) if type(value) is dict else {}
        code = error.get("code") if type(error) is dict else None
        code = code if type(code) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", code) else None
        receipt = {"event": "arm_response", "method": method, "path": parts.path,
            "status": status, "error_code": code, "response_sha256": hashlib.sha256(raw).hexdigest(),
            "request_id": response_headers.get("x-ms-request-id"),
            "correlation_id": response_headers.get("x-ms-correlation-request-id"),
            "failure_class": "authentication" if status == 401 else "authorization" if status == 403 else None}
        # A narrowly scoped caller may preserve network evidence here, before
        # downstream attestation can reject it and cleanup deletes the resource.
        # The callback never receives request headers or credentials.
        self.preserve_response(dict(receipt), raw)
        self.observe(receipt)
        if status in (401, 403):
            self.tokens.cache.pop(ARM, None)
            raise AuthenticationRejected(receipt)
        return status, response_headers, raw
