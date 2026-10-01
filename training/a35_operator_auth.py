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
    def __init__(self, subscription, tenant, principal, *, acquire=None, clock=time.time, observe=lambda _: None):
        need(all(type(x) is str and UUID.fullmatch(x) for x in (subscription, tenant, principal)),
             "pinned subscription, tenant and signed-in principal required")
        self.subscription, self.tenant, self.principal = subscription, tenant, principal
        self.acquire = acquire or self._cli
        self.clock, self.observe, self.cache = clock, observe, {}

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
        if not force and cached and now + MARGIN_SECONDS < cached[1] and now >= cached[2]:
            return cached[0]
        self.cache.pop(resource, None)
        data = self.acquire(resource)
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
            need(issued <= now and not_before <= now and expiry > now + MARGIN_SECONDS,
                 "credential expired, not yet valid or insufficient remaining lifetime")
        except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed credential metadata") from None
        self.cache[resource] = token, expiry, not_before
        self.observe({"event": "credential_validated", "resource": resource, "audience": claims["aud"],
            "subscription": self.subscription, "tenant": self.tenant, "principal": self.principal,
            "issued_at": issued, "not_before": not_before, "expires_on": expiry,
            "observed_at": now, "remaining_seconds": int(expiry - now)})
        return token


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


class AuthenticationRejected(OSError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("ARM " + receipt["failure_class"] + " rejected: HTTP " +
                         str(receipt["status"]) + " " + str(receipt["error_code"]))


class ArmClient:
    def __init__(self, tokens, *, opener=None, observe=lambda _: None):
        self.tokens, self.opener, self.observe = tokens, opener or build_opener(NoRedirect()), observe

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
        headers = {"Authorization": "Bearer " + self.tokens.get(ARM)}
        if body is not None:
            body = json.dumps(body, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = Request(url, method=method, data=body, headers=headers)
        try:
            response = self.opener.open(request, timeout=timeout)
        except HTTPError as error:
            response = error
        with response as response:
            raw = response.read(limit + 1)
            need(len(raw) <= limit, "ARM response exceeded evidence bound")
            status = response.status
            response_headers = {k.lower(): v for k, v in response.headers.items()}
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
        self.observe(receipt)
        if status in (401, 403):
            self.tokens.cache.pop(ARM, None)
            raise AuthenticationRejected(receipt)
        return status, response_headers, raw
