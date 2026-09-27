"""Concrete, disabled-by-default Azure HTTPS + managed-identity REST binding.

No Azure/OpenAI SDK, environment credential chain, local-login fallback, automatic
retry, token cache, or environment proxy. Import/construction contacts nothing.
Only reviewed server configuration may provide settings, address and API resource.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
import re
from time import time
from urllib.parse import quote, urlencode, urlsplit
from uuid import UUID

from worker.azure_container_apps import (
    AzureExecutionBlocked, AzureProtocolError, AzureRequest, AzureResponse,
    AzureSettings, _BEARER, _execution_guard, _reject_json_constant,
    _unique_json_object, make_azure_inference_client,
)
from worker.bounded_http import numeric_address, open_http_response
from worker.openai_protocol import OPENAI_CHAT_ROUTE


IDENTITY_API_VERSION = "2019-08-01"
IDENTITY_MAX_BYTES = 64 * 1024
TOKEN_CLOCK_SKEW_SECONDS = 30  # Credential safety allowance, not a model work budget.


def _require(condition, message):
    if not condition:
        raise AzureProtocolError(message)


@dataclass(frozen=True)
class AzureRuntimeConfig:
    """Server-owned target and access grant, never request-body configuration."""
    settings: AzureSettings
    destination_address: str
    resource: str
    identity_client_id: str | None = None
    network_execution_authorized: bool = False

    def __post_init__(self):
        _require(type(self.settings) is AzureSettings, "trusted Azure settings required")
        numeric_address(self.destination_address)
        _require(type(self.network_execution_authorized) is bool, "invalid network authorization")
        _require(isinstance(self.resource, str) and 0 < len(self.resource) <= 2048
                 and all(32 < ord(c) < 127 for c in self.resource)
                 and not self.resource.endswith("/.default"), "invalid API resource URI")
        try:
            parsed = urlsplit(self.resource)
            valid = (parsed.scheme in ("api", "https") and parsed.hostname
                     and not parsed.username and not parsed.password
                     and not parsed.query and not parsed.fragment)
            _ = parsed.port
        except ValueError:
            valid = False
        _require(valid, "invalid API resource URI")
        if self.identity_client_id is not None:
            try:
                client = UUID(self.identity_client_id)
            except (ValueError, TypeError, AttributeError):
                raise AzureProtocolError("invalid managed identity client ID") from None
            _require(str(client) == self.identity_client_id, "invalid managed identity client ID")

    def require_authorized(self):
        _execution_guard(self.settings)
        if not self.network_execution_authorized:
            raise AzureExecutionBlocked("Azure network execution remains disabled")


class AzureHTTPTransport:
    def __init__(self, config, *, cancelled=lambda: False):
        _require(type(config) is AzureRuntimeConfig, "trusted Azure runtime configuration required")
        _require(callable(cancelled), "cancellation callback required")
        self._config = config
        self._cancelled = cancelled

    def __call__(self, request, headers):
        config = self._config
        config.require_authorized()  # Must precede sockets, TLS and I/O.
        _require(type(request) is AzureRequest, "invalid Azure transport request")
        _require(request.url == config.settings.origin + OPENAI_CHAT_ROUTE
                 and request.method == "POST" and request.follow_redirects is False
                 and request.verify_tls is True and type(request.stream) is bool,
                 "Azure transport target or controls changed")
        _require(type(request.timeout_seconds) in (int, float)
                 and math.isfinite(request.timeout_seconds)
                 and 0 < request.timeout_seconds <= config.settings.timeout_seconds,
                 "invalid remaining transport budget")
        _require(isinstance(headers, dict) and set(headers) == {"Authorization", "Content-Type", "Accept"},
                 "invalid Azure transport headers")
        _require(headers["Content-Type"] == "application/json"
                 and headers["Accept"] == ("text/event-stream" if request.stream else "application/json"),
                 "invalid Azure request content types")
        authorization = headers["Authorization"]
        _require(isinstance(authorization, str) and authorization.startswith("Bearer ")
                 and len(authorization) <= 16391
                 and _BEARER.fullmatch(authorization[7:]) is not None,
                 "invalid Azure bearer credential")
        parsed = urlsplit(config.settings.origin)
        body = open_http_response(
            host=parsed.hostname, address=config.destination_address, port=443,
            target=OPENAI_CHAT_ROUTE, method="POST", headers=headers, body=request.body,
            timeout_seconds=request.timeout_seconds, cancelled=self._cancelled,
        )
        return AzureResponse(body.status, body.content_type, request.url, body, body.close)


def _identity_endpoint(environment):
    _require(isinstance(environment, Mapping), "identity environment unavailable")
    endpoint = environment.get("IDENTITY_ENDPOINT")
    header = environment.get("IDENTITY_HEADER")
    _require(isinstance(endpoint, str) and 0 < len(endpoint) <= 2048
             and all(32 < ord(c) < 127 for c in endpoint), "identity endpoint unavailable")
    _require(isinstance(header, str) and 0 < len(header) <= 16384
             and all(32 < ord(c) < 127 for c in header), "identity header unavailable")
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port if parsed.port is not None else 80
        valid = (parsed.scheme == "http" and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment and parsed.hostname
                 and re.fullmatch(r"/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+", parsed.path))
    except ValueError:
        valid = False
    _require(valid, "invalid local identity endpoint")
    address = numeric_address(parsed.hostname)
    _require(address.is_loopback or address.is_link_local, "identity endpoint is not local")
    _require(1 <= port <= 65535, "invalid local identity port")
    return parsed.hostname, port, parsed.path, header


class ManagedIdentityCredential:
    """Get a short-lived token from the local Container Apps identity endpoint.

    Resource and optional user-assigned client ID are explicit. Header rotation is
    respected by reading the platform environment per token request. There is no
    fallback to Azure OpenAI audiences, API keys, CLI login, or IMDS autodiscovery.
    Token signatures/issuer/roles must be verified by the receiving API; parsing a
    token response on this client does not prove caller authorization.
    """
    def __init__(self, config, *, environment=lambda: os.environ, cancelled=lambda: False,
                 wall_clock=time):
        _require(type(config) is AzureRuntimeConfig, "trusted Azure runtime configuration required")
        _require(all(callable(value) for value in (environment, cancelled, wall_clock)),
                 "trusted identity dependencies required")
        self._config = config
        self._environment = environment
        self._cancelled = cancelled
        self._wall_clock = wall_clock

    def __call__(self):
        config = self._config
        config.require_authorized()  # No environment or credential access when disabled.
        try:
            environment = self._environment()
        except Exception:
            raise AzureProtocolError("identity environment unavailable") from None
        host, port, path, header = _identity_endpoint(environment)
        params = {"resource": config.resource, "api-version": IDENTITY_API_VERSION}
        if config.identity_client_id is not None:
            params["client_id"] = config.identity_client_id
        body = open_http_response(
            host=host, address=host, port=port,
            target=path + "?" + urlencode(params, quote_via=quote), method="GET",
            headers={"X-IDENTITY-HEADER": header, "Accept": "application/json"}, body=b"",
            timeout_seconds=config.settings.timeout_seconds, max_response_bytes=IDENTITY_MAX_BYTES,
            tls=False, cancelled=self._cancelled,
        )
        try:
            _require(body.status == 200 and body.content_type.split(";", 1)[0].strip().lower() == "application/json",
                     "identity HTTP response rejected")
            chunks = []
            total = 0
            for part in body:
                _require(isinstance(part, bytes) and part, "invalid identity response bytes")
                total += len(part)
                _require(total <= IDENTITY_MAX_BYTES, "identity response too large")
                chunks.append(part)
            try:
                value = json.loads(b"".join(chunks).decode("utf-8"),
                                   parse_constant=_reject_json_constant, object_pairs_hook=_unique_json_object)
            except (ValueError, UnicodeError, RecursionError):
                raise AzureProtocolError("invalid identity response JSON") from None
            _require(isinstance(value, dict) and "error" not in value, "identity token request failed")
            token = value.get("access_token")
            _require(isinstance(token, str) and 0 < len(token) <= 16384
                     and _BEARER.fullmatch(token) is not None, "invalid identity access token")
            _require(value.get("token_type") == "Bearer" and value.get("resource") == config.resource,
                     "identity token resource or type mismatch")
            if config.identity_client_id is not None:
                _require(value.get("client_id") == config.identity_client_id,
                         "managed identity client mismatch")
            expiry = value.get("expires_on")
            _require(type(expiry) is int or (isinstance(expiry, str) and re.fullmatch(r"[0-9]{1,16}", expiry)),
                     "invalid identity token expiry")
            now = self._wall_clock()
            _require(type(now) in (int, float) and math.isfinite(now), "invalid credential clock")
            _require(int(expiry) > now + config.settings.timeout_seconds + TOKEN_CLOCK_SKEW_SECONDS,
                     "identity token expires before safe request completion")
            return token
        finally:
            body.close()


def make_azure_runtime_client(config, *, environment=lambda: os.environ, cancelled=lambda: False):
    """Connect concrete clients without authorizing or making any network calls."""
    _require(type(config) is AzureRuntimeConfig, "trusted Azure runtime configuration required")
    client = make_azure_inference_client(
        config.settings,
        AzureHTTPTransport(config, cancelled=cancelled),
        ManagedIdentityCredential(config, environment=environment, cancelled=cancelled),
        cancelled=cancelled,
    )

    def inference_client(engine_request):
        config.require_authorized()
        return client(engine_request)

    return inference_client
