"""Offline-capable Entra app-token verification for a future receiving API.

No listener, credentials, discovery HTTP request, provider call, or key refresh is
started here. A trusted server controller supplies the policy and separately
approved public-key snapshot. Returned service identity is NOT a Kova user, plan,
ExecutionGrant, or permission to spend on inference. Signature work uses PyJWT
and cryptography, not a custom cryptographic implementation.
"""

import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from time import time
from types import MappingProxyType
from urllib.parse import urlsplit
from uuid import UUID


MAX_TOKEN_CHARS = 16384
MAX_SNAPSHOT_BYTES = 128 * 1024
ALGORITHM = "RS256"
_B64 = re.compile(r"[A-Za-z0-9_-]+\Z", re.ASCII)
_SHA256 = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_KEY_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z", re.ASCII)
_ROLE = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z", re.ASCII)


class ReceiverAuthError(ValueError):
    """Authentication rejected without exposing tokens, claims or key bytes."""


class ReceiverAuthUnavailable(ReceiverAuthError):
    """Receiver authentication is disabled, stale or not configured."""


def _require(condition):
    if not condition:
        raise ReceiverAuthError("receiver authentication rejected")


def _guid(value):
    if not isinstance(value, str):
        return False
    try:
        return str(UUID(value)) == value and UUID(value).int != 0
    except ValueError:
        return False


def _integer(value, maximum=253402300799):
    return type(value) is int and 0 < value <= maximum


def _json(data):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value)
            value[key] = item
        return value
    def reject(_value):
        raise ReceiverAuthError("receiver authentication rejected")
    def finite(raw):
        value = float(raw)
        _require(math.isfinite(value))
        return value
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=reject, parse_float=finite)
        _require(isinstance(value, dict))
        pending, nodes = [(value, 0)], 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            _require(nodes <= 4096 and depth <= 16)
            if isinstance(item, dict):
                pending.extend((v, depth + 1) for pair in item.items() for v in pair)
            elif isinstance(item, list):
                pending.extend((v, depth + 1) for v in item)
            elif isinstance(item, str):
                item.encode("utf-8")
        return value
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ReceiverAuthError("receiver authentication rejected") from None


def _unbase64(value, maximum):
    _require(isinstance(value, str) and 0 < len(value) <= maximum and _B64.fullmatch(value))
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, base64.binascii.Error):
        raise ReceiverAuthError("receiver authentication rejected") from None
    _require(base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") == value)
    return decoded


@dataclass(frozen=True)
class AllowedService:
    client_id: str
    object_id: str

    def __post_init__(self):
        _require(_guid(self.client_id) and _guid(self.object_id))


@dataclass(frozen=True)
class ReceiverPolicy:
    """Server-owned, single-tenant, single-token-version app-only policy.

    Pair client ID with service-principal object ID; do not independently match
    two allowlists. Time values must be explicitly approved, not borrowed from
    model response-time targets. Only public Azure Entra issuer forms are enabled.
    """

    tenant_id: str
    audience: str
    token_version: str
    allowed_services: tuple[AllowedService, ...]
    required_roles: tuple[str, ...]
    keyset_sha256: str
    keyset_valid_until_epoch: int
    max_token_lifetime_seconds: int
    clock_skew_seconds: int
    enabled: bool = False

    def __post_init__(self):
        _require(_guid(self.tenant_id) and self.token_version in ("1.0", "2.0"))
        _require(isinstance(self.audience, str) and 0 < len(self.audience) <= 2048
                 and all(32 < ord(c) < 127 for c in self.audience))
        if self.token_version == "2.0":
            _require(_guid(self.audience))
        else:
            try:
                parsed = urlsplit(self.audience)
                _require(parsed.scheme in ("https", "api") and parsed.hostname
                         and not parsed.username and not parsed.password
                         and not parsed.query and not parsed.fragment and parsed.port is None)
            except ValueError:
                raise ReceiverAuthError("receiver authentication rejected") from None
        _require(type(self.allowed_services) is tuple and 1 <= len(self.allowed_services) <= 64
                 and all(type(s) is AllowedService for s in self.allowed_services)
                 and len(set(self.allowed_services)) == len(self.allowed_services))
        _require(type(self.required_roles) is tuple and 1 <= len(self.required_roles) <= 16
                 and all(isinstance(r, str) and _ROLE.fullmatch(r) for r in self.required_roles)
                 and len(set(self.required_roles)) == len(self.required_roles))
        _require(isinstance(self.keyset_sha256, str) and _SHA256.fullmatch(self.keyset_sha256))
        _require(_integer(self.keyset_valid_until_epoch)
                 and _integer(self.max_token_lifetime_seconds, 86400)
                 and type(self.clock_skew_seconds) is int and 0 <= self.clock_skew_seconds <= 300
                 and type(self.enabled) is bool)

    @property
    def issuer(self):
        if self.token_version == "1.0":
            return f"https://sts.windows.net/{self.tenant_id}/"
        return f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"


@dataclass(frozen=True)
class VerifiedServiceIdentity:
    tenant_id: str
    client_id: str
    object_id: str
    expires_at_epoch: int
    required_roles: tuple[str, ...]
    # Deliberately no user ID, subscription tier, raw token, or unfiltered claims.


def _load_keys(policy, snapshot):
    _require(isinstance(snapshot, bytes) and 0 < len(snapshot) <= MAX_SNAPSHOT_BYTES)
    _require(hashlib.sha256(snapshot).hexdigest() == policy.keyset_sha256)
    value = _json(snapshot)
    _require(set(value) == {"schema_version", "issuer", "keys"}
             and type(value["schema_version"]) is int and value["schema_version"] == 1
             and value["issuer"] == policy.issuer)
    keys = value["keys"]
    _require(isinstance(keys, list) and 1 <= len(keys) <= 16)
    # The reviewed snapshot is normalized from the configured tenant's signing
    # metadata. No URL or embedded key supplied by a request is ever followed.
    import jwt
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
    result = {}
    for key in keys:
        _require(isinstance(key, dict) and set(key) == {"kty", "use", "alg", "kid", "n", "e"})
        _require(key["kty"] == "RSA" and key["use"] == "sig" and key["alg"] == ALGORITHM
                 and isinstance(key["kid"], str) and _KEY_ID.fullmatch(key["kid"])
                 and key["kid"] not in result)
        modulus = _unbase64(key["n"], 1366)
        exponent = _unbase64(key["e"], 8)
        _require(256 <= len(modulus) <= 1024 and modulus[0] != 0
                 and int.from_bytes(exponent, "big") == 65537)
        try:
            public = jwt.PyJWK.from_dict(key, algorithm=ALGORITHM).key
        except Exception:
            raise ReceiverAuthError("receiver authentication rejected") from None
        _require(isinstance(public, RSAPublicKey) and 2048 <= public.key_size <= 8192)
        result[key["kid"]] = public
    return MappingProxyType(result)


@dataclass(frozen=True, init=False)
class ReceiverAuthenticator:
    policy: ReceiverPolicy
    _keys: object = field(repr=False)

    def __init__(self, policy, snapshot):
        _require(type(policy) is ReceiverPolicy)
        object.__setattr__(self, "policy", policy)
        # Disabled construction intentionally does not read/import the optional
        # cryptographic backend or parse an externally supplied key snapshot.
        if not policy.enabled:
            object.__setattr__(self, "_keys", MappingProxyType({}))
            return
        try:
            keys = _load_keys(policy, snapshot)
        except ImportError:
            raise ReceiverAuthUnavailable("receiver authentication backend unavailable") from None
        object.__setattr__(self, "_keys", keys)

    def _check_available(self):
        now = time()
        if (not self.policy.enabled or type(now) not in (int, float)
                or not math.isfinite(now) or now < 0
                or now >= self.policy.keyset_valid_until_epoch):
            raise ReceiverAuthUnavailable("receiver authentication unavailable")
        return now

    def __call__(self, token):
        self._check_available()
        try:
            _require(isinstance(token, str) and 0 < len(token) <= MAX_TOKEN_CHARS)
            parts = token.split(".")
            _require(len(parts) == 3)
            header = _json(_unbase64(parts[0], 2048))
            claims = _json(_unbase64(parts[1], MAX_TOKEN_CHARS))
            signature = _unbase64(parts[2], 1366)
            _require(256 <= len(signature) <= 1024)
            _require({"alg", "typ", "kid"} <= set(header)
                     and set(header) <= {"alg", "typ", "kid", "x5t"}
                     and header["alg"] == ALGORITHM and header["typ"] == "JWT"
                     and isinstance(header["kid"], str) and _KEY_ID.fullmatch(header["kid"]))
            if "x5t" in header:
                _unbase64(header["x5t"], 128)  # Metadata only, not a key selector.
            public = self._keys.get(header["kid"])
            _require(public is not None)
            client_claim = "appid" if self.policy.token_version == "1.0" else "azp"
            other_client_claim = "azp" if client_claim == "appid" else "appid"
            required = {"iss", "aud", "exp", "nbf", "iat", "ver", "tid", "oid", "sub", "idtyp", "roles", client_claim}
            _require(required <= set(claims) and other_client_claim not in claims)
            _require(not ({"scp", "nonce", "act", "may_act"} & set(claims)))
            _require(all(_integer(claims[k]) for k in ("iat", "nbf", "exp")))
            _require(claims["iat"] < claims["exp"] and claims["nbf"] < claims["exp"]
                     and claims["exp"] - claims["iat"] <= self.policy.max_token_lifetime_seconds)
            _require(claims["iss"] == self.policy.issuer and claims["aud"] == self.policy.audience
                     and claims["ver"] == self.policy.token_version and claims["tid"] == self.policy.tenant_id
                     and claims["idtyp"] == "app" and _guid(claims[client_claim]) and _guid(claims["oid"])
                     and isinstance(claims["sub"], str) and 0 < len(claims["sub"]) <= 256)
            roles = claims["roles"]
            _require(isinstance(roles, list) and 1 <= len(roles) <= 64
                     and all(isinstance(r, str) and _ROLE.fullmatch(r) for r in roles)
                     and len(set(roles)) == len(roles)
                     and set(self.policy.required_roles) <= set(roles))
            _require(AllowedService(claims[client_claim], claims["oid"]) in self.policy.allowed_services)
            # Above are cheap rejection filters only. No identity is accepted until
            # the maintained backend verifies the ORIGINAL signed compact token.
            import jwt
            verified = jwt.decode(
                token, public, algorithms=[ALGORITHM], issuer=self.policy.issuer,
                audience=self.policy.audience, leeway=self.policy.clock_skew_seconds,
                options={"require": sorted(required), "strict_aud": True,
                         "verify_signature": True, "verify_exp": True, "verify_nbf": True,
                         "verify_iat": True, "verify_iss": True, "verify_aud": True},
            )
            _require(verified == claims)
            self._check_available()
            return VerifiedServiceIdentity(self.policy.tenant_id, claims[client_claim], claims["oid"],
                                           claims["exp"], self.policy.required_roles)
        except ReceiverAuthUnavailable:
            raise
        except Exception:
            raise ReceiverAuthError("receiver authentication rejected") from None

    def from_headers(self, headers):
        """Consume raw header pairs so duplicate Authorization cannot be hidden.

        Trusted HTTP integration must supply the complete raw header list before
        collapsing duplicates and enforce TLS/body limits. Cookie, forwarded-user,
        owner, tier and x-ms-client-principal headers confer no identity here.
        """
        self._check_available()
        _require(type(headers) in (tuple, list) and len(headers) <= 64)
        authorization = []
        total = 0
        for pair in headers:
            _require(type(pair) in (tuple, list) and len(pair) == 2)
            name, value = pair
            _require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9-]{1,64}", name)
                     and isinstance(value, str) and len(value) <= MAX_TOKEN_CHARS + 7
                     and all(32 <= ord(c) < 127 for c in value))
            total += len(name) + len(value)
            _require(total <= 32768)
            if name.lower() == "authorization":
                authorization.append(value)
        _require(len(authorization) == 1 and authorization[0].startswith("Bearer "))
        return self(authorization[0][7:])
