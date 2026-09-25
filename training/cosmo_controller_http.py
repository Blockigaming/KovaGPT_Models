"""Authenticated WSGI endpoint for one Cosmo grant; does not start a server.

A separately approved TLS host loads build_application from its private config.
The app never initializes storage, creates resources, or releases the trainer.
Request bodies cannot select controller keys, quotes, Azure scopes or files.
"""
import hashlib
import hmac
from importlib import metadata
import os
from pathlib import Path
import re
import stat
import sys
from urllib.parse import urlsplit

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from training import cosmo_lifecycle_authority as authority
from training import cosmo_qlora_launch as launch
from training.cosmo_controller_azure import (AzureReadIO, AzureRequestVerifier, cli_token,
                                              managed_identity_token)
from training.cosmo_controller_grants import GrantIssuer
from training.cosmo_controller_ledger import AzureBlobIO, ControllerLedger, LedgerRejected, need, parse_json
from training.cosmo_adapter_preservation import (AdapterPreserver, MAX_REQUEST,
                                                  preservation_endpoint)


class GrantApplication:
    def __init__(self, *, issuer, endpoint, bearer_token, preserver=None,
                 trusted_ingress_mode=None):
        preserve = urlsplit(preservation_endpoint(endpoint)) if preserver is not None else None
        endpoint = urlsplit(endpoint)
        need(endpoint.scheme == "https" and endpoint.hostname and endpoint.path and
             not endpoint.query and not endpoint.fragment and not endpoint.username and
             not endpoint.password and endpoint.port in (None, 443), "pinned HTTPS endpoint required")
        need(type(bearer_token) is str and authority.TOKEN.fullmatch(bearer_token),
             "controller bearer token required")
        self.issuer, self.host, self.path = issuer, endpoint.netloc, endpoint.path
        self.preserver, self.preserve_path = preserver, preserve.path if preserve else None
        need(trusted_ingress_mode in (None, "azure_container_apps_https"),
             "unsupported controller ingress")
        self.proxy_https = trusted_ingress_mode == "azure_container_apps_https"
        self.token_hash = hashlib.sha256(bearer_token.encode("ascii")).digest()

    def __call__(self, environ, start_response):
        status, value = "403 Forbidden", {"error": "grant_request_rejected"}
        try:
            # ACA terminates TLS at its Envoy ingress. The proxy variant may
            # only run on a host with HTTPS-only ingress and no direct port.
            scheme = environ.get("wsgi.url_scheme")
            tls = (scheme == "https" or
                   (self.proxy_https and scheme == "http" and
                    environ.get("HTTP_X_FORWARDED_PROTO") == "https"))
            need(tls and
                 environ.get("HTTP_HOST") == self.host and
                 environ.get("PATH_INFO") in (self.path, self.preserve_path) and
                 not environ.get("QUERY_STRING") and
                 environ.get("REQUEST_METHOD") == "POST", "invalid endpoint")
            auth = environ.get("HTTP_AUTHORIZATION", "")
            need(type(auth) is str and auth.startswith("Bearer ") and len(auth) <= 16391 and
                 hmac.compare_digest(hashlib.sha256(auth[7:].encode("ascii")).digest(), self.token_hash),
                 "invalid endpoint credential")
            need(environ.get("CONTENT_TYPE", "").lower() in ("application/json", "application/json; charset=utf-8") and
                 not environ.get("HTTP_TRANSFER_ENCODING") and not environ.get("HTTP_CONTENT_ENCODING"),
                 "unsupported body encoding")
            length = environ.get("CONTENT_LENGTH", "")
            bound = MAX_REQUEST if environ.get("PATH_INFO") == self.preserve_path else 65536
            need(type(length) is str and re.fullmatch(r"[1-9][0-9]{0,7}", length) and
                 int(length) <= bound, "bounded body required")
            raw = environ["wsgi.input"].read(int(length))
            need(len(raw) == int(length), "incomplete body")
            request = parse_json(raw)
            value = (self.preserver.submit(request) if
                     environ.get("PATH_INFO") == self.preserve_path else self.issuer.issue(request))
            status = "200 OK"
        except Exception:
            # Never return token, Azure error, signed quote or ledger internals.
            # An uncertain append stays consumed; this handler does not retry it.
            pass
        raw = authority.canonical(value)
        start_response(status, [("Content-Type", "application/json"),
            ("Content-Length", str(len(raw))), ("Cache-Control", "no-store"),
            ("X-Content-Type-Options", "nosniff")])
        return [raw]


def private_file(path, root, maximum=65536):
    path = Path(path)
    repository = root.resolve(strict=True)
    need(path.is_absolute() and ".." not in path.parts and
         repository not in path.parents and path != repository,
         "private control file missing")
    directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        # Pin every ancestor by descriptor. A symlink or writable ancestor
        # cannot redirect the final open after its protection was checked.
        for name in path.parts[1:-1]:
            child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY |
                               os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
            directory = os.fstat(directory_fd)
            need(directory.st_uid in (0, os.geteuid()) and
                 (directory.st_mode & 0o022 == 0 or
                  (directory.st_uid == 0 and directory.st_mode & stat.S_ISVTX)),
                 "private control directory is writable by another principal")
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW |
                          os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=directory_fd)
        try:
            identity = os.fstat(file_fd)
            need(stat.S_ISREG(identity.st_mode) and identity.st_uid == os.geteuid() and
                 identity.st_nlink == 1 and identity.st_mode & 0o077 == 0 and
                 0 < identity.st_size <= maximum,
                 "control file must be protected and outside the repository")
            with os.fdopen(file_fd, "rb", closefd=False) as stream:
                raw = stream.read(maximum + 1)
            need(len(raw) == identity.st_size and os.fstat(file_fd).st_size == identity.st_size,
                 "control file changed during read")
            return raw
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


def verify_auth_runtime(root):
    need(sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12),
         "controller requires CPython 3.12")
    expected = {}
    for line in (root / "requirements/receiver-auth-py312-linux.lock").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_-]+)==([0-9.]+) --hash=sha256:[0-9a-f]{64}", line)
        need(match is not None and match[1] not in expected, "invalid authentication lock")
        expected[match[1]] = match[2]
    need(set(expected) == {"PyJWT", "cryptography", "cffi", "pycparser"},
         "authentication lock incomplete")
    try:
        need(all(metadata.version(name) == version for name, version in expected.items()),
             "controller authentication dependency version mismatch")
    except metadata.PackageNotFoundError:
        raise LedgerRejected("controller authentication dependency missing") from None


def build_application(config_path, *, root=launch.ROOT):
    """Wire the production read adapter, ledger, issuer and HTTP boundary.

    Merely building the app makes no Azure requests. The existing ledger must
    already have independently verified health/cost admissions before a POST.
    Host TLS/body/read timeouts and protected storage need separate approval.
    """
    verify_auth_runtime(root)
    config = parse_json(private_file(config_path, root))
    need(type(config) is dict and set(config) == {"ledger_context", "tenant_id", "token_version",
        "watchdog_resource_id", "signing_key_file", "preservation_public_key_hex",
        "cleanup_public_key_hex", "bearer_token_file", "quote_file", "runtime_evidence_file",
        "token_source", "artifact_container", "preservation_signing_key_file",
        "trusted_ingress_mode"},
        "controller configuration shape mismatch")
    token_source = config["token_source"]
    need(type(token_source) is dict, "explicit controller credential source required")
    if token_source == {"kind": "azure_cli"}:
        token_for = cli_token
    else:
        need(set(token_source) == {"kind", "client_id"} and
             token_source["kind"] == "container_app_managed_identity",
             "unsupported controller credential source")
        client_id = token_source["client_id"]
        need(type(client_id) is str and launch.UUID.fullmatch(client_id),
             "pinned controller managed identity required")
        token_for = lambda audience: managed_identity_token(audience, client_id=client_id)
    source = launch.clean_source_commit(root)
    need(config["ledger_context"]["source_commit"] == source, "controller source pin mismatch")
    trust = authority.load_trust_policy(root)
    need(trust["status"] == "authority_pinned", "controller trust not pinned")
    key = Ed25519PrivateKey.from_private_bytes(private_file(config["signing_key_file"], root, 32))
    try:
        token = private_file(config["bearer_token_file"], root, 16384).decode("ascii").strip()
    except UnicodeError:
        raise LedgerRejected("controller bearer token invalid") from None
    need(authority.TOKEN.fullmatch(token) is not None, "controller bearer token invalid")
    for name in ("quote_file", "runtime_evidence_file"):
        private_file(config[name], root)
    context = config["ledger_context"]
    io = AzureBlobIO(account=context["storage_account"], token_for=token_for)
    ledger = ControllerLedger(context=context, signing_key=key, transport=io,
        preservation_public_key=bytes.fromhex(config["preservation_public_key_hex"]),
        cleanup_public_key=bytes.fromhex(config["cleanup_public_key_hex"]))
    preservation_key = Ed25519PrivateKey.from_private_bytes(private_file(
        config["preservation_signing_key_file"], root, 32))
    preserver = AdapterPreserver(ledger=ledger,
        artifact_container=config["artifact_container"], signing_key=preservation_key,
        token_for=token_for)
    reader = AzureReadIO(account=context["storage_account"], tenant_id=config["tenant_id"],
                         token_for=token_for)
    verifier = AzureRequestVerifier(tenant_id=config["tenant_id"], token_version=config["token_version"],
        lifecycle=context["lifecycle"], watchdog_id=config["watchdog_resource_id"], read_json=reader)
    issuer = GrantIssuer(ledger=ledger, quote=Path(config["quote_file"]),
        runtime_evidence=Path(config["runtime_evidence_file"]), verify_live_request=verifier, root=root)
    return GrantApplication(issuer=issuer, endpoint=trust["endpoint"],
                            bearer_token=token, preserver=preserver,
                            trusted_ingress_mode=config["trusted_ingress_mode"])
