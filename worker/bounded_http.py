"""Small, deadline-bounded HTTP/1.1 transport over the standard-library parser.

No DNS, proxies, redirects, retries, credentials or network work at import time.
A reviewed numeric destination is supplied by server configuration; TLS still
verifies the canonical hostname/SNI. Live DNS refresh/ownership checks are a
separate integration gate. Nonblocking sockets bound connect, TLS, headers and
body reads with the same deadline and cancellation signal, without helper threads.
"""

import errno
import http.client
import io
import ipaddress
import math
import re
import select
import socket
import ssl
from time import monotonic

from worker.azure_container_apps import (
    AzureCancelled, AzureDeadlineExceeded, AzureProtocolError,
)


POLL_SECONDS = 0.05
READ_BYTES = 64 * 1024
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
# Used only by tests to map a numeric test connection to a loopback fixture port.
_SOCKET_FACTORY = socket.socket


def _require(condition, message):
    if not condition:
        raise AzureProtocolError(message)


def numeric_address(value):
    """Reject DNS names, scoped addresses and ambiguous/noncanonical IP literals."""
    _require(isinstance(value, str) and "%" not in value, "numeric destination required")
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise AzureProtocolError("numeric destination required") from None
    _require(str(address) == value and not address.is_unspecified and not address.is_multicast,
             "invalid numeric destination")
    return address


class _Budget:
    def __init__(self, seconds, cancelled):
        _require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds < 240,
                 "invalid HTTP time budget")
        _require(callable(cancelled), "cancellation callback required")
        self.deadline = monotonic() + seconds
        self.cancelled = cancelled

    def remaining(self):
        try:
            cancelled = self.cancelled()
        except Exception:
            raise AzureProtocolError("cancellation check failed") from None
        _require(type(cancelled) is bool, "invalid cancellation state")
        if cancelled:
            raise AzureCancelled("Azure HTTP operation cancelled")
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise AzureDeadlineExceeded("Azure HTTP operation deadline exceeded")
        return remaining


class _SocketReader(io.RawIOBase):
    def __init__(self, wire):
        self.wire = wire

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.wire.receive(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class _Wire:
    def __init__(self, host, address, port, context, budget):
        self.budget = budget
        self.socket = None
        self.closed = False
        try:
            self.budget.remaining()
            address = numeric_address(address)
            family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
            self.socket = _SOCKET_FACTORY(family, socket.SOCK_STREAM)
            self.socket.setblocking(False)
            result = self.socket.connect_ex((str(address), port))
            if result not in (0, errno.EISCONN):
                _require(result in (errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY),
                         "HTTP connection failed")
                self.wait(write=True)
                _require(self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) == 0,
                         "HTTP connection failed")
            if context is not None:
                self.socket = context.wrap_socket(
                    self.socket, server_hostname=host, do_handshake_on_connect=False,
                )
                self.socket.setblocking(False)
                while True:
                    self.budget.remaining()
                    try:
                        self.socket.do_handshake()
                        break
                    except ssl.SSLWantReadError:
                        self.wait()
                    except ssl.SSLWantWriteError:
                        self.wait(write=True)
            self.budget.remaining()
        except BaseException:
            self.close()
            raise

    def wait(self, *, write=False):
        while True:
            remaining = self.budget.remaining()
            _require(not self.closed, "HTTP connection is closed")
            readers, writers, _ = select.select(
                [] if write else [self.socket], [self.socket] if write else [], [],
                min(POLL_SECONDS, remaining),
            )
            if readers or writers:
                self.budget.remaining()
                return

    def send(self, data):
        view = memoryview(data)
        while view:
            self.budget.remaining()
            try:
                sent = self.socket.send(view)
                _require(sent > 0, "HTTP request write failed")
                view = view[sent:]
            except ssl.SSLWantReadError:
                self.wait()
            except (ssl.SSLWantWriteError, BlockingIOError):
                self.wait(write=True)

    def receive(self, count):
        while True:
            self.budget.remaining()
            try:
                return self.socket.recv(count)
            except ssl.SSLWantWriteError:
                self.wait(write=True)
            except (ssl.SSLWantReadError, BlockingIOError):
                self.wait()

    def makefile(self, mode):
        _require(mode == "rb", "unsupported HTTP reader mode")
        return io.BufferedReader(_SocketReader(self))

    def close(self):
        if not self.closed:
            self.closed = True
            if self.socket is not None:
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.socket.close()


class HTTPBody:
    """Owns the parser and socket; close is idempotent and stops further reads."""
    def __init__(self, response, wire, limit, expected_length, content_type):
        self.status = response.status
        self.content_type = content_type
        self._response = response
        self._wire = wire
        self._limit = limit
        self._expected_length = expected_length
        self._received = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            self._wire.budget.remaining()
            data = self._response.read1(min(READ_BYTES, self._limit - self._received + 1))
            self._wire.budget.remaining()
            self._received += len(data)
            _require(self._received <= self._limit, "HTTP response too large")
            if not data:
                _require(self._expected_length is None or self._received == self._expected_length,
                         "incomplete HTTP response body")
                self.close()
                raise StopIteration
            return data
        except StopIteration:
            raise
        except (AzureCancelled, AzureDeadlineExceeded, AzureProtocolError):
            self.close()
            raise
        except Exception:
            self.close()
            raise AzureProtocolError("HTTP response read failed") from None

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self._response.close()
            finally:
                self._wire.close()


def open_http_response(*, host, address, port, target, method, headers, body,
                       timeout_seconds, max_response_bytes=MAX_RESPONSE_BYTES,
                       tls=True, context=None, cancelled=lambda: False):
    """Open one HTTP response; caller must consume or close the returned body.

    Numeric address and hostname are distinct on purpose: connecting to a trusted
    address never changes the TLS verification name or HTTP Host. HTTP is allowed
    only for numeric local/link-local managed-identity endpoints, never inference.
    """
    address_object = numeric_address(address)
    _require(type(port) is int and 1 <= port <= 65535, "invalid destination port")
    _require(isinstance(host, str) and 0 < len(host) <= 253
             and re.fullmatch(r"[a-z0-9.:\-]+", host) is not None, "invalid HTTP hostname")
    _require(isinstance(target, str) and target.startswith("/") and not target.startswith("//")
             and len(target) <= 4096 and all(32 < ord(c) < 127 for c in target)
             and "#" not in target, "invalid HTTP request target")
    _require(method in ("GET", "POST") and isinstance(body, bytes)
             and len(body) <= MAX_REQUEST_BYTES and (method != "GET" or not body),
             "invalid HTTP request")
    _require(type(max_response_bytes) is int and 0 < max_response_bytes <= MAX_RESPONSE_BYTES,
             "invalid HTTP response limit")
    _require(type(tls) is bool, "invalid TLS flag")
    if tls:
        if context is None:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        _require(isinstance(context, ssl.SSLContext) and context.check_hostname
                 and context.verify_mode == ssl.CERT_REQUIRED
                 and context.minimum_version >= ssl.TLSVersion.TLSv1_2,
                 "verified TLS 1.2 or newer is required")
    else:
        _require(context is None and host == str(address_object)
                 and (address_object.is_loopback or address_object.is_link_local),
                 "plaintext HTTP is restricted to the local identity endpoint")
    _require(isinstance(headers, dict) and set(headers).issubset({
        "Authorization", "Content-Type", "Accept", "X-IDENTITY-HEADER",
    }), "unsupported HTTP headers")
    _require(all(isinstance(value, str) and 0 < len(value) <= 16391
                 and all(32 <= ord(c) < 127 for c in value) for value in headers.values()),
             "invalid HTTP header value")
    _require(tls or "Authorization" not in headers, "bearer credentials require TLS")
    _require(not tls or "X-IDENTITY-HEADER" not in headers,
             "identity endpoint header cannot be sent to inference")
    host_header = f"[{host}]" if ":" in host else host
    if port != (443 if tls else 80):
        host_header += f":{port}"
    lines = [f"{method} {target} HTTP/1.1", f"Host: {host_header}",
             "Connection: close", "Accept-Encoding: identity"]
    lines.extend(f"{key}: {value}" for key, value in headers.items())
    if method == "POST":
        lines.append(f"Content-Length: {len(body)}")
    message = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body
    budget = _Budget(timeout_seconds, cancelled)
    wire = response = None
    try:
        wire = _Wire(host, address, port, context, budget)
        wire.send(message)
        response = http.client.HTTPResponse(wire)
        response.begin()
        budget.remaining()
        _require(response.status == 200, "HTTP status rejected")
        for name in ("Content-Length", "Transfer-Encoding", "Content-Encoding", "Content-Type"):
            _require(len(response.headers.get_all(name, [])) <= 1, "ambiguous HTTP response headers")
        length = response.getheader("Content-Length")
        transfer = response.getheader("Transfer-Encoding")
        _require(length is None or transfer is None, "ambiguous HTTP response framing")
        expected_length = None
        if length is not None:
            _require(re.fullmatch(r"[0-9]{1,10}", length) is not None, "invalid HTTP content length")
            expected_length = int(length)
            _require(expected_length <= max_response_bytes, "HTTP response too large")
        _require(transfer is None or transfer.lower() == "chunked", "unsupported HTTP transfer encoding")
        encoding = response.getheader("Content-Encoding", "identity")
        _require(encoding.lower() == "identity", "compressed HTTP responses are disabled")
        content_type = response.getheader("Content-Type", "")
        _require(content_type.split(";", 1)[0].strip().lower() in ("application/json", "text/event-stream"),
                 "unsupported HTTP response type")
        return HTTPBody(response, wire, max_response_bytes, expected_length, content_type)
    except BaseException as error:
        if response is not None:
            response.close()
        if wire is not None:
            wire.close()
        if isinstance(error, (AzureCancelled, AzureDeadlineExceeded, AzureProtocolError)):
            raise
        if not isinstance(error, Exception):
            raise
        raise AzureProtocolError("HTTP exchange failed") from None
