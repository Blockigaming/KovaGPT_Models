"""Actual loopback socket/TLS tests. No external networking or Azure credentials.

A short-lived test CA/key is generated into a temporary directory and removed.
No key is checked into the repository. Production hostname verification stays on.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import socket
import ssl
import subprocess
import tempfile
import threading
from time import monotonic, sleep
import unittest
from unittest.mock import patch

from worker import bounded_http as http
from worker.azure_container_apps import AzureCancelled, AzureDeadlineExceeded, AzureProtocolError


HOST = "kova-core.internal.fixture.eastus.azurecontainerapps.io"
JSON_BODY = b'{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}'


class FixtureServer:
    def __init__(self, action, tls_context=None):
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._listener.listen(1)
        self._listener.settimeout(2)
        self.action = action
        self.context = tls_context
        self.request = b""
        self.error = None
        self.started = threading.Event()
        self.stop = threading.Event()
        self.sni = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._connection = None

    def _run(self):
        try:
            connection, _ = self._listener.accept()
            self._connection = connection
            connection.settimeout(2)
            if self.context:
                connection = self.context.wrap_socket(connection, server_side=True)
                self._connection = connection
            if self.action == "stall_headers":
                self.started.set()
                self.stop.wait(2)
                return
            while b"\r\n\r\n" not in self.request:
                chunk = connection.recv(8192)
                if not chunk:
                    return
                self.request += chunk
            self.started.set()
            self.action(connection, self)
        except (ConnectionError, ssl.SSLError, OSError) as error:
            # Expected for certificate rejection, timeout and consumer-close fixtures.
            self.error = type(error).__name__
        finally:
            if self._connection is not None:
                self._connection.close()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self._listener.close()
        self._thread.join(3)
        if self._thread.is_alive():
            raise AssertionError("loopback fixture did not stop")


def json_action(connection, _server):
    connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                       + str(len(JSON_BODY)).encode() + b"\r\n\r\n" + JSON_BODY)


def reply(wire):
    return lambda connection, _server: connection.sendall(wire)


class BoundedHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="kova-tls-fixture-")
        cls.cert = str(Path(cls.directory.name) / "cert.pem")
        cls.key = str(Path(cls.directory.name) / "key.pem")
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-keyout", cls.key, "-out", cls.cert, "-subj", f"/CN={HOST}",
            "-addext", f"subjectAltName=DNS:{HOST}",
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def server_context(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.cert, self.key)
        return context

    def client_context(self):
        context = ssl.create_default_context(cafile=self.cert)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        return context

    def open(self, server, **changes):
        options = dict(host="127.0.0.1", address="127.0.0.1", port=server.port,
                       target="/msi/token?resource=fixture", method="GET", headers={"Accept": "application/json"},
                       body=b"", timeout_seconds=1, tls=False)
        options.update(changes)
        return http.open_http_response(**options)

    def test_plain_local_identity_json_no_dns_or_proxy_lookup(self):
        with FixtureServer(json_action) as server, patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS called")):
            body = self.open(server)
            self.assertEqual(b"".join(body), JSON_BODY)
            self.assertTrue(body.closed)
            self.assertIn(b"Accept-Encoding: identity", server.request)
            self.assertIn(b"Connection: close", server.request)

    def test_real_tls_checks_hostname_and_uses_hostname_not_ip_for_host_and_sni(self):
        seen = []
        context = self.server_context()
        context.set_servername_callback(lambda _socket, hostname, _context: seen.append(hostname))
        with FixtureServer(json_action, context) as server:
            body = self.open(server, host=HOST, tls=True, context=self.client_context())
            self.assertEqual(b"".join(body), JSON_BODY)
            self.assertIn(f"Host: {HOST}:{server.port}\r\n".encode(), server.request)
            self.assertEqual(seen, [HOST])

    def test_bad_certificate_chain_is_rejected(self):
        with FixtureServer(json_action, self.server_context()) as server:
            with self.assertRaises(AzureProtocolError):
                self.open(server, host=HOST, tls=True)  # System roots do not trust the fixture CA.
            self.assertEqual(server.request, b"")
        self.assertEqual(server.error, "SSLError")

    def test_hostname_mismatch_is_rejected_even_with_trusted_test_ca(self):
        with FixtureServer(json_action, self.server_context()) as server:
            with self.assertRaises(AzureProtocolError):
                self.open(server, host="wrong.azurecontainerapps.io", tls=True, context=self.client_context())
            self.assertEqual(server.request, b"")
        self.assertEqual(server.error, "SSLError")

    def test_unverified_tls_context_rejected_before_socket_creation(self):
        context = ssl._create_unverified_context()
        with patch.object(http, "_SOCKET_FACTORY", side_effect=AssertionError("socket created")):
            with self.assertRaisesRegex(AzureProtocolError, "verified TLS"):
                http.open_http_response(host=HOST, address="127.0.0.1", port=443, target="/",
                                        method="GET", headers={}, body=b"", timeout_seconds=1, context=context)

    def test_deadline_interrupts_stalled_tls_handshake(self):
        with FixtureServer("stall_headers") as server:
            started = monotonic()
            with self.assertRaises(AzureDeadlineExceeded):
                self.open(server, host=HOST, tls=True, context=self.client_context(), timeout_seconds=0.15)
            self.assertLess(monotonic() - started, 1.5)

    def test_deadline_interrupts_stalled_request_write(self):
        original_socket = socket.socket
        def small_send_buffer(family, kind):
            sock = original_socket(family, kind)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            return sock
        with FixtureServer("stall_headers") as server, patch.object(http, "_SOCKET_FACTORY", small_send_buffer):
            started = monotonic()
            with self.assertRaises(AzureDeadlineExceeded):
                self.open(server, method="POST", body=b"x" * (4 * 1024 * 1024), timeout_seconds=0.15)
            self.assertLess(monotonic() - started, 1.5)

    def test_connection_failure_does_not_retry(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.close()
        sockets = []
        def tracked_socket(family, kind):
            sock = socket.socket(family, kind)
            sockets.append(sock)
            return sock
        with patch.object(http, "_SOCKET_FACTORY", tracked_socket):
            with self.assertRaises(AzureProtocolError):
                http.open_http_response(host="127.0.0.1", address="127.0.0.1", port=port,
                                        target="/msi/token", method="GET", headers={}, body=b"",
                                        timeout_seconds=1, tls=False)
        self.assertEqual(len(sockets), 1)
        self.assertEqual(sockets[0].fileno(), -1)

    def test_deadline_interrupts_stalled_headers(self):
        with FixtureServer("stall_headers") as server:
            started = monotonic()
            with self.assertRaises(AzureDeadlineExceeded):
                self.open(server, timeout_seconds=0.15)
            self.assertLess(monotonic() - started, 1.5)

    def test_cancellation_interrupts_a_blocked_header_read(self):
        cancelled = threading.Event()
        with FixtureServer("stall_headers") as server:
            timer = threading.Timer(0.1, cancelled.set)
            timer.start()
            try:
                with self.assertRaises(AzureCancelled):
                    self.open(server, cancelled=cancelled.is_set)
            finally:
                timer.cancel()
                timer.join()

    def test_deadline_and_cancellation_interrupt_stalled_body(self):
        def stalled(connection, server):
            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {}\n\n")
            server.stop.wait(2)
        for cancel in (False, True):
            with self.subTest(cancel=cancel), FixtureServer(stalled) as server:
                event = threading.Event()
                body = self.open(server, timeout_seconds=0.15 if not cancel else 1, cancelled=event.is_set)
                self.assertEqual(next(body), b"data: {}\n\n")
                timer = threading.Timer(0.1, event.set) if cancel else None
                if timer:
                    timer.start()
                try:
                    with self.assertRaises(AzureCancelled if cancel else AzureDeadlineExceeded):
                        next(body)
                    self.assertTrue(body.closed)
                finally:
                    if timer:
                        timer.cancel()
                        timer.join()
                    body.close()

    def test_trickle_bytes_do_not_reset_overall_deadline(self):
        def trickle(connection, server):
            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n")
            while not server.stop.wait(0.015):
                connection.sendall(b" ")
        with FixtureServer(trickle) as server:
            body = self.open(server, timeout_seconds=0.15)
            started = monotonic()
            with self.assertRaises(AzureDeadlineExceeded):
                list(body)
            self.assertTrue(body.closed)
            self.assertLess(monotonic() - started, 1.5)

    def test_consumer_close_aborts_socket_and_is_idempotent(self):
        def chunks(connection, server):
            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {}\n\n")
            server.stop.wait(2)
        with FixtureServer(chunks) as server:
            body = self.open(server)
            self.assertTrue(next(body))
            sock = body._wire.socket
            body.close()
            body.close()
            self.assertEqual(sock.fileno(), -1)
            self.assertEqual(list(body), [])

    def test_chunked_stream_is_decoded_without_buffering_full_response(self):
        data = b"data: {}\n\ndata: [DONE]\n\n"
        wire = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                + f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
        with FixtureServer(reply(wire)) as server:
            self.assertEqual(b"".join(self.open(server)), data)

    def test_status_redirects_framing_compression_and_types_fail_closed(self):
        wires = [
            b"HTTP/1.1 302 Found\r\nLocation: https://evil.example\r\nContent-Length: 0\r\n\r\n",
            b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n",
            b"HTTP/1.1 429 Slow\r\nContent-Length: 0\r\n\r\n",
        ]
        for headers in (
            b"Content-Length: 1\r\nContent-Length: 1\r\nContent-Type: application/json",
            b"Content-Length: -1\r\nContent-Type: application/json",
            b"Content-Length: 1\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json",
            b"Transfer-Encoding: unsupported\r\nContent-Type: application/json",
            b"Content-Encoding: gzip\r\nContent-Type: application/json",
            b"Content-Type: text/html",
            b"Content-Type: application/json\r\nContent-Type: text/event-stream",
        ):
            wires.append(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n\r\n")
        for wire in wires:
            with self.subTest(wire=wire[:80]), FixtureServer(reply(wire)) as server:
                with self.assertRaises(AzureProtocolError):
                    self.open(server)

    def test_truncated_content_length_and_chunked_bodies_fail(self):
        for wire in (
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 10\r\n\r\n{}",
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{",
        ):
            with self.subTest(wire=wire), FixtureServer(reply(wire)) as server:
                body = self.open(server)
                with self.assertRaises(AzureProtocolError):
                    list(body)
                self.assertTrue(body.closed)

    def test_length_and_chunked_byte_caps(self):
        for headers in (b"Content-Length: 20\r\n", b""):
            wire = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" + headers + b"\r\n" + b"x" * 20
            with self.subTest(headers=headers), FixtureServer(reply(wire)) as server:
                with self.assertRaisesRegex(AzureProtocolError, "too large"):
                    list(self.open(server, max_response_bytes=10))

    def test_precancelled_and_invalid_targets_never_create_socket(self):
        defaults = dict(host=HOST, address="127.0.0.1", port=443, target="/v1/chat/completions",
                        method="POST", headers={}, body=b"{}", timeout_seconds=1)
        changes = [
            {"address": "evil.example"}, {"address": "0.0.0.0"}, {"address": "127.000.0.1"},
            {"address": "ff02::1"}, {"port": True}, {"target": "/\nHeader: bad"},
            {"target": "//evil.example"}, {"headers": {"Host": "evil.example"}},
            {"headers": {"Authorization": "Bearer\r\nsecret"}}, {"tls": False},
            {"headers": {"X-IDENTITY-HEADER": "must-not-leave-local-endpoint"}},
            {"timeout_seconds": float("inf")}, {"timeout_seconds": True},
        ]
        with patch.object(http, "_SOCKET_FACTORY", side_effect=AssertionError("socket created")):
            for change in changes:
                with self.subTest(change=change), self.assertRaises(AzureProtocolError):
                    http.open_http_response(**(defaults | change))
            with self.assertRaises(AzureCancelled):
                http.open_http_response(**defaults, cancelled=lambda: True)


if __name__ == "__main__":
    unittest.main()
