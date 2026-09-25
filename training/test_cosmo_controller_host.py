"""Exercise accepted socket stalls against the intended bounded WSGI host."""
import socket
import threading
import unittest
from urllib.request import urlopen
from wsgiref.simple_server import make_server

from training.cosmo_controller_host import BoundedControllerServer


class HostTests(unittest.TestCase):
    def test_partial_request_cannot_block_another_request_indefinitely(self):
        def app(environ, start_response):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"ready"]

        with make_server("127.0.0.1", 0, app,
                         server_class=BoundedControllerServer) as server:
            server.connection_deadline_seconds = 0.25
            thread = threading.Thread(target=server.serve_forever,
                                      kwargs={"poll_interval": 0.01}, daemon=True)
            thread.start()
            try:
                stalled = socket.create_connection(server.server_address, timeout=2)
                with stalled:
                    stalled.settimeout(2)
                    stalled.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n")
                    with urlopen("http://127.0.0.1:%s/" % server.server_port,
                                 timeout=2) as response:
                        self.assertEqual(response.read(), b"ready")
                    self.assertEqual(stalled.recv(1024), b"")
            finally:
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
