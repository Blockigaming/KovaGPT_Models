"""Serve the independent Cosmo controller behind HTTPS-only Azure ingress.

The host has no provisioning or training route. ACA must terminate HTTPS,
block insecure traffic, and route to this private container port. Production
credentials and configuration are mounted outside the source checkout.
"""
import argparse
from pathlib import Path
import socket
from socketserver import ThreadingMixIn
from threading import BoundedSemaphore, Timer
from wsgiref.simple_server import WSGIServer
from wsgiref.simple_server import make_server

from training.cosmo_controller_http import build_application
from training.cosmo_controller_ledger import need


class BoundedControllerServer(ThreadingMixIn, WSGIServer):
    """Bound accepted requests by both concurrency and elapsed wall time."""
    daemon_threads = True
    request_queue_size = 8
    connection_deadline_seconds = 150

    def server_activate(self):
        super().server_activate()
        self._slots = BoundedSemaphore(4)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        request.settimeout(self.connection_deadline_seconds)
        def expire():
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        deadline = Timer(self.connection_deadline_seconds, expire)
        deadline.daemon = True
        deadline.start()
        try:
            super().process_request_thread(request, client_address)
        finally:
            deadline.cancel()
            self._slots.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    need(1 <= args.port <= 65535, "invalid controller port")
    app = build_application(args.config)
    need(app.proxy_https, "HTTPS-only managed ingress required")
    with make_server("0.0.0.0", args.port, app,
                     server_class=BoundedControllerServer) as server:
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
