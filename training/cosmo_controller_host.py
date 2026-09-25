"""Serve the independent Cosmo controller behind HTTPS-only Azure ingress.

The host has no provisioning or training route. ACA must terminate HTTPS,
block insecure traffic, and route to this private container port. Production
credentials and configuration are mounted outside the source checkout.
"""
import argparse
from pathlib import Path
from wsgiref.simple_server import make_server

from training.cosmo_controller_http import build_application
from training.cosmo_controller_ledger import need


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    need(1 <= args.port <= 65535, "invalid controller port")
    app = build_application(args.config)
    need(app.proxy_https, "HTTPS-only managed ingress required")
    with make_server("0.0.0.0", args.port, app) as server:
        server.socket.settimeout(120)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
