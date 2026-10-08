"""Bounded, create-only experimental evidence using the admitted VM identity."""

import hashlib
import json
import re
from urllib.request import Request, ProxyHandler, HTTPRedirectHandler, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


class Storage:
    def __init__(self, run_id):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("invalid run ID")
        self.prefix = "a35-nova-screen/" + run_id + "/"
        self.total = 0

    def put(self, name, body):
        if (not re.fullmatch(r"[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)?\.[a-z0-9]+", name)
                or type(body) is not bytes or not body or len(body) > 40 * 1024 * 1024
                or self.total + len(body) > 64 * 1024 * 1024):
            raise ValueError("bounded evidence path/body rejected")
        direct = build_opener(ProxyHandler({}), NoRedirect())
        request = Request("http://169.254.169.254/metadata/identity/oauth2/token"
            "?api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F",
            headers={"Metadata": "true"})
        with direct.open(request, timeout=10) as response:
            token = json.load(response)["access_token"]
        url = "https://kova42c1a27.blob.core.windows.net/cosmo-adapters/" + self.prefix + name
        headers = {"Authorization": "Bearer " + token, "x-ms-version": "2023-11-03"}
        digest = hashlib.sha256(body).hexdigest()
        blob = build_opener(NoRedirect())
        with blob.open(Request(url, method="PUT", data=body, headers=headers | {
                "If-None-Match": "*", "x-ms-blob-type": "BlockBlob",
                "x-ms-meta-sha256": digest, "Content-Type": "application/octet-stream"}), timeout=60) as response:
            if response.status != 201:
                raise ValueError("create-only evidence failed")
        self.total += len(body)
        with blob.open(Request(url, headers=headers), timeout=60) as response:
            returned = response.read(len(body) + 1)
        if returned != body or hashlib.sha256(returned).hexdigest() != digest:
            raise ValueError("evidence readback mismatch")
        return {"path": self.prefix + name, "sha256": digest, "bytes": len(body)}
