"""HTTP authentication and real signed-grant exchange; no listening socket."""
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from training import cosmo_lifecycle_authority as authority
from training.cosmo_controller_http import GrantApplication, build_application
from training import cosmo_controller_http as http
from training import test_cosmo_controller_grants as grant_tests


class HttpControllerTests(unittest.TestCase):
    def setUp(self):
        self.f = grant_tests.GrantIssuerTests('runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.secret = 'a' * 64
        lock = self.f.quote_fixture.root / 'requirements/receiver-auth-py312-linux.lock'
        lock.parent.mkdir()
        lock.write_bytes((http.launch.ROOT / 'requirements/receiver-auth-py312-linux.lock').read_bytes())
        self.app = GrantApplication(issuer=self.f.issuer,
            endpoint='https://authority.example.test/cosmo/grant', bearer_token=self.secret)

    def call(self, body=None, **changes):
        raw = authority.canonical(body or self.f.request)
        env = {'wsgi.url_scheme': 'https', 'HTTP_HOST': 'authority.example.test',
            'PATH_INFO': '/cosmo/grant', 'REQUEST_METHOD': 'POST',
            'HTTP_AUTHORIZATION': 'Bearer ' + self.secret, 'CONTENT_TYPE': 'application/json',
            'CONTENT_LENGTH': str(len(raw)), 'wsgi.input': BytesIO(raw), **changes}
        result = []
        payload = b''.join(self.app(env, lambda status, headers: result.append((status, dict(headers)))))
        return result[0], payload

    def test_private_config_wires_real_adapters_without_azure_requests(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            def private(name, raw):
                path = base / name
                path.write_bytes(raw)
                path.chmod(0o600)
                return str(path)
            config = {"ledger_context": self.f.f.context,
                "tenant_id": "12345678-1234-1234-1234-123456789abf", "token_version": "1.0",
                "watchdog_resource_id": self.f.f.context["lifecycle"]["watchdog_resource_group_id"] +
                    "/providers/Microsoft.Logic/workflows/kova-pilot-watchdog-test01",
                "signing_key_file": private('signer', self.f.f.key.private_bytes_raw()),
                "preservation_public_key_hex": self.f.f.verifier.public_key().public_bytes_raw().hex(),
                "cleanup_public_key_hex": self.f.f.verifier.public_key().public_bytes_raw().hex(),
                "bearer_token_file": private('bearer', self.secret.encode()),
                "quote_file": private('quote', self.f.quote.read_bytes()),
                "runtime_evidence_file": private('runtime', self.f.runtime_path.read_bytes())}
            path = private('config', json.dumps(config).encode())
            with patch('training.cosmo_controller_azure.cli_token', side_effect=AssertionError('unexpected credential read')):
                app = build_application(path, root=self.f.quote_fixture.root)
            self.assertIsInstance(app, GrantApplication)
            self.assertEqual(app.issuer.ledger.context, self.f.f.context)
            self.assertEqual(app.issuer.verify_live_request.watchdog, config['watchdog_resource_id'])
            Path(config['signing_key_file']).chmod(0o644)
            with self.assertRaises(ValueError):
                build_application(path, root=self.f.quote_fixture.root)

    def test_https_post_returns_committed_envelope_then_rejects_retry(self):
        result, raw = self.call()
        self.assertEqual(result[0], '200 OK')
        self.assertEqual(result[1]['Cache-Control'], 'no-store')
        envelope = __import__('json').loads(raw)
        self.f.ledger.public_key.verify(bytes.fromhex(envelope['signature']),
                                       authority.canonical(envelope['payload']))
        self.assertEqual(self.f.ledger.replay(self.f.f.io.body)[0]['sequence'], 3)
        self.assertEqual(self.call()[0][0], '403 Forbidden')
        self.assertNotIn(self.f.token.encode(), raw)

    def test_startup_rejects_interpreter_missing_or_changed_auth_packages_before_keys(self):
        root = self.f.quote_fixture.root
        with patch.object(http, 'private_file', side_effect=AssertionError('control file accessed')):
            with patch.object(http.sys, 'version_info', (3, 13, 0)), self.assertRaisesRegex(ValueError, 'CPython 3.12'):
                build_application('/missing', root=root)
            original = http.metadata.version
            for package in ('PyJWT', 'cryptography', 'cffi', 'pycparser'):
                def version(name):
                    return '0.0.0' if name == package else original(name)
                with self.subTest(package=package), patch.object(http.metadata, 'version', side_effect=version), self.assertRaisesRegex(ValueError, 'version mismatch'):
                    build_application('/missing', root=root)
            with patch.object(http.metadata, 'version', side_effect=http.metadata.PackageNotFoundError), self.assertRaisesRegex(ValueError, 'dependency missing'):
                build_application('/missing', root=root)

    def test_bad_http_inputs_never_touch_issuer(self):
        issuer = Mock()
        self.app.issuer = issuer
        for changes in ({'HTTP_AUTHORIZATION': 'Bearer wrong'}, {'HTTP_AUTHORIZATION': 'Bearer é'},
            {'wsgi.url_scheme': 'http', 'HTTP_X_FORWARDED_PROTO': 'https'},
            {'REQUEST_METHOD': 'GET'}, {'HTTP_HOST': 'wrong.test'}, {'PATH_INFO': '/admin'},
            {'CONTENT_LENGTH': '65537'}, {'CONTENT_LENGTH': '-1'}, {'CONTENT_LENGTH': ''},
            {'CONTENT_TYPE': 'text/plain'}, {'HTTP_TRANSFER_ENCODING': 'chunked'},
            {'wsgi.input': BytesIO(b'{"a":1,"a":2}')}, {'wsgi.input': BytesIO(b'')},
            {'QUERY_STRING': 'admin=1'}):
            with self.subTest(changes=changes):
                self.assertEqual(self.call(**changes)[0][0], '403 Forbidden')
        issuer.issue.assert_not_called()

    def test_uncertain_commit_error_is_sanitized_and_not_retried(self):
        self.f.f.io.lose_append_response = True
        result, raw = self.call()
        self.assertEqual(result[0], '403 Forbidden')
        self.assertEqual(raw, b'{"error":"grant_request_rejected"}')
        self.assertEqual(self.f.ledger.replay(self.f.f.io.body)[0]['sequence'], 3)
        self.assertEqual(self.call()[0][0], '403 Forbidden')


if __name__ == '__main__':
    unittest.main()
