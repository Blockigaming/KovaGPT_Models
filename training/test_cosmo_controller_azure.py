"""Signed Entra token + ARM snapshot + real issuer/guest protocol, offline."""
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key

from training import cosmo_controller_azure as azure
from training import cosmo_lifecycle_authority as authority
from training.cosmo_controller_ledger import LedgerRejected
from training import test_cosmo_controller_grants as grant_tests


class AzureVerifierTests(unittest.TestCase):
    def setUp(self):
        self.f = grant_tests.GrantIssuerTests('runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.now = self.f.f.now
        frozen_now = self.now
        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen_now
        clock = patch('jwt.api_jwt.datetime', FrozenDatetime)
        clock.start()
        self.addCleanup(clock.stop)
        self.tenant = '12345678-1234-1234-1234-123456789abf'
        self.rsa = generate_private_key(public_exponent=65537, key_size=2048)
        jwk = jwt.algorithms.RSAAlgorithm.to_jwk(self.rsa.public_key(), as_dict=True)
        jwk.update(kid='test-key', use='sig', alg='RS256')
        self.keys_url = f'https://login.microsoftonline.com/{self.tenant}/discovery/keys'
        self.documents = {self.keys_url: {'keys': [jwk]}}
        self.instance = self.f.vm
        self.network = authority.read_signed_record(self.f.runtime_path,
            expected_kind='kova_cosmo_qlora_runtime_preflight', repository_root=self.f.quote_fixture.root)[0]['azure_network']
        self.lifecycle = self.f.f.context['lifecycle']
        self.watchdog = self.lifecycle['watchdog_resource_group_id'] + '/providers/Microsoft.Logic/workflows/kova-pilot-watchdog-test01'
        self.trigger = self.f.admission['watchdog_cleanup_trigger_utc']
        self.claims = {'aud': self.f.issuer.audience, 'iss': f'https://sts.windows.net/{self.tenant}/',
            'tid': self.tenant, 'ver': '1.0', 'oid': self.instance['system_assigned_identity_principal_id'],
            'xms_mirid': self.instance['resource_id'], 'iat': int(self.now.timestamp()) - 10,
            'nbf': int(self.now.timestamp()) - 10, 'exp': int(authority.timestamp('2026-09-24T14:30:00Z').timestamp())}
        self.f.token = self.sign()
        self.f.token_digest = hashlib.sha256(self.f.token.encode()).hexdigest()
        self.f.request['azure_instance_identity_token'] = self.f.token
        def put(rid, api, props, **other):
            self.documents[azure.ARM + rid + '?api-version=' + api] = {'id': rid, 'properties': props, **other}
        n = self.network
        self.disk = self.lifecycle['pilot_resource_group_id'] + '/providers/Microsoft.Compute/disks/cosmo-os'
        put(self.instance['resource_id'], '2024-07-01', {
            'provisioningState': 'Succeeded', 'vmId': self.instance['vm_id'],
            'hardwareProfile': {'vmSize': 'Standard_NC4as_T4_v3'},
            'storageProfile': {'imageReference': {'publisher': 'Canonical', 'offer': 'ubuntu-24_04-lts',
                'sku': 'server', 'version': '24.04.202609040'},
                'osDisk': {'createOption': 'FromImage', 'deleteOption': 'Delete', 'diskSizeGB': 64,
                    'managedDisk': {'id': self.disk, 'storageAccountType': 'StandardSSD_LRS'}}},
            'networkProfile': {'networkInterfaces': [{'id': n['vm_nic_id']}]}}, location='eastus',
            identity={'type': 'SystemAssigned', 'tenantId': self.tenant,
                      'principalId': self.instance['system_assigned_identity_principal_id']})
        put(self.disk, '2024-03-02', {'diskSizeGB': 64, 'creationData': {'createOption': 'FromImage'},
            'provisioningState': 'Succeeded', 'diskState': 'Attached'},
            sku={'name': 'StandardSSD_LRS'}, managedBy=self.instance['resource_id'])
        put(n['vm_nic_id'], '2024-05-01', {'ipConfigurations': [{'properties': {'subnet': {'id': n['subnet_id']}}}]})
        put(n['subnet_id'], '2024-05-01', {'addressPrefix': '10.91.1.0/24',
            'provisioningState': 'Succeeded', 'privateEndpointNetworkPolicies': 'Disabled',
            'defaultOutboundAccess': False,
            'natGateway': {'id': n['nat_gateway_id']},
            'ipConfigurations': [{'id': n['vm_nic_id'] + '/ipConfigurations/private'}],
            'networkSecurityGroup': {'id': n['network_security_group_id']}})
        put(n['nat_gateway_id'], '2024-05-01', {'publicIpAddresses': [{'id': n['nat_gateway_public_ip_id']}]}, sku={'name': 'Standard'})
        put(n['nat_gateway_public_ip_id'], '2024-05-01', {'publicIPAllocationMethod': 'Static'}, sku={'name': 'Standard'})
        self.extension_id = self.instance['resource_id'] + '/extensions/NvidiaGpuDriverLinux'
        put(self.extension_id, '2024-03-01', {'provisioningState': 'Succeeded',
            'publisher': 'Microsoft.HpcCompute', 'type': 'NvidiaGpuDriverLinux',
            'typeHandlerVersion': '1.10', 'autoUpgradeMinorVersion': False,
            'enableAutomaticUpgrade': False,
            'settings': {'installCUDA': False, 'updateOS': False}})
        put(n['network_security_group_id'], '2024-05-01', {'securityRules': azure.expected_rules()})
        self.watchdog_principal = '12345678-1234-1234-1234-123456789aba'
        put(self.watchdog, '2019-05-01', {'state': 'Enabled', 'provisioningState': 'Succeeded',
            'parameters': {'deadlineUtc': {'value': self.trigger}},
            'definition': azure.watchdog_definition(self.instance['resource_id'], self.lifecycle['pilot_resource_group_id'], self.lifecycle['watchdog_resource_group_id'])},
            identity={'type': 'SystemAssigned', 'tenantId': self.tenant, 'principalId': self.watchdog_principal})
        put(self.watchdog + '/triggers/every_minute', '2019-05-01',
            {'state': 'Enabled', 'provisioningState': 'Succeeded'})
        pilot = self.lifecycle['pilot_resource_group_id']
        self.locks_url = azure.ARM + pilot.split('/resourceGroups/')[0] + '/providers/Microsoft.Authorization/locks?api-version=2016-09-01'
        self.documents[self.locks_url] = {'value': []}
        subscription = pilot.split('/resourceGroups/')[0]
        self.denials_url = azure.ARM + subscription + '/providers/Microsoft.Authorization/denyAssignments?api-version=2022-04-01'
        self.documents[self.denials_url] = {'value': []}
        self.at_scope_urls = [azure.ARM + group +
            '/providers/Microsoft.Authorization/denyAssignments?api-version=2022-04-01&' +
            urlencode({'$filter': 'atScope()'}) for group in
            (pilot, self.lifecycle['watchdog_resource_group_id'])]
        for url in self.at_scope_urls:
            self.documents[url] = {'value': []}
        self.guest_roles_url = (azure.ARM + pilot.split('/resourceGroups/')[0] +
            '/providers/Microsoft.Authorization/roleAssignments?api-version=2022-04-01&' +
            urlencode({'$filter': 'principalId eq ' + self.instance['system_assigned_identity_principal_id']}))
        self.documents[self.guest_roles_url] = {'value': []}
        self.effective_roles_url = (azure.ARM + pilot.split('/resourceGroups/')[0] +
            '/providers/Microsoft.Authorization/roleAssignments?api-version=2022-04-01&' +
            urlencode({'$filter': "assignedTo('" + self.instance['system_assigned_identity_principal_id'] + "')"}))
        self.documents[self.effective_roles_url] = {'value': []}
        self.roles_url = azure.ARM + pilot + '/providers/Microsoft.Authorization/roleAssignments?api-version=2022-04-01'
        pilot_role = pilot + '/providers/Microsoft.Authorization/roleAssignments/12345678-1234-1234-1234-123456789ac0'
        self.documents[self.roles_url] = {'value': [{'id': pilot_role, 'properties': {'principalId': self.watchdog_principal,
            'scope': pilot, 'roleDefinitionId': pilot.split('/resourceGroups/')[0] +
                '/providers/Microsoft.Authorization/roleDefinitions/' + azure.CONTRIBUTOR}}]}
        self.self_roles_url = azure.ARM + self.lifecycle['watchdog_resource_group_id'] + '/providers/Microsoft.Authorization/roleAssignments?api-version=2022-04-01'
        self.documents[self.self_roles_url] = deepcopy(self.documents[self.roles_url])
        self.documents[self.self_roles_url]['value'][0]['properties']['scope'] = self.lifecycle['watchdog_resource_group_id']
        self_role = self.lifecycle['watchdog_resource_group_id'] + '/providers/Microsoft.Authorization/roleAssignments/12345678-1234-1234-1234-123456789ac1'
        self.documents[self.self_roles_url]['value'][0]['id'] = self_role
        self.pilot_inventory_url = azure.ARM + pilot + '/resources?api-version=2021-04-01'
        self.watchdog_inventory_url = azure.ARM + self.lifecycle['watchdog_resource_group_id'] + '/resources?api-version=2021-04-01'
        self.documents[self.pilot_inventory_url] = {'value': [{'id': rid} for rid in (
            self.instance['resource_id'], self.disk, n['vm_nic_id'], n['nat_gateway_id'],
            n['nat_gateway_public_ip_id'], n['network_security_group_id'],
            n['subnet_id'].rsplit('/subnets/', 1)[0],
            self.instance['resource_id'] + '/extensions/NvidiaGpuDriverLinux', pilot_role)]}
        self.documents[self.watchdog_inventory_url] = {'value': [{'id': rid} for rid in (
            self.watchdog, self.watchdog + '/triggers/every_minute', self_role)]}
        self.reads = []
        self.verifier = azure.AzureRequestVerifier(tenant_id=self.tenant, token_version='1.0',
            lifecycle=self.lifecycle, watchdog_id=self.watchdog, read_json=self.read, clock=lambda: self.now)
        self.f.issuer.verify_live_request = self.verifier

    def sign(self, changes=None, headers=None, key=None):
        return jwt.encode({**self.claims, **(changes or {})}, key or self.rsa, algorithm='RS256',
                          headers={'kid': 'test-key', **(headers or {})})

    def read(self, url):
        self.reads.append(url)
        return deepcopy(self.documents[url])

    def observe(self, token=None):
        return self.verifier(token=token or self.f.token, audience=self.f.issuer.audience,
            instance=self.instance, network=self.network, cleanup_trigger_utc=self.trigger)

    def test_real_guest_issuer_azure_adapter_exchange(self):
        self.f.test_committed_response_matches_existing_guest_verifier()
        self.assertEqual(len(self.reads), 21)
        self.assertTrue(all(self.f.token not in url for url in self.reads))

    def test_foreign_expired_and_unsigned_identity_never_reads_arm(self):
        cases = [{'aud': 'wrong'}, {'tid': 'wrong'}, {'iss': 'https://attacker.invalid/'},
            {'oid': self.watchdog_principal}, {'xms_mirid': self.watchdog},
            {'exp': int(self.now.timestamp()) - 1}, {'nbf': int(self.now.timestamp()) + 60},
            {'ver': '2.0'}]
        for change in cases:
            self.reads.clear()
            with self.subTest(change=change), self.assertRaises(LedgerRejected):
                self.observe(self.sign(change))
            self.assertFalse(any(url.startswith(azure.ARM) for url in self.reads))
        bad_key = generate_private_key(public_exponent=65537, key_size=2048)
        for token in (self.sign(key=bad_key), self.sign(headers={'jku': 'https://attacker.invalid/keys'}),
                      jwt.encode(self.claims, key='', algorithm='none')):
            with self.assertRaises(LedgerRejected):
                self.observe(token)

    def test_network_watchdog_and_role_drift_prevents_durable_grant(self):
        originals = deepcopy(self.documents)
        before = self.f.f.io.body
        vm = azure.ARM + self.instance['resource_id'] + '?api-version=2024-07-01'
        nic = azure.ARM + self.network['vm_nic_id'] + '?api-version=2024-05-01'
        subnet = azure.ARM + self.network['subnet_id'] + '?api-version=2024-05-01'
        nsg = azure.ARM + self.network['network_security_group_id'] + '?api-version=2024-05-01'
        watchdog = azure.ARM + self.watchdog + '?api-version=2019-05-01'
        mutations = [
            (vm, lambda d: d['properties']['hardwareProfile'].update(vmSize='Standard_D4s_v3')),
            (vm, lambda d: d['properties']['storageProfile']['imageReference'].update(version='latest')),
            (vm, lambda d: d['identity'].update(principalId=self.watchdog_principal)),
            (vm, lambda d: d['properties'].update(diagnosticsProfile={
                'bootDiagnostics': {'enabled': True,
                                    'storageUri': 'https://untracked.blob.core.windows.net/'}})),
            (vm, lambda d: d['properties'].update(priority='Spot', evictionPolicy='Deallocate')),
            (vm, lambda d: d['properties'].update(billingProfile={'maxPrice': -1})),
            (nic, lambda d: d['properties']['ipConfigurations'][0]['properties'].update(publicIPAddress={'id': 'foreign'})),
            (subnet, lambda d: d['properties'].update(defaultOutboundAccess=True)),
            (subnet, lambda d: d['properties'].update(routeTable={'id': 'foreign'})),
            (subnet, lambda d: d['properties']['ipConfigurations'].append(
                {'id': '/subscriptions/other/resourceGroups/other/providers/Microsoft.Network/networkInterfaces/foreign/ipConfigurations/private'})),
            (subnet, lambda d: d['properties'].update(serviceAssociationLinks=[{'id': 'foreign'}])),
            (subnet, lambda d: d['properties'].update(resourceNavigationLinks=[{'id': 'foreign'}])),
            (subnet, lambda d: d['properties'].update(serviceEndpointPolicies=[{'id': 'foreign'}])),
            (subnet, lambda d: d['properties'].update(ipConfigurationProfiles=[{'id': 'foreign'}])),
            (subnet, lambda d: d['properties'].update(ipAllocations=[{'id': 'foreign'}])),
            (subnet, lambda d: d['properties'].update(serviceGateway={'id': 'foreign'})),
            (subnet, lambda d: d['properties'].update(sharingScope='Tenant')),
            (nsg, lambda d: d['properties']['securityRules'][0]['properties'].update(destinationPortRange='*')),
            (watchdog, lambda d: d['properties'].update(state='Disabled')),
            (watchdog, lambda d: d['properties']['parameters']['deadlineUtc'].update(value='2026-09-24T16:00:00Z')),
            (watchdog, lambda d: d['properties']['definition']['actions']['delete_pilot_group']['inputs'].update(uri='https://management.azure.com/foreign')),
            (self.roles_url, lambda d: d.update(value=[])),
            (self.self_roles_url, lambda d: d.update(value=[])),
            (watchdog, lambda d: d['properties']['definition']['actions'].pop('delete_watchdog_group')),
            (self.roles_url, lambda d: d.update(nextLink='https://management.azure.com/more')),
            (azure.ARM + self.network['nat_gateway_public_ip_id'] + '?api-version=2024-05-01',
             lambda d: d['properties'].update(ddosSettings={'protectionMode': 'Enabled'})),
            (azure.ARM + self.extension_id + '?api-version=2024-03-01',
             lambda d: d['properties']['settings'].update(installCUDA=True)),
            (azure.ARM + self.extension_id + '?api-version=2024-03-01',
             lambda d: d['properties'].update(typeHandlerVersion='latest')),
        ]
        for url, mutate in mutations:
            self.documents = deepcopy(originals)
            mutate(self.documents[url])
            with self.subTest(url=url), self.assertRaises(LedgerRejected):
                self.f.issuer.issue(self.f.request)
            self.assertEqual(before, self.f.f.io.body)

    def test_key_host_rejected_before_credential_or_network_access(self):
        transport = azure.AzureReadIO(account='kovatestledger', tenant_id=self.tenant,
                                     token_for=lambda _: self.fail('credential accessed'))
        with self.assertRaises(LedgerRejected):
            transport('https://attacker.invalid/keys')

    def test_disk_cost_and_trigger_drift_prevent_grant(self):
        originals = deepcopy(self.documents)
        before = self.f.f.io.body
        vm = azure.ARM + self.instance['resource_id'] + '?api-version=2024-07-01'
        disk = azure.ARM + self.disk + '?api-version=2024-03-02'
        trigger = azure.ARM + self.watchdog + '/triggers/every_minute?api-version=2019-05-01'
        mutations = [
            (vm, lambda d: d['properties']['storageProfile']['osDisk'].update(diskSizeGB=128)),
            (vm, lambda d: d['properties']['storageProfile']['osDisk'].update(deleteOption='Detach')),
            (vm, lambda d: d['properties']['storageProfile']['osDisk'].update(createOption='Attach')),
            (vm, lambda d: d['properties']['storageProfile']['osDisk']['managedDisk'].update(storageAccountType='Premium_LRS')),
            (vm, lambda d: d['properties']['storageProfile'].update(dataDisks=[{'lun': 0}])),
            (disk, lambda d: d['properties'].update(diskSizeGB=128)),
            (disk, lambda d: d['sku'].update(name='Premium_LRS')),
            (disk, lambda d: d.update(managedBy=self.watchdog)),
            (trigger, lambda d: d['properties'].update(state='Disabled')),
            (trigger, lambda d: d['properties'].update(state='Suspended')),
            (trigger, lambda d: d['properties'].pop('state')),
            (trigger, lambda d: d['properties'].update(provisioningState='Updating')),
        ]
        for url, mutate in mutations:
            self.documents = deepcopy(originals)
            mutate(self.documents[url])
            with self.subTest(url=url), self.assertRaises(LedgerRejected):
                self.f.issuer.issue(self.f.request)
            self.assertEqual(before, self.f.f.io.body)

    def test_inherited_group_and_child_locks_reject_without_consuming_grant(self):
        before = self.f.f.io.body
        pilot = self.lifecycle['pilot_resource_group_id']
        watchdog = self.lifecycle['watchdog_resource_group_id']
        for scope in (pilot.split('/resourceGroups/')[0], pilot, watchdog,
                      self.disk, self.instance['resource_id'], self.network['nat_gateway_id'], self.watchdog):
            for level in ('CanNotDelete', 'ReadOnly'):
                self.documents[self.locks_url] = {'value': [{'id': scope +
                    '/providers/Microsoft.Authorization/locks/protected', 'properties': {'level': level}}]}
                with self.subTest(scope=scope, level=level), self.assertRaises(LedgerRejected):
                    self.f.issuer.issue(self.f.request)
                self.assertEqual(before, self.f.f.io.body)
        for incomplete in ({}, {'value': [] , 'nextLink': 'https://management.azure.com/more'},
                           {'value': [{'id': 'malformed'}]}):
            self.documents[self.locks_url] = incomplete
            with self.assertRaises(LedgerRejected):
                self.f.issuer.issue(self.f.request)
            self.assertEqual(before, self.f.f.io.body)
        self.documents[self.locks_url] = {'value': [{'id': pilot + '-unrelated/providers/Microsoft.Authorization/locks/keep',
            'properties': {'level': 'ReadOnly'}}]}
        self.assertTrue(self.observe()['cleanup_scope_verified'])

    def test_inherited_and_child_deny_assignments_reject_cleanup_grant(self):
        before = self.f.f.io.body
        pilot = self.lifecycle['pilot_resource_group_id']
        for url in self.at_scope_urls:
            for result in ({'value': [{'properties': {'scope': pilot}}]},
                           {}, {'value': [], 'nextLink': 'https://management.azure.com/more'}):
                self.documents[url] = result
                with self.subTest(url=url, result=result), self.assertRaises(LedgerRejected):
                    self.f.issuer.issue(self.f.request)
                self.assertEqual(self.f.f.io.body, before)
            self.documents[url] = {'value': []}
        for scope in (pilot, self.disk, self.lifecycle['watchdog_resource_group_id'],
                      self.watchdog):
            self.documents[self.denials_url] = {'value': [{'properties': {'scope': scope}}]}
            with self.subTest(scope=scope), self.assertRaises(LedgerRejected):
                self.f.issuer.issue(self.f.request)
            self.assertEqual(self.f.f.io.body, before)
        for incomplete in ({}, {'value': [], 'nextLink': 'https://management.azure.com/more'},
                           {'value': [{'properties': {}}]}):
            self.documents[self.denials_url] = incomplete
            with self.assertRaises(LedgerRejected):
                self.f.issuer.issue(self.f.request)
            self.assertEqual(self.f.f.io.body, before)
        unrelated = pilot + '-unrelated'
        self.documents[self.denials_url] = {'value': [{'properties': {'scope': unrelated}}]}
        self.assertTrue(self.observe()['cleanup_scope_verified'])

    def test_pilot_identity_arm_roles_reject_before_grant_commit(self):
        before = self.f.f.io.body
        principal = self.instance['system_assigned_identity_principal_id']
        subscription = self.lifecycle['pilot_resource_group_id'].split('/resourceGroups/')[0]
        for scope in ('/providers/Microsoft.Management/managementGroups/root',
                      subscription, self.lifecycle['pilot_resource_group_id'], self.watchdog):
            self.documents[self.guest_roles_url] = {'value': [{
                'properties': {'principalId': principal, 'scope': scope,
                               'roleDefinitionId': subscription +
                               '/providers/Microsoft.Authorization/roleDefinitions/' + azure.CONTRIBUTOR}}]}
            with self.subTest(scope=scope), self.assertRaises(LedgerRejected):
                self.f.issuer.issue(self.f.request)
            self.assertEqual(self.f.f.io.body, before)
        for incomplete in ({}, {'value': [], 'nextLink': 'https://management.azure.com/more'}):
            self.documents[self.guest_roles_url] = incomplete
            with self.assertRaises(LedgerRejected):
                self.f.issuer.issue(self.f.request)
            self.assertEqual(self.f.f.io.body, before)
        self.documents[self.guest_roles_url] = {'value': []}
        for inherited in ({'value': [{'properties': {'principalId': 'security-group-id',
                            'scope': subscription}}]}, {},
                          {'value': [], 'nextLink': 'https://management.azure.com/more'}):
            self.documents[self.effective_roles_url] = inherited
            with self.assertRaises(LedgerRejected):
                self.f.issuer.issue(self.f.request)
            self.assertEqual(self.f.f.io.body, before)

    def test_unexpected_or_missing_cleanup_resources_reject_before_grant_commit(self):
        before = self.f.f.io.body
        original = deepcopy(self.documents)
        for url in (self.pilot_inventory_url, self.watchdog_inventory_url):
            group = (self.lifecycle['pilot_resource_group_id'] if url == self.pilot_inventory_url
                     else self.lifecycle['watchdog_resource_group_id'])
            for change in ('extra_vm', 'extra_disk', 'missing', 'next_page', 'duplicate'):
                self.documents = deepcopy(original)
                items = self.documents[url]['value']
                if change.startswith('extra_'):
                    kind = 'virtualMachines' if change == 'extra_vm' else 'disks'
                    items.append({'id': group + '/providers/Microsoft.Compute/' + kind + '/unpriced'})
                elif change == 'missing':
                    items.pop(0)
                elif change == 'next_page':
                    self.documents[url]['nextLink'] = azure.ARM + '/more'
                else:
                    items.append(deepcopy(items[0]))
                with self.subTest(url=url, change=change), self.assertRaises(LedgerRejected):
                    self.f.issuer.issue(self.f.request)
                self.assertEqual(self.f.f.io.body, before)
        self.documents = original
        self.assertTrue(self.observe()['cleanup_scope_verified'])


class ManagedIdentityTokenTests(unittest.TestCase):
    def test_local_managed_identity_is_pinned_to_client_and_audience(self):
        class Response(BytesIO):
            status = 200
        client = '12345678-1234-1234-1234-123456789abc'
        secret = '12345678-1234-1234-1234-123456789abd'
        env = {'IDENTITY_ENDPOINT': 'http://127.0.0.1:42314/msi/token',
               'IDENTITY_HEADER': secret}
        result = {'access_token': 'x' * 64, 'expires_on': str(int(azure.utc_now().timestamp()) + 180),
                  'token_type': 'Bearer', 'client_id': client, 'resource': azure.ARM + '/'}
        requests = []
        class Opener:
            def open(self, request, timeout):
                requests.append(request)
                assert timeout == 5
                return Response(json.dumps(result).encode())
        with patch.object(azure.urllib.request, 'build_opener', return_value=Opener()):
            self.assertEqual(azure.managed_identity_token(azure.ARM + '/', client_id=client, environ=env), 'x' * 64)
            self.assertEqual(requests[0].get_header('X-identity-header'), secret)
            self.assertEqual(parse_qs(urlsplit(requests[0].full_url).query),
                {'resource': [azure.ARM + '/'], 'api-version': ['2019-08-01'], 'client_id': [client]})
            for drift in ({'resource': 'https://storage.azure.com/'}, {'client_id': '0' * 36},
                          {'token_type': 'Basic'}, {'expires_on': '0'}, {'access_token': ''}):
                result.update(drift)
                with self.subTest(drift=drift), self.assertRaises(LedgerRejected):
                    azure.managed_identity_token(azure.ARM + '/', client_id=client, environ=env)
                result = {'access_token': 'x' * 64, 'expires_on': str(int(azure.utc_now().timestamp()) + 180),
                          'token_type': 'Bearer', 'client_id': client, 'resource': azure.ARM + '/'}

    def test_remote_or_missing_identity_endpoint_is_rejected_before_http(self):
        env = {'IDENTITY_ENDPOINT': 'http://attacker.invalid/msi/token',
               'IDENTITY_HEADER': '12345678-1234-1234-1234-123456789abd'}
        with patch.object(azure.urllib.request, 'build_opener', side_effect=AssertionError('HTTP accessed')):
            for endpoint in (env['IDENTITY_ENDPOINT'], 'https://127.0.0.1:42314/msi/token',
                             'http://127.0.0.1:42314/msi/token?extra=1', None):
                with self.subTest(endpoint=endpoint), self.assertRaises(LedgerRejected):
                    azure.managed_identity_token(azure.ARM + '/',
                        client_id='12345678-1234-1234-1234-123456789abc',
                        environ={**env, 'IDENTITY_ENDPOINT': endpoint})


if __name__ == '__main__':
    unittest.main()
