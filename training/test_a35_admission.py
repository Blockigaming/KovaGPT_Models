import http.client
from copy import deepcopy
import ssl
import unittest
import urllib.error
from unittest.mock import Mock
from training.a35_admission import bounded_request, mutate_once, transient, AmbiguousMutation, ClassifyingOpener, deployment_matches

URL = 'https://management.azure.com/subscriptions/test/resource?api-version=1'


class Admission(unittest.TestCase):
    def request(self, effects, method='GET', url=URL, **kwargs):
        call = Mock(side_effect=effects); log = Mock(); sleep = Mock()
        result = bounded_request(call, method, url, observe=log, sleep=sleep, **kwargs)
        return result, call, log, sleep

    def test_timeout_then_success(self):
        result, call, log, sleep = self.request([TimeoutError(), (200, {}, b'ok')])
        self.assertEqual(result[0], 200); self.assertEqual(call.call_count, 2)
        self.assertFalse(log.call_args.args[0]['paid_authorization_consumed_by_read'])
        sleep.assert_called_once_with(2)

    def test_two_retries_maximum(self):
        call = Mock(side_effect=TimeoutError()); log = Mock()
        with self.assertRaises(TimeoutError):
            bounded_request(call, 'GET', URL, observe=log, sleep=Mock())
        self.assertEqual(call.call_count, 3); self.assertEqual(log.call_count, 3)
        self.assertFalse(log.call_args.args[0]['retry_scheduled'])

    def test_transient_status_then_success(self):
        for status in (429,500,502,503,504):
            with self.subTest(status=status):
                result, call, _, _ = self.request([(status, {}, b'error'), (200, {}, b'ok')])
                self.assertEqual(call.call_count, 2); self.assertEqual(result[0], 200)

    def test_deterministic_status_never_retried(self):
        for status in (400,401,403,404,409,422,501,505):
            with self.subTest(status=status):
                _, call, _, sleep = self.request([(status, {}, b'error')])
                self.assertEqual(call.call_count, 1); sleep.assert_not_called()

    def test_write_never_replayed(self):
        for method in ('PUT','POST','PATCH','DELETE'):
            call = Mock(side_effect=TimeoutError())
            with self.subTest(method=method), self.assertRaises(TimeoutError):
                bounded_request(call, method, URL, observe=Mock(), sleep=Mock())
            self.assertEqual(call.call_count, 1)

    def test_foreign_host_no_retry(self):
        call = Mock(side_effect=TimeoutError())
        with self.assertRaises(TimeoutError):
            bounded_request(call, 'GET', 'https://foreign.example/', observe=Mock(), sleep=Mock())
        self.assertEqual(call.call_count, 1)

    def test_credential_and_certificate_failure_no_retry(self):
        for error in (ValueError('credential'), OSError('credential'), ssl.SSLError('certificate')):
            call = Mock(side_effect=error)
            with self.subTest(error=type(error)), self.assertRaises(type(error)):
                bounded_request(call, 'GET', URL, observe=Mock(), sleep=Mock())
            self.assertEqual(call.call_count,1)

    def test_typed_arm_receipt_is_retryable(self):
        class Error(OSError):
            receipt = {'failure_class':'transport','error_type':'TimeoutError'}
        result, call, _, _ = self.request([Error(), (200,{},b'ok')])
        self.assertEqual(call.call_count,2)

    def test_urlerror_reason_classification(self):
        for error, expected in [(TimeoutError(),True),(ConnectionResetError(),True),(ssl.SSLError(),False)]:
            self.assertEqual(transient(urllib.error.URLError(error)), expected)
        opener = ClassifyingOpener(Mock(open=Mock(side_effect=urllib.error.URLError(TimeoutError()))))
        with self.assertRaises(TimeoutError): opener.open(object(),timeout=20)

    def test_sanitized_failure_receipt(self):
        result, _, log, _ = self.request([(503,{},b'private'),(200,{},b'ok')], url=URL+'&secret=do-not-log')
        self.assertNotIn('do-not-log',str(log.call_args)); self.assertNotIn('private',str(log.call_args))

    def test_original_deadline_not_reset(self):
        call = Mock(return_value=(503,{},b''))
        with self.assertRaises(TimeoutError):
            bounded_request(call,'GET',URL,observe=Mock(),sleep=Mock(),clock=lambda:100,deadline=101)
        self.assertEqual(call.call_count,1); self.assertEqual(call.call_args.kwargs['timeout'],1)


class Reconciliation(unittest.TestCase):
    def test_lost_write_response_reconciles_success_without_replay(self):
        write=Mock(side_effect=TimeoutError()); read=Mock(return_value=(200,{'id':'expected'})); log=Mock()
        actual=mutate_once(write,read,expected=lambda x:x['id']=='expected',observe=log)
        self.assertEqual(actual['id'],'expected'); write.assert_called_once(); read.assert_called_once()

    def test_404_after_timeout_is_not_proof_of_no_mutation(self):
        write=Mock(side_effect=TimeoutError()); read=Mock(return_value=(404,{}))
        with self.assertRaises(AmbiguousMutation):
            mutate_once(write,read,expected=lambda _:True,observe=Mock(),sleep=Mock())
        write.assert_called_once(); self.assertEqual(read.call_count,3)

    def test_mismatched_resource_fails_closed(self):
        write=Mock(return_value=(201,{}))
        with self.assertRaises(AmbiguousMutation):
            mutate_once(write,lambda:(200,{'id':'foreign'}),expected=lambda x:x['id']=='expected',observe=Mock())
        write.assert_called_once()


class DeploymentReadback(unittest.TestCase):
    def setUp(self):
        self.rid = '/subscriptions/test/resourceGroups/exclusive/providers/Microsoft.Resources/deployments/a35-watchdog'
        self.template = {'parameters': {'provisionWatchdog': {'type':'bool','defaultValue':False},
                          'location': {'type':'string','defaultValue':'[resourceGroup().location]'},
                          'suffix': {'type':'string'},'deadlineUtc': {'type':'string'}}}
        self.parameters = {'provisionWatchdog':True,'suffix':'synthetic1','deadlineUtc':'2030-01-01T01:15:00Z'}
        self.observed = {'id':self.rid,'properties': {'mode':'Incremental','provisioningState':'Running',
                         'parameters': {'provisionWatchdog':{'type':'Bool','value':True},
                         'location':{'type':'String','value':'eastus'},
                         'suffix':{'type':'String','value':'synthetic1'},
                         'deadlineUtc':{'type':'String','value':'2030-01-01T01:15:00Z'}}}}

    def matches(self):
        return deployment_matches(self.observed,resource_id=self.rid,template=self.template,
                                  parameters=self.parameters,location='eastus')

    def test_reproduces_raw_equality_defect_and_accepts_typed_defaults(self):
        self.assertNotEqual(self.observed['properties']['parameters'], {k:{'value':v} for k,v in self.parameters.items()})
        self.assertTrue(self.matches())

    def test_each_changed_value_is_rejected(self):
        original=deepcopy(self.observed)
        for key in self.observed['properties']['parameters']:
            self.observed=deepcopy(original);self.observed['properties']['parameters'][key]['value']='foreign'
            with self.subTest(key=key): self.assertFalse(self.matches())

    def test_wrong_parameter_type_is_rejected(self):
        self.observed['properties']['parameters']['provisionWatchdog']['type']='String'
        self.assertFalse(self.matches())

    def test_boolean_integer_confusion_is_rejected(self):
        self.observed['properties']['parameters']['provisionWatchdog']['value']=1
        self.assertFalse(self.matches())

    def test_missing_parameter_is_rejected(self):
        del self.observed['properties']['parameters']['location'];self.assertFalse(self.matches())

    def test_extra_parameter_is_rejected(self):
        self.observed['properties']['parameters']['foreign']={'type':'String','value':'x'};self.assertFalse(self.matches())

    def test_missing_type_evidence_is_rejected(self):
        del self.observed['properties']['parameters']['location']['type'];self.assertFalse(self.matches())

    def test_unknown_default_expression_is_rejected(self):
        self.template['parameters']['location']['defaultValue']='[deployment().location]';self.assertFalse(self.matches())

    def test_wrong_resource_identity_is_rejected(self):
        self.observed['id']=self.rid+'-foreign';self.assertFalse(self.matches())

    def test_wrong_mode_is_rejected(self):
        self.observed['properties']['mode']='Complete';self.assertFalse(self.matches())

    def test_absent_state_is_rejected(self):
        del self.observed['properties']['provisioningState'];self.assertFalse(self.matches())

    def secure_fixture(self):
        self.template['parameters']['sshPublicKey']={'type':'securestring'}
        self.parameters['sshPublicKey']='synthetic public key only'
        self.observed['properties']['parameters']['sshPublicKey']={'type':'SecureString'}
        self.observed['properties']['correlationId']='00000000-0000-4000-8000-000000000001'
        self.observed['properties']['templateHash']='123456789'
        return deepcopy(self.observed)

    def secure_matches(self,ack):
        return deployment_matches(self.observed,resource_id=self.rid,template=self.template,
                                  parameters=self.parameters,location='eastus',acknowledgement=ack)

    def test_masked_ssh_requires_complete_acknowledgement(self):
        ack=self.secure_fixture()
        self.assertFalse(self.matches())
        self.assertTrue(self.secure_matches(ack))

    def test_masked_ssh_mismatched_acknowledgement_rejected(self):
        original=self.secure_fixture()
        for key,value in [('correlationId','foreign'),('templateHash','98765'),('mode','Complete')]:
            ack=deepcopy(original);ack['properties'][key]=value
            with self.subTest(key=key): self.assertFalse(self.secure_matches(ack))

    def test_arbitrary_masked_secret_not_allowed(self):
        ack=self.secure_fixture()
        self.template['parameters']['adminPassword']={'type':'securestring'}
        self.parameters['adminPassword']='test only'
        self.observed['properties']['parameters']['adminPassword']={'type':'SecureString'}
        ack=deepcopy(self.observed)
        self.assertFalse(self.secure_matches(ack))

    def test_lost_put_cannot_use_acknowledged_matcher(self):
        write=Mock(side_effect=TimeoutError());matched=Mock(return_value=True)
        with self.assertRaises(AmbiguousMutation):
            mutate_once(write,lambda:(200,{}),expected=lambda _:False,
                        acknowledged_expected=matched,observe=Mock())
        matched.assert_not_called();write.assert_called_once()

    def test_transient_put_status_cannot_use_acknowledged_matcher(self):
        matched=Mock(return_value=True)
        with self.assertRaises(AmbiguousMutation):
            mutate_once(lambda:(503,{}),lambda:(200,{}),expected=lambda _:False,
                        acknowledged_expected=matched,observe=Mock())
        matched.assert_not_called()

    def test_acknowledged_matcher_receives_actual_put_response(self):
        ack={'id':'expected','properties':{'correlationId':'observed'}};matched=Mock(return_value=True)
        mutate_once(lambda:(201,ack),lambda:(200,{}),expected=lambda _:False,
                    acknowledged_expected=matched,observe=Mock())
        matched.assert_called_once_with({},ack)

    def test_write_permission_failure_never_retried(self):
        write=Mock(return_value=(403,{})); read=Mock()
        with self.assertRaises(RuntimeError): mutate_once(write,read,expected=lambda _:True,observe=Mock())
        write.assert_called_once(); read.assert_not_called()

    def test_transient_write_status_reconciles(self):
        write=Mock(return_value=(503,{}))
        mutate_once(write,lambda:(200,{'id':'expected'}),expected=lambda x:x['id']=='expected',observe=Mock())
        write.assert_called_once()


if __name__=='__main__': unittest.main(verbosity=2)
