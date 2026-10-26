"""Private CPU-only contract/negative checks; never CUDA acceptance."""
import ast
import importlib.util
import logging
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'tools' / 'classic_capacity.py'

class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper_missing = not SOURCE.exists()
        if cls.helper_missing:
            return
        spec = importlib.util.spec_from_file_location('classic_capacity_probe_private', SOURCE)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def setUp(self):
        self.assertFalse(self.helper_missing, 'capacity helper implementation is missing')

    def test_exact_ladders_and_maximum_work(self):
        self.assertEqual(self.m.policy('vemg2pose', 600)['train_attempts'], 2)
        self.assertEqual(self.m.policy('neuropose', 256)['validation_samples'], 1576)
        self.assertEqual(self.m.work_budget(), dict(children=6, train_attempts=12,
            train_hand_samples=2192, validation_hand_samples=9456, validation_batches=197))

    def test_rejects_unreviewed_batches_and_overrides(self):
        for variant, batch in [('vemg2pose', 1), ('vemg2pose', 256), ('neuropose', 600), ('bad', 16), ('neuropose', True)]:
            with self.assertRaises(ValueError):
                self.m.policy(variant, batch)

    def test_memory_margin_requires_both_bounds(self):
        gib = 1024**3
        self.m.check_memory(dict(peak_reserved=80*gib, peak_allocated=75*gib,
            free=5*gib, total=100*gib))
        for entry in [dict(peak_reserved=86*gib, peak_allocated=75*gib, free=5*gib, total=100*gib),
                      dict(peak_reserved=80*gib, peak_allocated=75*gib, free=3*gib, total=100*gib),
                      dict(peak_reserved=float('nan'), peak_allocated=75*gib, free=5*gib, total=100*gib)]:
            with self.assertRaises(ValueError):
                self.m.check_memory(entry)

    def test_qualification_dictionary_cannot_authorize_gpu(self):
        with self.assertRaises(TypeError):
            self.m.require_qualification_files({'passed': True}, {'passed': True})

    def test_owned_output_claim_never_overwrites(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent, prefix='classic-capacity-test-') as d:
            destination = Path(d)/'entry'
            with self.m.OwnedOutput.claim(destination) as owned:
                owned.write_json('status.json', {'state':'running'})
                with self.assertRaises(FileExistsError):
                    self.m.OwnedOutput.claim(destination)
                owned.write_json('result.json', {'capacity_passed':True})
                owned.fail(ValueError('negative test'))
                self.assertFalse((destination/'result.json').exists())
                self.assertIn('failed', (destination/'status.json').read_text())

    def test_partial_or_self_asserted_capacity_result_rejected(self):
        p = self.m.policy('vemg2pose', 32)
        for result in ({'passed': True}, {'capacity_passed': True, 'attempts': 2},
                       {'format': self.m.FORMAT, 'scope': self.m.SCOPE, 'capacity_passed': True,
                        'scientific_eligible': False, 'production_ready': False, 'identity': {'policy': p}}):
            with self.assertRaises(ValueError):
                self.m.validate_result(result, p)

    def test_published_result_mutation_is_detected(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent, prefix='classic-capacity-test-') as d:
            destination = Path(d)/'entry'
            with self.m.OwnedOutput.claim(destination) as owned:
                expected = owned.write_json('result.json', {'value': 1})
                (destination/'result.json').write_text('{"value": 2}')
                with self.assertRaises(ValueError):
                    owned.verify_json('result.json', expected)

    def test_trainer_uses_only_the_actual_precision_plugin(self):
        tree = ast.parse(SOURCE.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'Trainer']
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg for keyword in calls[0].keywords}
        self.assertIn('plugins', keywords)
        self.assertNotIn('precision', keywords, 'PL rejects simultaneous precision keyword and precision plugin')

    def test_failure_still_revokes_success_status_when_result_unlink_fails(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent, prefix='classic-capacity-test-') as d:
            destination = Path(d)/'entry'
            with self.m.OwnedOutput.claim(destination) as owned:
                owned.write_json('result.json', {'capacity_passed': True})
                owned.write_json('status.json', {'state': 'capacity_entry_finished'})
                actual_unlink = self.m.os.unlink
                def unlink_with_fault(name, *args, **kwargs):
                    if name == 'result.json':
                        raise OSError('result unlink fault')
                    return actual_unlink(name, *args, **kwargs)
                with mock.patch.object(self.m.os, 'unlink', side_effect=unlink_with_fault):
                    owned.fail(ValueError('negative fault'))
                self.assertIn('failed', (destination/'status.json').read_text())

    def test_logging_cannot_swallow_log_cap_failure_into_success(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent, prefix='classic-capacity-test-') as d:
            with self.m.OwnedOutput.claim(Path(d)/'entry') as owned:
                log = self.m.CappedLog(owned)
                log.written = self.m.LOG_LIMIT
                handler = logging.StreamHandler(log)
                with mock.patch.object(handler, 'handleError'):
                    handler.emit(logging.LogRecord('capacity-test', logging.INFO, __file__, 1, 'over cap', (), None))
                with self.assertRaises(OSError):
                    self.m.close_log(log, owned)

    def test_late_log_close_failure_revokes_success(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent, prefix='classic-capacity-test-') as d:
            destination = Path(d)/'entry'
            with self.m.OwnedOutput.claim(destination) as owned:
                log = self.m.CappedLog(owned)
                owned.write_json('result.json', {'capacity_passed': True})
                owned.write_json('status.json', {'state': 'capacity_entry_finished'})
                with mock.patch.object(log, 'flush', side_effect=OSError('late fsync fault')):
                    with self.assertRaises(OSError):
                        self.m.close_log(log, owned)
                self.assertFalse((destination/'result.json').exists())
                self.assertIn('failed', (destination/'status.json').read_text())

    def test_claim_rejects_public_rebind_before_capability_capture(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent, prefix='classic-capacity-test-') as d:
            output=Path(d)/'entry';moved=Path(d)/'moved';real_open=self.m.os.open;fired=[]
            def opening(path,flags,*args,**kwargs):
                if not fired and Path(path)==output and flags&self.m.os.O_DIRECTORY:
                    fired.append(True);output.rename(moved);output.mkdir()
                    (output/'result.json').write_text('neighbor evidence')
                return real_open(path,flags,*args,**kwargs)
            with mock.patch.object(self.m.os,'open',side_effect=opening):
                with self.assertRaises((ValueError,OSError)):
                    self.m.OwnedOutput.claim(output)
            self.assertTrue(fired)
            self.assertEqual((output/'result.json').read_text(),'neighbor evidence')

    def test_claim_noreplace_preserves_a_concurrent_final_name(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent, prefix='classic-capacity-test-') as d:
            output=Path(d)/'entry';real_mkdir=self.m.os.mkdir;fired=[]
            def mkdir(path,*args,**kwargs):
                value=real_mkdir(path,*args,**kwargs)
                if str(path).startswith('.classic-capacity-stage-') and not fired:
                    fired.append(True);real_mkdir(output);(output/'result.json').write_text('concurrent evidence')
                return value
            with mock.patch.object(self.m.os,'mkdir',side_effect=mkdir):
                with self.assertRaises(FileExistsError):
                    self.m.OwnedOutput.claim(output)
            self.assertTrue(fired)
            self.assertEqual((output/'result.json').read_text(),'concurrent evidence')
            self.assertEqual(sorted(path.name for path in Path(d).iterdir()),['entry'])

    def test_claim_rejects_unsupported_atomic_platform(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent, prefix='classic-capacity-test-') as d:
            with mock.patch.object(self.m.sys,'platform','unsupported'):
                with self.assertRaises(RuntimeError):
                    self.m.OwnedOutput.claim(Path(d)/'entry')
            self.assertFalse((Path(d)/'entry').exists())

    def _assert_tail_rejects_source_fault(self,boundary,*,rebind=False):
        # CPU-only negative transaction fixture. No checkpoint/model/GPU path is run.
        import egoemg_repro
        from egoemg_repro import classic_emg_cohort as actual_cohort
        h=self.m;p=h.policy('vemg2pose',32);gib=1024**3
        memory=dict(peak_allocated=2*gib,peak_reserved=3*gib,free=77*gib,total=80*gib)
        sizes=[min(32,1576-i*32) for i in range(p['validation_batches'])]
        fit_result=dict(attempts=2,optimizer_executions=2,natural_overflows=0,
            events=[dict(batch=i,size=32,loss=1.0,optimizer_executed=True,natural_overflow=False,
                gradient=dict(finite=True,absolute_max=1.0),optimizer_step_before=i,optimizer_step_after=i+1,
                scale_before=65536.0,scale_after=65536.0) for i in range(2)],
            validation=dict(samples=1576,batches=len(sizes),batch_sizes=sizes,full_frozen_val_coverage=True,
                original_batch_union_fk=True,loss_selection_performed=False,scientific_eligible=False,
                val_loss=1.0,batch_weighted_loss=1.0),
            validation_events=[dict(batch=i,size=n,loss=1.0) for i,n in enumerate(sizes)],
            validation_order_sha256=h.digest(list(range(1576))),
            train_coverage=dict(samples=64,unique_samples=64,sorted_indices_sha256=h.digest(list(range(64)))),
            memory=dict(train=memory,validation=memory))
        with tempfile.TemporaryDirectory(dir=SOURCE.parent,prefix='classic-capacity-test-') as d:
            base=Path(d).resolve();config=base/'config.json';config.write_text('{}')
            parts=base/'partitions.json';parts.write_text('{}');receipt=base/'cohort.json';receipt.write_text('{}')
            dummy=base/'runtime.py';dummy.write_text('# CPU transaction fixture only\n')
            runtime={'torch_files_sha256':{}};recipe={'variant':'vemg2pose'}
            numeric={'sha256':'numeric-fixture','recipe':recipe,'runtime':runtime,'model':{}}
            for name in ('prefix','resumed'):
                folder=base/name;(folder/'checkpoints').mkdir(parents=True)
                for n in ('identity.json','status.json','result.json'):
                    (folder/n).write_text(json.dumps(numeric if n=='identity.json' else {}))
                (folder/'checkpoints/last.ckpt').write_text('CPU fixture; never loaded')
            guard=actual_cohort._StatGuard([config])
            fake_cohort=SimpleNamespace(_StatGuard=actual_cohort._StatGuard,cohort_stat_guard=lambda *a,**k:guard,
                _json=lambda *a:dict(recipe=recipe,footprints={k:{'hand_samples':v} for k,v in
                    [('train',101714),('val',1576),('test',4306)]},data_identity={'sha256':'data-fixture'}))
            fake_q=SimpleNamespace(RUNTIME_FILES=(),verify_cuda_check=lambda *a:dict(cuda_numeric_passed=True,
                strict_state_restore=True,identity_sha256='numeric-fixture'),
                datasets=SimpleNamespace(load_classic_config=lambda *a:recipe),_profile=lambda:(runtime,None),
                pl=SimpleNamespace(__file__=str(dummy)),storage=SimpleNamespace(FRAMEWORK_FILES=(),
                    _runtime=lambda:SimpleNamespace(__file__=str(dummy))),
                _prepare=lambda *a,**k:SimpleNamespace(identity=numeric),_factory=lambda *a:None,
                _check_prepared=lambda *a:None,_process_identity=lambda *a:dict(pid=os.getpid(),start_ticks=1))
            output=base/'out';moved=base/'original'
            args=SimpleNamespace(variant='vemg2pose',batch_size=32,reviewed_helper_sha256=h.IMPORTED_HELPER_SHA,
                numeric_prefix=base/'prefix',numeric_resume=base/'resumed',output=output,cohort_receipt=receipt,
                config=config,partitions=parts,data=base,upstream=base,finite_probe=None,previous=None)
            method={'flush':h.CappedLog.flush,'close':h.CappedLog.close,'readback':h.OwnedOutput.verify_json}[boundary]
            owner=h.OwnedOutput if boundary=='readback' else h.CappedLog
            name='verify_json' if boundary=='readback' else boundary
            fired=[]
            def fault(instance,*a,**k):
                value=method(instance,*a,**k)
                status=output/'status.json'
                if not fired and status.exists() and json.loads(status.read_text()).get('state')=='capacity_entry_finished':
                    fired.append(True);config.write_text('{"late_source_change":true}')
                    if rebind:
                        output.rename(moved);output.mkdir()
                        (output/'result.json').write_text('neighbor result')
                        (output/'status.json').write_text('neighbor status')
                return value
            with mock.patch.object(egoemg_repro,'classic_emg_cuda_check',fake_q,create=True), \
                 mock.patch.object(egoemg_repro,'classic_emg_cohort',fake_cohort), \
                 mock.patch.object(h,'_source_snapshot',return_value={'fixture':'source'}), \
                 mock.patch.object(h,'_fit',return_value=fit_result),mock.patch.object(owner,name,fault):
                with self.assertRaises((ValueError,OSError)):
                    h.run(args)
            self.assertTrue(fired)
            with self.assertRaises(ValueError):
                guard.check()
            original=moved if rebind else output
            self.assertFalse((original/'result.json').exists())
            status=original/'status.json'
            self.assertTrue(not status.exists() or json.loads(status.read_text()).get('state')!='capacity_entry_finished')
            if rebind:
                self.assertEqual((output/'result.json').read_text(),'neighbor result')
                self.assertEqual((output/'status.json').read_text(),'neighbor status')

    def test_run_guards_source_through_final_flush_readback_and_close(self):
        for boundary in ('flush','readback','close'):
            with self.subTest(boundary=boundary):
                self._assert_tail_rejects_source_fault(boundary)

    def test_final_close_rebind_cleanup_uses_original_owned_directory(self):
        self._assert_tail_rejects_source_fault('close',rebind=True)

    def test_rebound_output_does_not_modify_replacement(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent, prefix='classic-capacity-test-') as d:
            destination = Path(d)/'entry'
            owned = self.m.OwnedOutput.claim(destination)
            destination.rename(Path(d)/'original')
            destination.mkdir()
            sentinel = destination/'result.json'
            sentinel.write_text('unrelated')
            owned.fail(ValueError('rebound'))
            self.assertEqual(sentinel.read_text(), 'unrelated')
            owned.close()

if __name__ == '__main__':
    unittest.main()
