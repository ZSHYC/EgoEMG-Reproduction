"""Original-alignment reports and post-termination validation calibration.

Private sufficient statistics retain user keys. The validation commands admit
only genuinely terminated formal training and never evaluate the held-out test.
Native-test launch binding, test recomputation and sanitized scientific export
remain separate, unfinished stages; a validation profile is not completion.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from .metrics import AngleAccumulator

IMPORTED_REPORT_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PASS_FORMAT='classic_original_prediction_pass_v1'
PROFILE_FORMAT='classic_validation_exact_replay_profile_v1'
AGREEMENT='exact_typed_prediction_replay_v1'


JOINT_NAMES=(
    'THUMB_CMC_FE','THUMB_CMC_AA','THUMB_MCP_FE','THUMB_IP_FE',
    'INDEX_MCP_AA','INDEX_MCP_FE','INDEX_PIP_FE','INDEX_DIP_FE',
    'MIDDLE_MCP_AA','MIDDLE_MCP_FE','MIDDLE_PIP_FE','MIDDLE_DIP_FE',
    'RING_MCP_AA','RING_MCP_FE','RING_PIP_FE','RING_DIP_FE',
    'PINKY_MCP_AA','PINKY_MCP_FE','PINKY_PIP_FE','PINKY_DIP_FE',
    'wrist_pitch','wrist_yaw')
FINGERS={name:tuple(range(4*i,4*i+4)) for i,name in enumerate(('thumb','index','middle','ring','pinky'))}
PD_GROUPS=dict(proximal=(0,1,4,5,8,9,12,13,16,17),mid=(2,6,10,14,18),distal=(3,7,11,15,19))


def alignment(variant):
    if variant not in ('vemg2pose','neuropose'):raise ValueError('original classic variant required')
    left=1790 if variant=='vemg2pose' else 0
    return dict(input_length=7790,left_context=left,right_context=0,output_length=7790-left,
        original_center=3895,output_center=3895-left)


def validate_original_tuple(output,*,variant,original_target,original_mask):
    """Validate the unchanged tuple once, without re-cropping or interpolation."""
    a=alignment(variant)
    if type(output) is not tuple or len(output)!=3:raise ValueError('original three-part tuple required')
    prediction,target,mask=output
    if (not isinstance(original_target,np.ndarray) or original_target.ndim!=3 or
        original_target.shape[1:]!=(22,7790) or original_target.dtype!=np.float32 or
        not isinstance(original_mask,np.ndarray) or original_mask.dtype!=np.bool_ or
        original_mask.shape!=(len(original_target),7790)):
        raise ValueError('original complete target/mask input schema differs')
    shape=(len(original_target),22,a['output_length'])
    if (not len(original_target) or not isinstance(prediction,np.ndarray) or prediction.shape!=shape or
        prediction.dtype not in (np.float16,np.float32) or not isinstance(target,np.ndarray) or
        target.shape!=shape or target.dtype!=np.float32 or not isinstance(mask,np.ndarray) or
        mask.dtype!=np.bool_ or mask.shape!=(len(target),a['output_length']) or
        not np.isfinite(prediction).all() or not np.isfinite(target).all() or not np.isfinite(original_target).all() or
        not np.array_equal(target,original_target[...,a['left_context']:]) or
        not np.array_equal(mask,original_mask[...,a['left_context']:])):
        raise ValueError('original tuple dtype/shape/finite/target/generated mask differs')
    return a


def _record(error,count):
    return dict(absolute_error_sum_rad=float(error),valid_angle_count=int(count),
        mae_deg=float(error)/int(count)*180/math.pi if count else None)


def _conserves(records,total):
    if (sum(r['valid_angle_count'] for r in records)!=total['valid_angle_count'] or
        not math.isclose(math.fsum(r['absolute_error_sum_rad'] for r in records),
            total['absolute_error_sum_rad'],rel_tol=2e-12,abs_tol=1e-9)):
        raise ValueError('report sufficient-statistic partition does not conserve sums/counts')


class _AngleStatistics:
    """Arithmetic helper; small-width fixtures do not establish original alignment."""
    def __init__(self,*,center_index):
        if type(center_index) is not int or center_index<0:raise ValueError('exact center index required')
        self.center_index=center_index
        self.accumulators={mode:AngleAccumulator(mode) for mode in ('trajectory','center')}
        self.joint_sums={mode:np.zeros(22,np.float64) for mode in self.accumulators}
        self.joint_counts={mode:np.zeros(22,np.int64) for mode in self.accumulators}
        self.samples=0
    def update(self,prediction,target,mask,*,hand,split,user):
        if (not isinstance(prediction,np.ndarray) or prediction.ndim!=2 or prediction.shape[0]!=22 or
            prediction.dtype not in (np.float16,np.float32) or not isinstance(target,np.ndarray) or
            target.shape!=prediction.shape or target.dtype!=np.float32 or
            not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=(prediction.shape[1],) or
            self.center_index>=prediction.shape[1] or not np.isfinite(prediction).all() or not np.isfinite(target).all() or
            hand not in ('left','right') or split not in ('user','gesture','both') or not isinstance(user,str) or not user):
            raise ValueError('finite original 22D hand tuple/generated mask/metadata required')
        for mode,accumulator in self.accumulators.items():
            accumulator.update(prediction,target,mask,hand=hand,split=split,user=user,center_index=self.center_index)
            p,t,m=prediction,target,mask
            if mode=='center':
                span=slice(self.center_index,self.center_index+1);p=p[:,span];t=t[:,span];m=m[span]
            errors=np.abs(p.astype(np.float64)-t.astype(np.float64))
            self.joint_sums[mode]+=errors[:,m].sum(axis=1,dtype=np.float64)
            self.joint_counts[mode]+=int(m.sum())
        self.samples+=1
    def report(self):
        result=dict(format='classic_private_angle_statistics_v1',scientific_eligible=False,
            scope='arithmetic only; no inference, coverage, selected-best or calibration qualification',samples=self.samples)
        for mode,accumulator in self.accumulators.items():
            value=accumulator.report();sums=self.joint_sums[mode];counts=self.joint_counts[mode]
            value['by_joint']=[dict(index=i,name=JOINT_NAMES[i],**_record(sums[i],counts[i])) for i in range(22)]
            def groups(mapping):return {name:_record(sums[list(ids)].sum(),counts[list(ids)].sum()) for name,ids in mapping.items()}
            value['by_finger']=groups(FINGERS);value['by_pd_group']=groups(PD_GROUPS)
            _conserves(value['by_joint'],value['overall'])
            for key in ('by_hand','by_split','by_user','by_component'):_conserves(list(value[key].values()),value['overall'])
            _conserves([r for hand in value['by_hand_split'].values() for r in hand.values()],value['overall'])
            for key in ('by_finger','by_pd_group'):_conserves(list(value[key].values()),value['by_component'].get('fingers',_record(0,0)))
            result[mode]=value
        return result


class ClassicAngleReport(_AngleStatistics):
    """Fixed original-window center for each actual 22-angle classic variant."""
    def __init__(self,variant):
        self.alignment=alignment(variant)
        super().__init__(center_index=self.alignment['output_center'])
    def update(self,prediction,target,mask,*,hand,split,user):
        if not isinstance(prediction,np.ndarray) or prediction.shape!=(22,self.alignment['output_length']):
            raise ValueError('original context-cropped prediction width differs')
        super().update(prediction,target,mask,hand=hand,split=split,user=user)
    def report(self):return dict(super().report(),alignment=self.alignment.copy())


def _evaluation():
    # Arithmetic/status imports must not construct a model or initialize CUDA.
    from . import classic_emg_evaluate
    return classic_emg_evaluate


class ReportInputLedger:
    """Exact native input encoding, with separate EMG/metadata/prediction chains."""
    CHAINS=('target','mask','sample','emg','metadata','prediction')
    def __init__(self,windows,*,variant,batch_size,role,subjects):
        e=_evaluation()
        if (role not in ('val','test') or type(batch_size) is not int or batch_size<=0 or
            not windows or len(windows)>2153 or not isinstance(subjects,dict) or
            any(w.get('role')!=role or w['end']-w['start']!=7790 for w in windows) or
            len({(w['episode'],w['start'],w['end']) for w in windows})!=len(windows) or
            any(w['episode'] not in subjects or not isinstance(subjects[w['episode']],str) for w in windows)):
            raise ValueError('unique original validation/test windows and subject provenance required')
        self.windows=copy.deepcopy(windows);self.subjects=copy.deepcopy(subjects);self.role=role;self.variant=variant
        self.contract=e.cuda.models.classic_contract(variant);self.samples=2*len(windows);self.batch_size=batch_size
        self.coverage=e.formal.Coverage(self.samples,ordered=True,batch_size=batch_size)
        self.prepared_samples=0;self.prepared_batches=0;self.committed_batches=0;self.pending={}
        self.hashes={name:hashlib.sha256() for name in self.CHAINS};self.predictions=[]
    def _descriptors(self,batch,ids):
        torch=_evaluation().storage._runtime();size=len(ids);descriptors=[];metadata=[]
        for offset,index in enumerate(ids):
            w=self.windows[index//2];hand=('left','right')[index%2]
            for key in ('episode','start','end'):
                value=batch.get(key)
                if not torch.is_tensor(value) or value.dtype!=torch.int64 or value.shape!=(size,) or int(value[offset])!=w[key]:
                    raise ValueError('original report sample metadata differs: '+key)
            if (batch['hand'][offset]!=hand or batch.get('role',[None]*size)[offset]!=self.role or
                batch.get('split',[None]*size)[offset]!=w['split'] or
                batch.get('subject',[None]*size)[offset]!=self.subjects[w['episode']]):
                raise ValueError('original report hand/role/split/subject differs')
            descriptors.append(dict(sample_index=index,episode=w['episode'],start=w['start'],end=w['end'],hand=hand))
            metadata.append(dict(descriptors[-1],role=self.role,split=w['split'],subject=self.subjects[w['episode']]))
        return descriptors,metadata
    def prepare(self,batch):
        e=_evaluation();torch=e.storage._runtime();e.formal.admit_formal_batch(batch,self.contract)
        if any(batch[name].device.type!='cpu' for name in ('emg','joint_angles','label_valid_mask')):
            raise ValueError('original report inputs must be fingerprinted before transfer')
        ids=batch['classic_sample_index'].tolist();size=len(ids)
        if size!=min(self.batch_size,self.samples-self.prepared_samples) or ids!=list(range(self.prepared_samples,self.prepared_samples+size)):
            raise ValueError('complete canonical report input grouping/order differs')
        descriptors,metadata=self._descriptors(batch,ids)
        span=slice(self.contract.left_context,7790-self.contract.right_context)
        target=batch['joint_angles'][...,span];mask=batch['label_valid_mask'][...,span]
        receipt=dict(sequence=self.prepared_batches,batch_size=size,indices=ids,
            target_shape=list(target.shape),mask_shape=list(mask.shape),target_dtype=str(target.dtype),mask_dtype=str(mask.dtype),
            target_sha256=e._tensor_digest(target),mask_sha256=e._tensor_digest(mask),
            sample_sha256=e.artifacts.json_hash(descriptors),metadata_sha256=e.artifacts.json_hash(metadata),
            emg_sha256=e._tensor_digest(batch['emg']))
        self.pending[receipt['sequence']]=copy.deepcopy(receipt);self.prepared_batches+=1;self.prepared_samples+=size
        return receipt
    def commit(self,batch,receipt,output):
        e=_evaluation();torch=e.storage._runtime()
        if (not isinstance(receipt,dict) or receipt.get('sequence')!=self.committed_batches or
            self.pending.get(self.committed_batches)!=receipt or batch['classic_sample_index'].tolist()!=receipt['indices']):
            raise ValueError('report must consume each exact prepared input once')
        descriptors,metadata=self._descriptors(batch,receipt['indices'])
        if (e.artifacts.json_hash(descriptors)!=receipt['sample_sha256'] or
            e.artifacts.json_hash(metadata)!=receipt['metadata_sha256'] or e._tensor_digest(batch['emg'])!=receipt['emg_sha256']):
            raise ValueError('prepared report input or metadata changed before forward consumption')
        if type(output) is not tuple or len(output)!=3 or any(not torch.is_tensor(x) for x in output):
            raise ValueError('actual original tensor tuple required')
        prediction,target,mask=output
        validate_original_tuple(tuple(x.detach().cpu().numpy() for x in output),variant=self.variant,
            original_target=batch['joint_angles'].detach().cpu().numpy(),original_mask=batch['label_valid_mask'].detach().cpu().numpy())
        if e._tensor_digest(target)!=receipt['target_sha256'] or e._tensor_digest(mask)!=receipt['mask_sha256']:
            raise ValueError('forward tuple consumed different prepared targets/masks')
        prediction_sha=e._tensor_digest(prediction)
        self.predictions.append(dict(sequence=receipt['sequence'],batch_size=receipt['batch_size'],
            dtype=str(prediction.dtype),shape=list(prediction.shape),prediction_sha256=prediction_sha,
            **{name+'_sha256':receipt[name+'_sha256'] for name in self.CHAINS if name!='prediction'}))
        self.coverage.add(receipt['indices'])
        for name,digest in self.hashes.items():
            value=prediction_sha if name=='prediction' else receipt[name+'_sha256']
            header=e.artifacts.json_hash(dict(sequence=receipt['sequence'],batch_size=receipt['batch_size'],digest=value))
            digest.update(bytes.fromhex(header))
        del self.pending[self.committed_batches];self.committed_batches+=1
    def finish(self):
        e=_evaluation();coverage=self.coverage.finish();coverage.pop('batch_weighted_loss')
        if self.pending or self.prepared_samples!=self.samples or self.prepared_batches!=self.committed_batches:
            raise ValueError('unconsumed or incomplete report inputs')
        return dict(coverage,role=self.role,**{name+'_sha256':digest.hexdigest() for name,digest in self.hashes.items()},
            predictions=copy.deepcopy(self.predictions),
            fingerprint_encoding='ordered chain of typed canonical-batch target/mask and frozen-sample receipts v1',
            fingerprint_scope=e.FINGERPRINT_SCOPE,
            additional_fingerprints='separate exact EMG, role/split/subject and original-dtype prediction chains')


def compare_prediction_replay(left,right):
    """Fixed exact policy; the engineering prediction tolerance does not apply."""
    expected={'samples','batch_sizes','coverage_sha256','role','predictions','fingerprint_encoding',
        'fingerprint_scope','additional_fingerprints',*(name+'_sha256' for name in ReportInputLedger.CHAINS)}
    if type(left) is not dict or type(right) is not dict or set(left)!=expected or set(right)!=expected or left!=right:
        raise ValueError('exact original input/metadata/typed prediction replay differs')
    return dict(input_hashes_equal=True,prediction_hashes_equal=True,agreement_policy=AGREEMENT)


def _validate_saved_coverage(value,policy):
    """Rebuild every bounded receipt chain; outer readers fix the real cohort."""
    e=_evaluation()
    compare_prediction_replay(value,value)
    samples,batch=policy['samples'],policy['batch_size']
    if any(type(x) is not int or x<=0 for x in (samples,batch)):
        raise ValueError('positive canonical coverage dimensions required')
    sizes=[min(batch,samples-i) for i in range(0,samples,batch)]
    digest=hashlib.sha256()
    for i in range(samples):digest.update(i.to_bytes(8,'big'))
    if (type(value['samples']) is not int or value['samples']!=samples or
        value['role']!=policy['role'] or value['batch_sizes']!=sizes or
        any(type(x) is not int for x in value['batch_sizes']) or
        value['coverage_sha256']!=digest.hexdigest() or
        value['fingerprint_encoding']!='ordered chain of typed canonical-batch target/mask and frozen-sample receipts v1' or
        value['fingerprint_scope']!=e.FINGERPRINT_SCOPE or
        value['additional_fingerprints']!='separate exact EMG, role/split/subject and original-dtype prediction chains' or
        type(value['predictions']) is not list or len(value['predictions'])!=len(sizes)):
        raise ValueError('complete ordered original report coverage differs')
    chains={name:hashlib.sha256() for name in ReportInputLedger.CHAINS}
    fields={'sequence','batch_size','dtype','shape',*(name+'_sha256' for name in chains)}
    for sequence,(record,size) in enumerate(zip(value['predictions'],sizes)):
        if (type(record) is not dict or set(record)!=fields or
            type(record['sequence']) is not int or record['sequence']!=sequence or
            type(record['batch_size']) is not int or record['batch_size']!=size or
            record['dtype'] not in ('torch.float16','torch.float32') or
            record['shape']!=[size,22,alignment(policy['variant'])['output_length']] or
            any(type(x) is not int for x in record['shape'])):
            raise ValueError('typed original prediction batch receipt differs')
        for name,chain in chains.items():
            content=record[name+'_sha256'];e.artifacts._hex(content)
            chain.update(bytes.fromhex(e.artifacts.json_hash(dict(sequence=sequence,batch_size=size,digest=content))))
    for name,chain in chains.items():
        if value[name+'_sha256']!=chain.hexdigest():
            raise ValueError('saved '+name+' receipt chain differs')


def _process_key(value):
    if (type(value) is not dict or set(value)!={'pid','start_ticks','phase'} or
        any(type(value[k]) is not int or value[k]<=0 for k in ('pid','start_ticks')) or
        not isinstance(value['phase'],str) or not value['phase']):
        raise ValueError('actual process PID/start_ticks/phase required')
    return value['pid'],value['start_ticks']


def _merge_guard(guard,other):
    guard.check();other.check()
    if any(path in guard.paths and guard.paths[path]!=stamp for path,stamp in other.paths.items()):
        raise ValueError('artifact changed between independent admissions')
    guard.paths.update(other.paths);guard.parents.update(other.parents)
    guard.check()


def report_pass_policy(*,variant,role):
    if variant not in ('vemg2pose','neuropose') or role not in ('val','test'):
        raise ValueError('original classic variant and validation/test role required')
    batch=600 if variant=='vemg2pose' else 256;samples=1576 if role=='val' else 4306
    return dict(scope='post_termination_original_prediction_pass',scientific_eligible=False,variant=variant,role=role,
        samples=samples,batch_size=batch,final_batch_size=samples%batch,num_workers=0,shuffle=False,drop_last=False,
        precision='16-mixed',world_size=1,seed=42,loader_seed=46 if role=='val' else 45,
        prediction_agreement=AGREEMENT,bitwise_trajectory_guaranteed=False)


def _report_sources():
    e=_evaluation();root=Path(__file__).resolve().parents[1]
    names=(*e.SOURCE_FILES,'egoemg_repro/classic_emg_report.py','egoemg_repro/metrics.py')
    sources={name:e.file_sha256(root/name) for name in names}
    if sources['egoemg_repro/classic_emg_report.py']!=IMPORTED_REPORT_SHA256:
        raise ValueError('executed report source changed after import')
    return sources


def _assert_prediction_invariance(prepared,model_sha,scaler_sha):
    e=_evaluation()
    if (e.storage.state_digest(prepared.module.state_dict())!=model_sha or
        e.storage.state_digest(prepared.precision.state_dict())!=scaler_sha or
        any(p.grad is not None for p in prepared.module.parameters())):
        raise ValueError('prediction pass changed selected parameters/buffers/scaler or created gradients')


def _consume_prediction_batches(prepared,loader,ledger,*,device):
    """Generic batch mechanics; only the outer runner admits an actual GPU run."""
    e=_evaluation();torch=e.storage._runtime();module=prepared.module.to(device).eval()
    model_sha=prepared.selection['selected_model_sha256'];scaler_sha=e.storage.state_digest(prepared.selected_checkpoint['MixedPrecision'])
    _assert_prediction_invariance(prepared,model_sha,scaler_sha)
    statistics=ClassicAngleReport(ledger.variant) if ledger.role=='test' else None
    for original in loader:
        guard=getattr(prepared,'guard',None)
        if guard is not None:guard.check()
        receipt=ledger.prepare(original)
        batch={k:value.to(device) if torch.is_tensor(value) else value for k,value in original.items()}
        with torch.inference_mode(),prepared.precision.test_step_context():output=module.forward(batch)
        ledger.commit(original,receipt,output)
        if statistics is not None:
            prediction,target,mask=(x.detach().cpu().numpy() for x in output)
            for i in range(len(prediction)):
                statistics.update(prediction[i],target[i],mask[i],hand=original['hand'][i],
                    split=original['split'][i],user=str(original['subject'][i]))
        _assert_prediction_invariance(prepared,model_sha,scaler_sha)
    coverage=ledger.finish();_assert_prediction_invariance(prepared,model_sha,scaler_sha)
    return dict(coverage=coverage,statistics=statistics.report() if statistics is not None else None,
        model_before_sha256=model_sha,model_after_sha256=e.storage.state_digest(module.state_dict()),
        scaler_before_sha256=scaler_sha,scaler_after_sha256=e.storage.state_digest(prepared.precision.state_dict()))


def _prediction_identity(prepared,role):
    value=_identity_record(prepared.run,prepared.manifest,prepared.selection,role)
    sources=value['sources']
    for name in sources:prepared.guard.add(Path(__file__).resolve().parents[1]/name)
    if value['runtime']!=prepared.identity['runtime']:
        raise ValueError('prepared prediction runtime differs from terminated training')
    prepared.guard.check()
    return value


def _identity_record(run,manifest,selection,role):
    e=_evaluation();policy=report_pass_policy(variant=manifest['policy']['variant'],role=role)
    value=dict(format=PASS_FORMAT,scope=policy['scope'],scientific_eligible=False,production_ready=False,
        run=str(run),training_manifest_sha256=manifest['sha256'],
        training_result_sha256=e.file_sha256(run/'result.json'),selection=selection,
        cohort_bindings=manifest['state_identity']['bindings'],policy=policy,
        runtime=manifest['state_identity']['runtime'],sources=_report_sources(),
        evaluation_framework=e._framework(),alignment=alignment(policy['variant']))
    value['sha256']=e.artifacts.json_hash(value)
    return value


def _terminated_context(run):
    """Read actual selected state/data/source without constructing any model."""
    if not isinstance(run,(str,os.PathLike)):raise ValueError('terminated training artifact path required')
    e=_evaluation();run=e.artifacts._safe(run);root=Path(__file__).resolve().parents[1]
    sources=_report_sources()
    guard=e.cohort._StatGuard([run/name for name in ('identity.json','status.json','result.json','checkpoints/last.ckpt')])
    for name in sources:guard.add(root/name)
    selection=e.formal.verify_termination_and_select_best(run)
    manifest=e.formal._json(run/'identity.json');state=manifest['state_identity']
    last=e.artifacts.read_formal_last(run/'checkpoints/last.ckpt',state)
    best=e._selected_core(last,selection)
    provenance=manifest['provenance']
    if not provenance['partitions']:raise ValueError('actual frozen partition provenance required')
    complete=e.formal._cohort_guard(provenance['data'],provenance['partitions'],upstream=provenance['upstream'],
        config=provenance['config'],receipt=provenance['cohort_receipt'],finite_probe=provenance['finite_probe'])
    _merge_guard(guard,complete)
    receipt=e.formal._json(provenance['cohort_receipt'])
    partitions=e.formal._json(provenance['partitions'],limit=256*1024**2)
    if (manifest['sources']!={name:sources[name] for name in e.formal.SOURCES} or
        state['counts']!=e.formal.FULL_COUNTS or receipt['sha256']!=manifest['cohort_sha256'] or
        receipt['data_identity']['sha256']!=state['bindings']['data_sha256'] or
        e.file_sha256(provenance['cohort_receipt'])!=state['bindings']['cohort_receipt_sha256'] or
        e.storage._json_hash(partitions)!=state['bindings']['partition_sha256']):
        raise ValueError('terminated report source or complete frozen cohort differs')
    framework_root=Path(e.storage.pl.__file__).resolve().parent
    for name in e.EVALUATION_FRAMEWORK_FILES:guard.add(framework_root/name)
    guard.check()
    return dict(run=run,manifest=manifest,selection=selection,best=best,partitions=partitions,guard=guard)


def _read_prediction_pass(directory,*,role,expected_kind=None):
    """Reopen private raw sidecars and the actual terminated training artifacts."""
    if not isinstance(directory,(str,os.PathLike)):raise ValueError('raw prediction artifact path required')
    if role not in ('val','test'):raise ValueError('validation/test role required')
    e=_evaluation();directory=e.artifacts._safe(directory)
    guard=e.cohort._StatGuard([directory/name for name in ('identity.json','result.json','status.json')])
    try:
        identity=e.formal._json(directory/'identity.json');result=e.formal._json(directory/'result.json')
        status=e.formal._json(directory/'status.json')
        if (identity.get('format')!=PASS_FORMAT or result.get('format')!=PASS_FORMAT or
            e.artifacts.json_hash(status)!=e.artifacts.json_hash(dict(state='prediction_pass_finished',scope=PASS_FORMAT,
                scientific_eligible=False,production_ready=False,result_sha256=e.file_sha256(directory/'result.json')))):
            raise ValueError('prediction pass did not finish with bound original artifacts')
        context=_terminated_context(identity['run']);_merge_guard(guard,context['guard'])
        expected=_identity_record(context['run'],context['manifest'],context['selection'],role)
        if e.artifacts.json_hash(identity)!=e.artifacts.json_hash(expected):
            raise ValueError('prediction pass selection/data/runtime/source identity differs')
        fields={'coverage','statistics','model_before_sha256','model_after_sha256','scaler_before_sha256',
            'scaler_after_sha256','format','scope','scientific_eligible','production_ready','identity_sha256',
            'process','pass_kind','reference'}
        kind=result['pass_kind'];process=result['process'];_process_key(process)
        scaler=e.storage.state_digest(context['best']['MixedPrecision'])
        if (set(result)!=fields or kind not in ('reference','replay') or
            (expected_kind is not None and kind!=expected_kind) or
            process['phase']!='classic-'+role+'-'+kind or result['scope']!=expected['scope'] or
            result['scientific_eligible'] is not False or result['production_ready'] is not False or
            result['identity_sha256']!=identity['sha256'] or
            result['model_before_sha256']!=context['selection']['selected_model_sha256'] or
            result['model_after_sha256']!=result['model_before_sha256'] or
            result['scaler_before_sha256']!=scaler or result['scaler_after_sha256']!=scaler or
            (role=='val' and result['statistics'] is not None)):
            raise ValueError('original prediction process/state invariance or result schema differs')
        _validate_saved_coverage(result['coverage'],expected['policy'])
        if kind=='reference':
            if result['reference'] is not None:raise ValueError('reference pass cannot reuse another pass')
        else:
            ref=result['reference']
            if type(ref) is not dict or set(ref)!={'directory','result_sha256'}:
                raise ValueError('actual original prediction reference required')
            reference=e.artifacts._safe(ref['directory'])
            if reference==directory or ref['result_sha256']!=e.file_sha256(reference/'result.json'):
                raise ValueError('prediction reference artifact differs')
            first,baseline,first_guard=_read_prediction_pass(reference,role=role,expected_kind='reference')
            _merge_guard(guard,first_guard)
            if first!=identity or _process_key(baseline['process'])==_process_key(process):
                raise ValueError('same-selection prediction replay requires a fresh process')
            compare_prediction_replay(baseline['coverage'],result['coverage'])
        guard.check()
        return identity,result,guard
    except (KeyError,TypeError,IndexError,AttributeError) as exc:
        raise ValueError('incomplete original prediction pass artifact') from exc


def _build_validation_profile(reference,replay):
    e=_evaluation()
    first,a,guard=_read_prediction_pass(reference,role='val',expected_kind='reference')
    second,b,other=_read_prediction_pass(replay,role='val',expected_kind='replay')
    _merge_guard(guard,other)
    reference=e.artifacts._safe(reference);replay=e.artifacts._safe(replay)
    if (first!=second or reference==replay or _process_key(a['process'])==_process_key(b['process']) or
        b['reference']!=dict(directory=str(reference),result_sha256=e.file_sha256(reference/'result.json'))):
        raise ValueError('two independent validation passes of the same actual selected best required')
    agreement=compare_prediction_replay(a['coverage'],b['coverage'])
    artifacts=[]
    for directory,result in ((reference,a),(replay,b)):
        artifacts.append(dict(directory=str(directory),pass_kind=result['pass_kind'],process=result['process'],
            **{name+'_sha256':e.file_sha256(directory/(name+'.json')) for name in ('identity','result','status')}))
    profile=dict(format=PROFILE_FORMAT,scope='observed_full_validation_exact_prediction_replay',
        scientific_eligible=False,production_ready=False,observed_exact_replay=True,
        universal_determinism_proven=False,agreement=agreement,identity=first,artifacts=artifacts,
        validation_samples=1576,validation_batch_sizes=a['coverage']['batch_sizes'],
        test_policy=report_pass_policy(variant=first['policy']['variant'],role='test'))
    profile['sha256']=e.artifacts.json_hash(profile);guard.check()
    return profile,guard


def verify_validation_profile(path):
    """A pass flag, a dictionary or a profile without its raw evidence never opens test."""
    if not isinstance(path,(str,os.PathLike)):raise ValueError('frozen validation profile artifact path required')
    e=_evaluation();path=e.artifacts._safe(path);guard=e.cohort._StatGuard([path])
    profile=e.formal._json(path)
    try:
        records=profile['artifacts']
        if profile['format']!=PROFILE_FORMAT or type(records) is not list or len(records)!=2:
            raise ValueError('two original full-validation pass artifacts required')
        rebuilt,other=_build_validation_profile(records[0]['directory'],records[1]['directory'])
        _merge_guard(guard,other)
        if e.artifacts.json_hash(profile)!=e.artifacts.json_hash(rebuilt):
            raise ValueError('frozen validation profile differs from independently reopened raw evidence')
        guard.check()
        return profile,guard
    except (KeyError,TypeError,IndexError,AttributeError) as exc:
        raise ValueError('incomplete frozen validation profile artifact') from exc


def _retain_directory(directory):
    """Duplicate an already admitted capability without reopening its public name."""
    held=copy.copy(directory);held.owner=copy.copy(directory.owner)
    held.owner.fd=os.dup(directory.fd)
    return held


def _discard_fd(fd):
    # This last retained capability has no remaining semantic checks or writes.
    # Cleanup cannot replace an established success/failure with a close error.
    if fd is not None:
        try:os.close(fd)
        except OSError:pass


def _discard_directory(directory):
    if directory is not None:
        fd=directory.fd;directory.owner.fd=None;_discard_fd(fd)


def _check_profile_file(directory,name,publication):
    actual=os.stat(name,dir_fd=directory.fd,follow_symlinks=False)
    if (actual.st_dev,actual.st_ino)!=publication['identity']:
        raise ValueError('published profile was replaced; foreign inode cannot be adopted')


def _publish_profile_file(directory,name,value,publication):
    """Retain the new file's creation handle before publishing its first link."""
    if Path(name).name!=name or name in ('','.','..'):raise ValueError('one profile filename required')
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(raw)>8*1024**2:raise ValueError('bounded validation profile required')
    directory.check(len(raw)+4096)
    temp='.'+name+'.'+uuid.uuid4().hex+'.incomplete'
    fd=os.open(temp,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=directory.fd)
    publication['fd']=fd
    own=os.fstat(fd);publication['identity']=(own.st_dev,own.st_ino)
    try:
        for offset in range(0,len(raw),65536):
            block=raw[offset:offset+65536];directory.check(len(block))
            if os.write(fd,block)!=len(block):raise OSError('short profile write')
        os.fsync(fd);directory.check()
        os.link(temp,name,src_dir_fd=directory.fd,dst_dir_fd=directory.fd,follow_symlinks=False)
        os.fsync(directory.fd);directory.check();_check_profile_file(directory,name,publication)
    finally:
        try:os.unlink(temp,dir_fd=directory.fd)
        except FileNotFoundError:pass


def _revoke_profile(directory,name,publication):
    """Invalidate the owned inode even if moved; never adopt a replacement."""
    identity=publication.get('identity')
    if identity is None:return
    try:
        held=os.fstat(publication['fd'])
        if (held.st_dev,held.st_ino)==identity:
            os.ftruncate(publication['fd'],0);os.fsync(publication['fd'])
    except (OSError,KeyError,TypeError,ValueError):pass
    if directory is None:return
    try:
        actual=os.stat(name,dir_fd=directory.fd,follow_symlinks=False)
        if (actual.st_dev,actual.st_ino)==identity:
            os.unlink(name,dir_fd=directory.fd);os.fsync(directory.fd)
    except (OSError,TypeError,ValueError):pass


def freeze_validation_profile(reference,replay,*,output):
    """Freeze observed full-validation agreement before any scientific test launch."""
    if not isinstance(output,(str,os.PathLike)):raise ValueError('new frozen profile file path required')
    e=_evaluation();output=e.artifacts._safe(output)
    if output.exists():raise FileExistsError('frozen profile exists; never overwrite')
    profile,guard=_build_validation_profile(reference,replay)
    run=e.artifacts._safe(profile['identity']['run'])
    sources=[run,*(e.artifacts._safe(x['directory']) for x in profile['artifacts'])]
    if any(output==parent or parent in output.parents for parent in sources):
        raise ValueError('frozen profile must be separate from immutable training and prediction sidecars')
    owned=rescue=backup=None;publication={}
    try:
        owned=e.artifacts._Directory(output.parent,create=False)
        rescue=e.artifacts._Directory(output.parent,create=False)
        if owned.owner.identity!=rescue.owner.identity:raise ValueError('profile parent directory changed')
        backup=_retain_directory(rescue)
        guard.check()
        _publish_profile_file(owned,output.name,profile,publication)
        _check_profile_file(rescue,output.name,publication)
        rebuilt,other=verify_validation_profile(output);_merge_guard(guard,other)
        if e.artifacts.json_hash(rebuilt)!=e.artifacts.json_hash(profile):raise ValueError('frozen profile readback differs')
        owned.close();owned=None
        guard.check();rescue.check();_check_profile_file(rescue,output.name,publication)
        rescue.close();rescue=None
        guard.check();backup.check();_check_profile_file(backup,output.name,publication)
        return profile
    except BaseException:
        _revoke_profile(backup or rescue,output.name,publication);raise
    finally:
        try:
            if owned is not None:owned.close()
        except BaseException:
            _revoke_profile(backup or rescue,output.name,publication);raise
        finally:
            try:
                if rescue is not None:rescue.close()
            except BaseException:
                _revoke_profile(backup or rescue,output.name,publication);raise
            finally:
                _discard_fd(publication.pop('fd',None));_discard_directory(backup)


def _revoke_prediction_status(output,rescue,created_identity,exc):
    """A retained creation inode can revoke a moved owned directory's marker."""
    if rescue is None:return
    try:
        held=os.fstat(rescue.fd)
        if (held.st_dev,held.st_ino)!=created_identity:return
        try:os.unlink('status.json',dir_fd=rescue.fd)
        except FileNotFoundError:pass
        os.fsync(rescue.fd)
        if rescue.owner.is_current(output):
            _evaluation().artifacts._write_json(rescue,'status.json',dict(state='failed',scope=PASS_FORMAT,
                scientific_eligible=False,production_ready=False,error_type=type(exc).__name__,error=str(exc)))
    except (OSError,ValueError):pass


def _finish_prediction_publication(prepared,output,identity,result,owned,rescue,backup):
    """Filesystem finalization only; genuine CUDA admission belongs to the outer runner."""
    e=_evaluation()
    try:
        prepared.guard.check();owned.check()
        e.artifacts._write_json(owned,'result.json',result)
        e.artifacts._write_json(owned,'status.json',dict(state='prediction_pass_finished',scope=PASS_FORMAT,
            scientific_eligible=False,production_ready=False,result_sha256=e.file_sha256(output/'result.json')),replace=True)
        if e.artifacts.json_hash(e.formal._json(output/'result.json'))!=e.artifacts.json_hash(result):
            raise ValueError('prediction result readback differs')
        owned.close()
        prepared.guard.check();rescue.check()
        if _report_sources()!=identity['sources']:raise ValueError('prediction sources changed before publication')
        _,_,observed=_read_prediction_pass(output,role='val');_merge_guard(prepared.guard,observed)
        rescue.close()
        prepared.guard.check();backup.check()
        if _report_sources()!=identity['sources']:raise ValueError('prediction sources changed during final close')
        return result
    except BaseException as exc:
        _revoke_prediction_status(output,backup,owned.owner.identity,exc);raise


def run_validation_prediction(run,*,output,reference=None,device='cuda:0'):
    """Full validation only, after actual formal termination; never evaluates test."""
    if device!='cuda:0':raise ValueError('validation calibration requires actual CUDA cuda:0')
    e=_evaluation();output=e._sidecar_output(run,output)
    original=_read_prediction_pass(reference,role='val') if reference is not None else None
    prepared=e.prepare_selected_best_evaluation(run,device=device)
    identity=_prediction_identity(prepared,'val');torch=e.storage._runtime()
    if (not torch.cuda.is_available() or type(prepared.precision).__name__!='MixedPrecision' or
        not prepared.precision.scaler.is_enabled()):raise ValueError('genuine canonical CUDA MixedPrecision required')
    if original is not None and original[0]!=identity:raise ValueError('validation replay source/selection/runtime differs')
    if original is not None:_merge_guard(prepared.guard,original[2])
    policy=identity['policy'];windows=[w for w in prepared.partitions['windows'] if w['role']=='val']
    dataset=prepared.datasets['val'];subjects={w['episode']:str(dataset.store.episodes[w['episode']]['subject']) for w in windows}
    ledger=ReportInputLedger(windows,variant=policy['variant'],batch_size=policy['batch_size'],role='val',subjects=subjects)
    if len(dataset)!=1576 or ledger.samples!=1576:raise ValueError('complete1576-hand validation required')
    used=e.formal._Dataset(dataset,prepared,1576)
    loader=torch.utils.data.DataLoader(used,batch_size=policy['batch_size'],num_workers=0,shuffle=False,
        drop_last=False,generator=torch.Generator().manual_seed(policy['loader_seed']))
    owned=rescue=backup=None;created_identity=None
    try:
        owned=e.artifacts._Directory(output,create=True);created_identity=owned.owner.identity
        rescue=e.artifacts._Directory(output,create=False)
        if rescue.owner.identity!=created_identity:raise ValueError('prediction sidecar acquisition changed')
        backup=_retain_directory(rescue)
        e.artifacts._write_json(owned,'identity.json',identity)
        e.artifacts._write_json(owned,'status.json',dict(state='running',scope=PASS_FORMAT,scientific_eligible=False))
        result=_consume_prediction_batches(prepared,loader,ledger,device=device)
        process=e.cuda._process_identity('classic-val-replay' if original is not None else 'classic-val-reference')
        if original is not None:
            if _process_key(original[1]['process'])==_process_key(process):raise ValueError('validation replay requires a fresh process')
            compare_prediction_replay(original[1]['coverage'],result['coverage'])
        result.update(format=PASS_FORMAT,scope=policy['scope'],scientific_eligible=False,production_ready=False,
            identity_sha256=identity['sha256'],process=process,pass_kind='replay' if original is not None else 'reference',
            reference=dict(directory=str(e.artifacts._safe(reference)),result_sha256=e.file_sha256(Path(reference)/'result.json')) if original is not None else None)
        _finish_prediction_publication(prepared,output,identity,result,owned,rescue,backup)
        owned=rescue=None
        return result
    except BaseException as exc:
        _revoke_prediction_status(output,backup or rescue,created_identity,exc);raise
    finally:
        try:
            if owned is not None:owned.close()
        except BaseException as exc:
            _revoke_prediction_status(output,backup or rescue,created_identity,exc);raise
        finally:
            try:
                if rescue is not None:rescue.close()
            except BaseException as exc:
                _revoke_prediction_status(output,backup or rescue,created_identity,exc);raise
            finally:_discard_directory(backup)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    validation=commands.add_parser('validation-pass',help='one complete selected-best validation prediction pass')
    validation.add_argument('--run',type=Path,required=True);validation.add_argument('--output',type=Path,required=True)
    validation.add_argument('--reference',type=Path);validation.add_argument('--device',default='cuda:0')
    freeze=commands.add_parser('freeze-validation',help='freeze two fresh complete validation passes with exact replay')
    freeze.add_argument('--reference',type=Path,required=True);freeze.add_argument('--replay',type=Path,required=True)
    freeze.add_argument('--output',type=Path,required=True)
    verify=commands.add_parser('verify-validation',help='reopen a frozen profile and all of its raw evidence')
    verify.add_argument('--profile',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='validation-pass':
        result=run_validation_prediction(args.run,output=args.output,reference=args.reference,device=args.device)
    elif args.command=='freeze-validation':
        result=freeze_validation_profile(args.reference,args.replay,output=args.output)
    else:
        profile,_=verify_validation_profile(args.profile)
        result=dict(format=profile['format'],sha256=profile['sha256'],verified=True)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
