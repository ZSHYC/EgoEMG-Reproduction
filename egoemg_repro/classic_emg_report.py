"""Original-alignment reports and profile-bound post-termination testing.

Private sufficient statistics retain user keys. Validation admits only genuinely
terminated formal training and never evaluates the held-out test. Native testing
requires an actual-child/profile binding and inherited parent pipe release.
Independent test replay and sanitized publication reopen the whole raw chain;
only the last owned capsule commits one experiment, never the entire paper.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import time
import uuid

import numpy as np

from .metrics import AngleAccumulator

IMPORTED_REPORT_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PASS_FORMAT='classic_original_prediction_pass_v1'
PROFILE_FORMAT='classic_validation_exact_replay_profile_v1'
AGREEMENT='exact_typed_prediction_replay_v1'
NATIVE_LAUNCH_FORMAT='classic_native_test_launch_v1'
COMPLETED_FORMAT='classic_original_22d_completed_report_v1'
CAPSULE_FORMAT='classic_original_22d_completion_capsule_v1'
GO_PROTOCOL='inherited_pipe_G_sha256_33bytes_EOF_v1'


JOINT_NAMES=(
    'THUMB_CMC_FE','THUMB_CMC_AA','THUMB_MCP_FE','THUMB_IP_FE',
    'INDEX_MCP_AA','INDEX_MCP_FE','INDEX_PIP_FE','INDEX_DIP_FE',
    'MIDDLE_MCP_AA','MIDDLE_MCP_FE','MIDDLE_PIP_FE','MIDDLE_DIP_FE',
    'RING_MCP_AA','RING_MCP_FE','RING_PIP_FE','RING_DIP_FE',
    'PINKY_MCP_AA','PINKY_MCP_FE','PINKY_PIP_FE','PINKY_DIP_FE',
    'wrist_pitch','wrist_yaw')
FINGERS={name:tuple(range(4*i,4*i+4)) for i,name in enumerate(('thumb','index','middle','ring','pinky'))}
PD_GROUPS=dict(proximal=(0,1,4,5,8,9,12,13,16,17),mid=(2,6,10,14,18),distal=(3,7,11,15,19))
PUBLIC_NATIVE_METRICS=frozenset(('test_loss','test_mae','test_landmark/fingertip','test_landmark/all',
    'test_mae_wrist/flexion','test_mae_wrist/deviation',
    *('test_derivatives/'+name for name in ('vel','acc','jerk')),
    *('test_mae_per_finger/'+name for name in FINGERS),*('test_mae_per_pd/'+name for name in PD_GROUPS)))


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


def validate_test_statistics(value,*,variant):
    """Validate private sufficient statistics; no inference or cohort proof is inferred."""
    a=alignment(variant);samples=4306
    fields={'format','scientific_eligible','scope','samples','alignment','trajectory','center'}
    if (type(value) is not dict or set(value)!=fields or value['format']!='classic_private_angle_statistics_v1' or
        value['scientific_eligible'] is not False or
        value['scope']!='arithmetic only; no inference, coverage, selected-best or calibration qualification' or
        type(value['samples']) is not int or value['samples']!=samples or type(value['alignment']) is not dict or
        set(value['alignment'])!=set(a) or any(type(value['alignment'][k]) is not int or value['alignment'][k]!=v for k,v in a.items())):
        raise ValueError('complete original test statistic schema/alignment differs')
    def same_number(actual,expected):
        return (actual is None if expected is None else type(actual) is float and math.isfinite(actual) and
            math.isclose(actual,expected,rel_tol=2e-12,abs_tol=1e-9))
    def record(r,extra=()):
        if type(r) is not dict or set(r)!={'absolute_error_sum_rad','valid_angle_count','mae_deg',*extra}:
            raise ValueError('typed angular sufficient-statistic record required')
        error,count=r['absolute_error_sum_rad'],r['valid_angle_count']
        if (type(error) is not float or not math.isfinite(error) or error<0 or type(count) is not int or count<0 or
            (not count and error!=0.) or not same_number(r['mae_deg'],error/count*180/math.pi if count else None)):
            raise ValueError('angular sum/count/derived degree mean differs')
    for mode,width in (('trajectory',a['output_length']),('center',1)):
        m=value[mode];fields={'mode','overall','invalid_angle_count','by_hand_split','user_summary',
            'by_hand','by_split','by_user','by_component','by_joint','by_finger','by_pd_group'}
        if type(m) is not dict or set(m)!=fields or m['mode']!=mode:raise ValueError('original report mode schema differs')
        record(m['overall']);total=m['overall'];count=total['valid_angle_count']
        if (count%22 or count>samples*width*22 or type(m['invalid_angle_count']) is not int or
            m['invalid_angle_count']!=samples*width*22-count):raise ValueError('original generated-mask denominator differs')
        mappings={'by_hand':{'left','right'},'by_split':{'user','gesture','both'},'by_component':{'fingers','wrist'},
            'by_finger':set(FINGERS),'by_pd_group':set(PD_GROUPS)}
        for key,keys in mappings.items():
            if type(m[key]) is not dict or set(m[key])!=keys:raise ValueError('complete report grouping differs: '+key)
            for r in m[key].values():record(r)
        users=m['by_user']
        if (type(users) is not dict or not 1<=len(users)<=41 or
            any(type(k) is not str or not k for k in users)):raise ValueError('private user grouping required')
        for r in users.values():record(r)
        for key in ('by_hand','by_split','by_user','by_component'):_conserves(list(m[key].values()),total)
        cross=m['by_hand_split']
        if type(cross) is not dict or set(cross)!={'left','right'}:raise ValueError('complete hand/split grouping required')
        for hand,splits in cross.items():
            if type(splits) is not dict or set(splits)!={'user','gesture','both'}:raise ValueError('complete hand/split cells required')
            for r in splits.values():record(r)
            _conserves(list(splits.values()),m['by_hand'][hand])
        for split in ('user','gesture','both'):_conserves([cross[h][split] for h in ('left','right')],m['by_split'][split])
        joints=m['by_joint']
        if type(joints) is not list or len(joints)!=22:raise ValueError('all22 original joints required')
        for i,r in enumerate(joints):
            record(r,('index','name'))
            if (type(r['index']) is not int or r['index']!=i or r['name']!=JOINT_NAMES[i] or
                r['valid_angle_count']!=count//22):raise ValueError('original joint semantics/generated mask differs')
        _conserves(joints,total)
        for key,mapping in (('by_finger',FINGERS),('by_pd_group',PD_GROUPS),
            ('by_component',{'fingers':tuple(range(20)),'wrist':(20,21)})):
            for name,ids in mapping.items():_conserves([joints[i] for i in ids],m[key][name])
        summary=m['user_summary'];observed=[r['mae_deg'] for r in users.values() if r['valid_angle_count']]
        if (type(summary) is not dict or set(summary)!={'mean_mae_deg','std_mae_deg','user_count','aggregation'} or
            type(summary['user_count']) is not int or summary['user_count']!=len(observed) or
            summary['aggregation']!='unweighted per-user means; population standard deviation (ddof=0)' or
            not same_number(summary['mean_mae_deg'],float(np.mean(observed)) if observed else None) or
            not same_number(summary['std_mae_deg'],float(np.std(observed)) if observed else None)):
            raise ValueError('original unweighted user dispersion differs')
    if value['center']['overall']['valid_angle_count']>value['trajectory']['overall']['valid_angle_count']:
        raise ValueError('original center cannot have more valid angles than complete trajectory')


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


def _pid_identity(pid,*,phase):
    if type(pid) is not int or pid<=0 or not isinstance(phase,str) or not phase:
        raise ValueError('positive actual PID and process phase required')
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        value=dict(pid=pid,start_ticks=int(fields[19]),phase=phase);_process_key(value)
        return value
    except (OSError,IndexError,TypeError) as exc:
        raise ValueError('actual process identity unavailable') from exc


def _pipe_identity(fd):
    if type(fd) is not int or fd<=2:raise ValueError('dedicated inherited pipe descriptor required')
    value=os.fstat(fd)
    if not stat.S_ISFIFO(value.st_mode):raise ValueError('parent release must use an actual pipe')
    return dict(read_fd=fd,device=value.st_dev,inode=value.st_ino,protocol=GO_PROTOCOL)


def _receive_launch_go(fd):
    """Pure OS handshake: no model/runtime import before the exact parent release."""
    if type(fd) is not int or fd<=2:raise ValueError('dedicated inherited pipe descriptor required')
    try:
        channel=_pipe_identity(fd);deadline=time.monotonic()+60.;payload=b''
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([fd],[],[],remaining)[0]:raise TimeoutError('parent release timed out')
            block=os.read(fd,34-len(payload))
            if not block:break
            payload+=block
            if len(payload)>33:raise ValueError('trailing parent release payload')
        if len(payload)!=33 or payload[:1]!=b'G':raise ValueError('missing or malformed parent release')
        return payload[1:].hex(),channel
    finally:os.close(fd)


def _validate_native_coverage(value,*,variant):
    """Validate original native test aggregation, without inventing predictions."""
    e=_evaluation();policy=report_pass_policy(variant=variant,role='test')
    fields={'samples','batch_sizes','coverage_sha256','batch_weighted_loss','test_loss',
        'target_sha256','mask_sha256','sample_sha256','fingerprint_encoding','fingerprint_scope'}
    samples=policy['samples'];batch=policy['batch_size'];digest=hashlib.sha256()
    for i in range(samples):digest.update(i.to_bytes(8,'big'))
    if (type(value) is not dict or set(value)!=fields or type(value['samples']) is not int or
        value['samples']!=samples or value['batch_sizes']!=[min(batch,samples-i) for i in range(0,samples,batch)] or
        any(type(x) is not int for x in value['batch_sizes']) or value['coverage_sha256']!=digest.hexdigest() or
        value['fingerprint_encoding']!='ordered chain of typed canonical-batch target/mask and frozen-sample receipts v1' or
        value['fingerprint_scope']!=e.FINGERPRINT_SCOPE or
        any(type(value[k]) is not float or not math.isfinite(value[k]) for k in ('batch_weighted_loss','test_loss')) or
        not math.isclose(value['test_loss'],value['batch_weighted_loss'],rel_tol=2e-6,abs_tol=1e-7)):
        raise ValueError('complete original native test coverage/loss differs')
    for name in ('target','mask','sample'):e.artifacts._hex(value[name+'_sha256'])


def compare_native_prediction(native,prediction,*,variant):
    """Exact Task5 consumed-input agreement; Task5 did not record predictions."""
    e=_evaluation();_validate_native_coverage(native,variant=variant)
    _validate_saved_coverage(prediction,report_pass_policy(variant=variant,role='test'))
    keys=('samples','batch_sizes','coverage_sha256','target_sha256','mask_sha256','sample_sha256',
        'fingerprint_encoding','fingerprint_scope')
    if e.artifacts.json_hash({k:native[k] for k in keys})!=e.artifacts.json_hash({k:prediction[k] for k in keys}):
        raise ValueError('prediction pass differs from actual native target/mask/sample/grouping')
    return dict(samples=4306,native_target_mask_sample_equal=True,native_prediction_hash_available=False)


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
    monitor=next(value for key,value in last['callbacks'].items() if key.startswith('ModelCheckpoint{'))
    original_path=e.artifacts._safe(monitor['best_model_path'])
    if original_path.name=='last.ckpt':raise ValueError('last cannot substitute for selected best')
    original_best=dict(path=str(original_path),present=False,file_sha256=None)
    if original_path.exists():
        guard.add(original_path)
        original=e.artifacts._read(original_path,state,'best')
        if e.artifacts._seal(original)!=selection['selected_state_sha256']:
            raise ValueError('original best differs from verified carried selection')
        original_best.update(present=True,file_sha256=e.file_sha256(original_path))
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
    return dict(run=run,manifest=manifest,selection=selection,last=last,best=best,partitions=partitions,
        original_best=original_best,guard=guard)


def _native_identity_record(context):
    """Reconstruct unchanged Task5 identity from actual terminated evidence, without a factory."""
    e=_evaluation();run=context['run'];manifest=context['manifest'];selection=context['selection']
    state=manifest['state_identity'];bindings=state['bindings']
    value=dict(format=e.FORMAT,scope=e.SCOPE,scientific_eligible=False,production_ready=False,
        training_manifest_sha256=manifest['sha256'],training_result_sha256=e.file_sha256(run/'result.json'),
        last_checkpoint_sha256=e.file_sha256(run/'checkpoints/last.ckpt'),selection=selection,
        original_best=context['original_best'],policy=e.evaluation_policy(variant=manifest['policy']['variant']),
        runtime=state['runtime'],source_files_sha256=e._sources(),evaluation_framework_files_sha256=e._framework(),
        cohort_receipt_sha256=bindings['cohort_receipt_sha256'],cohort_sha256=manifest['cohort_sha256'],
        data_sha256=bindings['data_sha256'],partition_sha256=bindings['partition_sha256'],
        fingerprint_scope=e.FINGERPRINT_SCOPE,
        prediction_pass='not executed here; later post-termination report/recomputation stage')
    value['sha256']=e.artifacts.json_hash(value)
    return value


def _read_native_test(run,directory):
    """Reopen the actual Task5 snapshot, selection, full-input coverage and native metrics."""
    if any(not isinstance(p,(str,os.PathLike)) for p in (run,directory)):
        raise ValueError('real terminated run and immutable native sidecar paths required')
    e=_evaluation();run=e.artifacts._safe(run);directory=e.artifacts._safe(directory)
    guard=e.cohort._StatGuard([directory/name for name in
        ('identity.json','result.json','status.json','selected-best.json','selected-best.ckpt')])
    try:
        context=_terminated_context(run);_merge_guard(guard,context['guard'])
        identity=e.formal._json(directory/'identity.json');expected=_native_identity_record(context)
        if e.artifacts.json_hash(identity)!=e.artifacts.json_hash(expected):
            raise ValueError('native selected-best source/runtime/data/selection identity differs')
        result=e.formal._json(directory/'result.json');status=e.formal._json(directory/'status.json')
        reference=e.formal._json(directory/'selected-best.json');path=directory/'selected-best.ckpt'
        selected=e.artifacts._read(path,context['manifest']['state_identity'],'best')
        selection=context['selection']
        if (e.artifacts._seal(selected)!=selection['selected_state_sha256'] or
            e.storage.state_digest(selected['state_dict'])!=selection['selected_model_sha256'] or
            e.storage.state_digest(selected['MixedPrecision'])!=e.storage.state_digest(context['best']['MixedPrecision'])):
            raise ValueError('native immutable selected snapshot differs; no last fallback allowed')
        expected_reference=dict(format=e.FORMAT,scope='immutable_selected_best_reference',scientific_eligible=False,
            training_manifest_sha256=context['manifest']['sha256'],
            **{key:selection[key] for key in ('selected_state_sha256','selected_model_sha256','selected_event_sha256')},
            checkpoint_name=path.name,checkpoint_sha256=e.file_sha256(path),original_best=context['original_best'])
        if (e.artifacts.json_hash(reference)!=e.artifacts.json_hash(expected_reference) or
            e.artifacts.json_hash(status)!=e.artifacts.json_hash(dict(state='selected_best_test_finished',scope=e.SCOPE,
                scientific_eligible=False,production_ready=False,result_sha256=e.file_sha256(directory/'result.json')))):
            raise ValueError('native selected snapshot or finished status binding differs')
        fields={'format','scope','scientific_eligible','production_ready','identity_sha256','selection','metrics','coverage',
            'original_test_step','optimizer_executions','model_state_unchanged','scaler_state_unchanged','prediction_pass','process'}
        if (set(result)!=fields or result['format']!=e.FORMAT or result['scope']!=e.SCOPE or
            result['scientific_eligible'] is not False or result['production_ready'] is not False or
            result['identity_sha256']!=identity['sha256'] or
            e.artifacts.json_hash(result['selection'])!=e.artifacts.json_hash(reference) or
            result['original_test_step'] is not True or type(result['optimizer_executions']) is not int or
            result['optimizer_executions']!=0 or result['model_state_unchanged'] is not True or
            result['scaler_state_unchanged'] is not True or
            result['prediction_pass']!='not executed; subsequent report pass remains separate'):
            raise ValueError('native original test/invariance/result schema differs')
        _process_key(result['process'])
        if result['process']['phase']!='selected-best-test':raise ValueError('actual native test process phase differs')
        _validate_native_coverage(result['coverage'],variant=context['manifest']['policy']['variant'])
        metrics=result['metrics']
        if (type(metrics) is not dict or 'test_loss' not in metrics or
            any(type(k) is not str or type(v) is not float or not math.isfinite(v) for k,v in metrics.items()) or
            not math.isclose(metrics['test_loss'],result['coverage']['test_loss'],rel_tol=2e-6,abs_tol=1e-7)):
            raise ValueError('finite original native test metrics differ from consumed full-test loss')
        guard.check()
        return identity,result,guard
    except (KeyError,TypeError,IndexError,AttributeError,StopIteration) as exc:
        raise ValueError('incomplete original native test sidecar') from exc


def _native_command(path,fd):
    return [sys.executable,'-B','-m','egoemg_repro.classic_emg_report','_native-child',
        '--launch',str(path),'--go-fd',str(fd)]


def _native_namespaces(run,native,launch,profile):
    e=_evaluation();run=e.artifacts._safe(run);native=e.artifacts._safe(native);launch=e.artifacts._safe(launch)
    immutable=[run,*(e.artifacts._safe(item['directory']) for item in profile['artifacts'])]
    if (native==launch or native in launch.parents or launch in native.parents or
        any(path==root or root in path.parents for root in immutable for path in (native,launch))):
        raise ValueError('native and launch outputs must be separate from immutable training/calibration and one another')
    return run,native,launch


def _read_launch_manifest(path):
    """Validate the immutable before-GO manifest; no test or child completion is inferred."""
    if not isinstance(path,(str,os.PathLike)):raise ValueError('real pre-test launch manifest required')
    e=_evaluation();path=e.artifacts._safe(path)
    if path.name!='launch.json':raise ValueError('original launch.json artifact required')
    guard=e.cohort._StatGuard([path])
    try:
        value=e.formal._json(path)
        fields={'format','scope','scientific_eligible','production_ready','run','native_output','validation_profile',
            'identity_sha256','selection','sources','runtime','test_policy','directory_identity','parent','child','pipe','command','sha256'}
        if (set(value)!=fields or value['format']!=NATIVE_LAUNCH_FORMAT or value['scope']!='pre_test_process_profile_binding' or
            value['scientific_eligible'] is not False or value['production_ready'] is not False or
            value['sha256']!=e.artifacts.json_hash({k:v for k,v in value.items() if k!='sha256'})):
            raise ValueError('immutable pre-test manifest schema/seal differs')
        descriptor=value['validation_profile']
        if type(descriptor) is not dict or set(descriptor)!={'path','file_sha256','sha256'}:
            raise ValueError('actual frozen validation profile path and hashes required')
        profile_path=e.artifacts._safe(descriptor['path'])
        if descriptor['file_sha256']!=e.file_sha256(profile_path):raise ValueError('pre-test validation profile file differs')
        profile,other=verify_validation_profile(profile_path);_merge_guard(guard,other)
        context=_terminated_context(value['run']);_merge_guard(guard,context['guard'])
        expected=_identity_record(context['run'],context['manifest'],context['selection'],'val')
        if (descriptor['sha256']!=profile['sha256'] or
            e.artifacts.json_hash(profile['identity'])!=e.artifacts.json_hash(expected) or
            value['identity_sha256']!=expected['sha256'] or
            any(e.artifacts.json_hash(value[key])!=e.artifacts.json_hash(expected[key]) for key in ('selection','sources','runtime')) or
            e.artifacts.json_hash(value['test_policy'])!=e.artifacts.json_hash(profile['test_policy'])):
            raise ValueError('pre-test source/selected-best/runtime/profile binding differs')
        _native_namespaces(context['run'],value['native_output'],path.parent,profile)
        directory=value['directory_identity'];current=path.parent.stat()
        if (type(directory) is not list or len(directory)!=2 or any(type(x) is not int or x<0 for x in directory) or
            directory!=[current.st_dev,current.st_ino]):raise ValueError('launch directory creation identity differs')
        parent=value['parent'];child=value['child'];_process_key(parent);_process_key(child)
        channel=value['pipe']
        if (parent['phase']!='classic-native-test-parent' or child['phase']!='selected-best-test' or
            _process_key(parent)==_process_key(child) or type(channel) is not dict or
            set(channel)!={'read_fd','device','inode','protocol'} or
            any(type(channel[k]) is not int for k in ('read_fd','device','inode')) or
            channel['read_fd']<=2 or channel['device']<0 or channel['inode']<=0 or channel['protocol']!=GO_PROTOCOL or
            value['command']!=_native_command(path,channel['read_fd'])):
            raise ValueError('actual parent/child, inherited pipe or original child command binding differs')
        guard.check()
        return value,profile,context,guard
    except (KeyError,TypeError,IndexError,AttributeError) as exc:
        raise ValueError('incomplete immutable pre-test launch manifest') from exc


def _admit_native_child(path,go_sha256,channel):
    """Require actual self, live parent and argv after the inherited parent pipe release."""
    e=_evaluation();value,profile,context,guard=_read_launch_manifest(path)
    actual=_pid_identity(os.getpid(),phase='selected-best-test')
    parent=_pid_identity(os.getppid(),phase='classic-native-test-parent')
    if (go_sha256!=value['sha256'] or e.artifacts.json_hash(channel)!=e.artifacts.json_hash(value['pipe']) or
        actual!=value['child'] or parent!=value['parent'] or getattr(sys,'orig_argv',None)!=value['command']):
        raise ValueError('child did not receive its exact parent/manifest/pipe/argv admission')
    admission=_native_admission_record(path,value);guard.check()
    return value,admission,guard


def _native_admission_record(path,value):
    e=_evaluation()
    admission=dict(format=NATIVE_LAUNCH_FORMAT,scope='actual_child_admitted_frozen_profile_before_native_test',
        scientific_eligible=False,production_ready=False,manifest_sha256=value['sha256'],
        manifest_file_sha256=e.file_sha256(path),validation_profile_file_sha256=value['validation_profile']['file_sha256'],
        identity_sha256=value['identity_sha256'],process=value['child'],parent=value['parent'],pipe=value['pipe'],command=value['command'])
    admission['sha256']=e.artifacts.json_hash(admission)
    return admission


def _native_launch_result(path,value,identity):
    e=_evaluation();native=e.artifacts._safe(value['native_output'])
    return dict(format=NATIVE_LAUNCH_FORMAT,scope='profile_bound_original_native_test_finished',scientific_eligible=False,
        production_ready=False,manifest_sha256=value['sha256'],manifest_file_sha256=e.file_sha256(path),
        admission_file_sha256=e.file_sha256(path.parent/'child-admission.json'),native_identity_sha256=identity['sha256'],
        native_result_sha256=e.file_sha256(native/'result.json'),native_selected_checkpoint_sha256=e.file_sha256(native/'selected-best.ckpt'),
        parent=value['parent'],child=value['child'],child_exit_code=0,test_prediction_pass_executed=False)


def verify_native_test_launch(directory):
    """Reopen the complete parent/admission/native chain, without running any test."""
    if not isinstance(directory,(str,os.PathLike)):raise ValueError('real completed native launch artifact path required')
    e=_evaluation();directory=e.artifacts._safe(directory);path=directory/'launch.json'
    guard=e.cohort._StatGuard([directory/name for name in ('launch.json','child-admission.json','result.json','status.json')])
    try:
        value,profile,context,other=_read_launch_manifest(path);_merge_guard(guard,other)
        admission=e.formal._json(directory/'child-admission.json')
        if e.artifacts.json_hash(admission)!=e.artifacts.json_hash(_native_admission_record(path,value)):
            raise ValueError('actual child admission differs from before-GO process/profile/pipe binding')
        identity,native,other=_read_native_test(value['run'],value['native_output']);_merge_guard(guard,other)
        if e.artifacts.json_hash(native['process'])!=e.artifacts.json_hash(value['child']):
            raise ValueError('native test ran in a different child process than the before-GO binding')
        result=e.formal._json(directory/'result.json');status=e.formal._json(directory/'status.json')
        if (e.artifacts.json_hash(result)!=e.artifacts.json_hash(_native_launch_result(path,value,identity)) or
            e.artifacts.json_hash(status)!=e.artifacts.json_hash(dict(state='profile_bound_native_test_finished',scope=NATIVE_LAUNCH_FORMAT,
                scientific_eligible=False,production_ready=False,result_sha256=e.file_sha256(directory/'result.json')))):
            raise ValueError('parent/native finished result or status binding differs')
        guard.check()
        return value,identity,native,guard
    except (KeyError,TypeError,IndexError,AttributeError) as exc:
        raise ValueError('incomplete profile-bound native launch artifact') from exc


def _test_prediction_binding(run,evaluation_profile):
    if any(not isinstance(p,(str,os.PathLike)) for p in (run,evaluation_profile)):
        raise ValueError('actual run and completed native launch artifact paths required')
    value,_,native,guard=verify_native_test_launch(evaluation_profile)
    e=_evaluation();run=e.artifacts._safe(run);directory=e.artifacts._safe(evaluation_profile)
    if run!=e.artifacts._safe(value['run']):raise ValueError('test prediction run differs from bound native test')
    binding=dict(directory=str(directory),manifest_sha256=value['sha256'],
        result_sha256=e.file_sha256(directory/'result.json'),native_result_sha256=e.file_sha256(Path(value['native_output'])/'result.json'))
    profile=e.formal._json(value['validation_profile']['path'])
    immutable=[run,directory,e.artifacts._safe(value['native_output']),
        *(e.artifacts._safe(item['directory']) for item in profile['artifacts'])]
    guard.check()
    return dict(manifest=value,native=native,binding=binding,guard=guard,immutable=immutable)


def build_classic_status(run):
    """Read an append-only journal prefix; never infer live or scientifically complete."""
    if not isinstance(run,(str,os.PathLike)):raise ValueError('actual formal run directory required')
    e=_evaluation();run=e.artifacts._safe(run);directory=run/'journal'
    guard=e.cohort._StatGuard([run/'identity.json',run/'status.json',directory/'journal.json'])
    try:
        manifest=e.formal._json(run/'identity.json');status=e.formal._json(run/'status.json')
        state_identity=manifest['state_identity'];counts=state_identity['counts'];policy=manifest['policy']
        scope=manifest['scope'];variant=policy['variant'];batch=600 if variant=='vemg2pose' else 256
        expected_counts=dict(e.formal.FULL_COUNTS)
        if scope=='formal_engineering_acceptance':expected_counts['train']=2*batch
        if (manifest['format']!=e.formal.FORMAT or scope not in ('formal_training','formal_engineering_acceptance') or
            manifest['sha256']!=e.artifacts.json_hash({k:v for k,v in manifest.items() if k!='sha256'}) or
            manifest['sources']!=e.formal._source_snapshot() or variant not in ('vemg2pose','neuropose') or
            type(policy['batch_size']) is not int or policy['batch_size']!=batch or
            e.artifacts.json_hash(counts)!=e.artifacts.json_hash(expected_counts) or
            status['state'] not in ('running','failed','engineering_finished','training_finished')):
            raise ValueError('formal progress manifest/source/policy/status differs')
        journal_policy=e.artifacts._journal_policy(state_identity['sha256'],counts['train'],batch)
        header=e.formal._json(directory/'journal.json',limit=65536)
        if e.artifacts.json_hash(header)!=e.artifacts.json_hash(dict(format='formal_journal_v1',**journal_policy)):
            raise ValueError('progress journal header binding differs')
        state=e.artifacts._empty();paths=sorted(directory.glob('events-*.jsonl'));total=0;partial=False
        for index,path in enumerate(paths):
            if path.name!=f'events-{index:05d}.jsonl':raise ValueError('progress journal segment gap/order differs')
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
            try:
                before=os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=e.artifacts.MAX_SEGMENT_BYTES:
                    raise ValueError('bounded regular progress journal segment required')
                raw=b''
                while len(raw)<before.st_size:
                    block=os.read(fd,min(65536,before.st_size-len(raw)))
                    if not block:raise ValueError('short progress journal read')
                    raw+=block
                after=os.fstat(fd);current=path.stat(follow_symlinks=False)
                if ((after.st_dev,after.st_ino)!=(before.st_dev,before.st_ino) or after.st_size<before.st_size or
                    (current.st_dev,current.st_ino)!=(before.st_dev,before.st_ino)):
                    raise ValueError('progress journal changed identity or shrank during read')
            finally:os.close(fd)
            total+=len(raw)
            if total>journal_policy['total_budget_bytes']:raise ValueError('progress journal exceeds bounded total budget')
            lines=raw.splitlines(keepends=True)
            if lines and not lines[-1].endswith(b'\n'):
                if index!=len(paths)-1 or len(lines[-1])>e.artifacts.MAX_RECORD_BYTES:
                    raise ValueError('partial progress tail outside final bounded record')
                lines.pop();partial=True
            for line in lines:
                if len(line)>e.artifacts.MAX_RECORD_BYTES:raise ValueError('progress journal record exceeds bound')
                envelope=json.loads(line);record=envelope['record']
                if record.get('kind')=='epoch' and record.get('expected_samples')!=counts['val']:
                    raise ValueError('progress full-validation population differs')
                if line!=e.artifacts._line(record,state,journal_policy):raise ValueError('progress journal canonical chain differs')
        guard.check()
        if manifest['sources']!=e.formal._source_snapshot():raise ValueError('progress source changed during read')
        return dict(format='classic_readonly_progress_v1',scope='observed_bounded_journal_prefix',
            scientific_eligible=False,production_ready=False,training_completion_verified=False,process_liveness_verified=False,
            reported_state=status['state'],training_scope=scope,variant=variant,manifest_sha256=manifest['sha256'],
            attempts=state['attempts'],optimizer_executions=state['executions'],overflow_skips=state['skips'],
            completed_validation_epochs=state['epochs'],last_validation_loss=state['epoch_summaries'][-1]['val_loss'] if state['epochs'] else None,
            prefix_records=state['sequence'],prefix_sha256=state['chain_sha256'],partial_tail_ignored=partial)
    except (KeyError,TypeError,AttributeError,UnicodeError,json.JSONDecodeError) as exc:
        raise ValueError('invalid bounded formal progress artifacts') from exc


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
        if role=='test':fields.add('evaluation_profile')
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
        if role=='test':
            binding=result['evaluation_profile']
            if type(binding) is not dict or set(binding)!={'directory','manifest_sha256','result_sha256','native_result_sha256'}:
                raise ValueError('completed native launch binding required for test prediction')
            admitted=_test_prediction_binding(identity['run'],binding['directory']);_merge_guard(guard,admitted['guard'])
            if (e.artifacts.json_hash(binding)!=e.artifacts.json_hash(admitted['binding']) or
                _process_key(process) in (_process_key(admitted['manifest']['parent']),_process_key(admitted['native']['process']))):
                raise ValueError('test prediction native binding or fresh process differs')
            compare_native_prediction(admitted['native']['coverage'],result['coverage'],variant=expected['policy']['variant'])
            validate_test_statistics(result['statistics'],variant=expected['policy']['variant'])
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
            if role=='test' and (e.artifacts.json_hash(baseline['evaluation_profile'])!=e.artifacts.json_hash(result['evaluation_profile']) or
                e.artifacts.json_hash(baseline['statistics'])!=e.artifacts.json_hash(result['statistics'])):
                raise ValueError('independent test report statistics/native binding differs')
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
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    _publish_bytes_file(directory,name,raw,publication)


def _publish_bytes_file(directory,name,raw,publication):
    """Creation-owned, bounded, fsynced first link; no replacement of public files."""
    if Path(name).name!=name or name in ('','.','..'):raise ValueError('one publication filename required')
    if type(raw) is not bytes or len(raw)>8*1024**2:raise ValueError('bounded report/profile bytes required')
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


def _discard_publications(publications):
    for publication in publications.values():_discard_fd(publication.pop('fd',None))


def _revoke_prediction_status(output,rescue,created_identity,exc,publications,*,scope=PASS_FORMAT):
    """Invalidate only creation-owned result/status inodes, including moved files."""
    for name,publication in publications.items():_revoke_profile(rescue,name,publication)
    if rescue is None:return
    try:
        held=os.fstat(rescue.fd)
        if (held.st_dev,held.st_ino)!=created_identity:return
        if rescue.owner.is_current(output):
            # NOREPLACE preserves any foreign status, even within our directory.
            _evaluation().artifacts._write_json(rescue,'status.json',dict(state='failed',scope=scope,
                scientific_eligible=False,production_ready=False,error_type=type(exc).__name__,error=str(exc)))
    except (OSError,ValueError):pass


def _finish_prediction_publication(prepared,output,identity,result,owned,rescue,backup,publications):
    """Filesystem finalization only; genuine CUDA admission belongs to the outer runner."""
    e=_evaluation()
    try:
        prepared.guard.check();owned.check()
        _publish_profile_file(owned,'result.json',result,publications.setdefault('result.json',{}))
        _publish_profile_file(owned,'status.json',dict(state='prediction_pass_finished',scope=PASS_FORMAT,
            scientific_eligible=False,production_ready=False,result_sha256=e.file_sha256(output/'result.json')),
            publications.setdefault('status.json',{}))
        if e.artifacts.json_hash(e.formal._json(output/'result.json'))!=e.artifacts.json_hash(result):
            raise ValueError('prediction result readback differs')
        owned.close()
        prepared.guard.check();rescue.check()
        if _report_sources()!=identity['sources']:raise ValueError('prediction sources changed before publication')
        _,_,observed=_read_prediction_pass(output,role=identity['policy']['role']);_merge_guard(prepared.guard,observed)
        rescue.close()
        prepared.guard.check();backup.check()
        if _report_sources()!=identity['sources']:raise ValueError('prediction sources changed during final close')
        return result
    except BaseException as exc:
        _revoke_prediction_status(output,backup,owned.owner.identity,exc,publications);raise


def run_validation_prediction(run,*,output,reference=None,device='cuda:0'):
    """Full validation only, after actual formal termination; never evaluates test."""
    return _run_prediction(run,output=output,reference=reference,device=device,role='val')


def run_test_prediction(run,*,evaluation_profile,output,reference=None,device='cuda:0'):
    """Fresh original-forward test pass, only after the bound original native test."""
    if device!='cuda:0':raise ValueError('test prediction requires actual CUDA cuda:0')
    admitted=_test_prediction_binding(run,evaluation_profile)
    return _run_prediction(run,output=output,reference=reference,device=device,role='test',admitted=admitted)


def recompute_selected_best(run,*,device,output,evaluation_profile,reference):
    if not isinstance(reference,(str,os.PathLike)):raise ValueError('actual independent test prediction reference path required')
    return run_test_prediction(run,evaluation_profile=evaluation_profile,output=output,reference=reference,device=device)


def _run_prediction(run,*,output,reference,device,role,admitted=None):
    if device!='cuda:0' or role not in ('val','test') or (role=='test')!=(admitted is not None):
        raise ValueError('canonical CUDA validation or native-bound test required')
    e=_evaluation();output=e._sidecar_output(run,output)
    original=_read_prediction_pass(reference,role=role,expected_kind='reference') if reference is not None else None
    immutable=[e.artifacts._safe(run)]
    if reference is not None:immutable.append(e.artifacts._safe(reference))
    if admitted is not None:immutable.extend(admitted['immutable'])
    if any(output==parent or parent in output.parents for parent in immutable):
        raise ValueError('prediction output must be separate from immutable source/reference/native/calibration directories')
    prepared=e.prepare_selected_best_evaluation(run,device=device)
    identity=_prediction_identity(prepared,role);torch=e.storage._runtime()
    if (not torch.cuda.is_available() or type(prepared.precision).__name__!='MixedPrecision' or
        not prepared.precision.scaler.is_enabled()):raise ValueError('genuine canonical CUDA MixedPrecision required')
    if original is not None and original[0]!=identity:raise ValueError('prediction replay source/selection/runtime differs')
    if original is not None:_merge_guard(prepared.guard,original[2])
    policy=identity['policy'];kind='replay' if original is not None else 'reference'
    process=e.cuda._process_identity('classic-'+role+'-'+kind)
    if original is not None and _process_key(original[1]['process'])==_process_key(process):
        raise ValueError('prediction replay requires a fresh process')
    if admitted is not None:
        _merge_guard(prepared.guard,admitted['guard'])
        if (e.artifacts.json_hash(policy)!=e.artifacts.json_hash(admitted['manifest']['test_policy']) or
            _process_key(process) in (_process_key(admitted['manifest']['parent']),_process_key(admitted['native']['process'])) or
            (original is not None and e.artifacts.json_hash(original[1]['evaluation_profile'])!=e.artifacts.json_hash(admitted['binding']))):
            raise ValueError('test prediction profile/policy/native process differs')
    windows=[w for w in prepared.partitions['windows'] if w['role']==role]
    dataset=prepared.datasets[role];subjects={w['episode']:str(dataset.store.episodes[w['episode']]['subject']) for w in windows}
    ledger=ReportInputLedger(windows,variant=policy['variant'],batch_size=policy['batch_size'],role=role,subjects=subjects)
    if len(dataset)!=policy['samples'] or ledger.samples!=policy['samples']:raise ValueError('complete canonical prediction cohort required')
    used=e.formal._Dataset(dataset,prepared,policy['samples'])
    loader=torch.utils.data.DataLoader(used,batch_size=policy['batch_size'],num_workers=0,shuffle=False,
        drop_last=False,generator=torch.Generator().manual_seed(policy['loader_seed']))
    owned=rescue=backup=None;created_identity=None;publications={}
    try:
        owned=e.artifacts._Directory(output,create=True);created_identity=owned.owner.identity
        rescue=e.artifacts._Directory(output,create=False)
        if rescue.owner.identity!=created_identity:raise ValueError('prediction sidecar acquisition changed')
        backup=_retain_directory(rescue)
        e.artifacts._write_json(owned,'identity.json',identity)
        e.artifacts._write_json(owned,'progress.json',dict(state='running',scope=PASS_FORMAT,scientific_eligible=False))
        result=_consume_prediction_batches(prepared,loader,ledger,device=device)
        if admitted is not None:
            compare_native_prediction(admitted['native']['coverage'],result['coverage'],variant=policy['variant'])
            validate_test_statistics(result['statistics'],variant=policy['variant'])
            result['evaluation_profile']=admitted['binding']
        if original is not None:
            compare_prediction_replay(original[1]['coverage'],result['coverage'])
            if role=='test' and e.artifacts.json_hash(original[1]['statistics'])!=e.artifacts.json_hash(result['statistics']):
                raise ValueError('independent exact-prediction report statistics differ')
        result.update(format=PASS_FORMAT,scope=policy['scope'],scientific_eligible=False,production_ready=False,
            identity_sha256=identity['sha256'],process=process,pass_kind=kind,
            reference=dict(directory=str(e.artifacts._safe(reference)),result_sha256=e.file_sha256(Path(reference)/'result.json')) if original is not None else None)
        _finish_prediction_publication(prepared,output,identity,result,owned,rescue,backup,publications)
        owned=rescue=None
        return result
    except BaseException as exc:
        _revoke_prediction_status(output,backup or rescue,created_identity,exc,publications);raise
    finally:
        try:
            if owned is not None:owned.close()
        except BaseException as exc:
            _revoke_prediction_status(output,backup or rescue,created_identity,exc,publications);raise
        finally:
            try:
                if rescue is not None:rescue.close()
            except BaseException as exc:
                _revoke_prediction_status(output,backup or rescue,created_identity,exc,publications);raise
            finally:
                _discard_publications(publications);_discard_directory(backup)


def _native_test_child(path,fd):
    # The inherited pipe is consumed before importing the model runtime.
    go,channel=_receive_launch_go(fd)
    value,admission,guard=_admit_native_child(path,go,channel)
    e=_evaluation();path=e.artifacts._safe(path);directory=e.artifacts._Directory(path.parent,create=False)
    try:
        if list(directory.owner.identity)!=value['directory_identity']:raise ValueError('child launch directory was replaced')
        guard.check();e.artifacts._write_json(directory,'child-admission.json',admission)
        guard.add(path.parent/'child-admission.json')
        if e.artifacts.json_hash(e.formal._json(path.parent/'child-admission.json'))!=e.artifacts.json_hash(admission):
            raise ValueError('durable child admission readback differs')
        directory.check();guard.check()
    finally:directory.close()
    guard.check()
    result=e.run_selected_best_test(value['run'],output=value['native_output'],device='cuda:0')
    if result['process']!=value['child']:raise ValueError('actual native test child identity differs')
    guard.check()
    return result


def _stop_native_child(child):
    """Only this Popen child is stopped; it shares its controller's process group."""
    if child is None or child.poll() is not None:return
    child.terminate()
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        child.kill();child.wait(timeout=15)


def _finish_native_publication(path,value,identity,guard,owned,rescue,backup,publications):
    """Filesystem finalization only; actual CUDA native test is admitted by the outer runner."""
    e=_evaluation();result=_native_launch_result(path,value,identity)
    try:
        guard.check();owned.check()
        _publish_profile_file(owned,'result.json',result,publications.setdefault('result.json',{}))
        _publish_profile_file(owned,'status.json',dict(state='profile_bound_native_test_finished',scope=NATIVE_LAUNCH_FORMAT,
            scientific_eligible=False,production_ready=False,result_sha256=e.file_sha256(path.parent/'result.json')),
            publications.setdefault('status.json',{}))
        _,_,_,other=verify_native_test_launch(path.parent);_merge_guard(guard,other)
        owned.close();guard.check();rescue.check()
        rescue.close();guard.check();backup.check()
        return result
    except BaseException as exc:
        _revoke_prediction_status(path.parent,backup,owned.owner.identity,exc,publications,scope=NATIVE_LAUNCH_FORMAT);raise


def run_native_test(run,*,validation_profile,output,launch_output,device='cuda:0'):
    """Durably bind the frozen validation profile and actual child before releasing native test."""
    if device!='cuda:0':raise ValueError('native test launch requires actual CUDA cuda:0')
    profile,guard=verify_validation_profile(validation_profile)
    context=_terminated_context(run);_merge_guard(guard,context['guard'])
    e=_evaluation();expected=_identity_record(context['run'],context['manifest'],context['selection'],'val')
    if e.artifacts.json_hash(profile['identity'])!=e.artifacts.json_hash(expected):
        raise ValueError('native test run differs from actual calibrated selected best')
    output=e._sidecar_output(run,output);launch=e._sidecar_output(run,launch_output)
    run,output,launch=_native_namespaces(run,output,launch,profile)
    owned=rescue=backup=child=None;read=write=None;created=None;publications={}
    try:
        owned=e.artifacts._Directory(launch,create=True);created=owned.owner.identity
        rescue=e.artifacts._Directory(launch,create=False)
        if rescue.owner.identity!=created:raise ValueError('native launch acquisition changed')
        backup=_retain_directory(rescue)
        e.artifacts._write_json(owned,'progress.json',dict(state='running',scope=NATIVE_LAUNCH_FORMAT,
            scientific_eligible=False,production_ready=False))
        read,write=os.pipe();channel=_pipe_identity(read);path=launch/'launch.json';command=_native_command(path,read)
        child=subprocess.Popen(command,pass_fds=(read,),cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL)
        os.close(read);read=None
        process=_pid_identity(child.pid,phase='selected-best-test')
        parent=_pid_identity(os.getpid(),phase='classic-native-test-parent')
        value=dict(format=NATIVE_LAUNCH_FORMAT,scope='pre_test_process_profile_binding',scientific_eligible=False,production_ready=False,
            run=str(run),native_output=str(output),validation_profile=dict(path=str(e.artifacts._safe(validation_profile)),
                file_sha256=e.file_sha256(validation_profile),sha256=profile['sha256']),
            identity_sha256=expected['sha256'],selection=expected['selection'],sources=expected['sources'],runtime=expected['runtime'],
            test_policy=profile['test_policy'],directory_identity=list(created),parent=parent,child=process,pipe=channel,command=command)
        value['sha256']=e.artifacts.json_hash(value)
        guard.check();e.artifacts._write_json(owned,'launch.json',value);guard.add(path)
        if e.artifacts.json_hash(e.formal._json(path))!=e.artifacts.json_hash(value):raise ValueError('before-GO launch readback differs')
        # Full-cohort/profile verification finished before spawn. While the
        # child waits, use those SAME held guards plus durable manifest readback;
        # the child independently reopens everything after GO, before native test.
        guard.check();owned.check()
        if os.write(write,b'G'+bytes.fromhex(value['sha256']))!=33:raise OSError('short parent release write')
        os.close(write);write=None
        if child.wait(timeout=3600)!=0:raise RuntimeError('profile-bound native test child failed')
        guard.add(launch/'child-admission.json')
        admission=e.formal._json(launch/'child-admission.json')
        if e.artifacts.json_hash(admission)!=e.artifacts.json_hash(_native_admission_record(path,value)):
            raise ValueError('native child admission binding differs')
        identity,native,other=_read_native_test(run,output);_merge_guard(guard,other)
        if native['process']!=process:raise ValueError('native result does not belong to actual released child')
        result=_finish_native_publication(path,value,identity,guard,owned,rescue,backup,publications)
        owned=rescue=None
        return result
    except BaseException as exc:
        _discard_fd(write);write=None;_discard_fd(read);read=None
        try:_stop_native_child(child)
        finally:_revoke_prediction_status(launch,backup or rescue,created,exc,publications,scope=NATIVE_LAUNCH_FORMAT)
        raise
    finally:
        _discard_fd(write);_discard_fd(read)
        try:
            if owned is not None:owned.close()
        except BaseException as exc:
            _revoke_prediction_status(launch,backup or rescue,created,exc,publications,scope=NATIVE_LAUNCH_FORMAT);raise
        finally:
            try:
                if rescue is not None:rescue.close()
            except BaseException as exc:
                _revoke_prediction_status(launch,backup or rescue,created,exc,publications,scope=NATIVE_LAUNCH_FORMAT);raise
            finally:
                _discard_publications(publications);_discard_directory(backup)


def _public_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _completed_training(context,training):
    """Project counters from the already verified terminal LAST, never selected-best or result claims."""
    prefix=context['last']['formal']['journal_prefix'];selection=context['selection']
    expected=dict(completed_epochs=prefix['epochs'],attempts=prefix['attempts'],optimizer_executions=prefix['executions'],
        natural_overflows=prefix['skips'],stop_reason=selection['stop_reason'],interrupted=False,
        selection_scope='terminated-full-validation-loss',
        runtime_profile=context['manifest']['state_identity']['runtime']['runtime_profile'],
        trajectory_agreement_qualified=False,bitwise_trajectory_guaranteed=False)
    if (type(training) is not dict or prefix['epochs']!=context['last']['formal']['completed_epochs'] or
        prefix['epochs']!=selection['completed_epochs'] or
        _public_hash({k:training.get(k) for k in expected})!=_public_hash(expected)):
        raise ValueError('training counters, stop or runtime claims differ from verified terminal checkpoint/policy')
    return expected


def _public_completed_record(*,identity,prediction,manifest,training,native,subjects,data_summary,binding,recompute_result_sha256):
    """Projection of already admitted evidence; this formatter does not qualify a GPU run."""
    variant=identity['policy']['variant'];validate_test_statistics(prediction['statistics'],variant=variant)
    roster=sorted(set(subjects))
    if not roster or any(type(x) is not str or not x for x in roster):raise ValueError('private full-source subject roster required')
    aliases={name:f'participant_{i+1:03d}' for i,name in enumerate(roster)}
    statistics={key:copy.deepcopy(prediction['statistics'][key]) for key in ('samples','alignment','trajectory','center')}
    users=set(statistics['trajectory']['by_user'])
    if (not users<=set(aliases) or set(statistics['center']['by_user'])!=users or
        data_summary['source_subject_count']!=len(roster) or data_summary['test_subject_count']!=len(users)):
        raise ValueError('report users differ from full frozen metadata roster')
    for mode in ('trajectory','center'):
        statistics[mode]['by_user']={aliases[name]:record for name,record in statistics[mode]['by_user'].items()}
    metrics=native['metrics']
    if (type(metrics) is not dict or 'test_loss' not in metrics or not set(metrics)<=PUBLIC_NATIVE_METRICS or
        any(type(v) is not float or not math.isfinite(v) for v in metrics.values())):
        raise ValueError('public native metrics must use known original numeric names')
    state=manifest['state_identity'];recipe=state['recipe'];runtime=identity['runtime'];coverage=prediction['coverage']
    recipe_keys=('variant','upstream_experiment','window_length','train_stride','eval_stride','val_fraction','seed','jitter',
        'batch_size','num_workers','max_epochs','lr','weight_decay','eta_min','gradient_clip','precision','monitor','monitor_mode',
        'early_stopping_patience','augmentation','reader_field_selection','layout','normalization','mask_policy',
        'raw_loss_weights','effective_loss_weights')
    effective_keys=('precision','batch_size','global_batch_size','world_size','rank','num_workers','train_shuffle','seed',
        'log_every_n_steps','max_epochs','clip_value')
    # The pinned original VEMG/Neuro recipes both declare devices[0..5].
    # Recipe/effective fields have an exact contract in the verified checkpoint;
    # manifest adaptations alone were previously only self-hashed.
    adaptations=dict(declared_devices=[0,1,2,3,4,5],effective_devices=[0],world_size=state['effective']['world_size'],
        declared_per_rank_batch=recipe['batch_size'],effective_per_rank_batch=state['effective']['batch_size'],
        effective_global_batch=state['effective']['global_batch_size'],declared_num_workers=recipe['num_workers'],
        effective_num_workers=state['effective']['num_workers'])
    if (type(manifest['adaptations']) is not dict or
        _public_hash({k:manifest['adaptations'].get(k) for k in adaptations})!=_public_hash(adaptations)):
        raise ValueError('public adaptations differ from pinned declared and verified effective policy')
    runtime_keys=('runtime_profile','python','torch_build','cuda_build','cudnn','driver_version','device_name','capability',
        'total_memory','deterministic','cudnn_benchmark','cudnn_deterministic','matmul_precision','allow_tf32_matmul',
        'allow_tf32_cudnn','precision_plugin','scaler_enabled')
    versions=('torch','torchvision','numpy','scipy','hydra-core','omegaconf','pytorch-lightning','torchmetrics',
        'lightning-utilities','pandas','joblib')
    public_runtime={k:copy.deepcopy(runtime[k]) for k in runtime_keys if k in runtime}
    public_runtime['versions']={k:runtime['versions'][k] for k in versions if k in runtime['versions']}
    public_runtime['identity_sha256']=_public_hash(runtime)
    selection_keys=('stop_reason','completed_epochs','best_epoch','best_val_loss','selected_state_sha256',
        'selected_model_sha256','selected_event_sha256','last_checkpoint_sha256')
    training_keys=('completed_epochs','attempts','optimizer_executions','natural_overflows','stop_reason','interrupted','selection_scope',
        'runtime_profile','trajectory_agreement_qualified','bitwise_trajectory_guaranteed')
    sources=identity['sources']
    if any(Path(k).is_absolute() or '..' in Path(k).parts for k in sources):raise ValueError('repository-relative public source names required')
    record=dict(format=COMPLETED_FORMAT,scope='one_original_classic_experiment_on_released_legacy_data',
        scientific_eligible=True,production_ready=False,full_paper_complete=False,completion_requires_owned_capsule=True,
        variant=variant,selection={k:copy.deepcopy(identity['selection'][k]) for k in selection_keys},
        training={k:copy.deepcopy(training[k]) for k in training_keys},
        protocol=dict(recipe={k:copy.deepcopy(recipe[k]) for k in recipe_keys if k in recipe},
            effective={k:copy.deepcopy(state['effective'][k]) for k in effective_keys},
            adaptations=adaptations,
            test=copy.deepcopy(identity['policy']),validation_prediction_passes=2,test_traversals=3,
            selected_epoch_indexing='zero-based original checkpoint epoch',prediction_agreement=AGREEMENT),
        data=dict(copy.deepcopy(data_summary),hand_samples=copy.deepcopy(state['counts']),
            participant_labels='sorted unique full-source subject keys; mapping private; natural-person uniqueness not proven'),
        statistics=statistics,native_metrics=dict(values=copy.deepcopy(metrics),aggregation='unchanged original batch-weighted Trainer.test',
            angular_units='radians',landmark_units='original upstream coordinate units; not converted',
            derivative_units='original upstream sample-rate scaling; not reinterpreted'),
        runtime=public_runtime,sources_sha256=copy.deepcopy(sources),
        evidence=dict(training_manifest_sha256=identity['training_manifest_sha256'],training_result_sha256=identity['training_result_sha256'],
            test_prediction_identity_sha256=identity['sha256'],native_launch_manifest_sha256=binding['manifest_sha256'],
            native_launch_result_sha256=binding['result_sha256'],native_test_result_sha256=binding['native_result_sha256'],
            reference_result_sha256=prediction['reference']['result_sha256'],recompute_result_sha256=recompute_result_sha256,
            cohort_bindings={k:identity['cohort_bindings'][k] for k in ('cohort_receipt_sha256','data_sha256','partition_sha256',
                'normalization_sha256','numeric_qualification_artifact_sha256')},
            aggregate_test_fingerprints={k:copy.deepcopy(coverage[k]) for k in ('samples','batch_sizes','coverage_sha256',
                *(name+'_sha256' for name in ReportInputLedger.CHAINS))}),
        limitations=['Released legacy data/labels are not proven equivalent to PDF/current full release.',
            'Single rank and workers0 do not reproduce the original six-rank global batch or NeuroPose BatchNorm.',
            'Generated 2D label validity applies to all22 angles; raw marker/label regeneration is separate unfinished work.',
            'One seed42; no multi-seed or universal deterministic training trajectory claim.',
            'Exact prediction replay is observed under the frozen validation profile and this test pair only.',
            'Original FK remains batch dependent; native canonical batch is preserved.',
            'Subject overlap is counted from actual frozen windows; released condition names do not establish unseen-subject independence.'])
    record['sha256']=_public_hash(record)
    return record


def _completed_evidence(run,recompute):
    if any(not isinstance(p,(str,os.PathLike)) for p in (run,recompute)):
        raise ValueError('real terminated run and independent test recomputation artifact paths required')
    e=_evaluation();run=e.artifacts._safe(run);recompute=e.artifacts._safe(recompute)
    identity,prediction,guard=_read_prediction_pass(recompute,role='test',expected_kind='replay')
    if run!=e.artifacts._safe(identity['run']):raise ValueError('completed report run differs from actual test recomputation')
    admitted=_test_prediction_binding(run,prediction['evaluation_profile']['directory']);_merge_guard(guard,admitted['guard'])
    context=_terminated_context(run);_merge_guard(guard,context['guard'])
    manifest=context['manifest'];training=_completed_training(context,e.formal._json(run/'result.json'))
    data=e.artifacts._safe(manifest['provenance']['data'])
    guard.add(data/'manifest.json');guard.add(data/'metadata.npz')
    store=e.cuda.datasets.LegacyStore(data);subjects=sorted({str(item['subject']) for item in store.episodes})
    train=set();tests={name:set() for name in ('user','gesture','both')}
    for window in context['partitions']['windows']:
        subject=str(store.episodes[window['episode']]['subject'])
        if window['role']=='train':train.add(subject)
        elif window['role']=='test':tests[window['split']].add(subject)
    expected=set().union(*tests.values())
    if (set(prediction['statistics']['trajectory']['by_user'])!=expected or
        set(prediction['statistics']['center']['by_user'])!=expected):raise ValueError('actual test subject roster differs from report')
    summary=dict(source_subject_count=len(subjects),test_subject_count=len(expected),
        train_test_subject_overlap={name:len(users&train) for name,users in tests.items()})
    report=_public_completed_record(identity=identity,prediction=prediction,manifest=manifest,training=training,
        native=admitted['native'],subjects=subjects,data_summary=summary,binding=admitted['binding'],
        recompute_result_sha256=e.file_sha256(recompute/'result.json'))
    immutable=[*admitted['immutable'],recompute,e.artifacts._safe(prediction['reference']['directory'])]
    guard.check()
    return report,guard,immutable


def build_classic_completed(run,recompute):
    """Rebuild the public mapping; only a separate owned capsule commits publication."""
    report,guard,_=_completed_evidence(run,recompute);guard.check();return report


def render_classic_completed(report):
    lines=['# Classic selected-best experiment',
        '',f"Variant: **{report['variant']}**. This is one legacy-release experiment; the full paper remains incomplete.",
        '',f"Report SHA256: `{report['sha256']}`. Completion requires its bound `completion.json`.",
        '',f"Selection: original val_loss/min; epoch {report['selection']['best_epoch']} (zero-based); "
        f"{report['training']['completed_epochs']} completed epochs, {report['training']['optimizer_executions']} optimizer executions, "
        f"{report['training']['natural_overflows']} overflow skips.",
        '', 'Angular sums are radians; MAE is degrees. Aggregate sums/counts before division. Native metrics retain their own original aggregation.',
        '', '| Mode / group | Error sum (rad) | Valid angles | MAE (degrees) |','| --- | ---: | ---: | ---: |']
    def row(label,value):
        mean='n/a' if value['mae_deg'] is None else f"{value['mae_deg']:.9f}"
        lines.append(f"| {label} | {value['absolute_error_sum_rad']:.12g} | {value['valid_angle_count']} | {mean} |")
    for mode in ('trajectory','center'):
        statistics=report['statistics'][mode];row(mode+' / overall',statistics['overall'])
        for grouping in ('by_hand','by_split','by_component','by_finger','by_pd_group','by_user'):
            for name,value in statistics[grouping].items():row(mode+' / '+grouping+' / '+name,value)
        for value in statistics['by_joint']:row(mode+' / '+value['name'],value)
    lines.extend(['','Native original metrics (unchanged units; FK batch dependent):','','| Metric | Value |','| --- | ---: |'])
    for name,value in sorted(report['native_metrics']['values'].items()):lines.append(f'| {name} | {value:.12g} |')
    lines.extend(['','Limitations:','',*('- '+item for item in report['limitations']),
        '', 'Complete public aggregate evidence (unrounded JSON; same content as metrics.json):',
        '', '```json',json.dumps(report,sort_keys=True,indent=2,allow_nan=False),'```',''])
    return '\n'.join(lines)


def _completion_record(output,report):
    e=_evaluation()
    return dict(format=CAPSULE_FORMAT,scope=report['scope'],scientific_eligible=True,production_ready=False,
        full_paper_complete=False,report_sha256=report['sha256'],metrics_json_sha256=e.file_sha256(output/'metrics.json'),
        metrics_md_sha256=e.file_sha256(output/'metrics.md'),selected_state_sha256=report['selection']['selected_state_sha256'],
        recompute_result_sha256=report['evidence']['recompute_result_sha256'])


def verify_classic_completed(run,recompute,output):
    if not isinstance(output,(str,os.PathLike)):raise ValueError('actual completed publication directory required')
    report,guard,_=_completed_evidence(run,recompute);e=_evaluation();output=e.artifacts._safe(output)
    observed=e.cohort._StatGuard([output/name for name in ('metrics.json','metrics.md','completion.json')]);_merge_guard(guard,observed)
    if (e.artifacts.json_hash(e.formal._json(output/'metrics.json'))!=e.artifacts.json_hash(report) or
        (output/'metrics.md').read_bytes()!=render_classic_completed(report).encode() or
        e.artifacts.json_hash(e.formal._json(output/'completion.json'))!=e.artifacts.json_hash(_completion_record(output,report))):
        raise ValueError('public report/Markdown/completion differs from actual reopened evidence')
    guard.check();return report,guard


def _finish_completed_publication(run,recompute,output,report,guard,owned,rescue,backup,publications):
    """Filesystem commit only; the outer builder admits the genuine scientific evidence."""
    e=_evaluation()
    try:
        guard.check();owned.check()
        _publish_profile_file(owned,'metrics.json',report,publications.setdefault('metrics.json',{}))
        _publish_bytes_file(owned,'metrics.md',render_classic_completed(report).encode(),publications.setdefault('metrics.md',{}))
        rebuilt,other,_=_completed_evidence(run,recompute);_merge_guard(guard,other)
        if (e.artifacts.json_hash(rebuilt)!=e.artifacts.json_hash(report) or
            e.artifacts.json_hash(e.formal._json(output/'metrics.json'))!=e.artifacts.json_hash(report) or
            (output/'metrics.md').read_bytes()!=render_classic_completed(report).encode()):
            raise ValueError('public report readback or independent rebuild differs')
        guard.add(output/'metrics.json');guard.add(output/'metrics.md')
        owned.close();guard.check();rescue.check()
        rescue.close();guard.check();backup.check()
        # Both fallible semantic directory closes and the full evidence rebuild
        # precede the final capsule. Last raw FD disposal is nonthrowing cleanup.
        capsule=_completion_record(output,report)
        _publish_profile_file(backup,'completion.json',capsule,publications.setdefault('completion.json',{}))
        if e.artifacts.json_hash(e.formal._json(output/'completion.json'))!=e.artifacts.json_hash(capsule):
            raise ValueError('completion capsule readback differs')
        guard.add(output/'completion.json');guard.check();backup.check()
        for name,publication in publications.items():_check_profile_file(backup,name,publication)
        return report
    except BaseException as exc:
        _revoke_prediction_status(output,backup,owned.owner.identity,exc,publications,scope=COMPLETED_FORMAT);raise


def publish_classic_completed(run,recompute,*,output):
    """Publish sanitized JSON/Markdown, then its owned completion capsule last."""
    report,guard,immutable=_completed_evidence(run,recompute);e=_evaluation();output=e._sidecar_output(run,output)
    if any(output==parent or parent in output.parents for parent in immutable):
        raise ValueError('public report must be separate from immutable training/calibration/test sidecars')
    owned=rescue=backup=None;created=None;publications={}
    try:
        owned=e.artifacts._Directory(output,create=True);created=owned.owner.identity
        rescue=e.artifacts._Directory(output,create=False)
        if rescue.owner.identity!=created:raise ValueError('public report output acquisition changed')
        backup=_retain_directory(rescue)
        value=_finish_completed_publication(run,recompute,output,report,guard,owned,rescue,backup,publications)
        owned=rescue=None;return value
    except BaseException as exc:
        _revoke_prediction_status(output,backup or rescue,created,exc,publications,scope=COMPLETED_FORMAT);raise
    finally:
        try:
            if owned is not None:owned.close()
        except BaseException as exc:
            _revoke_prediction_status(output,backup or rescue,created,exc,publications,scope=COMPLETED_FORMAT);raise
        finally:
            try:
                if rescue is not None:rescue.close()
            except BaseException as exc:
                _revoke_prediction_status(output,backup or rescue,created,exc,publications,scope=COMPLETED_FORMAT);raise
            finally:
                _discard_publications(publications);_discard_directory(backup)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    status=commands.add_parser('status',help='read a bounded progress journal without constructing a model or claiming completion')
    status.add_argument('--run',type=Path,required=True)
    validation=commands.add_parser('validation-pass',help='one complete selected-best validation prediction pass')
    validation.add_argument('--run',type=Path,required=True);validation.add_argument('--output',type=Path,required=True)
    validation.add_argument('--reference',type=Path);validation.add_argument('--device',default='cuda:0')
    freeze=commands.add_parser('freeze-validation',help='freeze two fresh complete validation passes with exact replay')
    freeze.add_argument('--reference',type=Path,required=True);freeze.add_argument('--replay',type=Path,required=True)
    freeze.add_argument('--output',type=Path,required=True)
    verify=commands.add_parser('verify-validation',help='reopen a frozen profile and all of its raw evidence')
    verify.add_argument('--profile',type=Path,required=True)
    native=commands.add_parser('native-test',help='bind the frozen validation profile before launching original selected-best test')
    native.add_argument('--run',type=Path,required=True);native.add_argument('--validation-profile',type=Path,required=True)
    native.add_argument('--output',type=Path,required=True);native.add_argument('--launch-output',type=Path,required=True)
    native.add_argument('--device',default='cuda:0')
    verify_native=commands.add_parser('verify-native-test',help='reopen the completed parent/child/profile/native evidence chain')
    verify_native.add_argument('--launch-output',type=Path,required=True)
    test_pass=commands.add_parser('test-pass',help='one full original-forward test prediction pass after bound native testing')
    test_pass.add_argument('--run',type=Path,required=True);test_pass.add_argument('--evaluation-profile',type=Path,required=True)
    test_pass.add_argument('--output',type=Path,required=True);test_pass.add_argument('--reference',type=Path)
    test_pass.add_argument('--device',default='cuda:0')
    verify_test=commands.add_parser('verify-test-pass',help='reopen a test prediction pass and its complete native/reference chain')
    verify_test.add_argument('--output',type=Path,required=True)
    publish=commands.add_parser('publish-completed',help='reopen all evidence and publish sanitized metrics plus completion capsule last')
    publish.add_argument('--run',type=Path,required=True);publish.add_argument('--recompute',type=Path,required=True)
    publish.add_argument('--output',type=Path,required=True)
    verify_completed=commands.add_parser('verify-completed',help='compare public files/capsule with the actual private evidence chain')
    verify_completed.add_argument('--run',type=Path,required=True);verify_completed.add_argument('--recompute',type=Path,required=True)
    verify_completed.add_argument('--output',type=Path,required=True)
    child=commands.add_parser('_native-child',help='internal inherited-pipe child; cannot evaluate without parent release')
    child.add_argument('--launch',type=Path,required=True);child.add_argument('--go-fd',type=int,required=True)
    args=parser.parse_args(argv)
    if args.command=='status':result=build_classic_status(args.run)
    elif args.command=='validation-pass':
        result=run_validation_prediction(args.run,output=args.output,reference=args.reference,device=args.device)
    elif args.command=='freeze-validation':
        result=freeze_validation_profile(args.reference,args.replay,output=args.output)
    elif args.command=='verify-validation':
        profile,_=verify_validation_profile(args.profile)
        result=dict(format=profile['format'],sha256=profile['sha256'],verified=True)
    elif args.command=='native-test':
        result=run_native_test(args.run,validation_profile=args.validation_profile,output=args.output,
            launch_output=args.launch_output,device=args.device)
    elif args.command=='verify-native-test':
        manifest,_,_,_=verify_native_test_launch(args.launch_output)
        result=dict(format=NATIVE_LAUNCH_FORMAT,sha256=manifest['sha256'],verified=True)
    elif args.command=='test-pass':
        result=run_test_prediction(args.run,evaluation_profile=args.evaluation_profile,output=args.output,
            reference=args.reference,device=args.device)
    elif args.command=='verify-test-pass':
        _,value,_=_read_prediction_pass(args.output,role='test')
        result=dict(format=PASS_FORMAT,verified=True,role='test',pass_kind=value['pass_kind'],
            samples=value['coverage']['samples'],scientific_eligible=False)
    elif args.command=='publish-completed':result=publish_classic_completed(args.run,args.recompute,output=args.output)
    elif args.command=='verify-completed':
        report,_=verify_classic_completed(args.run,args.recompute,args.output)
        result=dict(format=COMPLETED_FORMAT,sha256=report['sha256'],verified=True,scientific_eligible=True,full_paper_complete=False)
    else:result=_native_test_child(args.launch,args.go_fd)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
