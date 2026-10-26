"""Pure NumPy center/trajectory arithmetic; caches are private, outputs aggregate only."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np

from .preflight import _safe,_stamp

MODELS=('s','m','rn18','fusion')
SPLITS=('user','gesture','both')
# Pinned upstream constants.py and generate_per_gesture_appendix.py definitions.
FAMILIES={'single':tuple(range(30)), 'symmetric':(30,31,32,35,36,37,38,39,40,42,44,46,47,48,49,50,51,52),
          'asymmetric':(33,34,41,43,45,53,54,55,56,57,58,59)}
GROUPS={**{n:tuple(range(i*4,i*4+4)) for i,n in enumerate(('thumb','index','middle','ring','pinky'))},
        'proximal':(0,1,4,5,8,9,12,13,16,17),'mid':(2,6,10,14,18),'distal':(3,7,11,15,19),
        'wrist_flexion':(20,),'wrist_deviation':(21,)}
COHORT_DTYPES={'ordinal':'<u8','hand':'u1','split':'u1','gesture':'<i4','active':'bool',
               'crop_present':'bool','user_ordinal':'<u4'}
MAX_ROWS=2792
MAX_CACHE=16*1024**2
DEG=180/math.pi


def _array(value,dtype,shape):
    if not isinstance(value,np.ndarray) or value.dtype!=np.dtype(dtype) or value.shape!=shape:
        raise ValueError('cache typed array schema differs')


def _cohort(target,valid,cohort):
    if not isinstance(target,np.ndarray) or target.ndim!=2 or not 0<len(target)<=MAX_ROWS:
        raise ValueError('center population shape differs')
    n=len(target);_array(target,'<f4',(n,22));_array(valid,'bool',(n,22))
    if set(cohort)!=set(COHORT_DTYPES):raise ValueError('cohort schema differs')
    for key,dtype in COHORT_DTYPES.items():_array(cohort[key],dtype,(n,))
    if (not np.array_equal(cohort['ordinal'],np.arange(n,dtype='<u8')) or
        np.any(cohort['hand']>1) or np.any(cohort['split']>2) or
        np.any((cohort['gesture']<-1)|(cohort['gesture']>59)) or not np.isfinite(target[valid]).all()):
        raise ValueError('cohort order, labels or finite mask differs')


def _digest(arrays):
    h=hashlib.sha256()
    for name,value in sorted(arrays.items()):
        header=json.dumps(dict(name=name,dtype=value.dtype.str,shape=list(value.shape)),sort_keys=True,separators=(',',':')).encode()
        h.update(len(header).to_bytes(8,'big'));h.update(header);h.update(value.tobytes(order='C'))
    return h.hexdigest()


def cohort_identity(target,valid,cohort):
    _cohort(target,valid,cohort)
    return _digest(dict(target=target,valid=valid,**cohort))


def _validate(frames,target,valid,cohort):
    identity=cohort_identity(target,valid,cohort)
    if set(frames)!=set(MODELS):raise ValueError('exact four model predictions required')
    for frame in frames.values():
        if set(frame)!= {'prediction','ordinal','cohort_sha256'}:raise ValueError('prediction frame schema differs')
        _array(frame['prediction'],'<f4',target.shape);_array(frame['ordinal'],'<u8',(len(target),))
        if (frame['cohort_sha256']!=identity or not np.array_equal(frame['ordinal'],cohort['ordinal']) or
            not np.isfinite(frame['prediction'][valid]).all()):raise ValueError('prediction cohort identity or finite state differs')


def _summary(values):
    return dict(mean_mae_deg=float(np.mean(values))*DEG if values else None,
                std_mae_deg=float(np.std(values,ddof=0))*DEG if values else None,user_count=len(values),ddof=0)


def _stats(error,valid,users):
    counts=valid.sum(axis=1);sums=error.sum(axis=1);present=counts>0
    samples=sums[present]/counts[present]
    user_values=[];user_sample_values=[]
    for user in np.unique(users[present]):
        selected=(users==user)&present
        user_values.append(float(sums[selected].sum()/counts[selected].sum()))
        user_sample_values.append(float(np.mean(sums[selected]/counts[selected])))
    count=int(counts.sum());total=float(sums.sum());sample_sum=float(samples.sum())
    return dict(requested_samples=len(valid),metric_sample_count=int(present.sum()),
        valid_angle_count=count,invalid_angle_count=int(valid.size-count),absolute_error_sum_rad=total,
        angle_pooled_mae_deg=total/count*DEG if count else None,sample_error_sum_rad=sample_sum,
        sample_weighted_mae_deg=sample_sum/len(samples)*DEG if len(samples) else None,
        user_summary=_summary(user_values),user_sample_weighted_summary=_summary(user_sample_values))


def _error(prediction,target,valid):
    result=np.zeros(target.shape,dtype=np.float64)
    np.subtract(prediction.astype(np.float64),target.astype(np.float64),out=result,where=valid)
    np.abs(result,out=result)
    return result


def reduce_centers(frames,target,valid,cohort):
    """Reduce exact common centers; never invent missing-model zero predictions."""
    _validate(frames,target,valid,cohort);n=len(target);models={}
    for name in MODELS:
        error=_error(frames[name]['prediction'],target,valid)
        def select(rows=None,angles=None):
            rows=np.ones(n,bool) if rows is None else rows
            e=error[rows];v=valid[rows]
            if angles is not None:e=e[:,angles];v=v[:,angles]
            return _stats(e,v,cohort['user_ordinal'][rows])
        gid=cohort['gesture'];known=gid>=0;defined=known&(gid!=50);crop=cohort['crop_present']
        models[name]=dict(overall=select(),by_joint={str(i):select(angles=[i]) for i in range(22)},
            by_group={k:select(angles=list(a)) for k,a in GROUPS.items()},
            by_hand={h:select(cohort['hand']==i) for i,h in enumerate(('left','right'))},
            by_split={s:select(cohort['split']==i) for i,s in enumerate(SPLITS)},
            by_hand_split={h:{s:select((cohort['hand']==j)&(cohort['split']==i)) for i,s in enumerate(SPLITS)}
                           for j,h in enumerate(('left','right'))},
            by_gesture={str(i):select(gid==i) for i in range(-1,60)},
            by_family={family:select(np.isin(gid,ids)) for family,ids in FAMILIES.items()},
            by_activity={key:select(cohort['active']==value) for key,value in [('active',True),('inactive',False)]},
            views={key:select(mask) for key,mask in dict(all_requested=np.ones(n,bool),known60=known,
                defined59=defined,raw50=gid==50,unknown_minus1=gid==-1,crop_present=crop,crop_missing=~crop,
                crop_present_defined59=crop&defined).items()})
    return dict(format='legacy_derived_aggregate_v1',scope='synthetic_or_cached_arithmetic_only',
                benchmark_eligible=False,inference_performed=False,models=models)


def physical_center(prediction,target,valid,*,window_length=12000,left_context=510,right_context=0):
    if (window_length,left_context,right_context)!=(12000,510,0):raise ValueError('frozen context differs')
    for value,dtype in [(prediction,'<f4'),(target,'<f4'),(valid,'bool')]:_array(value,dtype,(22,11490))
    center=window_length//2-left_context
    return tuple(x[:,center:center+1] for x in (prediction,target,valid))


def reduce_trajectory_chunk(prediction,target,valid,metadata,state):
    """Stream aligned hand-time points; no prediction/target arrays remain in state."""
    if not isinstance(target,np.ndarray) or target.ndim!=2 or target.shape[0]!=22 or not 0<target.shape[1]<=11490:
        raise ValueError('trajectory shape differs')
    for value,dtype in [(prediction,'<f4'),(target,'<f4'),(valid,'bool')]:_array(value,dtype,target.shape)
    if (set(metadata)!={'hand','split','user_ordinal'} or any(type(x) is not int for x in metadata.values()) or
        metadata['hand'] not in (0,1) or metadata['split'] not in (0,1,2) or metadata['user_ordinal']<0 or
        not np.isfinite(prediction[valid]).all() or not np.isfinite(target[valid]).all()):
        raise ValueError('trajectory finite mask or metadata differs')
    if not state:state.update(format='legacy_trajectory_reduction_v1',buckets={})
    if set(state)!={'format','buckets'} or state['format']!='legacy_trajectory_reduction_v1':raise ValueError('trajectory state schema differs')
    error=_error(prediction.T,target.T,valid.T);mask=valid.T
    hand=('left','right')[metadata['hand']];split=SPLITS[metadata['split']]
    selections={'overall':list(range(22)),**{'by_joint/'+str(i):[i] for i in range(22)},
                **{'by_group/'+g:list(a) for g,a in GROUPS.items()},'by_hand/'+hand:list(range(22)),
                'by_split/'+split:list(range(22)),'by_hand_split/'+hand+'/'+split:list(range(22))}
    for key,angles in selections.items():
        e=error[:,angles];v=mask[:,angles];counts=v.sum(axis=1);sums=e.sum(axis=1);present=counts>0
        sample_sum=float((sums[present]/counts[present]).sum());count=int(counts.sum());total=float(sums.sum())
        b=state['buckets'].setdefault(key,dict(requested=0,possible=0,count=0,total=0.,samples=0,sample_sum=0.,users={},hist=[0]*181))
        b['requested']+=len(v);b['possible']+=v.size;b['count']+=count;b['total']+=total
        b['samples']+=int(present.sum());b['sample_sum']+=sample_sum
        user=b['users'].setdefault(metadata['user_ordinal'],[0.,0,0.,0])
        for i,value in enumerate((total,count,sample_sum,int(present.sum()))):user[i]+=value
        hist=np.histogram(e[v]*DEG,bins=np.r_[np.arange(181),np.inf])[0].tolist()
        b['hist']=[a+c for a,c in zip(b['hist'],hist)]


def trajectory_report(state):
    if state.get('format')!='legacy_trajectory_reduction_v1':raise ValueError('trajectory state schema differs')
    def record(key):
        b=state['buckets'][key];count=b['count'];samples=b['samples']
        return dict(requested_samples=b['requested'],metric_sample_count=samples,valid_angle_count=count,
            invalid_angle_count=b['possible']-count,absolute_error_sum_rad=b['total'],
            angle_pooled_mae_deg=b['total']/count*DEG if count else None,sample_error_sum_rad=b['sample_sum'],
            sample_weighted_mae_deg=b['sample_sum']/samples*DEG if samples else None,
            user_summary=_summary([v[0]/v[1] for v in b['users'].values() if v[1]]),
            user_sample_weighted_summary=_summary([v[2]/v[3] for v in b['users'].values() if v[3]]),
            histogram_valid_angles=list(b['hist']))
    out=dict(format='legacy_trajectory_aggregate_v1',sampling='aligned_hand_time_points',benchmark_eligible=False,
        overall=record('overall'),by_joint={str(i):record('by_joint/'+str(i)) for i in range(22)},
        by_group={g:record('by_group/'+g) for g in GROUPS},histogram_edges_deg=list(range(181))+[None])
    for dimension in ('by_hand','by_split'):
        out[dimension]={k.split('/')[1]:record(k) for k in state['buckets'] if k.startswith(dimension+'/')}
    out['by_hand_split']={h:{s:record('by_hand_split/'+h+'/'+s) for s in SPLITS if 'by_hand_split/'+h+'/'+s in state['buckets']}
                          for h in ('left','right') if any(k.startswith('by_hand_split/'+h+'/') for k in state['buckets'])}
    return out


def _project_stats(value):
    counts=('requested_samples','metric_sample_count','valid_angle_count','invalid_angle_count')
    numbers=('absolute_error_sum_rad','angle_pooled_mae_deg','sample_error_sum_rad','sample_weighted_mae_deg')
    result={}
    for key in counts:
        v=value[key]
        if type(v) is not int or v<0:raise ValueError('invalid aggregate count')
        result[key]=v
    for key in numbers:
        v=value[key]
        if v is not None and (type(v) not in (int,float) or not math.isfinite(v) or v<0):raise ValueError('invalid aggregate value')
        result[key]=v
    for key in ('user_summary','user_sample_weighted_summary'):
        summary=value[key];record={}
        for name in ('mean_mae_deg','std_mae_deg'):
            v=summary[name]
            if v is not None and (type(v) not in (int,float) or not math.isfinite(v) or v<0):raise ValueError('invalid user aggregate')
            record[name]=v
        if type(summary['user_count']) is not int or summary['user_count']<0 or summary['ddof']!=0:
            raise ValueError('invalid user aggregate count')
        record.update(user_count=summary['user_count'],ddof=0);result[key]=record
    return result


def public_projection(reduced):
    dimensions=dict(by_joint=tuple(map(str,range(22))),by_group=tuple(GROUPS),by_hand=('left','right'),
        by_split=SPLITS,by_gesture=tuple(map(str,range(-1,60))),by_family=tuple(FAMILIES),
        by_activity=('active','inactive'),views=('all_requested','known60','defined59','raw50',
            'unknown_minus1','crop_present','crop_missing','crop_present_defined59'))
    result=dict(format='legacy_derived_aggregate_v1',scope='synthetic_or_cached_arithmetic_only',
                benchmark_eligible=False,inference_performed=False,models={})
    for model in MODELS:
        source=reduced['models'][model];body={'overall':_project_stats(source['overall'])}
        for dimension,keys in dimensions.items():body[dimension]={k:_project_stats(source[dimension][k]) for k in keys}
        body['by_hand_split']={h:{s:_project_stats(source['by_hand_split'][h][s]) for s in SPLITS} for h in ('left','right')}
        result['models'][model]=body
    return result


def _cache_arrays(frames,target,valid,cohort):
    arrays=dict(target=target,valid=valid,**{'cohort__'+k:v for k,v in cohort.items()})
    for name in MODELS:
        arrays['prediction__'+name]=frames[name]['prediction'];arrays['ordinal__'+name]=frames[name]['ordinal']
        arrays['identity__'+name]=np.frombuffer(bytes.fromhex(frames[name]['cohort_sha256']),dtype='u1')
    return arrays


def write_private_cache(path,frames,target,valid,cohort):
    _validate(frames,target,valid,cohort);path=_safe(path)
    stream=io.BytesIO();np.savez(stream,**_cache_arrays(frames,target,valid,cohort));raw=stream.getvalue()
    if len(raw)>MAX_CACHE:raise ValueError('private cache exceeds bounded size')
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(descriptor,'wb') as output:
            output.write(raw);output.flush();os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True);raise
    return dict(format='legacy_private_center_cache_v1',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def read_private_cache(path,*,expected_sha256=None):
    if not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or any(x not in '0123456789abcdef' for x in expected_sha256):
        raise ValueError('explicit private cache SHA256 receipt required')
    path=_safe(path);before=path.stat()
    if before.st_mode&0o777!=0o600 or before.st_size>MAX_CACHE or before.st_uid!=os.getuid():
        raise ValueError('private cache permissions or size differs')
    try:
        raw=path.read_bytes()
        if _stamp(path.stat())!=_stamp(before) or hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('private cache changed or receipt differs')
        keys={'target','valid',*('cohort__'+k for k in COHORT_DTYPES),
              *(p+'__'+m for m in MODELS for p in ('prediction','ordinal','identity'))}
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            infos=z.infolist()
            if len(infos)!=len(keys) or {i.filename for i in infos}!={k+'.npy' for k in keys} or sum(i.file_size for i in infos)>MAX_CACHE:
                raise ValueError('private cache archive schema differs')
            for info in infos:
                with z.open(info) as item:
                    version=np.lib.format.read_magic(item)
                    if version not in ((1,0),(2,0)):raise ValueError('private cache header version differs')
                    reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                    shape,fortran,dtype=reader(item)
                    if fortran or dtype.hasobject or len(shape)>2 or math.prod(shape)*dtype.itemsize>MAX_CACHE:
                        raise ValueError('private cache typed header differs')
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:values={k:archive[k] for k in keys}
        target=values['target'];valid=values['valid'];cohort={k:values['cohort__'+k] for k in COHORT_DTYPES}
        frames={}
        for m in MODELS:
            _array(values['identity__'+m],'u1',(32,))
            frames[m]=dict(prediction=values['prediction__'+m],ordinal=values['ordinal__'+m],
                           cohort_sha256=values['identity__'+m].tobytes().hex())
        _validate(frames,target,valid,cohort)
        if _stamp(_safe(path).stat())!=_stamp(before):raise ValueError('private cache changed during decoding')
        return frames,target,valid,cohort
    except (OSError,ValueError,KeyError,zipfile.BadZipFile,EOFError):
        raise ValueError('invalid private typed center cache') from None
