"""Descriptive released clock/webcam fields, preserving every aligned row.

Recorded timestamp differences describe quantization and jumps. The official
model time axis remains the nominal2kHz row index; this is not raw hardware
synchronization accuracy, a timestamp repair or a new model experiment.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .classic_emg_cohort import _safe,_json,_StatGuard

REVISION='f7cbd048b7092139852270491d3859cd472e0311'
FORMAT='legacy_released_clock_and_webcam_full_row_diagnostics_v1'
FIELDS={'timestamp_us':'int64','episode_index':'int64','frame_index':'int64',
 'image_webcam_frame_index':'int32','image_webcam_delta_ms':'int32','image_webcam_stale':'bool'}
SOURCE_PINS={
 'scripts/data/migrate_unified_memmap_v3.py':'716f9de6ef2c597bb9c5929da34b2305cc17b7cd4dfa0c3e87d8908af74a5dbf',
 'scripts/data/merge_datasets_to_unified_memmap.py':'868b53023142caa81c79c6359e7677c761a40f5f45111d9174f49e63c755a75c'}
MEASURES=('timestamp_step_us','emg_frame_step','camera_frame_step','recorded_delta_ms','metadata_usable_delta_ms','stale_delta_ms')
CAMERA_KEYS=('stale_rows','nonstale_rows','index_negative_rows','index_zero_rows','metadata_usable_rows',
 'int32max_delta_rows','stale_and_index_negative_rows','nonstale_and_index_negative_rows')


def _sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

IMPORTED_SHA=_sha(__file__)


def _add(counter,values):
    keys,counts=np.unique(values,return_counts=True);counter.update({int(k):int(n)for k,n in zip(keys,counts)})


def _summary(counter):
    items=sorted(counter.items());count=sum(counter.values());total=sum(k*n for k,n in items)
    def percentile(p):
        if not count:return None
        rank,remainder=divmod((count-1)*p,100);positions=(rank,rank+(remainder>0));found=[];offset=0
        for value,n in items:
            for position in positions[len(found):]:
                if offset<=position<offset+n:found.append(value)
                else:break
            if len(found)==2:break
            offset+=n
        return found[0]if not remainder else(found[0]*(100-remainder)+found[1]*remainder)/100
    return dict(count=count,min=items[0][0]if count else None,max=items[-1][0]if count else None,
        sum=total,mean=total/count if count else None,quantiles={'p'+str(p):percentile(p)for p in (0,50,95,99,100)})


def _steps(values,previous):
    values=values.astype('int64',copy=False)
    if previous is not None:values=np.concatenate((np.array([previous],dtype='int64'),values))
    if int(values.max())-int(values.min())>np.iinfo('int64').max:raise ValueError('recorded integer differences could overflow')
    return np.diff(values)


def timing_statistics(arrays,boundaries,chunk_rows=1048576,*,_check=None):
    """Exact descriptive counts/histograms; this helper grants no full permit."""
    if type(chunk_rows)is not int or chunk_rows<1 or type(arrays)is not dict or set(arrays)!=set(FIELDS):
        raise ValueError('exact six arrays and positive integer chunk required')
    n=len(arrays['timestamp_us'])
    for name,dtype in FIELDS.items():
        a=arrays[name];allowed=(np.dtype('bool'),np.dtype('uint8'))if name=='image_webcam_stale'else(np.dtype(dtype),)
        if not isinstance(a,np.ndarray)or a.shape!=(n,)or a.dtype not in allowed:raise ValueError('exact timing array dtype/shape required')
    end=0
    for b in boundaries:
        if(not isinstance(b,(list,tuple))or len(b)!=2 or any(type(v)is not int for v in b)
            or b[0]!=end or not b[0]<b[1]<=n):raise ValueError('contiguous complete half-open episode coverage required')
        end=b[1]
    if not boundaries or end!=n:raise ValueError('all timing rows must belong to metadata episodes')
    global_hist={k:Counter()for k in MEASURES};global_camera=Counter({k:0 for k in CAMERA_KEYS});episodes=[];histograms=[];span=0
    def finish(scope,count,counters,camera,elapsed):
        histograms.extend(dict(scope=scope,measure=k,value=int(v),count=int(c))for k in MEASURES for v,c in sorted(counters[k].items()))
        return dict(scope=scope,rows=count,nominal_duration_seconds=count/2000,
            recorded_timestamp_span_us=elapsed,camera=dict(camera),**{k:_summary(counters[k])for k in MEASURES})
    for episode,(start,end)in enumerate(boundaries):
        counters={k:Counter()for k in MEASURES};camera=Counter({k:0 for k in CAMERA_KEYS});previous=None
        for pos in range(start,end,chunk_rows):
            if _check is not None:_check()
            a={k:np.asarray(v[pos:min(end,pos+chunk_rows)])for k,v in arrays.items()}
            stale=a['image_webcam_stale'].view('uint8')
            if np.any(stale>1)or np.any(a['episode_index']!=episode):raise ValueError('raw stale bytes or episode identity differs')
            stale=stale.astype('bool',copy=False);index=a['image_webcam_frame_index'];delta=a['image_webcam_delta_ms']
            usable=(~stale)&(index>=0)
            conditions=(stale,~stale,index<0,index==0,usable,delta==np.iinfo('int32').max,stale&(index<0),(~stale)&(index<0))
            camera.update({k:int(np.count_nonzero(v))for k,v in zip(CAMERA_KEYS,conditions)})
            _add(counters['recorded_delta_ms'],delta);_add(counters['metadata_usable_delta_ms'],delta[usable]);_add(counters['stale_delta_ms'],delta[stale])
            for key,field in [('timestamp_step_us','timestamp_us'),('emg_frame_step','frame_index')]:
                _add(counters[key],_steps(a[field],None if previous is None else previous[field]))
            image=index.astype('int64');image=np.concatenate(([previous['image_webcam_frame_index']],image))if previous is not None else image
            _add(counters['camera_frame_step'],np.diff(image)[(image[:-1]>=0)&(image[1:]>=0)])
            previous={k:int(a[k][-1])for k in ('timestamp_us','frame_index','image_webcam_frame_index')}
        elapsed=int(arrays['timestamp_us'][end-1])-int(arrays['timestamp_us'][start]);span+=elapsed
        episodes.append(finish('episode_'+str(episode).zfill(3),end-start,counters,camera,elapsed))
        for k in MEASURES:global_hist[k].update(counters[k])
        global_camera.update(camera)
    whole=finish('all',n,global_hist,global_camera,span)
    if(whole['timestamp_step_us']['count']!=n-len(boundaries)or whole['emg_frame_step']['count']!=n-len(boundaries)
        or whole['recorded_delta_ms']['count']!=n or whole['metadata_usable_delta_ms']['count']!=whole['camera']['metadata_usable_rows']
        or whole['stale_delta_ms']['count']!=whole['camera']['stale_rows']):raise ValueError('timing denominators do not conserve')
    if _check is not None:_check()
    return dict(rows=n,episodes=len(boundaries),by_episode=episodes,histograms=histograms)|{'global':whole}


def _scan_context(root,upstream,*,require_full=True,chunk_rows=1048576):
    root=_safe(root);upstream=_safe(upstream);code=Path(__file__).resolve().parent
    inputs=[root/'manifest.json',root/'metadata.npz',*[root/(k+'.dat')for k in FIELDS]]
    code_paths=[Path(__file__),code/'classic_emg_cohort.py'];official=[upstream/n for n in SOURCE_PINS]
    guard=_StatGuard([*inputs,*code_paths,*official]);source_hashes={p.name:_sha(p)for p in code_paths}
    if source_hashes[Path(__file__).name]!=IMPORTED_SHA:raise ValueError('imported timing analysis source changed')
    official_hashes={n:_sha(upstream/n)for n in SOURCE_PINS}
    if official_hashes!=SOURCE_PINS:raise ValueError('fixed official timing semantics source differs')
    manifest=_json(inputs[0]);n=manifest['total_rows'];fields=manifest['fields']
    if manifest['format_version']!='egoemg_v2_memmap'or type(n)is not int or n<1:raise ValueError('original legacy timing manifest required')
    arrays={}
    try:
        for name,dtype in FIELDS.items():
            spec=fields[name];path=root/(name+'.dat')
            if(spec['filename']!=path.name or spec['dtype']!=dtype or spec['shape']!=[n]
                or path.stat().st_size!=n*np.dtype(dtype).itemsize):raise ValueError('original timing payload layout differs')
            arrays[name]=np.memmap(path,mode='r',dtype='uint8'if dtype=='bool'else dtype,shape=(n,))
        with np.load(inputs[1],allow_pickle=False)as meta:
            starts,ends,lengths=[meta[k].copy()for k in ('episode_start_idx','episode_end_idx','episode_length')]
        if(any(x.ndim!=1 or x.dtype!=np.dtype('int64')for x in (starts,ends,lengths))
            or starts.shape!=ends.shape or ends.shape!=lengths.shape or np.any(ends-starts!=lengths)):
            raise ValueError('complete original episode bounds required')
        boundaries=[(int(a),int(b))for a,b in zip(starts,ends)]
        if type(manifest['num_episodes'])is not int or manifest['num_episodes']!=len(boundaries):
            raise ValueError('manifest episode count differs from complete numeric metadata')
        full=n==66161725 and len(boundaries)==41
        if require_full and not full:raise ValueError('full timing scope requires actual66161725rows/41episodes')
        input_hashes={p.name:_sha(p)for p in inputs};guard.check()
        statistics=timing_statistics(arrays,boundaries,chunk_rows=chunk_rows,_check=guard.check)
        result=dict(format=FORMAT,analysis_complete=True,full_legacy_population_verified=full,full_paper_complete=False,
            hardware_synchronization_accuracy_established=False,timestamps_used_as_model_dt=False,
            original_imu_samples_verified=False,video_pixel_content_verified=False,scientific_settings_added=0,
            official_revision=REVISION,official_semantic_sources_sha256=official_hashes,source_files_sha256=source_hashes,
            input_files_sha256=input_hashes,input_payload_bytes=sum(p.stat().st_size for p in inputs[2:]),statistics=statistics,
            interpretation='all recorded differences/deltas preserved; nominal2kHz row-index time; no timestamp repair, hardware error/FPS/IMU-rate inference or crop/pixel availability claim; INT32_MAX candidate retained even in raw quantiles',
            model_runs=0)
        result['sha256']=_hash(result);guard.check();return result,guard
    finally:
        for value in arrays.values():value._mmap.close()


def scan_release_timing(root,upstream,*,require_full=True,chunk_rows=1048576):
    result,guard=_scan_context(root,upstream,require_full=require_full,chunk_rows=chunk_rows);guard.check();return result
