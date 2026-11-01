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


FILES=('metrics.json','episodes.csv','histograms.csv','README.md')


def _json_bytes(value):return(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def _csv(rows):
    import csv,io
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode()


def _artifact_bytes(report):
    s=report['statistics'];g=s['global'];camera=g['camera'];episodes=[]
    for ep in s['by_episode']:
        episodes.append(dict(episode=ep['scope'],rows=ep['rows'],nominal_duration_seconds=ep['nominal_duration_seconds'],
            recorded_timestamp_span_us=ep['recorded_timestamp_span_us'],timestamp_pairs=ep['timestamp_step_us']['count'],
            stale_rows=ep['camera']['stale_rows'],index_negative_rows=ep['camera']['index_negative_rows'],
            metadata_usable_rows=ep['camera']['metadata_usable_rows'],raw_delta_min_ms=ep['recorded_delta_ms']['min'],
            raw_delta_max_ms=ep['recorded_delta_ms']['max'],metadata_usable_delta_p50_ms=ep['metadata_usable_delta_ms']['quantiles']['p50'],
            metadata_usable_delta_p95_ms=ep['metadata_usable_delta_ms']['quantiles']['p95'],int32max_delta_rows=ep['camera']['int32max_delta_rows']))
    hist=s['histograms'];ties=sum(x['count']for x in hist if x['scope']=='all'and x['measure']=='timestamp_step_us'and x['value']==0)
    backwards=sum(x['count']for x in hist if x['scope']=='all'and x['measure']=='timestamp_step_us'and x['value']<0)
    lines=['# 已发布时钟与 webcam 对齐字段：全行诊断','',
        '完整legacy行人口已核验。'if report['full_legacy_population_verified']else'仅小样本工程诊断，未核验完整legacy人口。','',
        '| 字段统计 | 实际值 |','| --- | ---: |',f"| 行数／episode数 | {s['rows']:,}／{s['episodes']} |",
        f"| 名义2kHz行数对应秒数 | {g['nominal_duration_seconds']} |",f"| 段内相邻时间戳对 | {g['timestamp_step_us']['count']:,} |",
        f'| 重复时间戳对 | {ties:,} |',f'| 倒退时间戳对 | {backwards:,} |',f"| stale为True的行 | {camera['stale_rows']:,} |",
        f"| 相机索引负值行 | {camera['index_negative_rows']:,} |",f"| 非stale且索引非负行 | {camera['metadata_usable_rows']:,} |",
        f"| 原始delta等于INT32_MAX的行 | {camera['int32max_delta_rows']:,} |",'',
        '所有原始delta（含可能填充值）的分位数：`'+json.dumps(g['recorded_delta_ms']['quantiles'],sort_keys=True)+'`。',
        '非stale且索引非负子集的delta分位数：`'+json.dumps(g['metadata_usable_delta_ms']['quantiles'],sort_keys=True)+'`。','',
        '固定官方代码说明：timestamp_us为已发布时钟字段，存在量化／重复与跨段跳变；模型使用名义2kHz行索引。本分析保留所有段内原值差分，仅描述字段，不把差分用作模型dt或原始硬件同步精度。episode边界不计为相邻对，块边界仍保留相邻对。','',
        'stale与索引分别计数，非负索引不单独证明帧可用。元数据可用子集只定义为stale=False且frame_index>=0，未检查视频像素、crop存在或原始视频FPS。INT32_MAX候选值留在全部原始直方图与分位数中，未修复、删除或重写数据。','',
        '来源是旧版已对齐字段。v3关于Incre填充值的说明不等于本legacy数据的硬件生成证据；原IMU、设备时间戳、双视频／ZED与插值同步重建仍待核验。没有训练模型或新增科学成绩，整篇论文尚未完成。','',
        '- [完整计数、分位数、直方图与来源](metrics.json)','- [逐episode汇总](episodes.csv)',
        '- [全部整数直方图及确切分母](histograms.csv)','- [产物完整性记录](completion.json)','']
    return{'metrics.json':_json_bytes(report),'episodes.csv':_csv(episodes),'histograms.csv':_csv(hist),'README.md':'\n'.join(lines).encode()}


def _completion(report,files):
    return dict(format='released_timing_artifact_completion_v1',report_sha256=report['sha256'],
        files_sha256={k:hashlib.sha256(v).hexdigest()for k,v in files.items()},input_files_sha256=report['input_files_sha256'],full_paper_complete=False)


def _location(root,upstream,output,*,exists):
    out=_safe(output,exists=exists)
    for p in (_safe(root),_safe(upstream)):
        if out==p or out in p.parents or p in out.parents:raise ValueError('timing output overlaps immutable inputs')
    return out


def _directory_state(path):
    import stat
    path=_safe(path)
    def stamp():
        v=path.lstat()
        if not stat.S_ISDIR(v.st_mode):raise ValueError('timing output must remain a nonsymlink directory')
        return(v.st_dev,v.st_ino,v.st_mode,v.st_mtime_ns,v.st_ctime_ns)
    before=stamp();names=tuple(sorted(p.name for p in path.iterdir()))
    if stamp()!=before:raise ValueError('timing directory changed during enumeration')
    return before,names


def export_release_timing(root,upstream,output,*,require_full=True):
    import os
    out=_location(root,upstream,output,exists=False)
    if out.exists():raise ValueError('fresh immutable timing output required')
    report,guard=_scan_context(root,upstream,require_full=require_full);files=_artifact_bytes(report);guard.check();out.mkdir(mode=0o700)
    for name,raw in files.items():
        with(out/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    guard.check()
    with(out/'completion.json').open('xb')as f:f.write(_json_bytes(_completion(report,files)));f.flush();os.fsync(f.fileno())
    guard.check();return report


def verify_release_timing(root,upstream,output,*,require_full=True):
    out=_location(root,upstream,output,exists=True);before=_directory_state(out);names=(*FILES,'completion.json')
    if before[1]!=tuple(sorted(names)):raise ValueError('timing artifact membership differs')
    output_guard=_StatGuard([out/n for n in names]);report,input_guard=_scan_context(root,upstream,require_full=require_full)
    files=_artifact_bytes(report);input_guard.check();expected={**files,'completion.json':_json_bytes(_completion(report,files))}
    if any((out/n).read_bytes()!=raw for n,raw in expected.items()):raise ValueError('timing artifacts differ from independent full-input reconstruction')
    output_guard.check()
    if _directory_state(out)!=before:raise ValueError('timing output generation changed during verification')
    input_guard.check();return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('export','verify'))
    for name in ('data','upstream','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true',help='explicit fixture diagnostics; never a complete-corpus permit')
    a=parser.parse_args();value=(export_release_timing if a.action=='export'else verify_release_timing)(a.data,a.upstream,a.output,require_full=not a.allow_partial)
    print(json.dumps(dict(sha256=value['sha256'],rows=value['statistics']['rows'],episodes=value['statistics']['episodes'],
        full_legacy_population_verified=value['full_legacy_population_verified'],full_paper_complete=False)))


if __name__=='__main__':main()
