"""Source-bound population export from completed original standard geometry.

The immutable published execution receipt anchors actual native/replay evidence.
Every corresponding current private byte chain is reopened. No mesh operator,
learned pose model, GPU, dense-grid prediction or Figure13 calculation runs here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE_SHA='4020a7a37a621ddf429fed01fa20c674ecb1aa6c94e4de62ec0a01a05a83f0df'
RUNNER_SHA='c2507b6d6308f3c9609a4931391a7863e7c4f3811a206b1bd485ead5a15f1634'
EXPECTED_COUNTS=dict(val=1576,test=4306)
ARRAYS=dict(verts_cam=(np.float64,(778,3)),u_proj=(np.int32,(778,)),v_proj=(np.int32,(778,)),
    visible=(np.bool_,(778,)),area_weights=(np.float64,(778,)))


def _file_sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


IMPORTED_SHA=_file_sha(__file__)


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _typed_sha(a):
    h=hashlib.sha256(json.dumps([a.dtype.str,list(a.shape)],separators=(',',':')).encode())
    h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()


def _json(path):
    def reject(value):raise ValueError('nonfinite geometry report JSON: '+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def _source_check():
    if _file_sha(__file__)!=IMPORTED_SHA:raise ValueError('imported geometry report source changed')


def summarize_rows(rows,read_packet):
    """Summarize validated unique hand requests, retaining invalid denominators.

    The caller supplies already byte-bound small packet arrays. This pure
    function gives no provenance/full-role/native execution permit by itself.
    """
    if type(rows)is not list:raise ValueError('ordered geometry rows required')
    p={k:0 for k in('requests','native_geometry_rows','invalid_zero_pose_rows','mocap_tracked_rows','image_stale_rows',
        'markers_valid_rows','scored_mocap_tracked_rows','scored_image_stale_rows','scored_markers_valid_rows',
        'vertices_total','vertices_positive_depth','vertices_behind_or_near_camera','vertices_front_in_image',
        'vertices_front_offscreen','vertices_visible','native_rows_behind_camera_only',
        'native_rows_with_offscreen_vertices','native_rows_all_front_vertices_offscreen',
        'native_rows_with_visible_vertices','native_rows_zero_area_weight')}
    seen=set();previous=-1;scores=[]
    for row in rows:
        if (type(row.get('sequence'))is not int or row['sequence']<=previous
            or type(row.get('point_index'))is not int or row['point_index']<0
            or row.get('hand')not in('left','right')):raise ValueError('unique ordered geometry point keys required')
        previous=row['sequence'];key=(row['hand'],row['point_index'])
        if key in seen:raise ValueError('duplicate geometry request')
        seen.add(key);p['requests']+=1
        for name in('mocap_tracked','image_stale','markers_valid'):
            if type(row.get(name))is not bool:raise ValueError('strict geometry population flags required')
            p[name+'_rows']+=int(row[name])
        if row['status']=='invalid_zero_pose':
            if row['occlusion_score']is not None or row['visible_ratio']is not None or'packet_file'in row:
                raise ValueError('zero pose cannot acquire a score or native packet')
            p['invalid_zero_pose_rows']+=1;continue
        if row['status']!='native_geometry':raise ValueError('unknown geometry status')
        a=read_packet(row)
        if set(a)!=set(ARRAYS):raise ValueError('exact small geometry report array set required')
        for name,(dtype,shape)in ARRAYS.items():
            if a[name].dtype!=np.dtype(dtype)or a[name].shape!=shape:
                raise ValueError('native report array dtype/shape differs: '+name)
            if a[name].dtype.kind=='f'and not np.isfinite(a[name]).all():raise ValueError('nonfinite geometry report array')
        if np.any(a['area_weights']<0):raise ValueError('negative native area weights')
        front=a['verts_cam'][:,2]>1e-8
        inside=front&(a['u_proj']>=0)&(a['u_proj']<1920)&(a['v_proj']>=0)&(a['v_proj']<1080)
        outside=front&~inside;visible=a['visible'];weights=a['area_weights']
        if np.any(visible&~inside):raise ValueError('visible native vertex is behind camera or offscreen')
        total=float(weights.sum());ratio=0. if total<1e-12 else float(weights[visible].sum())/total
        expected=dict(positive_depth_vertices=int(front.sum()),visible_vertices=int(visible.sum()),area_weight_sum=total,
            visible_ratio=ratio,occlusion_score=1.-ratio,width=1920,height=1080)
        for name,value in expected.items():
            if type(row.get(name))is not type(value)or row[name]!=value:
                raise ValueError('native report row statistic differs: '+name)
        score=row['occlusion_score']
        if not math.isfinite(score)or not 0.<=score<=1.:raise ValueError('bounded native occlusion score required')
        scores.append(score);p['native_geometry_rows']+=1
        for flag in('mocap_tracked','image_stale','markers_valid'):p['scored_'+flag+'_rows']+=int(row[flag])
        p['vertices_total']+=778;p['vertices_positive_depth']+=int(front.sum())
        p['vertices_behind_or_near_camera']+=int((~front).sum());p['vertices_front_in_image']+=int(inside.sum())
        p['vertices_front_offscreen']+=int(outside.sum());p['vertices_visible']+=int(visible.sum())
        p['native_rows_behind_camera_only']+=int(not front.any())
        p['native_rows_with_offscreen_vertices']+=int(outside.any())
        p['native_rows_all_front_vertices_offscreen']+=int(front.any()and not inside.any())
        p['native_rows_with_visible_vertices']+=int(visible.any());p['native_rows_zero_area_weight']+=int(total<1e-12)
    values=np.asarray(scores,np.float64);n=len(values)
    p['native_occlusion']=dict(count=n,mean=float(values.mean())if n else None,
        sample_sd=float(values.std(ddof=1))if n>1 else None,minimum=float(values.min())if n else None,
        median=float(np.median(values))if n else None,maximum=float(values.max())if n else None)
    edges=np.linspace(0.,1.,21);counts,_=np.histogram(values,bins=edges)
    p['occlusion_histogram']=dict(edges=edges.tolist(),counts=counts.tolist(),
        policy='descriptive fixed0.05 bins, inclusive final edge; not Figure13 adaptive model-error bins')
    return p


def _evidence(path):
    from .classic_emg_cohort import _safe
    try:
        path=_safe(path)
        if _file_sha(path)!=EVIDENCE_SHA:raise ValueError('fixed actual complete-role evidence bytes required')
        v=_json(path)
    except OSError as exc:raise ValueError('fixed actual complete-role evidence unavailable')from exc
    if not v['both_standard_role_computations_complete']or not v['all_twenty_original_owners_closed']:
        raise ValueError('actual complete standard native roles required')
    return path,v


def _live_original(i):
    import os
    p=Path('/proc',str(i['pid']))
    try:z=(p/'stat').read_text().rsplit(')',1)[1].split();return z[19]==str(i['start_ticks'])and z[0]!='Z'and p.stat().st_uid==os.getuid()
    except(FileNotFoundError,ProcessLookupError):return False


def _load_role(root,role,evidence,guard):
    from .classic_emg_cohort import _safe
    from .emg2pose_memory_process import same_process
    root=_safe(root);expected=evidence['roles'][role]
    names=('execution-state.json','stage-receipt.json','workflow-result.json')
    for name in names:guard.add(root/'private'/name)
    if _file_sha(root/'private/workflow-result.json')!=expected['raw_workflow_file_sha256']:
        raise ValueError('actual native workflow anchor differs')
    state,stage,workflow=(_json(root/'private'/name)for name in names)
    if (state['status']!='complete'or state['actual_exit_code']!=0
        or any(_live_original(state[k])for k in('wrapper','child'))):raise ValueError('native role original owners not closed')
    if (_hash({k:v for k,v in stage.items()if k!='bundle_sha256'})!=expected['bundle_sha256']
        or stage['bundle_sha256']!=expected['bundle_sha256']or workflow['role']!=role
        or workflow['source_commit']!=expected['source_commit']):raise ValueError('native role capsule/source binding differs')
    if len(workflow['actual_owned_stages'])!=4:raise ValueError('two native/two verifier stages required')
    for owned in workflow['actual_owned_stages']:
        if (owned['status']!='success'or owned['worker_exit_code']!=0 or not owned['all_owned_consumers_exited']
            or any(same_process(owned[k])for k in('worker','guardian'))):raise ValueError('actual native stage not closed')
    for name,digest in stage['files_sha256'].items():
        p=_safe(root/name);guard.add(p)
        if _file_sha(p)!=digest:raise ValueError('native capsule source changed')
    runner=root/'scripts/paper/run_native_point_geometry.py'
    if _file_sha(runner)!=RUNNER_SHA:raise ValueError('known accepted native runner required')
    outputs=[]
    for seq,label in enumerate(('native-a','native-b')):
        out=root/'private'/label
        for name in('identity.json','result.json','points.jsonl'):guard.add(out/name)
        identity=_json(out/'identity.json');result=_json(out/'result.json')
        binding=workflow['outputs'][seq]
        if (identity['sha256']!=_hash({k:v for k,v in identity.items()if k!='sha256'})
            or identity['sha256']!=binding['identity_sha256']or result!=binding['result']
            or _file_sha(out/'result.json')!=binding['result_file_sha256']
            or _file_sha(out/'points.jsonl')!=expected['native_output_journal_file_sha256'][seq]
            or identity['role']!=role or identity['mode']!='complete_role'
            or identity['resolution_policy']!=evidence['resolution_policy']or identity['geometry_policy']!=evidence['geometry_policy']):
            raise ValueError('anchored native identity/result/journal differs')
        for key,p in identity['input_paths'].items():
            p=_safe(p);guard.add(p)
            if _file_sha(p)!=identity['input_sha256'][key]:raise ValueError('native report input bytes changed')
        for p,digest in identity['runtime']['source_files_sha256'].items():
            p=_safe(p);guard.add(p)
            if _file_sha(p)!=digest:raise ValueError('native runtime/asset source changed')
        rows=[json.loads(s)for s in(out/'points.jsonl').read_text().splitlines()]
        if len(rows)!=EXPECTED_COUNTS[role]or result['rows']!=len(rows)or not result['complete_standard_role']:
            raise ValueError('complete native standard role population differs')
        for row in rows:
            if row['status']=='native_geometry':
                p=_safe(out/row['packet_file']);guard.add(p)
                if p.parent!=out/'points'or _file_sha(p)!=row['packet_file_sha256']:
                    raise ValueError('actual raw native packet changed')
        outputs.append((out,rows))
    if outputs[0][1]!=outputs[1][1]:raise ValueError('full native replay rows differ')
    out,rows=outputs[0]
    def read_packet(row):
        p=_safe(out/row['packet_file'])
        with np.load(p,allow_pickle=False)as f:a={k:f[k]for k in ARRAYS}
        if any(_typed_sha(v)!=row['arrays_sha256'][k]for k,v in a.items()):raise ValueError('small native array content changed')
        return a
    populations={name:summarize_rows(rows if name=='all'else[r for r in rows if r['hand']==name],read_packet)
        for name in('all','left','right')}
    guard.check();return dict(role=role,workflow_file_sha256=expected['raw_workflow_file_sha256'],
        native_journal_sha256=expected['native_output_journal_file_sha256'],populations=populations)


def collect_geometry_report(val_root,test_root,*,evidence):
    """Reopen the pinned genuine execution chain and current native byte payloads."""
    from .classic_emg_cohort import _StatGuard
    _source_check();path,anchor=_evidence(evidence)
    sources={name:ROOT/name for name in('egoemg_repro/geometry_report.py','egoemg_repro/classic_emg_cohort.py',
        'egoemg_repro/emg2pose_memory_process.py')}
    guard=_StatGuard([path,*sources.values()]);source_hashes={k:_file_sha(p)for k,p in sources.items()}
    roles={role:_load_role(root,role,anchor,guard)for role,root in(('val',val_root),('test',test_root))}
    if any(_file_sha(p)!=source_hashes[k]for k,p in sources.items()):raise ValueError('report source changed during export')
    _source_check();guard.check()
    result=dict(format='original_standard_native_geometry_population_report_v1',roles=roles,
        complete_native_execution_evidence_sha256=EVIDENCE_SHA,source_files_sha256=source_hashes,
        resolution_policy=anchor['resolution_policy'],geometry_policy=anchor['geometry_policy'],
        statistical_policy='one unique hand request once; zero pose retained outside score denominator; original score includes behind/offscreen cases; sample SD ddof1',
        learned_pose_models_constructed=0,optimizer_updates=0,cuda_initialized=False,
        dense_grid_or_paired_gain_complete=False,figure13_reproduced=False,scientific_settings_added=0,
        scientific_eligible=False,full_paper_complete=False)
    result['sha256']=_hash(result);return result


def render_csv(report):
    stream=io.StringIO(newline='');fields=['role','population','requests','native_geometry_rows','invalid_zero_pose_rows',
        'mocap_tracked_rows','image_stale_rows','markers_valid_rows','vertices_total','vertices_positive_depth',
        'vertices_behind_or_near_camera','vertices_front_in_image','vertices_front_offscreen','vertices_visible',
        'native_rows_behind_camera_only','native_rows_with_offscreen_vertices','native_rows_all_front_vertices_offscreen',
        'native_rows_with_visible_vertices','native_rows_zero_area_weight','occlusion_mean','occlusion_sample_sd']
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for role,r in report['roles'].items():
        for name,p in r['populations'].items():
            values={k:p[k]for k in fields if k in p};values.update(role=role,population=name,
                occlusion_mean=p['native_occlusion']['mean'],occlusion_sample_sd=p['native_occlusion']['sample_sd'])
            writer.writerow(values)
    return stream.getvalue()


def render_markdown(report):
    lines=['# 原生标准网格几何人口统计','',
        '从已完成并独立重放的真实原生包重新核对全部来源、输入与包字节后导出。每个唯一手请求只计一次，零姿态保留且不赋造遮挡分数。','',
        '| 角色 | 手 | 请求 | 有效几何 | 零姿态 | 平均原生遮挡分数 | 样本SD | 全部顶点在相机后方/近相机 | 有前方顶点但全部越界 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for role,r in report['roles'].items():
        for name,p in r['populations'].items():
            d=p['native_occlusion'];mean='—'if d['mean']is None else f"{d['mean']:.6f}"
            sd='—'if d['sample_sd']is None else f"{d['sample_sd']:.6f}"
            lines.append(f"| {role} | {name} | {p['requests']} | {p['native_geometry_rows']} | {p['invalid_zero_pose_rows']} | {mean} | {sd} | {p['native_rows_behind_camera_only']} | {p['native_rows_all_front_vertices_offscreen']} |")
    lines+=['','统计条件为明确的官方1920×1080回退，实际episode视频尺寸未观测，原畸变未应用；左右镜像约定差异仍未证明物理一致。',
        '相机后方、图像越界和没有可见顶点分别计数。原始area-weighted遮挡分数保留这些情况，不能把全部高分直接解释为手部自遮挡。',
        '本统计不含模型误差、密1/4stride、paired gain、SEM趋势或Spearman，不是Figure13复现。直方图仅为描述性固定0.05箱，未冒充原绘图的自适应模型误差分箱。',
        '完整数据、原生数组及授权MANO模型保留在私有运行目录；公开JSON/CSV只有聚合统计及来源摘要。完整模型设置数未增加，全文尚未完成。','']
    return'\n'.join(lines)


def build_geometry_report(val_root,test_root,output,*,evidence):
    from .classic_emg_cohort import _safe
    _evidence(evidence);output=_safe(output,exists=False)
    if output.exists():raise ValueError('new geometry report output required')
    report=collect_geometry_report(val_root,test_root,evidence=evidence);output.mkdir(mode=0o700)
    files={'populations.json':json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n',
        'populations.csv':render_csv(report),'README.md':render_markdown(report)}
    for name,text in files.items():(output/name).write_text(text)
    completion=dict(report_sha256=report['sha256'],files_sha256={n:_file_sha(output/n)for n in files},full_paper_complete=False)
    (output/'completion.json').write_text(json.dumps(completion,sort_keys=True,indent=2)+'\n');return report


def verify_geometry_report(val_root,test_root,output,*,evidence):
    from .classic_emg_cohort import _safe
    output=_safe(output);report=collect_geometry_report(val_root,test_root,evidence=evidence)
    expected={'populations.json':json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n',
        'populations.csv':render_csv(report),'README.md':render_markdown(report)}
    if any((output/n).read_text()!=text for n,text in expected.items()):raise ValueError('public geometry population report differs from current raw chain')
    if _json(output/'completion.json')!=dict(report_sha256=report['sha256'],files_sha256={n:_file_sha(output/n)for n in expected},full_paper_complete=False):
        raise ValueError('geometry report completion differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('build','verify'))
    for name in('val-root','test-root','output','evidence'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();fn=build_geometry_report if a.action=='build'else verify_geometry_report
    report=fn(a.val_root,a.test_root,a.output,evidence=a.evidence)
    print(json.dumps(dict(report_sha256=report['sha256'],scientific_eligible=False,full_paper_complete=False)))


if __name__=='__main__':main()
