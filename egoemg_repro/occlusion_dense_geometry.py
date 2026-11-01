"""Authenticated original dense geometry inputs and native request selection.

Loading and selection execute no MANO, rasterizer, learned model or CUDA.
The dense native operator execution/independent verifier are separate actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import weakref

import numpy as np

from .classic_emg import _hash
from .classic_emg_cohort import _safe,_StatGuard
from .occlusion_dense_data import COLUMNS,_typed_sha
from .occlusion_dense_footprints import summarize_dense_footprints
from .occlusion_dense_geometry_data import _core,_metadata

ROOT=Path(__file__).resolve().parents[1]
INPUT_PROOF_SHA='d89f6298787e4eaae8a12207703f9a0b18f9959f90e8f03b69bf3dbbaaebbfdb'
RECIPES=('vits_s','rn18_s')
_LIVE=weakref.WeakKeyDictionary()


def _sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


IMPORTED_SHA=_sha(__file__)


def _json(path):
    def reject(v):raise ValueError('nonfinite native dense JSON: '+v)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def _source_paths():
    names=('egoemg_repro/occlusion_dense_geometry.py','egoemg_repro/occlusion_dense_geometry_data.py',
        'egoemg_repro/occlusion_dense_data.py','egoemg_repro/occlusion_dense_footprints.py',
        'egoemg_repro/classic_emg.py','egoemg_repro/classic_emg_cohort.py','scripts/paper/run_native_point_geometry.py')
    return {n:_safe(ROOT/n)for n in names}


def _proof(evidence):
    p=_safe(evidence);g=_StatGuard([p]);raw=p.read_bytes();g.check()
    if hashlib.sha256(raw).hexdigest()!=INPUT_PROOF_SHA:raise ValueError('fixed actual complete dense extraction proof required')
    v=json.loads(raw)
    if (v['format']!='actual_complete_dense_geometry_inputs_export_verified_v1'or v['actual_exit_code']!=0
        or v['actual_complete_raw_input_admissions']!=2 or not v['all_six_original_owners_closed']
        or not v['exact_complete_identity_and_typed_exports_rebuilt']or not v['all_capsule_and_export_bytes_after_exit_exact']
        or v['native_geometry_executed']or v['learned_pose_models_constructed']!=0 or v['full_paper_complete']):
        raise ValueError('actual fully closed dense numeric extraction required')
    return p,v


@dataclass(eq=False)
class LoadedDenseGeometry:
    data:dict
    selection:dict
    identity:dict
    paths:dict
    _guard:object
    _sources:dict
    _core:object

    def check(self):
        r=_LIVE.get(self)
        if r is None:raise ValueError('registered actual dense geometry packet required')
        self._guard.check()
        if(_sha(__file__)!=IMPORTED_SHA or self.identity.get('sha256')!=r['identity']
            or _hash({k:v for k,v in self.identity.items()if k!='sha256'})!=r['identity']
            or any(_sha(p)!=r['sources'][n]for n,p in self._sources.items())):
            raise ValueError('native dense loader identity/source changed')
        self._core.check_inputs(self.data)
        if self.data['_token']!=r['data_token']:raise ValueError('actual dense native input token changed')
        if(set(self.selection)!=set(r['selection'])or any(a.flags.writeable or _typed_sha(a)!=r['selection'][n]for n,a in self.selection.items())):
            raise ValueError('native dense source order/pool map changed')


def load_dense_geometry_inputs(export,evidence,*,recipe,metadata,calibration):
    """Consume exact actual derived bytes; never trust a caller-resealed proof."""
    if recipe not in RECIPES:raise ValueError('explicit vits_s or rn18_s dense recipe required')
    ep,proof=_proof(evidence);export=_safe(export);sources=_source_paths()
    paths=dict(evidence=ep,metadata=_safe(metadata),calibration=_safe(calibration),
        export_identity=_safe(export/'identity.json'),export_completion=_safe(export/'completion.json'),
        points=_safe(export/(recipe+'-points.npz')),selection=_safe(export/(recipe+'-selection.npz')))
    guard=_StatGuard([*paths.values(),*sources.values()]);source_sha={n:_sha(p)for n,p in sources.items()}
    if source_sha['egoemg_repro/occlusion_dense_geometry.py']!=IMPORTED_SHA:raise ValueError('imported native dense loader changed')
    core=_core(sources['scripts/paper/run_native_point_geometry.py']);guard.check()
    digests={k:_sha(p)for k,p in paths.items()}
    names={'export_identity':'identity.json','export_completion':'completion.json',
        'points':recipe+'-points.npz','selection':recipe+'-selection.npz'}
    if any(digests[k]!=proof['files_sha256'][name]for k,name in names.items()):
        raise ValueError('actual accepted exported dense input bytes differ')
    original=_json(paths['export_identity']);completion=_json(paths['export_completion'])
    if(original!=proof['identity']or original['sha256']!=_hash({k:x for k,x in original.items()if k!='sha256'})
        or set(original['grids'])!=set(RECIPES)or original['native_geometry_executed']or original['zero_beta_fallback_used']):
        raise ValueError('complete authentic dense input identity differs')
    expected=dict(format='complete_dense_geometry_numeric_export_v1',identity_sha256=original['sha256'],
        files_sha256={n:h for n,h in proof['files_sha256'].items()if n!='completion.json'},full_paper_complete=False)
    expected['sha256']=_hash(expected)
    if completion!=expected:raise ValueError('actual complete dense input export completion differs')
    if(digests['metadata']!=original['raw_input_files_sha256']['metadata']
        or digests['calibration']!=original['raw_input_files_sha256']['calibration']):
        raise ValueError('actual dense metadata/calibration bytes differ')
    with np.load(paths['points'],allow_pickle=False)as f:arrays={k:f[k].copy()for k in f.files}
    with np.load(paths['selection'],allow_pickle=False)as f:selection={k:f[k].copy()for k in f.files}
    grid=original['grids'][recipe];n=grid['unique_temporal_centers'];total=grid['unique_hand_requests']
    if type(n)is not int or type(total)is not int or n<=0 or total!=2*n:raise ValueError('complete paired dense counts required')
    core._typed_schema(arrays,n)
    if set(arrays)!=set(grid['arrays_sha256'])or any(_typed_sha(a)!=grid['arrays_sha256'][k]for k,a in arrays.items()):
        raise ValueError('actual native dense point arrays differ')
    if(set(selection)!={*COLUMNS,'pool_index'}or set(selection)!=set(grid['selection_sha256'])
        or any(_typed_sha(a)!=grid['selection_sha256'][k]for k,a in selection.items())):
        raise ValueError('actual native dense request arrays differ')
    points={k:selection[k]for k in COLUMNS};summarize_dense_footprints(points,dict(windows=[]))
    pool=selection['pool_index']
    if(pool.dtype!=np.dtype(np.int64)or pool.shape!=(total,)or np.any(pool<0)or np.any(pool>=n)
        or not np.array_equal(arrays['center'][pool],points['center'])or not np.array_equal(arrays['episode'][pool],points['episode'])):
        raise ValueError('native source request keys disagree with shared geometry pool')
    with np.load(paths['metadata'],allow_pickle=False)as f:meta={k:f[k].copy()for k in f.files}
    video=_metadata(meta,original['total_rows'],original['episode_records'],canonical_beta=True)
    if(np.any(points['episode']<0)or np.any(points['episode']>=original['episode_records'])
        or np.any(points['start']<meta['episode_start_idx'][points['episode']])
        or np.any(points['end']>meta['episode_end_idx'][points['episode']])):
        raise ValueError('native source request outside its episode')
    calib=_json(paths['calibration'])
    for a in(*arrays.values(),*selection.values(),*meta.values()):a.setflags(write=False)
    data=dict(role='native_dense_'+recipe,unique_hand_samples=total,arrays=arrays,metadata=meta,calibration=calib,
        admission=original,files_sha256=digests,_paths=dict(metadata=paths['metadata'],calibration=paths['calibration']),
        _guard=guard,_video_field=video)
    data['_token']=core._token(data);core.check_inputs(data);guard.check()
    identity=dict(format='actual_authenticated_dense_native_input_binding_v1',recipe=recipe,
        complete_geometry_input_proof_sha256=INPUT_PROOF_SHA,geometry_input_identity_sha256=original['sha256'],
        expected_complete_hand_requests=total,expected_unique_temporal_centers=n,files_sha256=digests,
        source_files_sha256=source_sha,resolution_policy=core.validate_resolution_policy('author_fallback_1920x1080'),
        geometry_policy=core.geometry_policy(),native_geometry_executed=False,learned_pose_models_constructed=0,
        optimizer_updates=0,cuda_initialized=False,scientific_eligible=False,full_paper_complete=False)
    identity['sha256']=_hash(identity);v=LoadedDenseGeometry(data,selection,identity,paths,guard,sources,core)
    _LIVE[v]=dict(identity=identity['sha256'],data_token=data['_token'],sources=source_sha,
        selection={k:_typed_sha(a)for k,a in selection.items()})
    v.check();return v


def select_dense_requests(loaded,mode):
    """All source requests, or an explicitly bounded actual nonzero probe."""
    loaded.check()
    if mode not in('complete_dense','bilateral_probe'):raise ValueError('explicit complete_dense or bilateral_probe mode required')
    s=loaded.selection;indices=list(range(len(s['sequence'])))
    if mode=='bilateral_probe':
        indices=[]
        for hi,hand in enumerate(('left','right')):
            candidates=[j for j in range(len(s['sequence']))if s['hand_index'][j]==hi
                and np.any(loaded.data['arrays'][hand+'_pose'][s['pool_index'][j]]!=0)]
            if not candidates:raise ValueError('no actual finite nonzero bilateral probe key for '+hand)
            indices.append(candidates[0])
        indices.sort()
    rows=[]
    for ordinal,j in enumerate(indices):
        rows.append(dict(sequence=ordinal,native_sequence=int(s['sequence'][j]),hand=('left','right')[int(s['hand_index'][j])],
            pool_index=int(s['pool_index'][j]),native_index=int(s['native_index'][j]),ep_idx=int(s['episode'][j]),
            center_idx=int(s['center'][j]),split_id=int(s['split_id'][j]),group_index=int(s['group_index'][j]),
            window_start=int(s['start'][j]),window_end=int(s['end'][j])))
    loaded.check();return rows


def _execution_identity(loaded,mode,runtime):
    requests=select_dense_requests(loaded,mode)
    inputs={k:str(loaded.paths[k])for k in('evidence','metadata','calibration')}
    inputs['export']=str(loaded.paths['points'].parent)
    v=dict(format='original_native_dense_geometry_execution_v1',recipe=loaded.identity['recipe'],mode=mode,
        input_binding=loaded.identity,input_paths=inputs,requested_count=len(requests),request_keys_sha256=_hash(requests),
        runtime=runtime,resolution_policy=loaded._core.validate_resolution_policy('author_fallback_1920x1080'),
        geometry_policy=loaded._core.geometry_policy(),accepted_native_math_source_sha256=loaded._core.IMPORTED_SHA,
        scientific_eligible=False,full_paper_complete=False)
    v['sha256']=_hash(v);return v


def _result(identity,journal_sha,native_rows,invalid,seconds):
    if type(seconds)is not float or not math.isfinite(seconds)or seconds<0:raise ValueError('finite native execution duration required')
    total=native_rows+invalid
    v=dict(format='original_native_dense_geometry_result_v1',identity_sha256=identity['sha256'],
        journal_file_sha256=journal_sha,recipe=identity['recipe'],mode=identity['mode'],rows=total,
        actual_mesh_calls=total,actual_occlusion_calls=native_rows,invalid_zero_pose_rows=invalid,
        complete_dense_geometry=identity['mode']=='complete_dense'and total==identity['input_binding']['expected_complete_hand_requests'],
        complete_standard_role=False,actual_right_mano_geometry_layers_constructed=identity['runtime']['actual_right_mano_geometry_layers_constructed'],
        learned_pose_models_constructed=0,optimizer_updates=0,cuda_initialized=False,
        resolution_policy=identity['resolution_policy'],all_input_and_source_generations_unchanged=True,seconds=seconds,
        dense_grid_or_paired_gain_complete=False,figure13_reproduced=False,scientific_eligible=False,full_paper_complete=False)
    v['sha256']=_hash(v);return v


def run_dense_geometry(loaded,output,*,mode,upstream,mano_assets,chumpy_dir):
    """Execute the unchanged accepted original runtime/frame on every request.

    A successful result records computation; the caller must also execute the
    separate fresh-process raw verifier and independent native replay.
    """
    loaded.check();requests=select_dense_requests(loaded,mode);output=_safe(output,exists=False)
    if output.exists():raise ValueError('new native dense geometry output required')
    output.mkdir(mode=0o700);(output/'points').mkdir(mode=0o700)
    videos=output/'unavailable-videos';videos.mkdir(mode=0o700);began=time.monotonic();core=loaded._core
    native,runtime_guard,runtime=core._native_runtime(SimpleNamespace(upstream=upstream,mano_assets=mano_assets,chumpy_dir=chumpy_dir))
    loaded.check();runtime_guard.check();identity=_execution_identity(loaded,mode,runtime);core._write_json(output/'identity.json',identity)
    valid=invalid=0
    with(output/'points.jsonl').open('x')as journal:
        for seq,request in enumerate(requests):
            loaded.check();runtime_guard.check()
            row,arrays=core.geometry_frame(native,loaded.data,request['hand'],request['pool_index'],video_root=videos,device='cpu')
            if arrays is None:
                if row['status']!='invalid_zero_pose':raise ValueError('native missing packet is not an admitted zero pose')
                invalid+=1
            else:
                if row['status']!='native_geometry':raise ValueError('native packet status differs')
                core._packet_schema(arrays);valid+=1
                p=output/'points'/f'{seq:06d}-{request["hand"]}-{row["center_idx"]}.npz'
                with p.open('xb')as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
                p.chmod(0o600);row.update(packet_file=str(p.relative_to(output)),packet_file_sha256=_sha(p))
            row.update(sequence=seq,identity_sha256=identity['sha256'])
            journal.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
            core._write_json(output/'progress.json',dict(completed_points=seq+1,expected_points=len(requests),mode=mode,recipe=identity['recipe']))
    loaded.check();runtime_guard.check()
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():raise ValueError('native dense geometry initialized CUDA')
    result=_result(identity,_sha(output/'points.jsonl'),valid,invalid,float(time.monotonic()-began))
    core._write_json(output/'result.json',result);return result


def _accepted_row_verifier(core):
    """Compile the exact closed verifier loop; no pixel/area math is rewritten.

    The loader pins the entire accepted source before this read. Only the
    input/scope adapters differ from the standard-role verifier. Its original
    strict row schema, calibrated projection, area weights and visibility
    checks execute verbatim, using the unchanged original kernel functions.
    """
    path=Path(core.__file__);tree=ast.parse(path.read_text());original=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='verify_geometry')
    assigns=[n for n in original.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in('native_rows','invalid','native_fields')for t in n.targets)]
    loops=[n for n in original.body if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)
        and isinstance(n.target.elts[0],ast.Name)and n.target.elts[0].id=='seq']
    if len(assigns)!=3 or len(loops)!=1:raise ValueError('accepted native row verifier source layout differs')
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n)for n in('data','selected','rows','output','identity','kernel','guard')],
        vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[])
    body=[*assigns,loops[0],ast.Return(value=ast.Tuple(elts=[ast.Name(id='native_rows',ctx=ast.Load()),ast.Name(id='invalid',ctx=ast.Load())],ctx=ast.Load()))]
    fn=ast.FunctionDef(name='_accepted_native_rows',args=args,body=body,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));namespace=dict(vars(core),np=np,_safe=_safe)
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['_accepted_native_rows']


def verify_dense_geometry(output):
    """Fresh raw verification without constructing MANO or a learned model."""
    output=_safe(output);ip=_safe(output/'identity.json');rp=_safe(output/'result.json');jp=_safe(output/'points.jsonl')
    initial_guard=_StatGuard([ip,rp,jp]);identity=_json(ip);result=_json(rp)
    if identity['format']!='original_native_dense_geometry_execution_v1':raise ValueError('explicit native dense geometry identity required')
    paths=identity['input_paths']
    loaded=load_dense_geometry_inputs(paths['export'],paths['evidence'],recipe=identity['recipe'],metadata=paths['metadata'],calibration=paths['calibration'])
    for p in(ip,rp,jp):loaded._guard.add(p)
    core=loaded._core;source=core._verify_runtime(identity['runtime'],loaded._guard);loaded._guard.add(source)
    expected=_execution_identity(loaded,identity['mode'],identity['runtime']);core._require_equal(identity,expected,'dense execution identity')
    requests=select_dense_requests(loaded,identity['mode']);rows=[json.loads(s)for s in jp.read_text().splitlines()]
    if len(rows)!=len(requests):raise ValueError('complete original dense request population differs')
    selected=[(r['hand'],r['pool_index'])for r in requests]
    kernel=core._load_source('_accepted_original_dense_occlusion_kernel',source);loaded._guard.check()
    native_rows,invalid=_accepted_row_verifier(core)(loaded.data,selected,rows,output,identity,kernel,loaded._guard)
    wanted=_result(identity,_sha(jp),native_rows,invalid,result['seconds']);core._require_equal(result,wanted,'dense native result')
    loaded.check();initial_guard.check();return result


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in('validate','run'):
        q=sub.add_parser(action)
        for name in('export','evidence','metadata','calibration'):q.add_argument('--'+name,required=True,type=Path)
        q.add_argument('--recipe',required=True,choices=RECIPES);q.add_argument('--mode',choices=('complete_dense','bilateral_probe'),default='complete_dense')
        if action=='run':
            for name in('upstream','mano-assets','chumpy-dir','output'):q.add_argument('--'+name,required=True,type=Path)
    q=sub.add_parser('verify');q.add_argument('--output',required=True,type=Path);a=vars(p.parse_args());action=a.pop('action')
    if action=='verify':result=verify_dense_geometry(**a)
    else:
        mode=a.pop('mode');run={k:a.pop(k)for k in('upstream','mano_assets','chumpy_dir','output')}if action=='run'else None
        loaded=load_dense_geometry_inputs(**a)
        if action=='validate':
            requests=select_dense_requests(loaded,mode)
            print(json.dumps(dict(input_binding_sha256=loaded.identity['sha256'],requests=len(requests),mode=mode,
                native_geometry_executed=False,learned_pose_models_constructed=0,cuda_initialized=False,full_paper_complete=False)));return
        result=run_dense_geometry(loaded,mode=mode,**run)
    print(json.dumps({k:result[k]for k in('sha256','recipe','mode','rows','actual_mesh_calls','actual_occlusion_calls','invalid_zero_pose_rows','complete_dense_geometry','full_paper_complete')}))


if __name__=='__main__':main()
