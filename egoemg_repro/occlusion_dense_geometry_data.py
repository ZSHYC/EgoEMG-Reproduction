"""Actual GT numeric inputs for the two authenticated native dense grids.

No model, mesh, rasterizer or CUDA executes here. Complete accepted raw bytes
are reopened before extraction and rehashed afterward. Selected readonly
arrays retain zero poses and every tracking/marker/image flag.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import weakref

import numpy as np

from .classic_emg import _hash
from .classic_emg_cohort import _safe,_StatGuard
from .occlusion_dense_data import COLUMNS,_typed_sha
from .occlusion_dense_footprints import summarize_dense_footprints

def _file_hash(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
GEOMETRY_REFERENCE_SHA='757eef807837c8c44a6be6cbe0b237473f30953a543477dfc13fc32543c96b6a'
INDEX_REFERENCE_SHA='b57de34dcf8c18fa94cb6390c1a92cf90c619f7b1e95dfc77ce3f78ca0417318'
RUNNER_SHA='c2507b6d6308f3c9609a4931391a7863e7c4f3811a206b1bd485ead5a15f1634'
CALIBRATION_SHA='2b94a3c887e5aff600dc2d555121b74ac6cbd6670667e2f4c5c8688a434b3f8a'
BETA_SHA=dict(left='cfc2b56a47e5a912cfd03e576a27920fb2c43a0152fc73e28b74f5ac9063d865',
    right='48e6889f51db4cb45623731f7ae98a79f2d6498fa45048abb0b7c3fbd6af2628')
RECIPES=('vits_s','rn18_s')
HANDS=('left','right')
_LIVE=weakref.WeakKeyDictionary()


def _json(path):
    def reject(v):raise ValueError('nonfinite dense geometry JSON: '+v)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def _schema(n):
    fields=dict(mocap_webcam_transform=(np.float32,(n,12)),mocap_webcam_tracked=(np.bool_,(n,)),
        image_webcam_stale=(np.bool_,(n,)),image_webcam_frame_index=(np.int32,(n,)))
    for h in HANDS:fields.update({f'generated_mano_{h}_pose':(np.float32,(n,48)),
        f'mocap_mano_{h}_world_transform':(np.float32,(n,12)),f'mocap_{h}_valid':(np.bool_,(n,21))})
    return fields


def _metadata(meta,n,episodes,*,canonical_beta):
    try:
        starts,ends,lengths,ids=(meta[k]for k in('episode_start_idx','episode_end_idx','episode_length','episode_beta_idx'))
        if (episodes<=0 or any(a.shape!=(episodes,)or a.dtype.kind not in'iu'for a in(starts,ends,lengths,ids))
            or starts[0]!=0 or ends[-1]!=n or np.any(ends<=starts)or not np.array_equal(ends-starts,lengths)
            or not np.array_equal(starts[1:],ends[:-1])or np.any(ids<0)or np.any(ids>=episodes)
            or(canonical_beta and not np.array_equal(ids,np.arange(episodes)))):
            raise ValueError('complete episode/beta indexing differs')
        video='episode_head_video_path'if'episode_head_video_path'in meta else'episode_webcam_video_path'
        a=meta[video]
        if a.shape!=(episodes,)or a.dtype.kind not in'SU':raise ValueError('released episode video mapping required')
        for value in a:
            p=Path(bytes(value).decode().rstrip('\x00')if a.dtype.kind=='S'else str(value))
            if p.is_absolute()or'..'in p.parts:raise ValueError('relative released episode video paths required')
        return video
    except(KeyError,TypeError,AttributeError)as exc:raise ValueError('typed complete dense geometry metadata required')from exc


def extract_dense_point_arrays(points,fields,betas,metadata):
    """Pure IO transform. Synthetic arrays here confer no provenance permit."""
    summarize_dense_footprints(points,dict(windows=[]))
    try:n=len(fields['image_webcam_stale']);episodes=len(metadata['episode_start_idx'])
    except(KeyError,TypeError)as exc:raise ValueError('complete numeric geometry fields required')from exc
    _metadata(metadata,n,episodes,canonical_beta=False)
    expected=_schema(n)
    if set(fields)!=set(expected)or set(betas)!=set(HANDS):raise ValueError('complete native field/beta set required')
    for name,(dtype,shape)in expected.items():
        a=fields[name]
        if not isinstance(a,np.ndarray)or a.dtype!=np.dtype(dtype)or a.shape!=shape:
            raise ValueError('dense source dtype/shape differs: '+name)
    for h,a in betas.items():
        if a.dtype!=np.dtype(np.float32)or a.shape!=(episodes,10)or not np.isfinite(a).all():
            raise ValueError('actual finite episode beta required: '+h)
    left=np.flatnonzero(points['hand_index']==0);centers=points['center'][left];ep=points['episode'][left]
    if (np.any(ep<0)or np.any(ep>=episodes)or np.any(centers<metadata['episode_start_idx'][ep])
        or np.any(centers>=metadata['episode_end_idx'][ep])
        or np.any(points['start']<metadata['episode_start_idx'][points['episode']])
        or np.any(points['end']>metadata['episode_end_idx'][points['episode']])):
        raise ValueError('native dense point/window outside recorded episode')
    pool={(int(e),int(c)):j for j,(e,c)in enumerate(zip(ep,centers))}
    if len(pool)!=len(left):raise ValueError('unique native temporal centers required')
    a=dict(center=np.tile(centers.astype(np.int64),2),episode=np.tile(ep.astype(np.int64),2),
        hand=np.repeat(np.array([0,1],np.int8),len(left)))
    for h in HANDS:
        a[h+'_pose']=fields['generated_mano_'+h+'_pose'][centers].copy()
        a[h+'_world_transform']=fields['mocap_mano_'+h+'_world_transform'][centers].copy()
        a[h+'_marker_valid']=fields['mocap_'+h+'_valid'][centers].copy()
        a[h+'_beta']=betas[h][metadata['episode_beta_idx'][ep]].copy()
    for dest,source in dict(camera_transform='mocap_webcam_transform',camera_tracked='mocap_webcam_tracked',
        image_stale='image_webcam_stale',image_frame_index='image_webcam_frame_index').items():a[dest]=fields[source][centers].copy()
    if any(v.dtype.kind=='f'and not np.isfinite(v).all()for v in a.values()):raise ValueError('nonfinite selected real geometry; no repair')
    for name in('camera_transform','left_world_transform','right_world_transform'):
        if np.any(np.abs(np.linalg.det(a[name][:,:9].astype(np.float64).reshape(-1,3,3)))<=1e-12):
            raise ValueError('selected native transform is singular')
    selection={k:v.copy()for k,v in points.items()}
    selection['pool_index']=np.array([pool[(int(e),int(c))]for e,c in zip(points['episode'],points['center'])],np.int64)
    for v in(*a.values(),*selection.values()):v.setflags(write=False)
    return a,selection


def _references(geometry_reference,index_reference):
    paths=dict(geometry_reference=_safe(geometry_reference),index_reference=_safe(index_reference))
    guard=_StatGuard(paths.values());digests={k:_file_hash(p)for k,p in paths.items()};guard.check()
    if digests!=dict(geometry_reference=GEOMETRY_REFERENCE_SHA,index_reference=INDEX_REFERENCE_SHA):
        raise ValueError('fixed actual full numeric/index reference bytes required')
    gr,ir=(_json(paths[k])for k in('geometry_reference','index_reference'))
    if (gr['format']!='actual_legacy_complete_geometry_numeric_scan_v1'or not gr['full_numeric_scan_complete']
        or not gr['structural_admission_passed']or not gr['actual_beta_fields_used']or gr['zero_beta_fallback_used']
        or gr['scientific_eligible']or gr['full_paper_complete']
        or ir['format']!='actual_complete_native_dense_index_admission_verified_v1'or ir['actual_exit_code']!=0
        or ir['actual_complete_index_passes']!=4 or not ir['all_ten_original_owners_closed']
        or not ir['exact_typed_point_arrays_and_identities_rebuilt']or ir['full_paper_complete']or set(ir['recipes'])!=set(RECIPES)):
        raise ValueError('actual complete source admissions required')
    return paths,digests,gr,ir


def _sources():
    names=('egoemg_repro/occlusion_dense_geometry_data.py','egoemg_repro/occlusion_dense_data.py',
        'egoemg_repro/occlusion_dense_footprints.py','egoemg_repro/classic_emg.py','egoemg_repro/classic_emg_cohort.py',
        'egoemg_repro/data.py','egoemg_repro/fusion_initializer_handoff.py','scripts/paper/run_native_point_geometry.py')
    return {name:_safe(ROOT/name)for name in names}


def _core(path):
    if _file_hash(path)!=RUNNER_SHA:raise ValueError('accepted frozen native geometry helper required')
    spec=importlib.util.spec_from_file_location('_accepted_dense_geometry_point_io_core',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def _grid(index_root,recipe,record,inputs,n,episodes,guard):
    root=_safe(index_root/'private'/(recipe+'-a'));ip=_safe(root/'identity.json');pp=_safe(root/'points.npz')
    guard.add(ip);guard.add(pp)
    if _file_hash(ip)not in record['identity_files_sha256']or _file_hash(pp)not in record['point_files_sha256']:
        raise ValueError('actual accepted dense identity/point packet differs')
    i=_json(ip)
    if (i['sha256']!=_hash({k:x for k,x in i.items()if k!='sha256'})or i['sha256']!=record['identity_sha256']
        or i['format']!='original_dense_occlusion_complete_index_v1'or i['protocol']['recipe']!=recipe
        or i['unique_hand_points']!=record['unique_hand_points']or i['total_rows']!=n or i['episode_records']!=episodes
        or i['input_files_sha256']!=inputs or record['input_files_sha256']!=inputs
        or i['protocol']['window_length']!=record['window_length']or i['protocol']['dense_stride']!=record['dense_stride']):
        raise ValueError('authenticated complete dense grid binding differs')
    with np.load(pp,allow_pickle=False)as f:points={k:f[k].copy()for k in f.files}
    if set(points)!=set(COLUMNS)or any(_typed_sha(a)!=i['arrays_sha256'][k]for k,a in points.items()):
        raise ValueError('actual dense typed point bytes differ')
    if len(points['sequence'])!=record['unique_hand_points']:raise ValueError('complete dense point count differs')
    return points,i,{'index:'+recipe+':identity':ip,'index:'+recipe+':points':pp}


@dataclass(eq=False)
class DenseGeometryInputs:
    data_by_recipe:dict
    selection_by_recipe:dict
    identity:dict
    _guard:object
    _sources:dict
    _core:object

    def check(self):
        r=_LIVE.get(self)
        if r is None:raise ValueError('registered actual dense geometry admission required')
        self._guard.check()
        if (_file_hash(__file__)!=IMPORTED_SHA or self.identity.get('sha256')!=r['identity']
            or _hash({k:x for k,x in self.identity.items()if k!='sha256'})!=r['identity']
            or set(self.data_by_recipe)!=set(RECIPES)or set(self.selection_by_recipe)!=set(RECIPES)
            or any(_file_hash(p)!=r['sources'][k]for k,p in self._sources.items())):
            raise ValueError('dense geometry identity/source changed')
        for recipe in RECIPES:
            d=self.data_by_recipe[recipe];self._core.check_inputs(d)
            if d['_token']!=r['tokens'][recipe]:raise ValueError('native dense input token changed')
            s=self.selection_by_recipe[recipe]
            if(set(s)!=set(r['selection'][recipe])or any(a.flags.writeable or _typed_sha(a)!=r['selection'][recipe][k]for k,a in s.items())):
                raise ValueError('native dense source order/pool mapping changed')


def admit_dense_geometry_inputs(data,geometry,index_root,*,geometry_reference,index_reference):
    """Reopen all actual raw bytes once for both complete accepted dense grids."""
    refs,refsha,gr,ir=_references(geometry_reference,index_reference)
    data,geometry,index_root=(_safe(p)for p in(data,geometry,index_root));sources=_sources()
    guard=_StatGuard([*refs.values(),*sources.values()]);source_sha={k:_file_hash(p)for k,p in sources.items()}
    if source_sha['egoemg_repro/occlusion_dense_geometry_data.py']!=IMPORTED_SHA:raise ValueError('imported dense geometry IO source changed')
    core=_core(sources['scripts/paper/run_native_point_geometry.py']);guard.check()
    paths=dict(refs,manifest=_safe(data/'manifest.json'),metadata=_safe(data/'metadata.npz'),
        frame_split_id=_safe(data/'frame_split_id.dat'),calibration=_safe(geometry/'meta/GX010023_standard_calibration.json'))
    for p in paths.values():guard.add(p)
    raw={k:_file_hash(p)for k,p in paths.items()};guard.check()
    if raw['manifest']!=gr['manifest_sha256']or raw['metadata']!=gr['metadata_sha256']or raw['calibration']!=CALIBRATION_SHA:
        raise ValueError('actual complete geometry metadata/calibration bytes differ')
    manifest=_json(paths['manifest']);n=gr['total_rows'];episodes=gr['episode_count']
    if manifest['total_rows']!=n or manifest['num_episodes']!=episodes:raise ValueError('complete source row/episode counts differ')
    with np.load(paths['metadata'],allow_pickle=False)as f:meta={k:f[k].copy()for k in f.files}
    video=_metadata(meta,n,episodes,canonical_beta=True)
    calib=_json(paths['calibration']);camera=gr['camera'];K=np.asarray(calib['camera_matrix'],np.float64);dist=np.asarray(calib['distortion_coefficients'],np.float64)
    if(K.shape!=(3,3)or not np.isfinite(K).all()or not np.isfinite(dist).all()or K.tolist()!=camera['matrix']
        or dist.size!=camera['distortion_coefficient_count']or[calib['image_width'],calib['image_height']]!=camera['declared_calibration_resolution']):
        raise ValueError('actual original calibration fields differ')
    fields={};wanted=_schema(n);bulk={a['field']:a for a in gr['bulk_files']}
    if len(gr['bulk_files'])!=8 or set(bulk)!=(set(wanted)-{'image_webcam_stale','image_webcam_frame_index'}):
        raise ValueError('exact accepted eight bulk fields required')
    for name,(dtype,shape)in wanted.items():
        spec=manifest['fields'][name]
        if tuple(spec['shape'])!=shape or np.dtype(spec['dtype'])!=np.dtype(dtype):raise ValueError('actual geometry manifest schema differs: '+name)
        path=_safe((data if name.startswith('image_')else geometry)/spec['filename']);guard.add(path);tag='field:'+name
        paths[tag]=path;raw[tag]=_file_hash(path)
        if path.stat().st_size!=int(np.prod(shape))*np.dtype(dtype).itemsize:raise ValueError('full raw geometry byte length differs')
        if name in bulk:
            a=bulk[name]
            if(a['filename']!=spec['filename']or a['shape']!=spec['shape']or np.dtype(a['dtype'])!=np.dtype(dtype)
                or raw[tag]!=a['actual_full_sha256']or a['sha256']!=a['actual_full_sha256']or a['expected_bytes']!=path.stat().st_size):
                raise ValueError('complete accepted bulk bytes/schema differ: '+name)
        fields[name]=np.memmap(path,mode='r',dtype=dtype,shape=shape)
    betas={}
    for h in HANDS:
        p=_safe(geometry/'meta'/('generated_mano_'+h+'_beta.dat'));guard.add(p);tag='beta:'+h;paths[tag]=p;raw[tag]=_file_hash(p)
        if p.stat().st_size!=episodes*10*4 or raw[tag]!=BETA_SHA[h]:raise ValueError('actual admitted beta bytes differ: '+h)
        betas[h]=np.memmap(p,mode='r',dtype=np.float32,shape=(episodes,10))
    sf=manifest['fields']['frame_split_id']
    if sf['filename']!='frame_split_id.dat'or sf['shape']!=[n]or np.dtype(sf['dtype']).kind not in'iu':raise ValueError('source split map schema differs')
    splits=np.memmap(paths['frame_split_id'],mode='r',dtype=sf['dtype'],shape=(n,))
    inputs={k:raw[k]for k in('manifest','metadata','frame_split_id')};data_by_recipe={};selection_by_recipe={};grids={}
    for a in meta.values():a.setflags(write=False)
    for recipe in RECIPES:
        guard.check();points,i,more=_grid(index_root,recipe,ir['recipes'][recipe],inputs,n,episodes,guard)
        paths.update(more);raw.update({k:_file_hash(p)for k,p in more.items()})
        if(np.any(splits[points['center']]!=points['split_id'])or np.any(points['group_index']+1!=points['split_id'])
            or np.any(points['end']-points['start']!=i['protocol']['window_length'])
            or np.any((points['start']-meta['episode_start_idx'][points['episode']])%i['protocol']['dense_stride'])):
            raise ValueError('actual original center/split/episode-start grid differs')
        arrays,selection=extract_dense_point_arrays(points,fields,betas,meta);core._typed_schema(arrays,len(arrays['camera_transform']))
        d=dict(role='native_dense_'+recipe,unique_hand_samples=i['unique_hand_points'],arrays=arrays,metadata=meta,
            calibration=calib,admission=gr,files_sha256=dict(raw),_paths=dict(metadata=paths['metadata'],calibration=paths['calibration']),
            _guard=guard,_video_field=video)
        d['_token']=core._token(d);core.check_inputs(d);data_by_recipe[recipe]=d;selection_by_recipe[recipe]=selection
        stats={h:dict(points=len(arrays[h+'_pose']),finite_nonzero_pose_rows=int(np.any(arrays[h+'_pose']!=0,axis=1).sum()),
            invalid_zero_pose_rows=int((~np.any(arrays[h+'_pose']!=0,axis=1)).sum()),
            camera_tracked_rows=int(arrays['camera_tracked'].sum()),image_stale_rows=int(arrays['image_stale'].sum()),
            marker_any_valid_rows=int(arrays[h+'_marker_valid'].any(axis=1).sum()),negative_image_frame_indices=int((arrays['image_frame_index']<0).sum()))for h in HANDS}
        grids[recipe]=dict(index_identity_sha256=i['sha256'],unique_hand_requests=i['unique_hand_points'],
            unique_temporal_centers=len(arrays['camera_transform']),by_hand=stats,
            arrays_sha256={k:_typed_sha(a)for k,a in arrays.items()},selection_sha256={k:_typed_sha(a)for k,a in selection.items()})
    # Do not rely on filesystem timestamp granularity for byte immutability.
    if any(_file_hash(p)!=raw[k]for k,p in paths.items()):raise ValueError('raw source bytes changed during dense extraction')
    if any(_file_hash(p)!=source_sha[k]for k,p in sources.items()):raise ValueError('dense extraction provider bytes changed')
    guard.check();identity=dict(format='actual_complete_native_dense_geometry_numeric_inputs_v1',reference_files_sha256=refsha,
        raw_input_files_sha256=raw,raw_input_file_bytes={k:p.stat().st_size for k,p in paths.items()},source_files_sha256=source_sha,
        total_rows=n,episode_records=episodes,grids=grids,actual_episode_beta_used=True,zero_beta_fallback_used=False,
        all_raw_bytes_rehashed_after_extraction=True,full_numeric_scan_inherited_from_exact_accepted_bytes=True,
        native_geometry_executed=False,learned_pose_models_constructed=0,optimizer_updates=0,cuda_initialized=False,
        model_or_geometry_permit=False,scientific_eligible=False,full_paper_complete=False)
    identity['sha256']=_hash(identity);result=DenseGeometryInputs(data_by_recipe,selection_by_recipe,identity,guard,sources,core)
    _LIVE[result]=dict(identity=identity['sha256'],sources=source_sha,tokens={r:d['_token']for r,d in data_by_recipe.items()},
        selection={r:g['selection_sha256'].copy()for r,g in grids.items()})
    result.check();return result


def _write(path,value):
    with path.open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush()
    import os
    with path.open('rb')as f:os.fsync(f.fileno())


def export_dense_geometry_inputs(output,**kwargs):
    output=_safe(output,exists=False)
    if output.exists():raise ValueError('new dense geometry export directory required')
    v=admit_dense_geometry_inputs(**kwargs);v.check();output.mkdir(mode=0o700);files={}
    for recipe in RECIPES:
        for kind,arrays in(('points',v.data_by_recipe[recipe]['arrays']),('selection',v.selection_by_recipe[recipe])):
            p=output/(recipe+'-'+kind+'.npz')
            with p.open('xb')as f:
                np.savez(f,**arrays);f.flush()
                import os
                os.fsync(f.fileno())
            files[p.name]=_file_hash(p)
    _write(output/'identity.json',v.identity);files['identity.json']=_file_hash(output/'identity.json')
    v.check();completion=dict(format='complete_dense_geometry_numeric_export_v1',identity_sha256=v.identity['sha256'],
        files_sha256=files,full_paper_complete=False)
    completion['sha256']=_hash(completion);_write(output/'completion.json',completion);return v.identity


def verify_dense_geometry_export(output,**kwargs):
    output=_safe(output);v=admit_dense_geometry_inputs(**kwargs);v.check();identity=_json(_safe(output/'identity.json'))
    if identity!=v.identity:raise ValueError('fresh dense geometry input identity differs')
    files={'identity.json':_file_hash(output/'identity.json')}
    for recipe in RECIPES:
        for kind,expected in(('points',v.data_by_recipe[recipe]['arrays']),('selection',v.selection_by_recipe[recipe])):
            p=_safe(output/(recipe+'-'+kind+'.npz'))
            with np.load(p,allow_pickle=False)as f:
                if set(f.files)!=set(expected)or any(_typed_sha(f[k])!=_typed_sha(a)for k,a in expected.items()):
                    raise ValueError('complete typed dense geometry export differs')
            files[p.name]=_file_hash(p)
    c=dict(format='complete_dense_geometry_numeric_export_v1',identity_sha256=v.identity['sha256'],files_sha256=files,full_paper_complete=False)
    c['sha256']=_hash(c)
    if _json(_safe(output/'completion.json'))!=c:raise ValueError('dense geometry completion differs')
    v.check();return v.identity
