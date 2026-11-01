"""Same-key RGB/three predictions/GT panels with original local FK meshes.

These numeric helpers never authenticate a trained producer or full cohort.
The report writer must separately retain real input and closed raw-run guards.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .classic_emg import _hash
from .classic_emg_cohort import _safe
from .data import file_sha256
from .native_ik_inputs import _Guard
from .occlusion_dense_data import _typed_sha
from .qualitative_geometry import native_qualitative_meshes

IMPORTED_SHA=file_sha256(__file__)
ROLES=('emg','vision','fusion','gt')
COLORS=dict(emg='#dfad49',vision='#c66888',fusion='#5796bd',gt='#5d9d71')
MEAN=255*np.array([.485,.456,.406],np.float32)
STD=255*np.array([.229,.224,.225],np.float32)
ROOT=Path(__file__).resolve().parents[1]
PREDICTION_REFERENCE=ROOT/'results/native-qualitative/frozen-predictions-complete-20261101.json'
PREDICTION_REFERENCE_SHA='1021f819d6c5371a43643715d4b3495bdf4f2017b087fcb028c8c4fdaf349e71'
FLAG_REFERENCE=ROOT/'results/native-qualitative/boolean-flags-complete-20261101.json'
FLAG_REFERENCE_SHA='dc63fa66693b45f5928a81a3d51577c3e590930943665cef77f054dc8a43e1d9'


def _guard():
    guard=_Guard([_safe(__file__)],[])
    if file_sha256(__file__)!=IMPORTED_SHA:raise ValueError('imported qualitative renderer changed')
    return guard


def _rgb(image):
    if not isinstance(image,np.ndarray)or image.dtype!=np.float32 or image.shape!=(3,256,256)or not np.isfinite(image).all():
        raise ValueError('exact original finite normalized RGB input required')
    restored=np.empty(image.shape,np.float32)
    for c in range(3):restored[c]=image[c]*STD[c]+MEAN[c]
    if np.any(restored<-.001)or np.any(restored>255.001):raise ValueError('original normalized uint8 RGB range required')
    pixels=np.rint(restored).astype(np.uint8);roundtrip=pixels.astype(np.float32)
    for c in range(3):roundtrip[c]=(roundtrip[c]-MEAN[c])/STD[c]
    if not np.array_equal(roundtrip,image):raise ValueError('image is not an exact original uint8 normalization; no pixel repair')
    return pixels.transpose(1,2,0).copy()


def _panel_arrays(panel):
    arrays=dict(rgb=panel['rgb'])
    arrays.update({'pose:'+r:a for r,a in panel['poses'].items()})
    for role,mesh in panel['meshes'].items():
        for n in('vertices','faces','unscaled_vertices'):arrays[role+':'+n]=mesh[n]
    return arrays


def _check_panel(panel):
    if(not isinstance(panel,dict)or panel.get('metadata_sha256')!=_hash(panel.get('metadata'))
        or {n:_typed_sha(a)for n,a in _panel_arrays(panel).items()}!=panel['metadata']['arrays_sha256']):
        raise ValueError('bound qualitative RGB, pose, geometry or metadata changed')


def assemble_qualitative_panel(sample,prediction,row,*,upstream):
    import torch
    guard=_guard()
    try:
        key=sample['key'];a=prediction['arrays'];f=sample['fusion'];e=sample['standalone_emg']
        if key!=prediction['key']or key!=row['key']or key['hand']not in('left','right'):
            raise ValueError('same full original window, hand and point required')
        if not row['source_filter_pass']or not row['crop_valid']or not bool(f['vision_valid_mask'][0]):
            raise ValueError('a source-filtered sample with actual RGB is needed for display; retain other rank rows')
        required={'targets','vision_prediction','emg_prediction','fusion_prediction','emg_full_prediction',
            'label_valid_mask','crop_valid_mask','mae_vision_rad','mae_emg_rad','mae_fusion_rad'}
        if set(a)!=required:raise ValueError('all original row prediction arrays required')
        for n in('targets','vision_prediction','emg_prediction','fusion_prediction'):
            if not isinstance(a[n],np.ndarray)or a[n].dtype!=np.float32 or a[n].shape!=(22,)or not np.isfinite(a[n]).all():
                raise ValueError('all four exact finite22-angle FP32 poses required')
        if (f['joint_angles'].dtype!=np.float32 or e['joint_angles'].dtype!=np.float32
            or f['joint_angles'].shape!=(22,1)or e['joint_angles'].shape!=(22,1)
            or a['targets'].tobytes()!=f['joint_angles'][:,0].tobytes()
            or a['targets'].tobytes()!=e['joint_angles'][:,0].tobytes()):
            raise ValueError('same exact original center GT for RGB and both EMG inputs required')
        for n,actual in(('label_valid_mask',f['label_valid_mask']),('crop_valid_mask',f['vision_valid_mask'])):
            if a[n].dtype!=np.bool_ or a[n].shape!=(1,)or not np.array_equal(a[n],actual):
                raise ValueError('same original label and crop masks required')
        if (not np.array_equal(e['label_valid_mask'],f['label_valid_mask'])
            or row['label_valid']!=bool(a['label_valid_mask'][0])or row['crop_valid']!=bool(a['crop_valid_mask'][0])):
            raise ValueError('rank row original masks differ')
        full=a['emg_full_prediction']
        if (not isinstance(full,np.ndarray)or full.dtype!=np.float32 or full.ndim!=2 or full.shape[0]!=22
            or full.shape[1]<1 or not np.isfinite(full).all()or full[:,full.shape[1]//2].tobytes()!=a['emg_prediction'].tobytes()):
            raise ValueError('original full-EMG midpoint required')
        typed={'fusion:'+n:_typed_sha(f[n])for n in('emg','emg_channel_mask','joint_angles','label_valid_mask','vision_img','vision_valid_mask','vision_frame_indices')}
        typed.update({'standalone:'+n:_typed_sha(e[n])for n in('emg','emg_channel_mask','joint_angles','label_valid_mask')})
        if typed!=sample['typed_input_sha256']or _hash(typed)!=sample['verified_input_sha256']:
            raise ValueError('actual original point input tensor bytes changed')
        errors={}
        for role in('emg','vision','fusion'):
            value=float((torch.from_numpy(a[role+'_prediction'])-torch.from_numpy(a['targets'])).abs().mean())
            if value!=a['mae_'+role+'_rad']or value!=row['mae_'+role]:
                raise ValueError('original unmasked22 error differs between raw, rank and display')
            errors[role]=value
        if row['delta_mae']!=errors['vision']-errors['fusion']:raise ValueError('original unrounded gain differs')
        rgb=_rgb(f['vision_img'])
    except(KeyError,IndexError,TypeError,AttributeError)as exc:
        raise ValueError('complete aligned original sample, raw prediction and rank row required')from exc
    poses={r:a['targets'if r=='gt'else r+'_prediction'].copy()for r in ROLES}
    geometry=native_qualitative_meshes(poses,hand=key['hand'],upstream=upstream)
    meshes=geometry['meshes'];all_vertices=np.concatenate([m['vertices']for m in meshes.values()])
    lo=all_vertices.min(axis=0);hi=all_vertices.max(axis=0);center=(lo+hi)/2;radius=float(np.max(hi-lo)/2)*1.07
    view=dict(elevation=20.,azimuth=-70.,projection='ortho',zoom=1.5,
        limits=[[float(c-radius),float(c+radius)]for c in center])
    panel=dict(rgb=rgb,poses=poses,meshes=meshes)
    metadata=dict(format='same_key_original_qualitative_numeric_panel_v1',key=copy.deepcopy(key),
        errors_rad=errors,errors_deg={r:v*57.3 for r,v in errors.items()},delta_rad=row['delta_mae'],
        numeric_error_angles=22,mesh_finger_angles=20,wrist_angles_displayed=False,
        label_valid=bool(a['label_valid_mask'][0]),crop_valid=True,shared_view=view,
        normalized_rgb_sha256=_typed_sha(f['vision_img']),input_typed_sha256=typed,
        geometry_source_files_sha256=geometry['source_files_sha256'],geometry_adapter_source_sha256=geometry['adapter_source_sha256'],
        native_mesh_normalization=geometry['source_display_normalization'],
        arrays_sha256={n:_typed_sha(x)for n,x in _panel_arrays(panel).items()},
        left_rgb_already_canonical=True,physical_rgb_projection_proven=False,
        trained_source_admitted=False,figure7_reproduced=False,scientific_eligible=False,full_paper_complete=False)
    panel.update(metadata=metadata,metadata_sha256=_hash(metadata))
    guard.check();_check_panel(panel);return panel


def render_qualitative_panel(panel,output,*,caption=''):
    """Draw unchanged vertices under one common view and bounds for all poses."""
    guard=_guard();_check_panel(panel);path=_safe(output,exists=False)
    if path.exists():raise FileExistsError(path)
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    meta=panel['metadata'];fig=plt.figure(figsize=(13.5,3.2),dpi=160,facecolor='white')
    try:
        grid=fig.add_gridspec(1,5,width_ratios=[1.08,1,1,1,1],wspace=.015)
        ax=fig.add_subplot(grid[0,0]);ax.imshow(panel['rgb']);ax.set_axis_off();ax.set_title('Input image',fontsize=12,pad=4)
        titles=dict(emg='EMG only',vision='Vision only',fusion='EMG + vision',gt='GT')
        view=meta['shared_view'];light=np.array([.4,-.6,.7]);light=light/np.linalg.norm(light)
        for column,role in enumerate(ROLES,1):
            ax=fig.add_subplot(grid[0,column],projection='3d');mesh=panel['meshes'][role]
            triangles=mesh['vertices'][mesh['faces'].astype(np.int64)]
            normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);length=np.linalg.norm(normal,axis=1)
            normal=normal/np.maximum(length[:,None],1e-12)
            shade=.35+.65*np.maximum(normal@light,0)
            colors=np.asarray(to_rgb(COLORS[role]))[None,:]*shade[:,None]
            ax.add_collection3d(Poly3DCollection(triangles,facecolors=colors,edgecolors='none',linewidths=0,zsort='average'))
            ax.set_proj_type(view['projection']);ax.view_init(elev=view['elevation'],azim=view['azimuth'])
            ax.set_xlim(*view['limits'][0]);ax.set_ylim(*view['limits'][1]);ax.set_zlim(*view['limits'][2]);ax.set_box_aspect((1,1,1),zoom=view['zoom'])
            ax.set_axis_off();ax.set_title(titles[role],fontsize=12,pad=4)
            if role!='gt':ax.text2D(.5,-.025,f"MAE {meta['errors_deg'][role]:.2f}°",transform=ax.transAxes,ha='center',fontsize=11)
        fig.subplots_adjust(left=.012,right=.995,bottom=.17,top=.82)
        fig.suptitle(caption,fontsize=12,y=.975)
        fig.text(.5,.035,'Shared display view · meshes: 20 finger angles · MAE: all 22 angles',ha='center',fontsize=9,color='#555555')
        fig.savefig(path,dpi=160,facecolor='white')
    finally:plt.close(fig)
    _check_panel(panel);guard.check()
    with Image.open(path)as image:pixels=np.asarray(image.convert('RGBA')).copy()
    return dict(file_sha256=file_sha256(path),png_pixels_sha256=_typed_sha(pixels),
        panel_metadata=copy.deepcopy(meta),renderer_source_sha256=IMPORTED_SHA,
        trained_source_admitted=False,figure7_reproduced=False,full_paper_complete=False)


def export_qualitative_candidates(inputs,first,second,*,flags,output,limit=120):
    """Real closed sources produce a review pool, never automatic case labels."""
    from .qualitative_run import (QualitativeRunInputs,_RunGuard,read_qualitative_pass,
        compare_qualitative_passes)
    from .qualitative_flags import QualitativeFlags
    from .qualitative_analysis import rank_qualitative_predictions
    from .qualitative_geometry import SCRIPT,ASSET,LOADER,COMMON
    from .occlusion_dense_run import _closed_workflow,_write_json,_seal
    from .emg2pose_memory_process import process_identity
    if type(inputs)is not QualitativeRunInputs or type(flags)is not QualitativeFlags:
        raise ValueError('registered actual ordinary inputs and raw flag sources required')
    inputs.check();flags.check();e=inputs.evidence
    if type(limit)is not int or limit<1:raise ValueError('positive integer display candidate limit required')
    guard=_RunGuard([_safe(__file__),_safe(PREDICTION_REFERENCE),_safe(FLAG_REFERENCE)],[])
    if(file_sha256(__file__)!=IMPORTED_SHA or file_sha256(PREDICTION_REFERENCE)!=PREDICTION_REFERENCE_SHA
        or file_sha256(FLAG_REFERENCE)!=FLAG_REFERENCE_SHA):
        raise ValueError('fixed independently verified complete prediction/boolean source proofs required')
    accepted=json.loads(PREDICTION_REFERENCE.read_text());flag_proof=json.loads(FLAG_REFERENCE.read_text())
    if (flags.identity!=flag_proof['identity']
        or flags.identity['source_evidence_identity_sha256']!=_hash(e.identity)):
        raise ValueError('same exact original ordinary point population and actual raw flags required')
    first,second=_safe(first),_safe(second);root=first.parent.parent
    if first!=root/'private/pass-a'or second!=root/'private/pass-b':
        raise ValueError('distinct actual complete source pass-a and pass-b required')
    workflow=_closed_workflow(root,accepted['raw_workflow_file_sha256'],guard)
    manifest=root/'private/stage-receipt.json';guard.add(manifest);stage=json.loads(manifest.read_text())
    if (_hash({k:v for k,v in stage.items()if k!='bundle_sha256'})!=accepted['bundle_sha256']
        or stage['source_commit']!=accepted['source_commit']or workflow['bundle_sha256']!=accepted['bundle_sha256']):
        raise ValueError('same cryptographically bound actual neural source capsule required')
    for name,digest in stage['files_sha256'].items():
        path=_safe(root/name);guard.add(path)
        if file_sha256(path)!=digest:raise ValueError('frozen source capsule bytes changed')
    for name,digest in accepted['files_sha256'].items():
        path=_safe(root/name);guard.add(path)
        if file_sha256(path)!=digest:raise ValueError('independently accepted neural raw graph changed')
    for actor in accepted['actual_closed_original_actors']:
        current=process_identity(actor['pid'])
        if current is not None and current['start_ticks']==str(actor['start_ticks']):
            raise ValueError('actual original neural/loader owner remains live')
    for name in(SCRIPT,ASSET,LOADER,COMMON+'hand.py',COMMON+'hand_skinning.py',COMMON+'pytorch3d_transforms_so3.py'):
        guard.add(e.upstream/name)
    comparison=compare_qualitative_passes(first,second,evidence=e)
    raw=read_qualitative_pass(first,evidence=e)
    if raw['identity']['sha256']!=accepted['prediction_identity_sha256']:
        raise ValueError('same complete accepted neural identity required')
    if {n:_typed_sha(a)for n,a in raw['arrays'].items()}!=accepted['complete_prediction_arrays_sha256']:
        raise ValueError('same independently accepted complete raw predictions required')
    ranking=rank_qualitative_predictions(e.keys(),raw['arrays'],flags.records(),recipe='vits_s',upstream=e.upstream)
    directory=_safe(output,exists=False)
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(mode=0o700)
    with(directory/'ranking.csv').open('x',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(ranking['csv_rows'][0]));writer.writeheader();writer.writerows(ranking['csv_rows'])
    _write_json(directory/'ranking.json',ranking)
    samples=[];skipped=[]
    for rank,row in enumerate(ranking['ranked_rows'],1):
        if not row['crop_valid']:
            skipped.append(dict(rank=rank,key=row['key'],reason='missing actual input RGB; retained in original rank population'))
            continue
        if len(samples)>=limit:break
        index=row['sequence'];sample=inputs[index]
        prediction=dict(key=e.keys()[index],arrays={n:a[index]for n,a in raw['arrays'].items()})
        panel=assemble_qualitative_panel(sample,prediction,row,upstream=e.upstream)
        base=f"rank_{rank:04d}_seq_{index:06d}"
        png=base+'.png';arrays_file=base+'.npz';meta_file=base+'.json'
        with(directory/arrays_file).open('xb')as stream:
            np.savez_compressed(stream,**_panel_arrays(panel));stream.flush()
        caption=f"Rank {rank} | {row['split']} / {row['hand']} | gain {row['delta_mae']*57.3:.2f}°"
        drawn=render_qualitative_panel(panel,directory/png,caption=caption)
        _write_json(directory/meta_file,dict(drawn,rank=rank,source_model_identity_sha256=accepted['source_model_identity_sha256'],
            accepted_prediction_proof_sha256=PREDICTION_REFERENCE_SHA,accepted_flag_proof_sha256=FLAG_REFERENCE_SHA,
            paired_input_identity_sha256=inputs.identity['paired_input_identity_sha256']))
        samples.append(dict(rank=rank,key=copy.deepcopy(row['key']),png_file=png,arrays_file=arrays_file,metadata_file=meta_file,
            png_file_sha256=file_sha256(directory/png),arrays_file_sha256=file_sha256(directory/arrays_file),
            metadata_file_sha256=file_sha256(directory/meta_file),panel_metadata_sha256=panel['metadata_sha256']))
        inputs.check();flags.check();raw['_guard'].check();raw['_packet_guard'].check();guard.check()
    payloads=inputs.hash_payloads()
    result=_seal(dict(format='actual_original_ordinary_qualitative_candidate_pool_v1',actor=process_identity(),
        source_evidence_identity_sha256=_hash(e.identity),source_model_identity_sha256=accepted['source_model_identity_sha256'],
        accepted_prediction_proof_sha256=PREDICTION_REFERENCE_SHA,accepted_boolean_proof_sha256=FLAG_REFERENCE_SHA,
        raw_prediction_identity_sha256=raw['identity']['sha256'],full_source_payload_sha256_after_render=payloads,
        counts=ranking['counts'],rendered_candidates=len(samples),requested_display_limit=limit,
        selection_policy='stable original gain rank; first displayable actual crops up to explicit review-pool limit',
        samples=samples,skipped_for_display=skipped,condition_examples_selected=False,
        dual_full_frozen_passes_verified=comparison['complete_frozen_predictions_verified'],
        renderer_source_sha256=IMPORTED_SHA,ranking_csv_file_sha256=file_sha256(directory/'ranking.csv'),
        ranking_json_file_sha256=file_sha256(directory/'ranking.json'),
        physical_rgb_projection_proven=False,figure7_reproduced=False,scientific_eligible=False,full_paper_complete=False))
    inputs.check();flags.check();raw['_guard'].check();raw['_packet_guard'].check();guard.check()
    _write_json(directory/'index.json',result)
    inputs.check();flags.check();raw['_guard'].check();raw['_packet_guard'].check();guard.check()
    return result
