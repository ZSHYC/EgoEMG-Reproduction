"""Readonly RN18 ordinary-cache admission and actual RGB/pose review panels.

The completed dense producer keeps its original identity and keys. Only the
independently accepted exact ordinary-key cache may feed this visual consumer.
Existing numerical/FK/render helpers are unchanged; this module never predicts.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import weakref

import numpy as np

from .classic_emg import _hash
from .classic_emg_cohort import _safe
from .data import file_sha256
from .occlusion_dense_data import _typed_sha
from .qualitative_run import _RunGuard, _admit_capsule

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'results/native-qualitative/rn18-cache-source-read-20261105.json'
REFERENCE_SHA='c7a36ad430a9d080ea1fa12f9d4718e0bd506186e46836a3028ce84e1056ea69'
IMPORTED_SHA=file_sha256(__file__)
_CACHES=weakref.WeakKeyDictionary()
_PHYSICAL=('split','hand','ep_idx','center_idx','window_start','window_end')
_FIELDS=dict(fusion=('emg','emg_channel_mask','joint_angles','label_valid_mask','vision_img','vision_valid_mask','vision_frame_indices'),
    standalone=('emg','emg_channel_mask','joint_angles','label_valid_mask'))


def _state(cache):
    if type(cache)is not CachedRN18 or cache not in _CACHES:
        raise ValueError('registered actual RN18 ordinary cache required')
    return _CACHES[cache]


def _closed(actor):
    from .emg2pose_memory_process import process_identity
    current=process_identity(actor['pid'])
    if current is not None and str(current['start_ticks'])==str(actor['start_ticks']):
        raise ValueError('accepted cache source owner still present')


@dataclass(frozen=True,eq=False)
class CachedRN18:
    def check(self):
        s=_state(self);s['guard'].check()
        if (file_sha256(__file__)!=IMPORTED_SHA or _hash(s['identity'])!=s['identity_hash']
            or any(a.flags.writeable or _typed_sha(a)!=s['array_hashes'][n]for n,a in s['arrays'].items())):
            raise ValueError('RN18 cache source, identity or prediction arrays changed')

    @property
    def identity(self):self.check();return copy.deepcopy(_state(self)['identity'])

    def ranked_rows(self):
        self.check();return json.loads(_state(self)['ranking'])['ranked_rows']

    def row(self,index):
        self.check()
        if type(index)is not int or not 0<=index<2208:raise IndexError(index)
        return {n:a[index].copy()for n,a in _state(self)['arrays'].items()}


def open_rn18_cache(source_root):
    """Accept the actual closed source graph and every original raw artifact."""
    if file_sha256(REFERENCE)!=REFERENCE_SHA:
        raise ValueError('fixed actual completed RN18 cache proof required')
    proof=json.loads(REFERENCE.read_text());root=_safe(source_root)
    guard=_RunGuard([_safe(__file__),_safe(REFERENCE),root/'private/stage-receipt.json',
        root/'private/execution-state.json',root/'private/workflow-result.json',
        root/'private/source-audit.json',root/'private/verification.json'],[])
    if file_sha256(__file__)!=IMPORTED_SHA:raise ValueError('imported RN18 visual consumer changed')
    manifest=json.loads((root/'private/stage-receipt.json').read_text())
    if manifest['source_commit']!=proof['source_commit']:raise ValueError('same actual accepted cache producer required')
    _admit_capsule(root,manifest,proof['source_bundle_sha256'],guard)
    state=json.loads((root/'private/execution-state.json').read_text())
    if state!=proof['actual_terminal_state']or state['status']!='complete'or state['actual_exit_code']!=0:
        raise ValueError('actual successful closed source execution required')
    for name,field in [('workflow-result.json','workflow_result_file_sha256'),
            ('source-audit.json','source_audit_file_sha256'),('verification.json','verification_file_sha256')]:
        if file_sha256(root/'private'/name)!=proof[field]:raise ValueError('accepted complete cache graph changed')
    for actor in [*proof['actual_closed_owning_actors'],proof['collector_actor']]:_closed(actor)
    directory=root/'private/selected'
    for name,digest in proof['source_audit']['files_sha256'].items():
        path=_safe(directory/name);guard.add(path)
        if file_sha256(path)!=digest:raise ValueError('accepted RN18 raw cache file changed: '+name)
    records=json.loads((directory/'ordinary-records.json').read_text())
    joins=json.loads((directory/'cache-join.json').read_text());ranking=json.loads((directory/'ranking.json').read_text())
    with np.load(directory/'arrays.npz',allow_pickle=False)as archive:arrays={n:archive[n]for n in archive.files}
    hashes={n:_typed_sha(a)for n,a in arrays.items()}
    if (len(records)!=2208 or len(joins)!=2208 or len(ranking['raw_rows'])!=2208
        or hashes!=proof['source_audit']['all10_selected_prediction_arrays_sha256']
        or ranking['counts']!=dict(selected=2208,tracked_filter_pass=2119,rejected=89,crop_invalid_retained=0,label_invalid_retained=0)):
        raise ValueError('complete original ordinary RN18 arrays and rankings required')
    for i,(record,join)in enumerate(zip(records,joins)):
        if record['key']['sequence']!=i or join['ordinary_sequence']!=i or join['ordinary_key']!=record['key']:
            raise ValueError('actual original ordered ordinary keys required')
    for a in arrays.values():a.flags.writeable=False
    identity=dict(format='registered_RN18_ordinary_cached_visual_source_v1',source_root=str(root),
        selected=2208,source_filter_pass=2119,source_model_identity_sha256=proof['source_audit']['source_model_identity_sha256'],
        accepted_cache_proof_sha256=REFERENCE_SHA,source_bundle_sha256=proof['source_bundle_sha256'],
        all10_selected_array_sha256=hashes,trained_model_or_inference_permit=False,figure7_reproduced=False,full_paper_complete=False)
    cache=CachedRN18();_CACHES[cache]=dict(identity=identity,identity_hash=_hash(identity),guard=guard,root=root,
        records=tuple(json.dumps(r,sort_keys=True)for r in records),joins=tuple(json.dumps(j,sort_keys=True)for j in joins),
        ranking=json.dumps(ranking,sort_keys=True),arrays=arrays,array_hashes=hashes,
        request=json.loads((root/'private/request.json').read_text())['dense_request'])
    cache.check();return cache


def materialize_rn18_cache_row(record,join,arrays,native_sample):
    """Pure exact-key/tensor join; it grants no trained-source admission."""
    try:
        ordinary=record['key'];dense=join['dense_key'];actual=native_sample['key']
        if (ordinary!=join['ordinary_key']or dense!=actual
            or join['ordinary_sequence']!=ordinary['sequence']or join['dense_sequence']!=dense['sequence']
            or any(ordinary[n]!=dense[n]for n in _PHYSICAL)
            or ordinary['window_end']-ordinary['window_start']!=12000
            or ordinary['center_idx']!=ordinary['window_start']+6000):
            raise ValueError('exact physical RN18 ordinary/dense window and original sequence mapping required')
        f=native_sample['fusion'];e=native_sample['standalone_emg']
        typed={prefix+':'+n:_typed_sha(packet[n])for prefix,packet in(('fusion',f),('standalone',e))for n in _FIELDS[prefix]}
        digest=_hash(typed)
        if (typed!=record['typed_input_sha256']or digest!=record['verified_input_sha256']
            or digest!=join['verified_input_sha256']or digest!=native_sample['verified_input_sha256']):
            raise ValueError('all eleven actual RN18 input tensors must match the accepted ordinary cache')
        if (not np.array_equal(arrays['targets'],f['joint_angles'][:,0])
            or not np.array_equal(arrays['targets'],e['joint_angles'][:,0])
            or not np.array_equal(arrays['label_valid_mask'],f['label_valid_mask'])
            or not np.array_equal(arrays['crop_valid_mask'],f['vision_valid_mask'])):
            raise ValueError('same actual original center targets and masks required')
        sample=dict(native_sample,key=copy.deepcopy(ordinary),typed_input_sha256=copy.deepcopy(typed))
        return dict(sample=sample,prediction=dict(key=copy.deepcopy(ordinary),arrays={n:a.copy()for n,a in arrays.items()}),
            cache_lineage=copy.deepcopy(join),trained_source_admitted=False)
    except(KeyError,TypeError,IndexError,AttributeError)as exc:
        raise ValueError('complete actual RN18 cached row and original native sample required')from exc


def select_rn18_render_rows(ranked_rows,*,limit=120,extra_ranks=()):
    """Keep source rank numbers; diagnostic extras never change the ranking."""
    rows=list(ranked_rows);extra=tuple(extra_ranks)
    if type(limit)is not int or limit<1 or len(set(extra))!=len(extra):
        raise ValueError('positive native review-pool cap and distinct explicit extra ranks required')
    selected=[]
    for rank,row in enumerate(rows,1):
        if not row['source_filter_pass']:raise ValueError('original source-filtered ranking required')
        if row['crop_valid']and len(selected)<limit:
            selected.append(dict(rank=rank,row=copy.deepcopy(row),diagnostic_extra=False))
    default={s['rank']for s in selected}
    for rank in extra:
        if type(rank)is not int or not 1<=rank<=len(rows)or rank in default or not rows[rank-1]['crop_valid']:
            raise ValueError('extra rank must be distinct, in range and displayable; no silent repair')
        selected.append(dict(rank=rank,row=copy.deepcopy(rows[rank-1]),diagnostic_extra=True))
    return selected


def export_rn18_candidates(cache,*,upstream,data,crops,output,limit=120,extra_ranks=()):
    """Render real cached poses against the same admitted original RGB/inputs."""
    _state(cache);cache.check();s=_state(cache);destination=_safe(output,exists=False)
    if destination.exists():raise FileExistsError(destination)
    selected=select_rn18_render_rows(cache.ranked_rows(),limit=limit,extra_ranks=extra_ranks)
    request=copy.deepcopy(s['request'])
    if any(_safe(actual)!=_safe(request[name])for name,actual in [('upstream',upstream),('data',data),('crops',crops)]):
        raise ValueError('same original source paths for cached RN18 predictions required')
    request['evidence']=str(ROOT/'results/native-dense-occlusion')
    from .modality_control_run import load_control_evidence
    from .rn18_occlusion_run import admit_dense_run_inputs,_runtime_budget,_write_json,_seal
    from .qualitative_render import assemble_qualitative_panel,render_qualitative_panel,_panel_arrays
    from .emg2pose_memory_process import process_identity
    _runtime_budget(destination);evidence=load_control_evidence(request)
    if evidence.identity['baseline_model_identity_sha256']!=cache.identity['source_model_identity_sha256']:
        raise ValueError('same actual completed RN18 source triplet required')
    inputs=admit_dense_run_inputs(evidence.native,data=data,crops=crops);destination.mkdir(mode=0o700)
    samples=[]
    for choice in selected:
        _runtime_budget(destination);row=choice['row'];index=row['sequence']
        record=json.loads(s['records'][index]);join=json.loads(s['joins'][index])
        sample=inputs[join['dense_sequence']]
        bound=materialize_rn18_cache_row(record,join,cache.row(index),sample)
        panel=assemble_qualitative_panel(bound['sample'],bound['prediction'],row,upstream=upstream)
        base=f"rank_{choice['rank']:04d}_seq_{index:06d}";png=base+'.png';raw=base+'.npz';meta=base+'.json'
        with(destination/raw).open('xb')as f:np.savez_compressed(f,**_panel_arrays(panel))
        caption=f"RN18 | Rank {choice['rank']} | {row['split']} / {row['hand']} | gain {row['delta_mae']*57.3:.2f}°"
        drawn=render_qualitative_panel(panel,destination/png,caption=caption)
        _write_json(destination/meta,dict(drawn,rank=choice['rank'],diagnostic_extra=choice['diagnostic_extra'],
            cache_lineage=bound['cache_lineage'],accepted_cache_proof_sha256=REFERENCE_SHA,
            source_model_identity_sha256=cache.identity['source_model_identity_sha256']))
        samples.append(dict(rank=choice['rank'],key=copy.deepcopy(row['key']),diagnostic_extra=choice['diagnostic_extra'],
            png_file=png,arrays_file=raw,metadata_file=meta,
            png_file_sha256=file_sha256(destination/png),arrays_file_sha256=file_sha256(destination/raw),metadata_file_sha256=file_sha256(destination/meta)))
        cache.check();inputs.check();evidence.check()
    final_hashes=inputs.hash_payloads();cache.check();inputs.check();evidence.check()
    result=_seal(dict(format='actual_RN18_ordinary_cached_RGB_pose_candidate_pool_v1',actor=process_identity(),
        accepted_cache_proof_sha256=REFERENCE_SHA,cache_identity=cache.identity,
        source_model_identity_sha256=cache.identity['source_model_identity_sha256'],
        selected_points=2208,source_filter_pass=2119,rendered_candidates=len(samples),
        default_rank_candidates=sum(not v['diagnostic_extra']for v in samples),diagnostic_extra_candidates=sum(v['diagnostic_extra']for v in samples),
        requested_default_cap=limit,requested_extra_ranks=list(extra_ranks),samples=samples,
        full_source_payload_files=len(inputs.paths),full_source_payload_bytes=inputs.identity['actual_complete_payload_bytes'],
        complete_payload_sha256_after_render=final_hashes,renderer_source_sha256=IMPORTED_SHA,
        accepted_cache_dual_full_raw_readback_verified=True,condition_examples_selected=False,
        physical_rgb_projection_proven=False,figure7_reproduced=False,full_paper_complete=False))
    _write_json(destination/'index.json',result);cache.check();inputs.check();evidence.check();return result
