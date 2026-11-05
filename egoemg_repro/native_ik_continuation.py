"""Read-only admission of genuinely completed, interrupted original IK batches.

Full mode pins the published 68-batch audit and original failed capsule.
Engineering carriers exercise persistence only: neither mode grants geometry,
optimization or label-completion authority merely by opening old packets.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import stat
import weakref

from .classic_emg_cohort import _safe
from .data import file_sha256
from .emg2pose_memory_process import process_identity
from .native_ik_inputs import _hash, HANDS
from . import native_ik_results as results

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'results/native-mano-ik/interrupted-prefix-raw-verified-20261104.json'
REFERENCE_SHA='22b00fb5a7e35e47e8021a015a8f289303d177f8d71ea0683dca4afdad84928c'
FULL_ORIGINAL_ROOT=Path('/data/shihao/egoemg-native-ik-full-data-bdf59e746e0d-v2-20261102')
IMPORTED_SHA=file_sha256(__file__)
_PREFIXES=weakref.WeakKeyDictionary()


def _closed(actor):
    # Historical outer recorders carry PID/start only; packet actors also carry
    # uid/session/pgrp and are checked by the unchanged original inspector.
    if (type(actor)is not dict or type(actor.get('pid'))is not int or actor['pid']<1
        or type(actor.get('start_ticks'))is not str or not actor['start_ticks'].isdigit()):
        raise ValueError('actual original prefix actor required')
    current=process_identity(actor['pid'])
    if current is not None and current['start_ticks']==actor['start_ticks']:
        raise ValueError('original prefix actor must actually exit')


def _json(path,bindings,expected=None,*,sealed=False):
    bindings.add_file(path,expected)
    value=json.loads(path.read_text(),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite prefix JSON')))
    bindings.check()
    if type(value)is not dict or(sealed and value.get('sha256')!=_hash({k:v for k,v in value.items()if k!='sha256'})):
        raise ValueError('complete original prefix JSON/seal required')
    return value


def _inventory(root):
    """Bind directory generations, trailing partial files and the entire prefix.

    Only paths within the old isolated root are watched; sibling jobs may still
    create their own outputs without invalidating this interrupted workflow.
    """
    flow=root/'private/full-label-workflow/flow';directories=[];partial=[];complete=[]
    def directory(p):
        p=_safe(p);s=p.lstat()
        if not stat.S_ISDIR(s.st_mode):raise ValueError('original prefix directory required')
        directories.append((str(p),(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink)))
    for p in (root,root/'private',flow.parent,flow):directory(p)
    allowed={f'{hand}-shard-{shard}'for hand in HANDS for shard in range(6)}
    if any(p.name not in allowed for p in flow.iterdir()):raise ValueError('unexpected original prefix logical shard')
    for hand in HANDS:
        for shard in range(6):
            branch=flow/f'{hand}-shard-{shard}'
            if not branch.exists():continue
            directory(branch);base=branch/'produce'
            if not base.exists():raise ValueError('original prefix producer directory missing')
            directory(base)
            for p in sorted(base.iterdir()):
                if not p.name.startswith('batch-')or not p.name[6:].isdigit()or p.name!=f'batch-{int(p.name[6:]):05d}':
                    raise ValueError('exact original prefix packet directory required')
                directory(p);index=int(p.name[6:]);names={q.name for q in p.iterdir()}
                if names-{'state.npz','result.json'}:raise ValueError('unexpected original prefix packet file')
                if names=={'state.npz','result.json'}:complete.append((hand,shard,index,p))
                else:partial.extend(sorted(p.iterdir()))
    return tuple(directories),tuple(complete),tuple(partial)


def _state(prefix):
    if type(prefix)is not IKPrefix or prefix not in _PREFIXES:
        raise ValueError('registered original IK prefix required')
    return _PREFIXES[prefix]


@dataclass(frozen=True,eq=False)
class IKPrefix:
    def check(self,*,rehash=False):
        s=_state(self);s['inputs'].check();s['bindings'].check(rehash=rehash)
        if (file_sha256(__file__)!=IMPORTED_SHA or _hash(s['identity'])!=s['identity_hash']
            or _inventory(s['root'])!=s['inventory']):
            raise ValueError('original IK prefix source, identity or generation changed')
        for actor in s['actors']:_closed(actor)
        if s['sources']is not None:s['sources'].check()
        if s['qualification']is not None:s['qualification'].check(rehash=rehash)
        s['bindings'].check()

    @property
    def identity(self):
        self.check();return copy.deepcopy(_state(self)['identity'])

    def batch_prefix(self,hand,shard):
        self.check()
        if hand not in HANDS or type(shard)is not int or not 0<=shard<6:
            raise ValueError('explicit right/left hand and original shard0..5 required')
        return _state(self)['batches'][(hand,shard)]


def admit_ik_prefix(inputs,original_root,*,proof=None,engineering=False):
    """Authenticate and guard the old contiguous prefix without writing it."""
    if type(engineering)is not bool:raise ValueError('explicit engineering boolean required')
    proof_path=_safe(proof or REFERENCE)
    bindings=results.IKArtifactBindings()
    accepted=_json(proof_path,bindings,None if engineering else REFERENCE_SHA,sealed=engineering)
    if not engineering and file_sha256(REFERENCE)!=REFERENCE_SHA:
        raise ValueError('fixed pinned full prefix proof required')
    inputs.check();identity=inputs.identity
    if identity['engineering']is not engineering or identity['full_available_release']is engineering:
        raise ValueError('same registered original full/engineering IK input scope required')
    root=_safe(original_root)
    if not engineering and root!=FULL_ORIGINAL_ROOT:
        raise ValueError('exact original failed full prefix root required')
    if engineering:
        if (accepted.get('format')!='original_native_ik_prefix_engineering_IO_audit_v1'
            or accepted.get('engineering')is not True):raise ValueError('explicit engineering IO prefix proof required')
    elif (accepted.get('format')!='actual_interrupted_original_native_IK68batch_prefix_postexit_raw_read_v1'
        or accepted['actual_completed_original_batches']!=68 or accepted['actual_completed_original_rows']!=686528):
        raise ValueError('fixed actual complete 68-batch prefix proof required')
    if any(accepted.get(k)is not False for k in ('full_frame_labels_regenerated','scientific_eligible','full_paper_complete')):
        raise ValueError('prefix IO cannot grant geometry, labels or full scientific completion')
    if (accepted['original_input_identity_sha256']!=_hash(identity)
        or accepted['original_batch_size']!=identity['batch_size']):
        raise ValueError('same original prefix input identity and batch size required')
    state=_json(root/'private/execution-state.json',bindings,sealed=engineering)
    if state!=accepted['original_terminal_state']or state['status']!='failed'or state['actual_exit_code']!=1:
        raise ValueError('same actually stopped original workflow required')
    actors=copy.deepcopy(accepted['all_recorded_original_owning_actors_physically_closed'])
    if not actors:raise ValueError('original prefix owners required')
    if not engineering:
        actors.extend([accepted['collector_actor'],accepted['transport_collector_actor']])
        actors.extend(accepted['postexit_raw_reader_workflow_terminal_state'][k]for k in ('wrapper','child'))
        manifest=_json(root/'private/stage-receipt.json',bindings)
        if (manifest['source_commit']!=accepted['original_source_commit']
            or manifest['bundle_sha256']!=accepted['original_source_bundle_sha256']
            or _hash({k:v for k,v in manifest.items()if k!='bundle_sha256'})!=accepted['original_source_bundle_sha256']):
            raise ValueError('complete actual original source capsule body differs')
        for name,digest in manifest['files_sha256'].items():
            p=Path(name)
            if p.is_absolute()or'..'in p.parts:raise ValueError('original capsule member must stay within root')
            bindings.add_file(_safe(root/p),digest)
        bindings.add_file(REFERENCE,REFERENCE_SHA)
    for actor in actors:_closed(actor)
    config_path=root/'private/full-label-workflow/configuration.json'
    config=_json(config_path,bindings,accepted['original_configuration_file_sha256'],sealed=True)
    if (config['action']!='workflow' or config['request']['engineering']is not engineering
        or config['request']['batch_size']!=identity['batch_size']
        or config['input_identity_sha256']!=_hash(identity)
        or config['source_identity_sha256']!=accepted['original_native_source_identity_sha256']
        or config['qualified_runtime_identity']!=accepted['original_qualified_runtime_identity']
        or config['qualification_file_sha256']!=accepted['original_qualification_file_sha256']):
        raise ValueError('exact original prefix configuration/runtime required')
    sources=qualification=None
    if not engineering:
        from .native_ik_runtime import admit_ik_sources
        from scripts.paper.run_native_mano_ik import _check_qualification,workflow_sources
        if config['sources_sha256']!=workflow_sources():raise ValueError('original qualified scientific source snapshot changed')
        sources=admit_ik_sources(*(config['request'][n]for n in ('upstream','manotorch','chumpy','mano_assets')))
        if _hash(sources.identity)!=accepted['original_native_source_identity_sha256']:
            raise ValueError('actual original native source/assets identity differs')
        sha,qualification,runtime=_check_qualification(config['request'],inputs,sources,config['physical_device'],workflow_sources())
        if sha!=accepted['original_qualification_file_sha256']or runtime!=accepted['original_qualified_runtime_identity']:
            raise ValueError('same original closed native10096 qualification required')
    inventory=_inventory(root);actual=inventory[1];records=accepted['original_batches']
    hand,shard=accepted['actual_hand'],accepted['actual_shard']
    expected=[(hand,shard,index)for index in range(accepted['actual_completed_original_batches'])]
    if ([a[:3]for a in actual]!=expected or len(records)!=len(expected)
        or accepted['first_uncompleted_batch_index']!=len(expected)
        or [(hand,shard,a.get('batch_index'))for a in records]!=expected):
        raise ValueError('complete ordered contiguous original prefix required; holes or extras forbidden')
    if not expected:raise ValueError('nonempty completed original prefix required')
    rows=0;batches={(h,s):()for h in HANDS for s in range(6)}
    for record,(_,_,index,directory)in zip(records,actual):
        for name,field in [('result.json','result_file_sha256'),('state.npz','state_file_sha256')]:
            target=directory/name
            if target.lstat().st_nlink!=1:raise ValueError('old IK packet must not acquire hardlinks')
            bindings.add_file(target,record[field])
        packet=results.inspect_ik_batch(inputs,directory);meta=packet.metadata
        selected=results.select_ik_batch(inputs,hand,shard,index).identity
        if (meta['batch']!=selected or meta['actor']!=record['actual_original_producer_actor']
            or not any(meta['actor']==actor for actor in actors)
            or meta['runtime_identity']!=accepted['original_qualified_runtime_identity']
            or any(selected[k]!=record[k]for k in ('rows','start','end'))
            or meta['record']['actual_ik_optimizer_steps']!=record['original_native_optimizer_steps']
            or meta['record']['actual_ik_closure_evaluations']!=record['original_native_closure_evaluations']
            or meta['statistics']['raw_zero_pose_rows']!=record['raw_zero_pose_rows']):
            raise ValueError('original prefix packet actor/runtime/membership/work differs')
        packet.check(rehash=True);rows+=selected['rows'];batches[(hand,shard)]+= (directory,)
    if rows!=accepted['actual_completed_original_rows']:raise ValueError('original prefix row count differs')
    for path in inventory[2]:bindings.add_file(_safe(path))
    bindings.add_file(_safe(__file__),IMPORTED_SHA);bindings.check(rehash=True);inputs.check()
    value=dict(format='registered_readonly_original_IK_prefix_v1',original_root=str(root),
        prefix_proof_file_sha256=file_sha256(proof_path),
        input_identity_sha256=_hash(identity),original_configuration_file_sha256=accepted['original_configuration_file_sha256'],
        original_source_bundle_sha256=accepted['original_source_bundle_sha256'],
        runtime_identity=copy.deepcopy(accepted['original_qualified_runtime_identity']),
        reused_batches=len(records),reused_rows=rows,engineering=engineering,full_available_release=not engineering,
        native_optimizer_execution_verified=False,full_frame_labels_regenerated=False,scientific_eligible=False,full_paper_complete=False)
    prefix=IKPrefix();_PREFIXES[prefix]=dict(inputs=inputs,root=root,bindings=bindings,inventory=inventory,
        identity=value,identity_hash=_hash(value),batches=batches,actors=actors,sources=sources,qualification=qualification,
        original_configuration=copy.deepcopy(config),audit=copy.deepcopy(accepted))
    prefix.check();return prefix
