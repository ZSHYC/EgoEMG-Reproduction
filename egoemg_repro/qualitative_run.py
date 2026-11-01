"""Ordinary Figure7 frozen predictions, exact input joins and fresh readback.

Actual full inputs and completed native trained producers are separate
admissions. Numeric packet helpers never certify either. No sample filtering,
training, zero branch, dense-grid substitution or Figure7 promotion occurs.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import weakref

import numpy as np

from .classic_emg import _hash, _approved_upstream
from .classic_emg_cohort import _safe, _json
from .data import file_sha256
from .native_ik_inputs import _Guard, _nine
from .occlusion_dense_data import _typed_sha
from .occlusion_dense_inference import native_batch_ranges, predict_dense_batch
from .occlusion_dense_predictions import KEY_FIELDS, _key
from .occlusion_dense_run import (MODEL_ID, _seal, _sealed, _write_json, _canonical,
    _closed_workflow, _actor_closed, _cpu_hardware, _managed_loader,
    write_dense_batch, read_dense_batch, compare_dense_batches)

ROOT=Path(__file__).resolve().parents[1]
PROOF_SHA='c61e85df0b043e591ea52d9ab20984183894cbf9dc32dc9e555a50940fcdd312'
PROOF='ordinary-paired-inputs-complete-20261101.json'
IMPORTED_SHA=file_sha256(__file__)
LOCAL=('qualitative_run','qualitative_data','qualitative_inputs','occlusion_dense_run',
    'occlusion_dense_inference','occlusion_dense_models','occlusion_dense_inputs',
    'occlusion_dense_data','native_ik_inputs','emg2pose_memory_process','vision_literal_runtime')
FIELDS=dict(fusion=('emg','joint_angles','label_valid_mask','vision_img','vision_valid_mask','vision_frame_indices'),
    standalone=('emg','joint_angles','label_valid_mask'))
_EVIDENCE=weakref.WeakKeyDictionary();_INPUTS=weakref.WeakKeyDictionary()


class _RunGuard(_Guard):
    def add(self,path):
        self.check();path=_safe(path)
        if path not in self.stamps:
            self.stamps[path]=_nine(path);self.parents.update(path.parents)
        self.check()


def _guard():
    paths={f'egoemg_repro/{n}.py':_safe(ROOT/'egoemg_repro'/(n+'.py'))for n in LOCAL}
    guard=_RunGuard(paths.values(),[])
    if file_sha256(__file__)!=IMPORTED_SHA:raise ValueError('imported ordinary prediction source changed')
    hashes={n:file_sha256(p)for n,p in paths.items()};guard.check()
    return guard,hashes


def verify_qualitative_collated_batch(batch,expected_records):
    """Verify real collated tensor bytes, not just worker-reported hashes."""
    import torch
    guard,_=_guard();expected=list(expected_records);keys=[r['key']for r in expected]
    if (not 1<=len(keys)<=64 or not isinstance(batch,dict)
        or batch.get('keys')!=keys or batch.get('verified_input_sha256')!=[_hash(r['typed_input_sha256'])for r in expected]):
        raise ValueError('exact original batch keys and input fingerprints required')
    fingerprints={'row:'+str(k['sequence']):_hash(r['typed_input_sha256'])for k,r in zip(keys,expected)}
    for role,fields in FIELDS.items():
        payload=batch.get(role)
        if not isinstance(payload,dict)or set(payload)!=set(fields)|{'target_hand'}:
            raise ValueError('literal original collated tensor fields required; no extra features')
        if payload['target_hand']!=[k['hand']for k in keys]:raise ValueError('same original batch hand metadata required')
        for name in fields:
            a=payload[name]
            if (not isinstance(a,torch.Tensor)or a.device.type!='cpu' or a.shape[0]!=len(keys)
                or a.dtype!=(torch.bool if name.endswith('_mask')else torch.int64 if name=='vision_frame_indices'else torch.float32)
                or not torch.isfinite(a).all()):raise ValueError('finite exact CPU native tensor dtype required')
            arrays=a.numpy();prefix='fusion'if role=='fusion'else'standalone'
            if any(_typed_sha(arrays[i])!=r['typed_input_sha256'][prefix+':'+name]for i,r in enumerate(expected)):
                raise ValueError('actual collated bytes differ from admitted original sample')
            fingerprints[prefix+':'+name]=_typed_sha(arrays)
    guard.check();return fingerprints


def read_qualitative_batches(directory,records,*,expected_records):
    """Ordered numeric join, with no actual trained-source admission."""
    guard,_=_guard();directory=_safe(directory);expected=list(expected_records)
    keys=[r['key']for r in expected];ranges=native_batch_ranges(keys);records=list(records)
    if len(records)!=len(ranges):raise ValueError('every original group batch and tail required')
    chunks=[];missing=invalid=0
    for ordinal,((a,b),record)in enumerate(zip(ranges,records)):
        if record.get('ordinal')!=ordinal or record.get('file')!=f'{ordinal:06d}.npz':
            raise ValueError('complete original ordered packet sequence required')
        guard.add(directory/record['file'])
        arrays=read_dense_batch(directory,record,expected_keys=keys[a:b])
        row_names={n for n in record['inputs_sha256']if n.startswith('row:')}
        if row_names!={'row:'+str(k['sequence'])for k in keys[a:b]}:
            raise ValueError('exact complete input fingerprint rows required')
        for j,original in enumerate(expected[a:b]):
            k=original['key'];typed=original['typed_input_sha256']
            if (record['inputs_sha256']['row:'+str(k['sequence'])]!=_hash(typed)
                or _typed_sha(arrays['targets'][j,:,None])!=typed['fusion:joint_angles']
                or _typed_sha(arrays['label_valid_mask'][j])!=typed['fusion:label_valid_mask']
                or _typed_sha(arrays['crop_valid_mask'][j])!=typed['fusion:vision_valid_mask']):
                raise ValueError('saved raw predictions no longer join exact original inputs/targets/masks')
        chunks.append(arrays);missing+=int((~arrays['crop_valid_mask']).sum());invalid+=int((~arrays['label_valid_mask']).sum())
    guard.check()
    return dict(arrays={n:np.concatenate([a[n]for a in chunks])for n in chunks[0]},
        counts=dict(points=len(keys),batches=len(ranges),crop_missing=missing,label_invalid=invalid),
        trained_source_admitted=False,_guard=guard)


def _registered(value,registry,cls):
    if type(value)is not cls or value not in registry:raise ValueError('registered original qualitative evidence/inputs required')
    return registry[value]


@dataclass(frozen=True,eq=False)
class QualitativeEvidence:
    def check(self):
        s=_registered(self,_EVIDENCE,QualitativeEvidence);s['guard'].check()
        if _hash(s['identity'])!=s['identity_hash']:raise ValueError('ordinary prediction evidence identity changed')

    @property
    def identity(self):self.check();return copy.deepcopy(_EVIDENCE[self]['identity'])

    @property
    def upstream(self):self.check();return _EVIDENCE[self]['upstream']

    def records(self):self.check();return [json.loads(line)for line in _EVIDENCE[self]['records']]

    def keys(self):return [r['key']for r in self.records()]


def load_qualitative_evidence(*,evidence,paired_root,upstream):
    guard,sources=_guard();path=_safe(Path(evidence)/PROOF);guard.add(path)
    if file_sha256(path)!=PROOF_SHA:raise ValueError('fixed independently reconstructed ordinary input proof required')
    proof=_json(path);root=_safe(paired_root)
    for name in('private/workflow-result.json','private/execution-state.json','private/stage-receipt.json'):guard.add(root/name)
    # The trusted independent proof anchors actors, bundle and immutable IO
    # identity/journal hashes. The workflow itself supplies current closure.
    workflow=_closed_workflow(root,file_sha256(root/'private/workflow-result.json'),guard)
    state=_json(root/'private/execution-state.json');manifest=_json(root/'private/stage-receipt.json')
    actors=[state[n]for n in('wrapper','child')]
    for stage in workflow['actual_owned_stages']:actors.extend(stage[n]for n in('worker','guardian','owner'))
    actor_keys=lambda rows:{(r['pid'],str(r['start_ticks']))for r in rows}
    if (actor_keys(actors)!=actor_keys(proof['actual_closed_original_actors'])
        or workflow['source_commit']!=proof['source_commit']or manifest['source_commit']!=proof['source_commit']
        or workflow['bundle_sha256']!=proof['bundle_sha256']or manifest['bundle_sha256']!=proof['bundle_sha256']
        or workflow['actual_complete_io_passes']!=4):raise ValueError('same complete actual ordinary source graph required')
    directory=root/'private/vits_s-a';identity_path=directory/'identity.json';journal=directory/'samples.jsonl'
    guard.add(identity_path);guard.add(journal)
    identity=_json(identity_path);accepted=proof['recipes']['vits_s']
    if _hash(identity)!=accepted['identity_sha256']or file_sha256(journal)!=accepted['sample_journal_file_sha256']:
        raise ValueError('complete admitted original ordinary IO identity/journal differs')
    lines=tuple(journal.read_text().splitlines());records=[json.loads(line)for line in lines];keys=[r['key']for r in records]
    ranges=native_batch_ranges(keys)
    if (len(keys)!=2322 or len(ranges)!=38 or identity['engineering']
        or identity['complete_payload_sha256']!=proof['complete_payload_sha256']
        or sum(not r['crop_valid']for r in records)!=80 or sum(not r['label_valid']for r in records)!=64):
        raise ValueError('all2322 original ordinary requests and retained masks required')
    upstream=_approved_upstream(upstream)
    scripts=[upstream/'scripts/eval'/n for n in('find_fusion_wins.py','evaluate_occlusion_fusion.py')]
    nodes=[]
    for p in scripts:
        guard.add(p);tree=ast.parse(p.read_bytes())
        nodes.append(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_collate_fusion'))
    if ast.dump(nodes[0],include_attributes=False)!=ast.dump(nodes[1],include_attributes=False):
        raise ValueError('original ordinary/dense executable collation semantics differ')
    value=dict(format='admitted_complete_original_ordinary_prediction_evidence_v1',recipe='vits_s',
        accepted_input_proof_sha256=PROOF_SHA,paired_input_identity=identity,source_files_sha256=sources,
        complete_payload_sha256=proof['complete_payload_sha256'],source_keys_sha256=_hash(keys),
        complete_io_journal_file_sha256=file_sha256(journal),batch_ranges=[list(r)for r in ranges],
        selected_hand_points=2322,crop_missing_retained=80,label_invalid_retained=64,
        source_model_identity_sha256=MODEL_ID,ordinary_dense_collation_AST_exact=True,
        trained_model_or_inference_permit=False,figure7_reproduced=False,full_paper_complete=False)
    result=QualitativeEvidence();_EVIDENCE[result]=dict(guard=guard,identity=value,identity_hash=_hash(value),records=lines,upstream=upstream)
    result.check();return result


@dataclass(frozen=True,eq=False)
class QualitativeRunInputs:
    def check(self):
        s=_registered(self,_INPUTS,QualitativeRunInputs);s['evidence'].check();s['dataset'].check()
        if _hash(s['identity'])!=s['identity_hash']:raise ValueError('ordinary run input identity changed')

    @property
    def evidence(self):self.check();return _INPUTS[self]['evidence']

    @property
    def identity(self):self.check();return copy.deepcopy(_INPUTS[self]['identity'])

    def __len__(self):self.check();return len(_INPUTS[self]['records'])

    def __getitem__(self,index):
        self.check();s=_INPUTS[self]
        if type(index)is not int or not 0<=index<len(s['records']):raise IndexError(index)
        packet=s['dataset'][index];expected=json.loads(s['records'][index])
        typed={'fusion:'+n:_typed_sha(packet['fusion'][n])for n in(*FIELDS['fusion'],'emg_channel_mask')}
        typed.update({'standalone:'+n:_typed_sha(packet['standalone_emg'][n])for n in(*FIELDS['standalone'],'emg_channel_mask')})
        if packet['key']!=expected['key']or typed!=expected['typed_input_sha256']:
            raise ValueError('fresh original ordinary window tensors differ from independent complete IO')
        self.check();return dict(packet,verified_input_sha256=_hash(typed))

    def hash_payloads(self):self.check();return _INPUTS[self]['dataset'].hash_payloads()


def admit_qualitative_run_inputs(evidence,*,data,crops):
    _registered(evidence,_EVIDENCE,QualitativeEvidence);evidence.check()
    from .fusion_literal import compose_fusion
    from .qualitative_data import admit_qualitative_selection
    from .qualitative_inputs import admit_qualitative_inputs
    cfg=compose_fusion(evidence.upstream)
    selected=admit_qualitative_selection(data,upstream=evidence.upstream,config=cfg,recipe='vits_s')
    ds=admit_qualitative_inputs(selected,data=data,crops=crops,upstream=evidence.upstream,config=cfg,recipe='vits_s')
    if ds.identity!=evidence.identity['paired_input_identity']:
        raise ValueError('same full original ordinary selection, payloads and branch normalizations required')
    identity=_seal(dict(format='actual_ordinary_complete_frozen_input_admission_v1',
        evidence_identity_sha256=_hash(evidence.identity),paired_input_identity_sha256=_hash(ds.identity),
        complete_payload_sha256=ds.identity['complete_payload_sha256'],
        complete_pre_forward_payload_hashes_verified=True,actual_complete_payload_files=54,
        actual_complete_payload_bytes=61379659366,data=str(_safe(data)),crops=str(_safe(crops)),
        trained_model_or_inference_permit=False,figure7_reproduced=False,full_paper_complete=False))
    result=QualitativeRunInputs();_INPUTS[result]=dict(evidence=evidence,dataset=ds,identity=identity,
        identity_hash=_hash(identity),records=tuple(json.dumps(r,sort_keys=True)for r in evidence.records()))
    result.check();return result


def read_qualitative_pass(directory,*,evidence):
    _registered(evidence,_EVIDENCE,QualitativeEvidence);evidence.check();directory=_safe(directory)
    guard,_=_guard()
    for name in('identity.json','result.json','batches.jsonl'):guard.add(directory/name)
    identity=_sealed(_json(directory/'identity.json'));result=_sealed(_json(directory/'result.json'))
    p=evidence.identity;admission=_sealed(identity['input_admission']);model=_sealed(identity['model_identity'])
    if (identity['format']!='actual_original_ordinary_frozen_pass_v1'
        or identity['evidence_identity_sha256']!=_hash(p)or identity['source_files_sha256']!=p['source_files_sha256']
        or model['sha256']!=MODEL_ID or identity['batch_ranges']!=p['batch_ranges']
        or identity['hardware']!=_cpu_hardware()or result['identity_sha256']!=identity['sha256']
        or admission['evidence_identity_sha256']!=_hash(p)
        or admission['paired_input_identity_sha256']!=_hash(p['paired_input_identity'])
        or admission['complete_payload_sha256']!=p['complete_payload_sha256']
        or not admission['complete_pre_forward_payload_hashes_verified']
        or(result['points'],result['batches'],result['crop_missing'],result['label_invalid'])!=(2322,38,80,64)
        or result['complete_payload_sha256_after_forward']!=p['complete_payload_sha256']
        or result['journal_file_sha256']!=file_sha256(directory/'batches.jsonl')
        or result['model_state_sha256_before']!=model['model_state_sha256']
        or result['model_state_sha256_after']!=result['model_state_sha256_before']
        or not result['all_model_states_before_after_exact']or result['actual_optimizer_updates']!=0 or result['cuda_initialized']):
        raise ValueError('complete source-bound original ordinary trained prediction pass required')
    _actor_closed(result['actor'])
    groups=result['loader_groups'];wanted=[(s,h)for s in('user','gesture','both')for h in('left','right')]
    if [(g['split'],g['hand'])for g in groups]!=wanted:raise ValueError('all six original source loader groups required')
    keys=evidence.keys()
    for group in groups:
        selected=[k for k in keys if(k['split'],k['hand'])==(group['split'],group['hand'])]
        if (len(group['workers'])!=4 or not group['all_workers_exited']or group['points']!=len(selected)
            or group['batches']!=(len(selected)+63)//64):raise ValueError('all actual original B64/4worker groups and tails required')
        for actor in group['workers']:_actor_closed(actor)
    records=[json.loads(line)for line in(directory/'batches.jsonl').read_text().splitlines()]
    value=read_qualitative_batches(directory/'batches',records,expected_records=evidence.records())
    guard.check();value['_guard'].check();evidence.check()
    return dict(directory=str(directory),identity=identity,result=result,records=records,arrays=value['arrays'],
        _guard=guard,_packet_guard=value['_guard'],_result_file_sha256=file_sha256(directory/'result.json'))


def compare_qualitative_passes(first,second,*,evidence):
    a=read_qualitative_pass(first,evidence=evidence);b=read_qualitative_pass(second,evidence=evidence)
    if (a['directory']==b['directory']or a['identity']!=b['identity']
        or(a['result']['actor']['pid'],a['result']['actor']['start_ticks'])==(b['result']['actor']['pid'],b['result']['actor']['start_ticks'])
        or not b['result']['fresh_model_forward_replay_verified']
        or b['result']['replay_of_result_file_sha256']!=a['_result_file_sha256']
        or b['result']['replay_of_identity_sha256']!=a['identity']['sha256']):
        raise ValueError('distinct fresh producing actors and exact preceding raw pass required')
    for n in a['arrays']:
        if a['arrays'][n].dtype!=b['arrays'][n].dtype or a['arrays'][n].tobytes()!=b['arrays'][n].tobytes():
            raise ValueError('fresh original ordinary raw prediction arrays differ')
    for item in(a,b):item['_guard'].check();item['_packet_guard'].check()
    return dict(points=2322,batches_per_pass=38,all_ten_raw_arrays_exact=True,
        model_identity_sha256=MODEL_ID,complete_frozen_predictions_verified=True,
        figure7_reproduced=False,scientific_eligible=False,full_paper_complete=False)


def run_qualitative_pass(inputs,fusion_fit,output,*,replay_of=None):
    _registered(inputs,_INPUTS,QualitativeRunInputs);inputs.check();evidence=inputs.evidence
    previous=read_qualitative_pass(replay_of,evidence=evidence)if replay_of is not None else None
    import torch
    from .occlusion_dense_models import load_dense_trained_models
    from .vision_literal_runtime import vision_state_sha256
    from .emg2pose_memory_process import process_identity
    torch.set_num_threads(2);trained=load_dense_trained_models(fusion_fit,upstream=evidence.upstream,recipe='vits_s')
    p=inputs.identity
    if(trained.identity['sha256']!=MODEL_ID or _safe(trained.data_locations['data'])!=Path(p['data'])
        or _safe(trained.data_locations['crops'])!=Path(p['crops'])):raise ValueError('actual completed native triplet and exact data roots required')
    inputs.check();trained.check();keys=evidence.keys();records=evidence.records();ranges=native_batch_ranges(keys)
    output=_safe(output,exists=False);output.mkdir(mode=0o700);(output/'batches').mkdir(mode=0o700)
    identity=_seal(dict(format='actual_original_ordinary_frozen_pass_v1',
        evidence_identity_sha256=_hash(evidence.identity),input_admission=p,
        source_files_sha256=evidence.identity['source_files_sha256'],model_identity=copy.deepcopy(trained.identity),
        batch_ranges=[list(r)for r in ranges],hardware=_cpu_hardware(),
        scientific_eligible=False,figure7_reproduced=False,full_paper_complete=False))
    if previous is not None and previous['identity']!=identity:raise ValueError('same complete ordinary source/model/batch identity required for replay')
    _write_json(output/'identity.json',identity);started=time.monotonic()
    before={n:vision_state_sha256(m.state_dict())for n,m in trained.models.items()}
    ordinal=points=missing=invalid=0;groups=[]
    with(output/'batches.jsonl').open('x')as journal:
        for split in('user','gesture','both'):
            for hand in('left','right'):
                current=[(a,b)for a,b in ranges if(keys[a]['split'],keys[a]['hand'])==(split,hand)]
                indices=[i for a,b in current for i in range(a,b)];count=0
                with _managed_loader(inputs,indices,upstream=evidence.upstream)as(iterator,workers,actual_pin):
                    for batch in iterator:
                        if count>=len(current):raise ValueError('extra original ordinary source batch')
                        a,b=current[count];fingerprints=verify_qualitative_collated_batch(batch,records[a:b])
                        predicted=predict_dense_batch(trained,batch['fusion'],batch['standalone'])
                        arrays=dict(predicted['arrays'],**predicted['errors'],crop_valid_mask=batch['fusion']['vision_valid_mask'].numpy().copy())
                        record=write_dense_batch(output/'batches',ordinal,keys=keys[a:b],arrays=arrays,inputs_sha256=fingerprints)
                        read_dense_batch(output/'batches',record,expected_keys=keys[a:b])
                        if previous is not None:
                            previous['_guard'].check();previous['_packet_guard'].check()
                            compare_dense_batches(Path(previous['directory'])/'batches',previous['records'][ordinal],output/'batches',record,expected_keys=keys[a:b])
                        journal.write(_canonical(record)+'\n');journal.flush();os.fsync(journal.fileno())
                        ordinal+=1;count+=1;points+=b-a
                        missing+=int((~arrays['crop_valid_mask']).sum());invalid+=int((~arrays['label_valid_mask']).sum())
                        progress=dict(batches=ordinal,total_batches=38,points=points,total_points=2322,split=split,hand=hand,seconds=time.monotonic()-started)
                        (output/'progress.json').write_text(json.dumps(progress,sort_keys=True)+'\n');print(json.dumps(progress),flush=True)
                        inputs.check()
                if count!=len(current):raise ValueError('missing original ordinary group tail')
                for actor in workers:_actor_closed(actor)
                groups.append(dict(split=split,hand=hand,workers=workers,all_workers_exited=True,
                    pin_memory_requested=True,actual_pin_memory=actual_pin,points=len(indices),batches=count))
    inputs.check();trained.check();after={n:vision_state_sha256(m.state_dict())for n,m in trained.models.items()}
    if before!=after or torch.cuda.is_initialized()or(ordinal,points,missing,invalid)!=(38,2322,80,64):
        raise ValueError('complete unchanged frozen models and all original ordinary inputs required')
    payloads=inputs.hash_payloads()
    if previous is not None:previous['_guard'].check();previous['_packet_guard'].check()
    result=_seal(dict(format='actual_original_ordinary_frozen_pass_result_v1',identity_sha256=identity['sha256'],
        actor=process_identity(),points=points,batches=ordinal,crop_missing=missing,label_invalid=invalid,
        journal_file_sha256=file_sha256(output/'batches.jsonl'),loader_groups=groups,
        model_state_sha256_before=before,model_state_sha256_after=after,all_model_states_before_after_exact=True,
        complete_payload_sha256_after_forward=payloads,fresh_model_forward_replay_verified=previous is not None,
        replay_of_result_file_sha256=previous['_result_file_sha256']if previous is not None else None,
        replay_of_identity_sha256=previous['identity']['sha256']if previous is not None else None,
        actual_optimizer_updates=0,cuda_initialized=False,seconds=time.monotonic()-started,
        scientific_eligible=False,figure7_reproduced=False,full_paper_complete=False))
    _write_json(output/'result.json',result);return result
