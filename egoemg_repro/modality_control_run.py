"""Paired full-population control sources and lossless condition packets.

Input and packet admission are separate from learned execution. Neither an
engineering packet nor completed original predictions certify a control run.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import weakref

import numpy as np

from .classic_emg import _hash
from .classic_emg_cohort import _safe, _StatGuard, _json
from .occlusion_dense_data import _typed_sha
from .occlusion_dense_inference import native_batch_ranges
from .occlusion_dense_run import _seal, _sealed, _canonical

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = tuple((e,v) for e in ('aligned','normalized_zero','half_window_roll') for v in ('native','missing'))
LOCAL = ('modality_controls','modality_control_inference','modality_control_run')
REQUIRED = ('recipe','evidence','upstream','index_root','paired_root','geometry_root','baseline','data','crops','fit')
_SOURCES = weakref.WeakKeyDictionary()


def _sha(path):
    with Path(path).open('rb')as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


IMPORTED_SHA = _sha(__file__)


def _route(recipe):
    if recipe not in ('vits_s','rn18_s'):
        raise ValueError('explicit original ViTS or RN18 control recipe required')
    return importlib.import_module('.'+('occlusion_dense_run'if recipe=='vits_s'else'rn18_occlusion_run'),__package__)


def _accepted_baseline(recipe):
    if recipe=='rn18_s':
        # Pin its real postexit proof only after the live original workflow and
        # collector finish. A qualification milestone is not a full baseline.
        raise ValueError('completed RN18 original baseline pending; full postexit proof required')
    if recipe!='vits_s':
        raise ValueError('explicit original control recipe required')
    path=_safe(ROOT/'results/native-dense-occlusion/vits-complete-prediction-report-20261101.json')
    if _sha(path)!='337e10528f518a9cf2fbfaa7fd560fd57b6dd25a2d272d110a737e9991233c18':
        raise ValueError('fixed actual completed original baseline proof bytes required')
    return path,_json(path)


def _source_guard():
    paths=[ROOT/'egoemg_repro'/(n+'.py')for n in LOCAL]
    if _sha(__file__)!=IMPORTED_SHA:
        raise ValueError('imported control workflow source changed')
    return _StatGuard(paths),{str(p.relative_to(ROOT)):_sha(p)for p in paths}


def _closed(handle):
    from .emg2pose_memory_process import process_identity
    if (not isinstance(handle,dict)or type(handle.get('pid'))is not int
        or not str(handle.get('start_ticks','')).isdigit()):
        raise ValueError('exact original source actor identity required')
    current=process_identity(handle['pid'])
    if current is not None and current['start_ticks']==str(handle['start_ticks']):
        raise ValueError('original baseline source actor still present')


@dataclass(eq=False)
class ControlEvidence:
    native:object
    request:dict
    baseline_records:tuple
    identity:dict
    _guard:object

    def check(self):
        saved=_SOURCES.get(self)
        if saved is None:
            raise ValueError('registered actual complete control evidence required')
        if (self.native is not saved['native']or self._guard is not saved['guard']
            or self.baseline_records is not saved['records']or self.request!=saved['request']
            or _sealed(self.identity)['sha256']!=saved['identity']):
            raise ValueError('control source identity, providers, guard or fixed request changed')
        self._guard.check();self.native.check()

    def keys(self):
        self.check()
        return self.native.keys()

    def baseline_packet(self,ordinal):
        self.check()
        if type(ordinal)is not int or not 0<=ordinal<len(self.baseline_records):
            raise ValueError('exact complete native baseline batch ordinal required')
        ranges=native_batch_ranges(self.keys());start,stop=ranges[ordinal]
        record=json.loads(self.baseline_records[ordinal])
        value=_route(self.identity['recipe']).read_dense_batch(
            Path(self.request['baseline'])/'full-a/batches',record,expected_keys=self.keys()[start:stop])
        self.check()
        return value


def load_control_evidence(request):
    """Authenticate complete original sources without importing Torch/models."""
    if not isinstance(request,dict)or set(request)!=set(REQUIRED):
        raise ValueError('complete fixed control source request required; no subset or batch overrides')
    recipe=request['recipe'];proof_path,proof=_accepted_baseline(recipe)
    guard,sources=_source_guard();guard.add(proof_path)
    route=_route(recipe)
    native=route.load_dense_evidence(**{n:request[n]for n in('evidence','upstream','index_root','paired_root','geometry_root')})
    if (not proof['available_legacy_figure13_pipeline_complete']or proof['actual_exit_code']!=0
        or proof['actual_complete_prediction_passes']!=2
        or proof['actual_complete_points_each']!=len(native.keys())
        or proof['model_identity_sha256']!=native.identity['expected_trained_model_identity_sha256']
        or proof['evidence_identity_sha256']!=native.identity['sha256']
        or not proof['postexit_complete_raw_prediction_replay_exact']
        or not proof['postexit_full_source_keys_targets_flags_errors_verified']
        or not proof['postexit_collector_also_exited']):
        raise ValueError('actual complete original prediction/replay and postexit acceptance required')
    for actor in [*proof['actual_closed_owning_and_dataworker_identities'],proof['collector_actor']]:
        _closed(actor)
    normalized={n:str(_safe(request[n]))for n in REQUIRED if n!='recipe'}
    normalized['recipe']=recipe;baseline=Path(normalized['baseline'])
    archives=proof['private_raw_artifact_sha256']
    for name,digest in archives.items():
        path=_safe(baseline/name);guard.add(path)
        if _sha(path)!=digest:
            raise ValueError('original complete baseline packet or metadata bytes differ: '+name)
    keys=native.keys();ranges=native_batch_ranges(keys);passes=[]
    for label in('full-a','full-b'):
        identity=_sealed(_json(baseline/label/'identity.json'))
        result=_sealed(_json(baseline/label/'result.json'))
        records=[_sealed(json.loads(s))for s in(baseline/label/'batches.jsonl').read_text().splitlines()]
        if (identity['scope']!='complete'or identity['evidence_identity_sha256']!=native.identity['sha256']
            or identity['model_identity']['sha256']!=proof['model_identity_sha256']
            or identity['source_files_sha256']!=native.identity['source_files_sha256']
            or identity['batch_ranges']!=[list(r)for r in ranges]
            or result['identity_sha256']!=identity['sha256']or result['points']!=len(keys)
            or result['batches']!=len(ranges)or len(records)!=len(ranges)
            or result['journal_file_sha256']!=_sha(baseline/label/'batches.jsonl')
            or not result['all_model_states_before_after_exact']
            or result['model_state_sha256_before']!=identity['model_identity']['model_state_sha256']
            or result['model_state_sha256_after']!=result['model_state_sha256_before']):
            raise ValueError('complete original baseline identity/population/state differs')
        _closed(result['actor'])
        for ordinal,((start,stop),record)in enumerate(zip(ranges,records)):
            if (record['ordinal']!=ordinal or record['file']!=f'{ordinal:06d}.npz'
                or record['keys']!=keys[start:stop]):
                raise ValueError('full native baseline key ordering differs')
            for key in record['keys']:
                expected=json.loads(native.records[key['sequence']])['typed_input_sha256']
                if record['inputs_sha256']['row:'+str(key['sequence'])]!=_hash(expected):
                    raise ValueError('baseline and original complete input fingerprints differ')
        passes.append((identity,result,records))
    first,second=passes
    if first[0]!=second[0]or not second[1]['fresh_model_forward_replay_verified']:
        raise ValueError('same complete native baseline and fresh replay required')
    for left,right in zip(first[2],second[2]):
        if (left['arrays_sha256']!=right['arrays_sha256']or left['inputs_sha256']!=right['inputs_sha256']):
            raise ValueError('original complete baseline raw replay differs')
    identity=_seal(dict(format='registered_original_full_control_sources_v1',recipe=recipe,
        native_evidence_identity_sha256=native.identity['sha256'],baseline_proof_sha256=_sha(proof_path),
        baseline_model_identity_sha256=proof['model_identity_sha256'],original_points=len(keys),
        original_batches=len(ranges),condition_order=[list(x)for x in CONDITIONS],request=normalized,
        all_baseline_payload_sha256_verified=True,baseline_artifact_sha256=copy.deepcopy(archives),
        source_files_sha256=sources,full_inference_permit=False,full_paper_complete=False))
    value=ControlEvidence(native,copy.deepcopy(normalized),tuple(_canonical(r)for r in first[2]),identity,guard)
    _SOURCES[value]=dict(native=native,guard=guard,records=value.baseline_records,
        request=copy.deepcopy(normalized),identity=identity['sha256'])
    value.check()
    return value


ARRAYS=('fusion_prediction','vision_prediction','targets','label_valid_mask',
    'natural_crop_valid_mask','condition_vision_valid_mask','mae_fusion_rad','mae_vision_rad')


def _validate_control_arrays(arrays,keys,baseline):
    import torch
    if not keys:
        raise ValueError('nonempty native control packet required')
    length=keys[0]['window_end']-keys[0]['window_start']
    recipe='vits_s'if length==7790 else'rn18_s'if length==12000 else None
    _route(recipe)._validate_arrays(baseline,keys)
    size=len(keys)
    if not isinstance(arrays,dict)or set(arrays)!=set(ARRAYS):
        raise ValueError('all eight original control arrays required')
    for name in ARRAYS:
        shape=((6,size,22)if name.endswith('_prediction')else(6,size,1)if name=='condition_vision_valid_mask'
            else(size,1)if name.endswith('_mask')else(6,size)if name.startswith('mae_')else(size,22))
        dtype=np.dtype(bool if name.endswith('_mask')else np.float64 if name.startswith('mae_')else np.float32)
        x=arrays[name]
        if not isinstance(x,np.ndarray)or x.shape!=shape or x.dtype!=dtype or not np.isfinite(x).all():
            raise ValueError('finite complete original control shapes/dtypes required: '+name)
    for name,old in(('targets','targets'),('label_valid_mask','label_valid_mask'),('natural_crop_valid_mask','crop_valid_mask')):
        if not np.array_equal(arrays[name],baseline[old]):
            raise ValueError('same complete native baseline targets and natural masks required')
    for c,(_,vision)in enumerate(CONDITIONS):
        expected=baseline['crop_valid_mask']if vision=='native'else np.zeros_like(baseline['crop_valid_mask'])
        if not np.array_equal(arrays['condition_vision_valid_mask'][c],expected):
            raise ValueError('artificial vision flags changed original label validity or condition')
        for role in('fusion','vision'):
            expected=np.array([float((torch.from_numpy(arrays[role+'_prediction'][c,i])
                -torch.from_numpy(arrays['targets'][i])).abs().mean())for i in range(size)],np.float64)
            if not np.array_equal(expected,arrays['mae_'+role+'_rad'][c]):
                raise ValueError('literal unmasked22-angle raw control error differs')
    for role in('fusion','vision'):
        if (not np.array_equal(arrays[role+'_prediction'][0],baseline[role+'_prediction'])
            or not np.array_equal(arrays['mae_'+role+'_rad'][0],baseline['mae_'+role+'_rad'])):
            raise ValueError('aligned native condition differs from actual original complete baseline')
    for c in range(2,6):
        if not np.array_equal(arrays['vision_prediction'][c],arrays['vision_prediction'][c%2]):
            raise ValueError('same vision prediction must not depend on changed EMG')


def write_control_batch(directory,ordinal,*,keys,arrays,baseline,baseline_ordinal,inputs_sha256):
    _validate_control_arrays(arrays,keys,baseline)
    if (type(ordinal)is not int or ordinal<0 or type(baseline_ordinal)is not int or baseline_ordinal<0
        or not isinstance(inputs_sha256,dict)or not inputs_sha256
        or any(not isinstance(k,str)or not isinstance(v,str)or not re.fullmatch('[a-f0-9]{64}',v)
            for k,v in inputs_sha256.items())):
        raise ValueError('exact ordinals and complete original input fingerprints required')
    directory=_safe(directory);filename=f'{ordinal:06d}.npz'
    with(directory/filename).open('xb')as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    return _seal(dict(format='native_six_condition_control_batch_v1',ordinal=ordinal,file=filename,
        baseline_ordinal=baseline_ordinal,file_sha256=_sha(directory/filename),keys=copy.deepcopy(keys),
        condition_order=[list(x)for x in CONDITIONS],inputs_sha256=copy.deepcopy(inputs_sha256),
        arrays_sha256={n:_typed_sha(a)for n,a in arrays.items()},
        baseline_arrays_sha256={n:_typed_sha(a)for n,a in baseline.items()},full_inference_permit=False))


def read_control_batch(directory,record,*,expected_keys,baseline,baseline_ordinal):
    try:
        record=_sealed(record);ordinal=record['ordinal']
        if (record['format']!='native_six_condition_control_batch_v1'or type(ordinal)is not int or ordinal<0
            or record['file']!=f'{ordinal:06d}.npz'or record['keys']!=expected_keys
            or record['baseline_ordinal']!=baseline_ordinal
            or record['condition_order']!=[list(x)for x in CONDITIONS]or record['full_inference_permit']is not False
            or record['baseline_arrays_sha256']!={n:_typed_sha(a)for n,a in baseline.items()}):
            raise ValueError('exact original native keys, baseline, conditions and packet identity required')
        path=_safe(Path(directory)/record['file']);guard=_StatGuard([path])
        if _sha(path)!=record['file_sha256']:
            raise ValueError('control raw packet bytes changed')
        with np.load(path,allow_pickle=False)as archive:
            arrays={n:archive[n]for n in archive.files}
        _validate_control_arrays(arrays,expected_keys,baseline)
        if {n:_typed_sha(a)for n,a in arrays.items()}!=record['arrays_sha256']:
            raise ValueError('control raw typed arrays changed')
        guard.check()
        return arrays
    except(OSError,KeyError,TypeError)as exc:
        raise ValueError('complete original control raw packet unavailable')from exc


_INPUTS = weakref.WeakKeyDictionary()
_TRANSPORT = weakref.WeakKeyDictionary()


@dataclass(eq=False)
class ControlInputs:
    evidence:ControlEvidence
    native:object
    normalization:dict
    identity:dict

    def check(self):
        saved=_INPUTS.get(self)
        if saved is None:
            raise ValueError('registered complete native control inputs required')
        if (self.evidence is not saved['evidence']or self.native is not saved['native']
            or self.normalization!=saved['normalization']or _sealed(self.identity)['sha256']!=saved['identity']):
            raise ValueError('control input provider, normalization or identity changed')
        self.evidence.check();self.native.check()
        if self.native.evidence is not self.evidence.native:
            raise ValueError('actual control input and original evidence providers differ')
        for hand,view in zip(('left','right'),self.native.dataset.fusion_views):
            if self.normalization[hand]!=dict(vision_mean=view.vision_mean.tolist(),vision_std=view.vision_std.tolist()):
                raise ValueError('actual frozen native image normalization changed')

    def __len__(self):
        self.check()
        return len(self.native)

    def __getitem__(self,index):
        self.check()
        value=self.native[index]
        self.check()
        return value

    def hash_payloads(self):
        self.check()
        return self.native.hash_payloads()


def admit_control_inputs(evidence):
    if not isinstance(evidence,ControlEvidence):
        raise ValueError('registered complete control evidence required')
    evidence.check();request=evidence.request
    native=_route(request['recipe']).admit_dense_run_inputs(evidence.native,data=request['data'],crops=request['crops'])
    norm={hand:dict(vision_mean=view.vision_mean.tolist(),vision_std=view.vision_std.tolist())
        for hand,view in zip(('left','right'),native.dataset.fusion_views)}
    identity=_seal(dict(format='registered_complete_control_input_admission_v1',
        evidence_identity_sha256=evidence.identity['sha256'],native_input_admission=copy.deepcopy(native.identity),
        normalization=copy.deepcopy(norm),full_inference_permit=False,full_paper_complete=False))
    _validate_control_input_identity(evidence,identity)
    value=ControlInputs(evidence,native,copy.deepcopy(norm),identity)
    _INPUTS[value]=dict(evidence=evidence,native=native,normalization=copy.deepcopy(norm),identity=identity['sha256'])
    value.check()
    return value


def _validate_control_input_identity(evidence,admission):
    """A resealed outer record cannot relax original whole-input admission."""
    if not isinstance(evidence,ControlEvidence):
        raise ValueError('registered actual complete control evidence required')
    evidence.check();admission=_sealed(admission)
    try:
        native=_sealed(admission['native_input_admission'])
        baseline=_sealed(_json(Path(evidence.request['baseline'])/'full-a/identity.json'))
        original=_sealed(baseline['input_admission'])
        if native!=original:
            raise ValueError('exact actual original full-input admission required, including counts/bytes/hash policy')
        # These are the fixed RGB-byte ImageNet constants in the admitted
        # _PointView source. Real preparation also compares the native views.
        rgb=dict(vision_mean=(255.*np.array([.485,.456,.406],np.float32)).tolist(),
            vision_std=(255.*np.array([.229,.224,.225],np.float32)).tolist())
        expected=_seal(dict(format='registered_complete_control_input_admission_v1',
            evidence_identity_sha256=evidence.identity['sha256'],native_input_admission=copy.deepcopy(original),
            normalization={h:copy.deepcopy(rgb)for h in('left','right')},
            full_inference_permit=False,full_paper_complete=False))
        if admission!=expected:
            raise ValueError('same native normalization and complete control input metadata required')
    except(OSError,KeyError,TypeError)as exc:
        raise ValueError('actual original nested control input admission unavailable')from exc
    evidence.check()
    return admission


def control_batch_ranges(evidence,scope):
    if not isinstance(evidence,ControlEvidence):
        raise ValueError('registered complete control evidence required')
    evidence.check()
    return _route(evidence.identity['recipe'])._scope_batches(evidence.native,scope)


def load_control_models(inputs):
    if not isinstance(inputs,ControlInputs):
        raise ValueError('registered complete control inputs required')
    inputs.check();request=inputs.evidence.request
    import torch
    torch.set_num_threads(2)
    if request['recipe']=='vits_s':
        from .occlusion_dense_models import load_dense_trained_models as load
    else:
        from .rn18_occlusion_models import load_rn18_dense_trained_models as load
    trained=load(request['fit'],upstream=request['upstream'],recipe=request['recipe'])
    _route(request['recipe'])._bind_models(inputs.native,trained)
    if trained.identity['sha256']!=inputs.evidence.identity['baseline_model_identity_sha256']:
        raise ValueError('same actual trained baseline model identity required')
    return trained


def _baseline_ordinal(evidence,keys):
    all_keys=evidence.keys();ranges=native_batch_ranges(all_keys)
    for ordinal,(start,stop)in enumerate(ranges):
        if keys and keys[0]['sequence']==start:
            if keys!=all_keys[start:stop]:
                break
            return ordinal
    raise ValueError('exact original full native batch or group-tail keys required')


def _validate_control_fingerprints(inputs,keys,fingerprints):
    """Recompute all condition tensors from authenticated original samples.

    Hashes of original rows alone cannot prove zero/roll/black transport. Cache
    only within this registered, generation-guarded input provider; no model,
    caller-supplied expected hashes or persisted cache can authorize admission.
    """
    if not isinstance(inputs,ControlInputs):
        raise ValueError('registered complete native control inputs required')
    inputs.check();evidence=inputs.evidence;ordinal=_baseline_ordinal(evidence,keys)
    if (not isinstance(fingerprints,dict)or not fingerprints
        or any(not isinstance(k,str)or not isinstance(v,str)or not re.fullmatch('[a-f0-9]{64}',v)
            for k,v in fingerprints.items())):
        raise ValueError('complete exact condition tensor fingerprint schema required')
    cache=_TRANSPORT.setdefault(inputs,{})
    if ordinal not in cache:
        import torch
        from .occlusion_dense_run import _collate_pairs
        from .modality_controls import apply_input_condition
        samples=[inputs[k['sequence']]for k in keys]
        batch=_collate_pairs(samples,upstream=evidence.native.upstream)
        if batch['keys']!=keys:
            raise ValueError('fresh transport input keys differ from original native batch')
        expected={'row:'+str(k['sequence']):s['verified_input_sha256']for k,s in zip(keys,samples)}
        norm=inputs.normalization[keys[0]['hand']]
        for c,(emg,vision)in enumerate(CONDITIONS):
            transported=apply_input_condition(batch['fusion'],emg=emg,vision=vision,**norm)['batch']
            expected.update({'condition:'+str(c)+':'+n:_typed_sha(x.numpy())
                for n,x in transported.items()if isinstance(x,torch.Tensor)})
        inputs.check();cache[ordinal]=_canonical(expected)
    if fingerprints!=json.loads(cache[ordinal]):
        raise ValueError('persisted six-condition tensors differ from deterministic admitted native inputs')
    inputs.check()


def predict_control_batch_set(inputs,trained,batch):
    """Run all fixed conditions on one complete original native batch/tail."""
    if not isinstance(inputs,ControlInputs):
        raise ValueError('registered complete control inputs required')
    inputs.check();evidence=inputs.evidence
    _route(evidence.identity['recipe'])._bind_models(inputs.native,trained)
    from .modality_control_inference import predict_control_batch
    from .modality_controls import apply_input_condition
    import torch
    try:
        keys=batch['keys'];ordinal=_baseline_ordinal(evidence,keys)
        fingerprints={}
        expected_fingerprints=[]
        for i,key in enumerate(keys):
            typed=json.loads(evidence.native.records[key['sequence']])['typed_input_sha256']
            expected_fingerprints.append(_hash(typed))
            for label,names in(('fusion',('emg','vision_img','vision_valid_mask','vision_frame_indices','joint_angles','label_valid_mask')),
                    ('standalone',('emg','joint_angles','label_valid_mask'))):
                for name in names:
                    tensor=batch[label][name]
                    if (not isinstance(tensor,torch.Tensor)or tensor.device.type!='cpu'or tensor.requires_grad
                        or _typed_sha(tensor[i].numpy())!=typed[label+':'+name]):
                        raise ValueError('collated control input differs from actual original row: '+label+':'+name)
            if batch['fusion'].get('target_hand',[key['hand']]*len(keys))[i]!=key['hand']:
                raise ValueError('original source target hand changed after collation')
            fingerprints['row:'+str(key['sequence'])]=_hash(typed)
        if batch['verified_input_sha256']!=expected_fingerprints:
            raise ValueError('original source row fingerprint ordering changed')
    except(KeyError,TypeError,AttributeError,IndexError)as exc:
        raise ValueError('complete verified original collated batch required')from exc
    baseline=evidence.baseline_packet(ordinal);norm=inputs.normalization[keys[0]['hand']]
    outputs=[]
    for c,(emg,vision)in enumerate(CONDITIONS):
        out=predict_control_batch(trained,batch['fusion'],emg=emg,vision=vision,**norm)
        if out['trained_model_identity_sha256']!=evidence.identity['baseline_model_identity_sha256']:
            raise ValueError('native condition used a different trained producer')
        outputs.append(out['arrays'])
        # The same deterministic transport is used by the native adapter. Keep
        # its actual tensor schema and byte identity without retaining images.
        transported=apply_input_condition(batch['fusion'],emg=emg,vision=vision,**norm)['batch']
        fingerprints.update({'condition:'+str(c)+':'+n:_typed_sha(x.numpy())
            for n,x in transported.items()if isinstance(x,torch.Tensor)})
    first=outputs[0]
    arrays={n:np.stack([a[n]for a in outputs])for n in('fusion_prediction','vision_prediction',
        'mae_fusion_rad','mae_vision_rad','condition_vision_valid_mask')}
    arrays.update({n:first[n].copy()for n in('targets','label_valid_mask','natural_crop_valid_mask')})
    _validate_control_arrays(arrays,keys,baseline)
    inputs.check();trained.check()
    return dict(keys=copy.deepcopy(keys),arrays=arrays,baseline_ordinal=ordinal,
        inputs_sha256=fingerprints,actual_forward_condition_count=6,full_inference_permit=False)


def _write(path,value):
    with Path(path).open('x')as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def _budget(output):
    from .rn18_occlusion_run import _runtime_budget
    _runtime_budget(output)


def read_control_pass(directory,*,evidence,inputs=None):
    """Reopen the complete scope after its producer and all loaders exit."""
    if not isinstance(evidence,ControlEvidence):
        raise ValueError('registered complete control evidence required')
    evidence.check()
    try:
        directory=_safe(directory);journal=directory/'batches.jsonl'
        guard=_StatGuard([directory/'identity.json',directory/'result.json',journal])
        identity=_sealed(_json(directory/'identity.json'));result=_sealed(_json(directory/'result.json'))
        ranges=control_batch_ranges(evidence,identity['scope']);route=_route(evidence.identity['recipe'])
        model=_sealed(identity['model_identity']);admission=_validate_control_input_identity(evidence,identity['input_admission'])
        if (identity['format']!='actual_native_six_condition_pass_v1'
            or identity['evidence_identity_sha256']!=evidence.identity['sha256']
            or identity['condition_order']!=[list(x)for x in CONDITIONS]
            or identity['batch_ranges']!=[list(x)for x in ranges]
            or identity['hardware']!=route._cpu_hardware()
            or model['sha256']!=evidence.identity['baseline_model_identity_sha256']
            or admission['evidence_identity_sha256']!=evidence.identity['sha256']
            or admission['native_input_admission']['complete_payload_sha256']!=evidence.native.identity['complete_payload_sha256']
            or result['format']!='actual_native_six_condition_pass_result_v1'
            or result['identity_sha256']!=identity['sha256']or result['scope']!=identity['scope']
            or result['journal_file_sha256']!=_sha(journal)
            or result['points']!=sum(b-a for a,b in ranges)or result['batches']!=len(ranges)
            or result['condition_predictions']!=6*result['points']
            or result['model_state_sha256_before']!=model['model_state_sha256']
            or result['model_state_sha256_after']!=result['model_state_sha256_before']
            or not result['all_model_states_before_after_exact']
            or result['complete_payload_sha256_after_forward']!=evidence.native.identity['complete_payload_sha256']
            or result['actual_optimizer_updates']!=0 or result['cuda_initialized']):
            raise ValueError('same complete original control inputs/model/scope/result required')
        _closed(result['actor'])
        keys=evidence.keys();records=[_sealed(json.loads(s))for s in journal.read_text().splitlines()]
        if len(records)!=len(ranges)or len(result['loader_groups'])!=6:
            raise ValueError('all exact native control batches and six loader groups required')
        for group in result['loader_groups']:
            selected=[(a,b)for a,b in ranges if(keys[a]['split'],keys[a]['hand'])==(group['split'],group['hand'])]
            if (len(group['workers'])!=4 or not group['all_workers_exited']
                or not group['pin_memory_requested']or group['batches']!=len(selected)
                or group['points']!=sum(b-a for a,b in selected)):
                raise ValueError('same four-worker native control group with complete tails required')
            for actor in group['workers']:_closed(actor)
        if [(g['split'],g['hand'])for g in result['loader_groups']]!=[(s,h)for s in('user','gesture','both')for h in('left','right')]:
            raise ValueError('six original loader groups must remain ordered and unique')
        if inputs is None:inputs=admit_control_inputs(evidence)
        if not isinstance(inputs,ControlInputs)or inputs.evidence is not evidence:
            raise ValueError('same actual registered native inputs required for condition transport replay')
        inputs.check()
        missing=invalid=0
        for ordinal,((start,stop),record)in enumerate(zip(ranges,records)):
            expected=keys[start:stop];base_ordinal=_baseline_ordinal(evidence,expected)
            if record['ordinal']!=ordinal or record['file']!=f'{ordinal:06d}.npz':
                raise ValueError('control batch ordinal ordering changed')
            baseline=evidence.baseline_packet(base_ordinal)
            guard.add(_safe(directory/'batches'/record['file']))
            arrays=read_control_batch(directory/'batches',record,expected_keys=expected,
                baseline=baseline,baseline_ordinal=base_ordinal)
            for key in expected:
                typed=json.loads(evidence.native.records[key['sequence']])['typed_input_sha256']
                if record['inputs_sha256']['row:'+str(key['sequence'])]!=_hash(typed):
                    raise ValueError('original control row fingerprint changed')
            _validate_control_fingerprints(inputs,expected,record['inputs_sha256'])
            missing+=int((~arrays['natural_crop_valid_mask']).sum())
            invalid+=int((~arrays['label_valid_mask']).sum())
        if (result['crop_missing'],result['label_invalid'])!=(missing,invalid):
            raise ValueError('natural missing/invalid requests were lost in control output')
        if identity['scope']=='complete'and(missing,invalid)!=(evidence.native.identity['counts']['crop_missing'],evidence.native.identity['counts']['label_invalid']):
            raise ValueError('complete native missing/invalid population required')
        evidence.check();guard.check()
        return dict(directory=str(directory),identity=identity,result=result,records=records,
            _guard=guard,_result_file_sha256=_sha(directory/'result.json'),_inputs=inputs)
    except(OSError,KeyError,TypeError)as exc:
        raise ValueError('complete control pass unavailable or malformed')from exc


def _compare_control_passes(first,second,evidence):
    if (_safe(first['directory'])==_safe(second['directory'])or first['identity']!=second['identity']
        or (first['result']['actor']['pid'],first['result']['actor']['start_ticks'])
            ==(second['result']['actor']['pid'],second['result']['actor']['start_ticks'])
        or not second['result']['fresh_model_forward_replay_verified']
        or second['result']['replay_of_result_file_sha256']!=first['_result_file_sha256']):
        raise ValueError('independent fresh complete control replay required')
    for item in(first,second):_closed(item['result']['actor']);item['_guard'].check()
    for left,right in zip(first['records'],second['records']):
        if (left['keys']!=right['keys']or left['arrays_sha256']!=right['arrays_sha256']
            or left['inputs_sha256']!=right['inputs_sha256']or left['baseline_arrays_sha256']!=right['baseline_arrays_sha256']):
            raise ValueError('native six-condition replay differs in raw predictions or inputs')
    evidence.check()


def accept_control_qualification(first,second,*,evidence,output):
    a=read_control_pass(first,evidence=evidence);b=read_control_pass(second,evidence=evidence,inputs=a['_inputs'])
    if a['identity']['scope']!='qualification'or b['identity']['scope']!='qualification':
        raise ValueError('actual original batch64 and group-tail control qualification required')
    _compare_control_passes(a,b,evidence)
    value=_seal(dict(format='actual_six_condition_CPU_qualification_v1',
        evidence_identity_sha256=evidence.identity['sha256'],condition_order=[list(x)for x in CONDITIONS],
        actual_batch_size=64,actual_workers=4,actual_points_each=a['result']['points'],actual_batches_each=12,
        first=str(_safe(first)),second=str(_safe(second)),first_result_file_sha256=a['_result_file_sha256'],
        second_result_file_sha256=b['_result_file_sha256'],separate_fresh_replay_exact=True,
        full_population_predicted=False,scientific_eligible=False,full_paper_complete=False))
    _write(output,value)
    return value


def _qualification(path,evidence,inputs=None):
    if path is None:
        raise ValueError('actual six-condition batch64 qualification required before complete control inference')
    try:
        path=_safe(path);guard=_StatGuard([path]);v=_sealed(_json(path))
        ranges=control_batch_ranges(evidence,'qualification')
        if (v['format']!='actual_six_condition_CPU_qualification_v1'
            or v['evidence_identity_sha256']!=evidence.identity['sha256']
            or v['condition_order']!=[list(x)for x in CONDITIONS]
            or(v['actual_batch_size'],v['actual_workers'],v['actual_points_each'],v['actual_batches_each'])
                !=(64,4,sum(b-a for a,b in ranges),12)or not v['separate_fresh_replay_exact']):
            raise ValueError('same actual native six-condition qualification required')
        a=read_control_pass(v['first'],evidence=evidence,inputs=inputs)
        b=read_control_pass(v['second'],evidence=evidence,inputs=a['_inputs'])
        if (a['identity']['scope']!='qualification'or b['identity']['scope']!='qualification'
            or a['_result_file_sha256']!=v['first_result_file_sha256']or b['_result_file_sha256']!=v['second_result_file_sha256']):
            raise ValueError('actual original qualification scope and result bytes required')
        _compare_control_passes(a,b,evidence);guard.check()
        return v
    except(OSError,KeyError,TypeError)as exc:
        raise ValueError('actual same-source control qualification unavailable')from exc


def run_control_pass(inputs,output,*,scope,qualification=None,replay_of=None):
    """Consume the full original scope; no configurable batch or sample limit."""
    if not isinstance(inputs,ControlInputs):
        raise ValueError('registered complete native control inputs required')
    inputs.check();evidence=inputs.evidence;ranges=control_batch_ranges(evidence,scope)
    if scope=='complete':_qualification(qualification,evidence,inputs)
    _budget(output)
    previous=read_control_pass(replay_of,evidence=evidence,inputs=inputs)if replay_of is not None else None
    import time
    import torch
    from .emg2pose_memory_process import process_identity
    from .vision_literal_runtime import vision_state_sha256
    route=_route(evidence.identity['recipe']);trained=load_control_models(inputs)
    output=_safe(output,exists=False);output.mkdir(mode=0o700);(output/'batches').mkdir(mode=0o700)
    identity=_seal(dict(format='actual_native_six_condition_pass_v1',scope=scope,
        evidence_identity_sha256=evidence.identity['sha256'],input_admission=copy.deepcopy(inputs.identity),
        condition_order=[list(x)for x in CONDITIONS],batch_ranges=[list(x)for x in ranges],
        model_identity=copy.deepcopy(trained.identity),hardware=route._cpu_hardware(),
        scientific_eligible=False,full_paper_complete=False))
    if previous is not None and previous['identity']!=identity:
        raise ValueError('fresh control replay must use identical source/model/input/condition identities')
    _write(output/'identity.json',identity)
    before={n:vision_state_sha256(m.state_dict())for n,m in trained.models.items()}
    keys=evidence.keys();groups=[];ordinal=points=missing=invalid=0;started=time.monotonic()
    with(output/'batches.jsonl').open('x')as journal:
        for split in('user','gesture','both'):
            for hand in('left','right'):
                selected=[(a,b)for a,b in ranges if(keys[a]['split'],keys[a]['hand'])==(split,hand)]
                indices=[i for a,b in selected for i in range(a,b)];count=0
                with route._managed_loader(inputs,indices,upstream=evidence.native.upstream)as(iterator,workers,pinned):
                    for batch in iterator:
                        _budget(output);inputs.check()
                        if count>=len(selected)or batch['keys']!=keys[selected[count][0]:selected[count][1]]:
                            raise ValueError('native control loader order or group tail changed')
                        value=predict_control_batch_set(inputs,trained,batch)
                        baseline=evidence.baseline_packet(value['baseline_ordinal'])
                        record=write_control_batch(output/'batches',ordinal,keys=value['keys'],arrays=value['arrays'],
                            baseline=baseline,baseline_ordinal=value['baseline_ordinal'],inputs_sha256=value['inputs_sha256'])
                        read_control_batch(output/'batches',record,expected_keys=value['keys'],baseline=baseline,baseline_ordinal=value['baseline_ordinal'])
                        if previous is not None:
                            previous['_guard'].check();old=previous['records'][ordinal]
                            if (old['arrays_sha256']!=record['arrays_sha256']or old['inputs_sha256']!=record['inputs_sha256']):
                                raise ValueError('fresh native six-condition predictions differ')
                        journal.write(_canonical(record)+'\n');journal.flush();os.fsync(journal.fileno())
                        ordinal+=1;count+=1;points+=len(value['keys'])
                        missing+=int((~value['arrays']['natural_crop_valid_mask']).sum())
                        invalid+=int((~value['arrays']['label_valid_mask']).sum())
                        progress=dict(scope=scope,batches=ordinal,total_batches=len(ranges),points=points,
                            total_points=sum(b-a for a,b in ranges),condition_predictions=6*points,split=split,hand=hand,seconds=time.monotonic()-started)
                        (output/'progress.json').write_text(json.dumps(progress,sort_keys=True)+'\n');print(json.dumps(progress),flush=True)
                if count!=len(selected):raise ValueError('native control loader omitted complete group batches')
                for actor in workers:_closed(actor)
                groups.append(dict(split=split,hand=hand,workers=workers,all_workers_exited=True,
                    pin_memory_requested=True,actual_pin_memory=pinned,points=len(indices),batches=count))
    if ordinal!=len(ranges)or points!=sum(b-a for a,b in ranges):
        raise ValueError('complete original control population was not consumed')
    inputs.check();trained.check();after={n:vision_state_sha256(m.state_dict())for n,m in trained.models.items()}
    if before!=after or torch.cuda.is_initialized():raise ValueError('frozen control model state or CPU policy changed')
    payloads=inputs.hash_payloads()
    if previous is not None:previous['_guard'].check()
    result=_seal(dict(format='actual_native_six_condition_pass_result_v1',identity_sha256=identity['sha256'],
        actor=process_identity(),scope=scope,points=points,batches=ordinal,condition_predictions=6*points,
        crop_missing=missing,label_invalid=invalid,journal_file_sha256=_sha(output/'batches.jsonl'),
        model_state_sha256_before=before,model_state_sha256_after=after,all_model_states_before_after_exact=True,
        complete_payload_sha256_after_forward=payloads,loader_groups=groups,
        fresh_model_forward_replay_verified=previous is not None,
        replay_of_result_file_sha256=previous['_result_file_sha256']if previous is not None else None,
        actual_optimizer_updates=0,cuda_initialized=False,scientific_eligible=False,full_paper_complete=False,
        seconds=time.monotonic()-started))
    _write(output/'result.json',result)
    return result


def summarize_control_errors(fusion_rad,vision_rad,label_valid_mask,crop_valid_mask,*,selection=None):
    """Descriptive paired arithmetic; raw arrays alone never certify a run."""
    if (not isinstance(fusion_rad,np.ndarray)or fusion_rad.ndim!=2 or fusion_rad.shape[0]!=6
        or not isinstance(vision_rad,np.ndarray)or vision_rad.shape!=fusion_rad.shape
        or fusion_rad.dtype!=np.float64 or vision_rad.dtype!=np.float64
        or not np.isfinite(fusion_rad).all()or not np.isfinite(vision_rad).all()):
        raise ValueError('six complete finite float64 raw-radian error vectors required')
    size=fusion_rad.shape[1]
    for mask in(label_valid_mask,crop_valid_mask):
        if not isinstance(mask,np.ndarray)or mask.dtype!=np.bool_ or mask.shape!=(size,1):
            raise ValueError('original per-request boolean label/crop validity required')
    if selection is None:selection=np.ones(size,bool)
    if not isinstance(selection,np.ndarray)or selection.dtype!=np.bool_ or selection.shape!=(size,):
        raise ValueError('explicit per-request boolean group selection required')
    valid=selection&label_valid_mask[:,0];n=int(valid.sum());rows=[]
    for c,(emg,vision)in enumerate(CONDITIONS):
        reference=0 if vision=='native'else 1
        penalty=fusion_rad[c,valid]-fusion_rad[reference,valid]
        gain=vision_rad[c,valid]-fusion_rad[c,valid]
        rows.append(dict(condition=c,emg=emg,vision=vision,valid_label_requests=n,
            fusion_mae_deg=float(fusion_rad[c,valid].mean()*57.3)if n else None,
            vision_mae_deg=float(vision_rad[c,valid].mean()*57.3)if n else None,
            literal_all_requests_fusion_mae_deg=float(fusion_rad[c,selection].mean()*57.3)if selection.any()else None,
            vision_gain_deg=float(gain.mean()*57.3)if n else None,
            fusion_better_than_vision_requests=int((gain>0).sum()),fusion_worse_than_vision_requests=int((gain<0).sum()),
            aligned_reference_condition=reference,penalty_vs_aligned_deg=float(penalty.mean()*57.3)if n else None,
            median_penalty_vs_aligned_deg=float(np.median(penalty)*57.3)if n else None,
            aligned_better_requests=int((penalty>0).sum()),aligned_worse_requests=int((penalty<0).sum()),
            aligned_equal_requests=int((penalty==0).sum())))
    return dict(requests=int(selection.sum()),valid_label_requests=n,
        invalid_label_requests=int((selection&~label_valid_mask[:,0]).sum()),
        natural_missing_valid_requests=int((valid&~crop_valid_mask[:,0]).sum()),conditions=rows,
        uncertainty='descriptive only; participant/recording clustered inference not yet provided',
        full_inference_permit=False)


def build_control_report(first,second,*,evidence,output):
    """Read only the two complete passes; never construct a model or optimizer."""
    import csv
    from .emg2pose_memory_process import process_identity
    a=read_control_pass(first,evidence=evidence);b=read_control_pass(second,evidence=evidence,inputs=a['_inputs'])
    if a['identity']['scope']!='complete'or b['identity']['scope']!='complete':
        raise ValueError('two actual complete six-condition passes required for a control report')
    _compare_control_passes(a,b,evidence);_budget(output)
    keys=evidence.keys();collected={n:[]for n in ARRAYS}
    for record in a['records']:
        baseline=evidence.baseline_packet(record['baseline_ordinal'])
        values=read_control_batch(Path(first)/'batches',record,expected_keys=record['keys'],
            baseline=baseline,baseline_ordinal=record['baseline_ordinal'])
        for n in ARRAYS:collected[n].append(values[n])
    arrays={n:np.concatenate(values,axis=1 if n.endswith('_prediction')or n.startswith('mae_')or n=='condition_vision_valid_mask'else 0)
        for n,values in collected.items()}
    count=len(keys)
    if arrays['targets'].shape!=(count,22):raise ValueError('complete ordered control report targets required')
    groups={'all':np.ones(count,bool),'natural_crop_missing':~arrays['natural_crop_valid_mask'][:,0],
        'natural_crop_present':arrays['natural_crop_valid_mask'][:,0].copy()}
    for split in('user','gesture','both'):
        groups['split:'+split]=np.array([k['split']==split for k in keys],bool)
        for hand in('left','right'):
            groups['split:'+split+':'+hand]=np.array([k['split']==split and k['hand']==hand for k in keys],bool)
    geometry=evidence.native.geometry()
    scores=np.array([np.nan if g['occlusion_score']is None else g['occlusion_score']for g in geometry],np.float64)
    for name,mask in(('occlusion_below_0_6',np.isfinite(scores)&(scores<.6)),
            ('occlusion_0_6_to_0_8',(scores>=.6)&(scores<.8)),('occlusion_0_8_to_1_inclusive',(scores>=.8)&(scores<=1))):
        groups[name]=mask&arrays['natural_crop_valid_mask'][:,0]
    summaries={name:summarize_control_errors(arrays['mae_fusion_rad'],arrays['mae_vision_rad'],
        arrays['label_valid_mask'],arrays['natural_crop_valid_mask'],selection=mask)for name,mask in groups.items()}
    metrics=dict(format='actual_complete_native_six_condition_descriptive_metrics_v1',recipe=evidence.identity['recipe'],
        evidence_identity_sha256=evidence.identity['sha256'],model_identity_sha256=evidence.identity['baseline_model_identity_sha256'],
        original_baseline_proof_sha256=evidence.identity['baseline_proof_sha256'],condition_order=[list(x)for x in CONDITIONS],
        actual_points=count,actual_conditions=6,actual_complete_prediction_passes=2,groups=summaries,
        metric='per-request native22-angle MAE from raw radians times57.3; valid labels primary, unmasked literal mean separate',
        limitations=['Available legacy data; PDF population, labels and physical camera equivalence not established.',
            'Original center-only dense windows retain disclosed training-input overlap.',
            'Natural geometric occlusion, genuine crop loss and induced full-image loss are distinct.',
            'Normalized EMG zero is the native force_zero_emg input, not resting/raw voltage zero.',
            'Half-window circular roll retains signal information and creates an artificial temporal seam.',
            'Native standalone vision and fusion consume vision-valid flags differently; same-checkpoint EMG comparisons isolate that factor.',
            'Descriptive estimates only; correlated requests are not independent participants.'],
        full_paper_complete=False)
    output=_safe(output,exists=False);output.mkdir(mode=0o700);_write(output/'metrics.json',metrics)
    rows=[dict(group=group,**row)for group,value in summaries.items()for row in value['conditions']]
    with(output/'condition-results.csv').open('x')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pending=_seal(dict(format='pending_complete_control_report_v1',evidence_identity_sha256=evidence.identity['sha256'],
        actual_points=count,actual_conditions=6,first_result_file_sha256=a['_result_file_sha256'],second_result_file_sha256=b['_result_file_sha256'],
        actual_complete_raw_replay_exact=True,artifacts_sha256={n:_sha(output/n)for n in('metrics.json','condition-results.csv')},
        report_actor=process_identity(),scientific_eligible=False,full_paper_complete=False))
    _write(output/'pending-report.json',pending);evidence.check()
    return pending


def _pending_control_metadata(value,evidence):
    """Only the reporting actor and seal may differ after a fresh rebuild."""
    value=_sealed(value)
    fields={'format','evidence_identity_sha256','actual_points','actual_conditions',
        'first_result_file_sha256','second_result_file_sha256','actual_complete_raw_replay_exact',
        'artifacts_sha256','report_actor','scientific_eligible','full_paper_complete','sha256'}
    try:
        hashes=[value[n]for n in('first_result_file_sha256','second_result_file_sha256')]
        artifacts=value['artifacts_sha256'];actor=value['report_actor']
        if (set(value)!=fields or value['format']!='pending_complete_control_report_v1'
            or value['evidence_identity_sha256']!=evidence.identity['sha256']
            or type(value['actual_points'])is not int or value['actual_points']!=len(evidence.keys())
            or type(value['actual_conditions'])is not int or value['actual_conditions']!=6
            or value['actual_complete_raw_replay_exact']is not True
            or value['scientific_eligible']is not False or value['full_paper_complete']is not False
            or not isinstance(artifacts,dict)or set(artifacts)!={'metrics.json','condition-results.csv'}
            or any(not isinstance(h,str)or not re.fullmatch('[a-f0-9]{64}',h)for h in [*hashes,*artifacts.values()])
            or not isinstance(actor,dict)or type(actor.get('pid'))is not int
            or not str(actor.get('start_ticks','')).isdigit()):
            raise ValueError('exact complete control report metadata required')
    except(KeyError,TypeError)as exc:
        raise ValueError('complete pending control report unavailable')from exc
    return {k:v for k,v in value.items()if k not in('sha256','report_actor')}


def _same_actor(actual,expected):
    return isinstance(actual,dict)and isinstance(expected,dict)and all(
        actual.get(n)==expected.get(n)for n in('pid','start_ticks'))


def _verification_receipt(original_path,rebuilt_path,pending,*,evidence,actor):
    return _seal(dict(format='independently_rebuilt_complete_control_report_v1',
        pending_report_file_sha256=_sha(original_path),rebuilt_pending_report_file_sha256=_sha(rebuilt_path),
        artifacts_sha256=pending['artifacts_sha256'],evidence_identity_sha256=evidence.identity['sha256'],
        all_numeric_json_and_csv_bytes_exact=True,actual_points=pending['actual_points'],actual_conditions=6,
        first_result_file_sha256=pending['first_result_file_sha256'],second_result_file_sha256=pending['second_result_file_sha256'],
        verifier_actor=actor,scientific_eligible=False,full_paper_complete=False))


def verify_control_report(first,second,report,*,evidence,output):
    from .emg2pose_memory_process import process_identity
    original_path=_safe(Path(report)/'pending-report.json');guard=_StatGuard([original_path])
    original=_sealed(_json(original_path));original_metadata=_pending_control_metadata(original,evidence)
    _closed(original['report_actor'])
    for n,h in original['artifacts_sha256'].items():
        path=_safe(Path(report)/n);guard.add(path)
        if _sha(path)!=h:raise ValueError('original control report artifact changed')
    rebuilt=build_control_report(first,second,evidence=evidence,output=output)
    if original_metadata!=_pending_control_metadata(rebuilt,evidence):
        raise ValueError('fresh control report scientific/provenance metadata or artifacts differ')
    actor=process_identity()
    if not _same_actor(rebuilt['report_actor'],actor):
        raise ValueError('rebuilt control report must belong to the actual verifier')
    guard.check();evidence.check()
    result=_verification_receipt(original_path,_safe(Path(output)/'pending-report.json'),rebuilt,evidence=evidence,actor=actor)
    _write(Path(output)/'verification.json',result)
    return result


def _promote_verified_control_report(directory,*,evidence,report_actor,verifier_actor):
    """Require both actual closed stage workers and all rebuilt metadata."""
    directory=_safe(directory)
    paths=[_safe(directory/n)for n in('report/pending-report.json','report-rebuilt/pending-report.json',
        'report-rebuilt/verification.json')]
    guard=_StatGuard(paths);pending,rebuilt,verified=[_sealed(_json(p))for p in paths]
    if _same_actor(report_actor,verifier_actor):raise ValueError('separate actual report/verifier workers required')
    for actor in(report_actor,verifier_actor):_closed(actor)
    if (_pending_control_metadata(pending,evidence)!=_pending_control_metadata(rebuilt,evidence)
        or not _same_actor(pending['report_actor'],report_actor)
        or not _same_actor(rebuilt['report_actor'],verifier_actor)
        or not _same_actor(verified.get('verifier_actor'),verifier_actor)
        or verified!=_verification_receipt(paths[0],paths[1],rebuilt,evidence=evidence,actor=verified.get('verifier_actor'))):
        raise ValueError('actual independent full control metadata/worker verification required')
    for folder in('report','report-rebuilt'):
        for n,h in pending['artifacts_sha256'].items():
            path=_safe(directory/folder/n);guard.add(path)
            if _sha(path)!=h:raise ValueError('verified native control artifact changed')
    guard.check();evidence.check()
    return _seal(dict({k:v for k,v in pending.items()if k!='sha256'},scientific_eligible=True,
        available_legacy_control_pipeline_complete=True,verification_file_sha256=_sha(paths[2])))


def run_control_workflow(request,output):
    """Model-free controller; each learned or numeric stage is a new process."""
    evidence=load_control_evidence(request)
    import sys
    import time
    from .emg2pose_memory_process import process_identity,same_process,run_owned_stage
    output=_safe(output,exists=False);_budget(output)
    cli=_safe(ROOT/'scripts/paper/run_modality_controls.py')
    guard,sources=_source_guard();guard.add(cli);source_cli_sha=_sha(cli)
    output.mkdir(mode=0o700);_write(output/'request.json',evidence.request);guard.add(output/'request.json')
    parent=process_identity(os.getppid());started=time.monotonic();owned=[]
    def check():
        guard.check();evidence.check();_budget(output)
        if 'torch'in sys.modules or not same_process(parent):
            raise ValueError('control controller must remain model-free with its actual parent')
    common=[sys.executable,'-B',str(cli)];request_args=['--request',str(output/'request.json')]
    qa,qb=output/'qual-a',output/'qual-b';a,b=output/'full-a',output/'full-b';qualification=output/'qualification.json'
    stages=[('qual-a',['predict',*request_args,'--scope','qualification','--output',str(qa)]),
        ('qual-b',['predict',*request_args,'--scope','qualification','--replay-of',str(qa),'--output',str(qb)]),
        ('accept-qualification',['accept-qualification',*request_args,'--first',str(qa),'--second',str(qb),'--output',str(qualification)]),
        ('full-a',['predict',*request_args,'--scope','complete','--qualification',str(qualification),'--output',str(a)]),
        ('full-b',['predict',*request_args,'--scope','complete','--qualification',str(qualification),'--replay-of',str(a),'--output',str(b)]),
        ('report',['report',*request_args,'--first',str(a),'--second',str(b),'--output',str(output/'report')]),
        ('verify-report',['verify-report',*request_args,'--first',str(a),'--second',str(b),'--report',str(output/'report'),'--output',str(output/'report-rebuilt')])]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    for name,args in stages:
        check();print(json.dumps(dict(phase=name,elapsed_seconds=time.monotonic()-started)),flush=True)
        result=run_owned_stage([*common,*args],output/('owned-'+name),descriptors={},guard=check,
            environment=environment,poll_seconds=2.,terminate_grace=15.)
        if (result['status']!='success'or result['worker_exit_code']!=0 or not result['all_owned_consumers_exited']
            or any(same_process(result[n])for n in('worker','guardian'))):
            raise ValueError('owned native control stage failed: '+name)
        owned.append(result);check()
    report=_promote_verified_control_report(output,evidence=evidence,
        report_actor=owned[-2]['worker'],verifier_actor=owned[-1]['worker'])
    if source_cli_sha!=_sha(cli)or any(_sha(ROOT/n)!=h for n,h in sources.items()):
        raise ValueError('frozen control execution source changed')
    check()
    _write(output/'report/report.json',report)
    value=_seal(dict(format='actual_complete_owned_native_control_workflow_v1',
        evidence_identity_sha256=evidence.identity['sha256'],source_files_sha256=sources,cli_sha256=source_cli_sha,
        actual_owned_stages=owned,all_owned_consumers_exited=True,actual_points=report['actual_points'],actual_conditions=6,
        report_file_sha256=_sha(output/'report/report.json'),available_legacy_control_pipeline_complete=True,
        parent_torch_unimported=True,scientific_eligible=True,full_paper_complete=False,seconds=time.monotonic()-started))
    _write(output/'workflow-result.json',value)
    return value


def _inference_only(function,*,deny_modules=False):
    """An inference process must fail rather than accidentally create an optimizer."""
    from functools import wraps
    @wraps(function)
    def wrapped(*args,**kwargs):
        import torch
        optimizer=torch.optim.Optimizer.__init__;module=torch.nn.Module.__init__
        def forbidden(*a,**k):raise ValueError('control numeric/inference stage forbids learned or optimizer construction')
        torch.optim.Optimizer.__init__=forbidden
        if deny_modules:torch.nn.Module.__init__=forbidden
        try:return function(*args,**kwargs)
        finally:
            torch.optim.Optimizer.__init__=optimizer
            if deny_modules:torch.nn.Module.__init__=module
    return wrapped


run_control_pass=_inference_only(run_control_pass)
build_control_report=_inference_only(build_control_report,deny_modules=True)
verify_control_report=_inference_only(verify_control_report,deny_modules=True)
