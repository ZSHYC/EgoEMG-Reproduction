"""Admission and typed new-reference behavior; no alternate model mathematics."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest


def api():
    try:
        return importlib.import_module('egoemg_repro.center_reference_eval')
    except ModuleNotFoundError as exc:
        if exc.name == 'egoemg_repro.center_reference_eval':
            pytest.fail('selected native same-center prediction entry missing')
        raise


def sample(source_index=0):
    hand = 'left' if source_index < 2141 else 'right'
    center = 6000 + (source_index % 2141) * 7790
    return dict(emg=np.ones((8, 12000), np.float32), joint_angles=np.full((22, 1), .5, np.float32),
        label_valid_mask=np.array([source_index != 0], bool), episode=0,
        start=center-6000, end=center+6000, center=center, hand=hand,
        split='gesture', subject='synthetic', role='test', jitter_offset=0,
        reference_sample_index=source_index, reference_unique_sample_index=source_index,
        reference_repeat_index=0)


def admitted_record(s, index):
    # Independent definition of the already recorded Task1 byte commitments.
    meta={k:s[k] for k in ('episode','start','end','center','hand','split','subject')}
    sha=lambda b:hashlib.sha256(b).hexdigest()
    return dict(index=index,emg_sha256=sha(s['emg'].tobytes()),
        target_mask_sha256=sha(s['joint_angles'].tobytes()+s['label_valid_mask'].tobytes()),
        identity_sha256=sha(json.dumps(meta,sort_keys=True,separators=(',',':')).encode()))


def test_full_scope_cannot_be_replaced_by_four_boundary_probes():
    m=api()
    assert m.reference_source_indices(4282,'engineering') == [0,2140,2141,4281]
    assert m.reference_source_indices(4282,'full') == list(range(4282))
    for n,scope in ((4,'full'),(4282,'mini'),(True,'engineering'),(2792,'full')):
        with pytest.raises(ValueError):m.reference_source_indices(n,scope)


@pytest.mark.parametrize('damage',['center','hand','mask','emg','target','dtype','index','repeat'])
def test_request_mutation_fails_against_actual_input_commitment(damage):
    m=api();s=sample();expected=admitted_record(s,0)
    assert isinstance(m.validate_reference_sample(s,expected,0),str)
    if damage=='center':s['center']+=1
    elif damage=='hand':s['hand']='right'
    elif damage=='mask':s['label_valid_mask'][0]=True
    elif damage=='emg':s['emg'][0,0]+=1
    elif damage=='target':s['joint_angles'][0,0]+=1
    elif damage=='dtype':s['emg']=s['emg'].astype(np.float64)
    elif damage=='index':s['reference_sample_index']=1
    else:s['reference_repeat_index']=1
    with pytest.raises(ValueError):m.validate_reference_sample(s,expected,0)


def test_native_raw_token_midpoint_is_selected_without_interpolation():
    torch=pytest.importorskip('torch');raw=torch.arange(5,dtype=torch.float32).reshape(1,1,5).expand(1,22,5).clone()
    got=api().reference_prediction(raw)
    assert got.shape==(1,22,1) and got.dtype==torch.float32 and torch.equal(got,torch.full((1,22,1),2.))


@pytest.mark.parametrize('damage',['bf16','nan','channels','batch','empty'])
def test_reference_predictions_reject_other_precision_or_incomplete_shape(damage):
    torch=pytest.importorskip('torch');raw=torch.zeros((1,22,5))
    if damage=='bf16':raw=raw.bfloat16()
    elif damage=='nan':raw[0,0,0]=float('nan')
    elif damage=='channels':raw=raw[:,:20]
    elif damage=='batch':raw=raw.expand(2,22,5)
    else:raw=raw[:,:,:0]
    with pytest.raises(ValueError):api().reference_prediction(raw)


def packet():
    torch=pytest.importorskip('torch');indices=[0,2140,2141,4281];parts=[];identities=[]
    records=[dict(index=i,emg_sha256='0'*64,target_mask_sha256='0'*64,identity_sha256='0'*64)for i in range(4282)]
    for local,source in enumerate(indices):
        s=sample(source);record=admitted_record(s,source);records[source]=record
        token=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        parts.append(dict(pred=torch.full((1,22,1),.6),target=torch.from_numpy(s['joint_angles']).unsqueeze(0),
            mask=torch.from_numpy(s['label_valid_mask']).unsqueeze(0),
            index=torch.tensor([local]),unique=torch.tensor([local]),repeat=torch.tensor([0]),
            center=torch.tensor([s['center']]),episode=torch.tensor([s['episode']]),
            hand=[s['hand']],split=[s['split']],user=[s['subject']],sample_input_sha256=[token],
            input_sha256=hashlib.sha256(json.dumps([record],sort_keys=True,separators=(',',':')).encode()).hexdigest()))
        identities.append({k:s[k]for k in('episode','start','end','center','hand','split','subject')})
    return dict(parts=parts,source_indices=indices,request_identities=identities),records


def test_typed_engineering_retains_invalid_requests_without_full_claim():
    payload,records=packet();v=api().validate_reference_payload(payload,records,scope='engineering')
    assert v['consumed_samples']==v['unique_samples']==4 and v['duplicate_consumptions']==0
    assert v['all']['overall']['valid_angle_count']==66


@pytest.mark.parametrize('damage',['missing','duplicate','bf16','mask','user','source_index','identity','full_scope'])
def test_typed_payload_rejects_partial_reordered_or_unbound_consumption(damage):
    payload,records=packet();scope='engineering'
    if damage=='missing':payload['parts'].pop()
    elif damage=='duplicate':payload['parts'][1]['index'][0]=0
    elif damage=='bf16':payload['parts'][0]['pred']=payload['parts'][0]['pred'].bfloat16()
    elif damage=='mask':payload['parts'][0]['mask'][0,0]=True
    elif damage=='user':payload['parts'][0]['user'][0]='different'
    elif damage=='source_index':payload['source_indices'][0]=1
    elif damage=='identity':payload['request_identities'][0]['center']+=1
    else:scope='full'
    with pytest.raises(ValueError):api().validate_reference_payload(payload,records,scope=scope)


@pytest.mark.parametrize('size',['small','middle','large'])
def test_original_real_200_epoch_metadata_is_accepted_without_loading_weights(size,tmp_path):
    # These are exact accepted real metadata bytes fetched from the server,
    # not a short/fake fit relabeled as full. No learned weights are fixtures.
    root=Path(os.environ.get('EGOEMG_REFERENCE_FIT_FIXTURES',
        Path(__file__).parent/'fixtures/native-center-reference-real-fit-metadata'))/size
    assert root.is_dir(),'actual accepted metadata fixture must be staged, never skipped'
    copied=tmp_path/'source';shutil.copytree(root,copied)
    fit=copied/'private/full-fit';got=api().source_fit_metadata(fit)
    assert got['size']==size and got['completed_epochs']==200 and got['optimizer_updates']==25000
    assert got['selected_epoch']=={'small':169,'middle':124,'large':142}[size]
    journal=[json.loads(line)for line in (fit/'epochs.jsonl').read_text().splitlines()]
    expected_state=journal[{'small':168,'middle':123,'large':141}[size]]['record']['model_sha256']
    assert got.get('selected_model_state_sha256')==expected_state
    assert got['learned_weights_loaded'] is False
    altered=json.loads((fit/'fit-result.json').read_text());altered['scope']='cpu_engineering'
    (fit/'fit-result.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError):api().source_fit_metadata(fit)


def test_wrong_selected_weight_bytes_reject_before_deserialization(tmp_path):
    root=Path(os.environ.get('EGOEMG_REFERENCE_FIT_FIXTURES',
        Path(__file__).parent/'fixtures/native-center-reference-real-fit-metadata'))/'small'
    copied=tmp_path/'source';shutil.copytree(root,copied);fit=copied/'private/full-fit'
    meta=api().source_fit_metadata(fit);weights=fit/'checkpoints'/meta['selected_best']['file'];weights.parent.mkdir()
    weights.write_bytes(b'this is not the accepted trained checkpoint')
    with pytest.raises(ValueError,match='checkpoint|checksum|weight'):
        api().load_reference_runtime(meta,device='cpu')


def test_same_center_loop_preserves_all_four_requests_and_fails_on_sampling_error():
    torch=pytest.importorskip('torch');indices=[0,2140,2141,4281]
    records=[dict(index=i,emg_sha256='0'*64,target_mask_sha256='0'*64,identity_sha256='0'*64)for i in range(4282)]
    samples={i:sample(i)for i in indices}
    for i,s in samples.items():records[i]=admitted_record(s,i)
    class NativeShape(torch.nn.Module):
        def forward(self,batch):return batch['emg'][:,:1,:5].expand(1,22,5)
    payload,lengths=api().predict_reference_dataset(NativeShape(),samples,records,
        indices,device='cpu')
    assert len(payload['parts'])==4 and payload['source_indices']==indices and lengths==[5]
    assert all(torch.equal(p['pred'],torch.ones((1,22,1)))for p in payload['parts'])
    assert payload['parts'][0]['mask'].item() is False
    class Unavailable:
        def __getitem__(self,index):
            if index==2140:raise OSError('input file became unavailable')
            return samples[index]
    with pytest.raises(OSError,match='unavailable'):
        api().predict_reference_dataset(NativeShape(),Unavailable(),records,indices,device='cpu')


def replay_records():
    base=dict(scope='engineering',fit_result_sha256='1'*64,input_admission_sha256='2'*64,
        source_indices=[0,2140,2141,4281],model_before_sha256='3'*64,model_after_sha256='3'*64,
        prediction_sha256='4'*64,runtime_policy=dict(device='cpu',precision='32-true',batch_size=1),
        sources_sha256={'source':'5'*64})
    reference=dict(base,kind='native',process=dict(pid=2147483647,start_ticks='0',uid=0),finished_ns=10)
    candidate=dict(base,kind='replay',process=dict(pid=2147483646,start_ticks='0',uid=0),started_ns=11)
    return reference,candidate


@pytest.mark.parametrize('damage',['same_process','source_alive','chronology','fit','scope','precision','indices','prediction'])
def test_replay_requires_closed_distinct_process_and_exact_predeclared_context(damage):
    reference,candidate=replay_records();api().validate_replay_identity(reference,candidate)
    if damage=='same_process':candidate['process']=reference['process']
    elif damage=='source_alive':
        fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        reference['process']=dict(pid=os.getpid(),start_ticks=fields[19],uid=os.getuid())
    elif damage=='chronology':candidate['started_ns']=9
    elif damage=='fit':candidate['fit_result_sha256']='9'*64
    elif damage=='scope':candidate['scope']='full'
    elif damage=='precision':candidate['runtime_policy']=dict(candidate['runtime_policy'],precision='bf16-mixed')
    elif damage=='indices':candidate['source_indices']=[0,1,2,3]
    else:candidate['prediction_sha256']='9'*64
    with pytest.raises(ValueError):api().validate_replay_identity(reference,candidate)


def test_source_changed_during_hash_is_rejected_before_guard_can_adopt_it(tmp_path,monkeypatch):
    p=tmp_path/'input';p.write_bytes(b'accepted')
    expected=hashlib.sha256(b'accepted').hexdigest();m=api();original=m._file_hash
    def moving(path):
        digest=original(path)
        if Path(path)==p:p.write_bytes(b'changed after the hash')
        return digest
    monkeypatch.setattr(m,'_file_hash',moving)
    with pytest.raises(ValueError):m.guarded_source_files({str(p):expected})


@pytest.mark.parametrize('damage',['state','forwards','optimizer','precision'])
def test_execution_claim_requires_selected_journal_state_and_actual_full_counts(damage):
    m=api();state='a'*64
    source=dict(selected_model_state_sha256=state)
    value=dict(runtime_policy=dict(precision='32-true',batch_size=1,autocast_enabled=False,
        matmul_precision='high',point_selector='raw[...,T//2:T//2+1]',device='cpu',gpu_uuid=None,
        cuda_matmul_allow_tf32=True,cudnn_allow_tf32=True),samples=4,batch_size=1,
        neural_forwards=4,optimizer_updates=0,optimizer_construction_attempts=0,
        native_runtime_construction_calls=1,model_before_sha256=state,model_after_sha256=state,
        started_ns=1,finished_ns=2,cuda_initialized=False,scientific_eligible=False,full_paper_complete=False)
    m.validate_reference_execution(value,source)
    if damage=='state':value.update(model_before_sha256='b'*64,model_after_sha256='b'*64)
    elif damage=='forwards':value['neural_forwards']=3
    elif damage=='optimizer':value['optimizer_construction_attempts']=1
    else:value['runtime_policy']['precision']='bf16-mixed'
    with pytest.raises(ValueError):m.validate_reference_execution(value,source)
