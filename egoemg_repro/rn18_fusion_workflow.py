"""Exact producer exit gate and CPU-only full trained RN18 fusion handoff.

Queue readiness grants permission only to independently read the actual native
producer. It is never a trained-weight, GPU-qualification or scientific permit.
The four CLI actions reopen full source/data/producers, preserving native LAST
vision plus validation-BEST EMG. They never train or construct an optimizer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re

from .classic_emg_cohort import _safe,_json,_StatGuard
from .emg2pose_memory_process import process_identity

STAGES=('qual-prefix','qual-resume','qual-export','qual-reopen','full-fit','full-export','full-reopen')
PRODUCER_FORMAT='actual_rn18_cotrain_original_batch_qualification_and_full150_workflow_v1'
FORMAT='full_trained_rn18_fusion_cpu_handoff_action_v1'


def _sha(path):
    with _safe(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _same(handle):
    if(type(handle)is not dict or type(handle.get('pid'))is not int or handle['pid']<=0
        or type(handle.get('start_ticks'))is not str or not handle['start_ticks'].isdigit()):
        raise ValueError('exact PID/start pair required')
    current=process_identity(handle['pid'])
    return current is not None and current['start_ticks']==handle['start_ticks']


def _owner_pair(value,expected):
    if(set(expected)!={'wrapper','child'}or any(value.get(k)!=expected[k]for k in expected)
        or expected['wrapper']==expected['child']):raise ValueError('original producer owner identity differs')
    return {k:_same(expected[k])for k in expected}


def cotrain_dependency_status(root,*,expected_bundle,expected_handles):
    """Observe the same actual handles; never retry or infer successful exit."""
    try:
        root=_safe(root);private=root/'private';manifest=_json(private/'stage-receipt.json')
        body={k:v for k,v in manifest.items()if k!='bundle_sha256'}
        digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if(manifest['bundle_sha256']!=expected_bundle or digest!=expected_bundle
            or type(manifest['source_commit'])is not str or re.fullmatch('[0-9a-f]{40}',manifest['source_commit'])is None
            or manifest['full_paper_complete']is not False or not manifest['files_sha256']):
            raise ValueError('pinned original producer source capsule differs')
        paths=[private/'stage-receipt.json']
        for name in manifest['files_sha256']:
            if Path(name).is_absolute()or '..'in Path(name).parts:raise ValueError('relative nonsymlink producer source member required')
            paths.append(_safe(root/name))
        guard=_StatGuard(paths)
        if _json(private/'stage-receipt.json')!=manifest:raise ValueError('producer capsule generation changed')
        for name,sha in manifest['files_sha256'].items():
            if _sha(root/name)!=sha:raise ValueError('original producer source member differs')
        guard.check();state=_json(private/'execution-state.json');live=_owner_pair(state,expected_handles)
        if state['status']=='running':
            if not live['wrapper']:
                # The recorder atomically commits exit before it exits. Re-read
                # once to allow an observation spanning that actual boundary.
                newer=_json(private/'execution-state.json')
                if newer!=state:
                    state=newer;live=_owner_pair(state,expected_handles)
                else:raise ValueError('original producer wrapper missing without durable terminal exit')
            if state['status']=='running':
                guard.check();return dict(status='waiting',ready_for_independent_admission=False,
                    original_owners_live=live,full_trained_handoff_admitted=False,scientific_eligible=False,full_paper_complete=False)
        if state['status']!='complete'or type(state.get('actual_exit_code'))is not int or state['actual_exit_code']!=0:
            raise ValueError('original producer did not terminate successfully')
        if any(live.values()):
            guard.check();return dict(status='waiting',ready_for_independent_admission=False,
                original_owners_live=live,full_trained_handoff_admitted=False,scientific_eligible=False,full_paper_complete=False)
        guard.add(private/'execution-state.json')
        if _json(private/'execution-state.json')!=state:raise ValueError('producer terminal generation changed')
        guard.add(private/'workflow-result.json');workflow=_json(private/'workflow-result.json')
        if(workflow['format']!=PRODUCER_FORMAT or workflow['source_commit']!=manifest['source_commit']
            or workflow['bundle_sha256']!=expected_bundle or len(workflow['actual_owned_stages'])!=len(STAGES)
            or any(workflow[k]is not True for k in ('all_owned_consumers_exited','all_frozen_members_unchanged','trained_full_initializer_complete'))
            or workflow['scientific_settings_added']!=0 or workflow['full_paper_complete']is not False):
            raise ValueError('complete seven-stage native initializer workflow required')
        actors=set()
        for label,recorded in zip(STAGES,workflow['actual_owned_stages']):
            directory=private/('owned-'+label)
            for name in ('started.json','stage-result.json'):guard.add(directory/name)
            started=_json(directory/'started.json');result=_json(directory/'stage-result.json')
            if(result!=recorded or result['status']!='success'or type(result['worker_exit_code'])is not int
                or result['worker_exit_code']!=0 or result['all_owned_consumers_exited']is not True
                or any(result[k]!=started[k]for k in ('owner','worker','guardian'))
                or any(result['owner'][k]!=expected_handles['child'][k]for k in ('pid','start_ticks'))):
                raise ValueError('actual owned initializer stage/exit record differs')
            for key in ('worker','guardian'):
                h=started[key];pair=(h['pid'],h['start_ticks'])
                if _same(h)or pair in actors:raise ValueError('original initializer consumer is live or reused')
                actors.add(pair)
            if label in ('qual-prefix','qual-resume','full-fit'):
                observed=workflow['actual_nvml_observations'][label]
                if(observed['worker']!=result['worker']or observed['actual_nvml_pid']!=result['worker']['pid']):
                    raise ValueError('actual owned native GPU observation differs')
        selected=workflow['actual_selected_full_initializer'];fit=selected['fit_result'];export=selected['selected_export']
        if(fit['scope']!='full'or fit['stage_complete']is not True or fit['full_training_terminated']is not True
            or type(fit['completed_epochs'])is not int or fit['completed_epochs']!=150
            or type(fit['optimizer_executions'])is not int or fit['optimizer_executions']!=36150
            or export['full_initializer']is not True or len(export['tensor_inventory'])!=110):
            raise ValueError('actual complete150/36150 producer and110tensor export required')
        guard.check();return dict(status='producer_closed',ready_for_independent_admission=True,
            terminal_state=state,source_commit=manifest['source_commit'],bundle_sha256=expected_bundle,
            workflow_file_sha256=_sha(private/'workflow-result.json'),closed_owned_stages=list(STAGES),closed_owner_handles=len(actors)+2,
            full_trained_handoff_admitted=False,scientific_eligible=False,full_paper_complete=False)
    except(OSError,KeyError,TypeError,IndexError,AttributeError)as exc:
        raise ValueError('complete pinned producer source, owners and durable stage records required')from exc


def run_handoff_action(action,request,output):
    """Full-data/full-producer CPU admission or two-real-validation-hand probe."""
    if action not in ('admit','reopen','construct','restore')or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('explicit CPU-only RN18 handoff action required before importing ML')
    required={'data','crops','upstream','cohort_receipt','vision_fit','emg_fit','vision_source_root','emg_source_root',
        'admission','runtime_packet','construction_record'}
    if type(request)is not dict or set(request)!=required:raise ValueError('exact full-data/producer locators required')
    import socket
    import tempfile
    import time
    import uuid
    import torch
    from .rn18_fusion_literal import compose_rn18_fusion
    from .rn18_fusion_cohort import admit_rn18_fusion_cohort,FULL_COUNTS
    from .rn18_fusion_trained_handoff import admit_rn18_fusion_initializers,read_rn18_fusion_initializers
    from .rn18_fusion_runtime import build_rn18_fusion_runtime
    from .vision_literal_runtime import vision_state_sha256
    from .vision_literal_train import _write
    from . import classic_emg_train as storage
    if torch.cuda.is_initialized():raise ValueError('CPU handoff action must not initialize CUDA')
    output=_safe(output,exists=False)
    if output.exists():raise ValueError('fresh handoff action record required')
    q=copy.deepcopy(request);start=time.monotonic();torch.set_num_threads(2)
    counters=dict(neural_constructors=0,optimizer_constructors=0,learned_forwards=0,optimizer_updates=0)
    original_module=torch.nn.Module.__init__;original_optimizer=torch.optim.Optimizer.__init__;original_connect=socket.socket.connect
    def module_init(module,*args,**kwargs):
        counters['neural_constructors']+=1
        if action in ('admit','reopen'):raise ValueError('model-free handoff reader reached a neural constructor')
        return original_module(module,*args,**kwargs)
    def no_optimizer(*a,**k):counters['optimizer_constructors']+=1;raise ValueError('handoff action cannot construct an optimizer')
    def no_network(*a,**k):raise ValueError('trained handoff and self-contained reload must remain offline')
    torch.nn.Module.__init__=module_init;torch.optim.Optimizer.__init__=no_optimizer;socket.socket.connect=no_network
    try:
        cfg=compose_rn18_fusion(q['upstream']);expected=_json(q['cohort_receipt'])
        c=admit_rn18_fusion_cohort(q['data'],q['crops'],upstream=q['upstream'],config=cfg,expected=expected,require_full=True)
        if c.identity['counts']!=FULL_COUNTS:raise ValueError('complete registered full fusion cohort required')
        args=dict(cohort=c,vision_fit=q['vision_fit'],emg_fit=q['emg_fit'],output=q['admission'],
            vision_source_root=q['vision_source_root'],emg_source_root=q['emg_source_root'])
        initializer=admit_rn18_fusion_initializers(**args)if action=='admit'else read_rn18_fusion_initializers(**args)
        sources=Path(q['admission'])/'producer-sources.json'
        if action=='admit':_write(sources,initializer.producer_sources)
        elif _json(sources)!=initializer.producer_sources:raise ValueError('complete trained producer CLI locators differ')
        result=dict(format=FORMAT,action=action,actor=process_identity(),full_cohort_sha256=c.identity['sha256'],
            full_counts=copy.deepcopy(c.identity['counts']),source_payload_files=c.identity['payload_file_count'],
            source_payload_bytes=sum(c.identity['payload_file_bytes'].values()),trained_initializers=copy.deepcopy(initializer.identity),
            producer_sources_sha256=_sha(sources),scientific_eligible=False,full_paper_complete=False,
            actual_cuda_qualification=False,formal_fusion_fit_started=False)
        if action=='construct':
            packet_path=_safe(q['runtime_packet'],exists=False)
            if packet_path.exists():raise ValueError('fresh trained construction packet required')
            prepared=build_rn18_fusion_runtime(c.upstream,c.config,initializers=initializer)
            from egoemg.datamodule import make_data_module
            from torch.utils.data import Subset
            ds=c.datasets['val'];indices=[0,len(ds)//2];dm=make_data_module(c.config)
            dm.batch_size=2;dm.num_workers=dm.eval_num_workers=0;dm.pin_memory=False;dm.persistent_workers=False;dm.prefetch_factor=None
            batch=next(iter(dm._make_loader(Subset(ds,indices),shuffle=False,num_workers=0)))
            if(batch['rn18_fusion_sample_index'].tolist()!=indices or batch['hand']!=['left','right']
                or batch['emg'].shape!=(2,8,12000)or batch['vision_img'].shape!=(2,3,256,256)):
                raise ValueError('actual source first validation anchor must include both hands')
            prepared.module.eval();before=vision_state_sha256(prepared.module.state_dict());input_before=storage.state_digest(batch)
            with torch.inference_mode():prediction,target,mask=[x.detach().cpu().clone()for x in prepared.module.forward(batch)]
            counters['learned_forwards']+=1
            if(prediction.shape!=(2,22,1)or not bool(torch.isfinite(prediction).all())
                or not torch.equal(target,batch['joint_angles'])or not torch.equal(mask,batch['label_valid_mask'])):
                raise ValueError('full trained native center forward differs from original targets/masks')
            after=vision_state_sha256(prepared.module.state_dict())
            if before!=after or storage.state_digest(batch)!=input_before:raise ValueError('native prediction probe changed source tensors')
            prepared.check();temporary=packet_path.with_name('.'+uuid.uuid4().hex+'.pt')
            try:
                with temporary.open('xb')as f:
                    torch.save(dict(format='full_trained_rn18_fusion_cpu_constructor_packet_v1',construction=prepared.identity,
                        state_dict=prepared.module.state_dict(),batch=batch,prediction=prediction,target=target,mask=mask),f)
                    f.flush();os.fsync(f.fileno())
                os.link(temporary,packet_path)
            finally:
                if temporary.exists():temporary.unlink()
            result.update(runtime_packet_sha256=_sha(packet_path),construction=prepared.identity,model_state_sha256=before,
                actual_validation_hand_requests=2,validation_sample_indices=indices,complete_native_transfer_verified=True,
                actual_native_factory_mode=prepared.mode)
        elif action=='restore':
            original=_json(q['construction_record']);packet_path=_safe(q['runtime_packet'])
            if(original['action']!='construct'or original['trained_initializers']!=initializer.identity
                or _sha(packet_path)!=original['runtime_packet_sha256']):raise ValueError('original complete trained constructor packet differs')
            packet=torch.load(packet_path,map_location='cpu',weights_only=False)
            if(packet['format']!='full_trained_rn18_fusion_cpu_constructor_packet_v1'or packet['construction']!=original['construction']):
                raise ValueError('source-bound complete RN18 constructor packet required')
            names=('HF_HOME','TORCH_HOME','HUGGINGFACE_HUB_CACHE');previous={k:os.environ.get(k)for k in names}
            try:
                with tempfile.TemporaryDirectory(prefix='rn18-trained-fusion-empty-caches-')as scratch:
                    for name in names:os.environ[name]=str(Path(scratch)/name)
                    prepared=build_rn18_fusion_runtime(c.upstream,c.config,state=packet['state_dict'],expected_initialization=packet['construction'])
                    prepared.module.eval()
                    with torch.inference_mode():actual=prepared.module.forward(packet['batch'])
                    counters['learned_forwards']+=1
                    if(prepared.mode!='strict_self_contained_restore'or
                        any(not torch.equal(x.detach().cpu(),packet[k])for x,k in zip(actual,('prediction','target','mask')))
                        or vision_state_sha256(prepared.module.state_dict())!=original['model_state_sha256']):
                        raise ValueError('cold-cache complete trained state/prediction restoration differs')
                    prepared.check()
            finally:
                for name,value in previous.items():
                    if value is None:os.environ.pop(name,None)
                    else:os.environ[name]=value
            result.update(runtime_packet_sha256=_sha(packet_path),model_state_sha256=original['model_state_sha256'],
                actual_validation_hand_requests=2,validation_sample_indices=original['validation_sample_indices'],
                strict_self_contained_state_prediction_exact=True,empty_asset_caches=True,actual_native_factory_mode=prepared.mode)
        c.check(require_full=True);initializer.check(require_full=True)
        if torch.cuda.is_initialized()or counters['optimizer_constructors']or counters['optimizer_updates']:
            raise ValueError('CPU trained handoff may not initialize CUDA or an optimizer')
        result.update(counters=counters,cuda_initialized=False,seconds=time.monotonic()-start)
        _write(output,result);return result
    finally:
        torch.nn.Module.__init__=original_module;torch.optim.Optimizer.__init__=original_optimizer;socket.socket.connect=original_connect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('admit','reopen','construct','restore'))
    parser.add_argument('--request',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();value=run_handoff_action(args.action,_json(args.request),args.output)
    print(json.dumps({k:value[k]for k in ('format','action','seconds','counters','cuda_initialized','scientific_eligible')}))


if __name__=='__main__':main()
