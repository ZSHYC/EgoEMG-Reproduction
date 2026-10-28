"""Selected native WiLoR evaluation, exact replay and source-bound acceptance.

Input fingerprints are coupled to actual original source indices. CPU engineering
uses one admitted window per hand with both original repetitions, never a full
corpus or CUDA certificate. No original forward/loss/precision operator changes.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
import math
import os
from pathlib import Path
import re
import time

from . import wilor_native_train as train
from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='native_wilor_selected_pass_v1'
PROFILE='native_wilor_pretest_profile_v1'
QUALIFICATION='native_wilor_raw_cuda_qualification_v1'
_EVAL_FILES=('egoemg_repro/wilor_native_eval.py','egoemg_repro/wilor_native_train.py',
    'egoemg_repro/wilor_literal_runtime.py','egoemg_repro/vision_literal_eval.py',
    'egoemg_repro/vision_literal_data.py','egoemg_repro/classic_emg_train.py',
    'egoemg_repro/classic_emg_report.py','egoemg_repro/metrics.py')


def _policy():
    return dict(prediction='exact typed prediction/target/mask/input replay',native_mae_atol_rad=2e-6,
        native_mae_rtol=2e-6,native_aggregation='sample-count weighted original per-batch masked L1, float32 accumulation',
        pooled_aggregation='float64 absolute-error sum / exact valid-angle count; degrees',
        scope='fixed before validation; no test-driven tolerance or checkpoint changes')


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported native WiLoR evaluation source changed')
    return {n:_file_hash(ROOT/n) for n in _EVAL_FILES}


def _capture_sources():
    guard=_StatGuard([ROOT/n for n in _EVAL_FILES]);value=_sources();guard.check();return guard,value


def _json(path):
    from .vision_literal_train import _json as read
    return read(path)


def _write(path,value):
    from .vision_literal_train import _write as write
    return write(path,value)


def _sealed(value):
    from .vision_literal_train import _sealed as seal
    return seal(value)


def _check(value,format):
    if (type(value) is not dict or value.get('format')!=format or
        value.get('sha256')!=_hash({k:v for k,v in value.items() if k!='sha256'}) or
        value.get('scientific_eligible') is not False or value.get('full_paper_complete') is not False):
        raise ValueError('native WiLoR artifact seal/format/scope differs')


def _path(value):
    if not isinstance(value,(str,os.PathLike)):raise ValueError('actual native artifact path required')
    return _safe(value)


def _evaluation_layout(scope,role,source_samples):
    if (type(scope) is not str or scope not in train.SCOPES or role not in ('val','test') or
        type(source_samples) is not int or source_samples<=0 or source_samples%4):
        raise ValueError('explicit original repeated evaluation population required')
    if scope=='cpu_engineering':
        indices=[0,source_samples//4,source_samples//2,3*source_samples//4]
        description='bounded CPU: first admitted window per hand, both original repetitions; not full population'
    else:
        if source_samples!=train.FULL_COUNTS[role]:raise ValueError('actual complete registered GPU evaluation population required')
        indices=list(range(source_samples));description='complete original source order, both hands and repetitions'
    return train._sealed(dict(format='native_wilor_evaluation_population_v1',scope=scope,role=role,
        source_samples=source_samples,samples=len(indices),source_indices=indices,batch_size=1 if scope=='cpu_engineering' else 18,
        selection=description,scientific_eligible=False,full_paper_complete=False))


def _digest(value):
    from .classic_emg_train import state_digest
    return state_digest(value)


def wilor_statistics(parts,*,samples,batch_size,role,source_samples,source_indices):
    import torch
    from .vision_literal_eval import vision_statistics
    try:
        if (type(source_samples) is not int or source_samples<=0 or source_samples%4 or type(samples) is not int or
            samples<=0 or samples%4 or type(source_indices) is not list or any(type(i) is not int for i in source_indices)):
            raise ValueError('complete native repeated source population required')
        expected=list(range(source_samples)) if samples==source_samples else [0,source_samples//4,source_samples//2,3*source_samples//4]
        if source_indices!=expected or len(source_indices)!=samples:raise ValueError('actual bounded or complete source-index projection differs')
        first={};position=0;anchors=source_samples//4
        for part in parts:
            n=len(part['pred']);src=part['source_index'];unique=part['source_unique'];repeat=part['source_repeat'];window=part['source_window']
            expected_src=torch.tensor(source_indices[position:position+n],dtype=torch.int64)
            if (any(not torch.is_tensor(t) or t.dtype!=torch.int64 or t.shape!=(n,) for t in (src,unique,repeat,window)) or
                not torch.equal(src,expected_src) or not torch.equal(unique,src%(source_samples//2)) or
                not torch.equal(repeat,src//(source_samples//2)) or not torch.equal(window,src%anchors) or
                len(part['sample_input_sha256'])!=n or re.fullmatch('[0-9a-f]{64}',str(part['input_sha256'])) is None):
                raise ValueError('typed native source indices/repetition/input digest differs')
            for j in range(n):
                digest=part['sample_input_sha256'][j]
                if type(digest) is not str or re.fullmatch('[0-9a-f]{64}',digest) is None:raise ValueError('actual sample input SHA required')
                hand='left' if int(unique[j])<anchors else 'right'
                if part['hand'][j]!=hand:raise ValueError('original hand/source-index binding differs')
                packet=dict(target=part['target'][j],mask=part['mask'][j],vision=part['vision'][j],hand=hand,
                    split=part['split'][j],user=part['user'][j],center=part['center'][j],episode=part['episode'][j],
                    source_window=window[j],source_unique=unique[j],input_sha256=digest)
                key=int(unique[j]);paired=_digest(packet)
                if int(repeat[j])==0:
                    if key in first:raise ValueError('duplicated first-repetition input')
                    first[key]=paired
                elif first.get(key)!=paired:raise ValueError('repeated WiLoR input/target/generated mask changed')
            position+=n
        if position!=samples or len(first)!=samples//2:raise ValueError('partial paired source population')
        value=vision_statistics(parts,samples=samples,batch_size=batch_size,role=role)
        users=sorted({u for part in parts for u in part['user']});aliases={u:'user_'+str(i).zfill(3) for i,u in enumerate(users)}
        def anonymous(item):
            if isinstance(item,dict):
                return {k:({aliases[u]:anonymous(cell) for u,cell in v.items()} if k=='by_user' else anonymous(v)) for k,v in item.items()}
            if isinstance(item,list):return [anonymous(v) for v in item]
            return item
        value=anonymous(value)
        value.update(format='native_wilor_center_statistics_v1',source_population_samples=source_samples,
            source_indices_sha256=_hash(source_indices),scientific_eligible=False,full_paper_complete=False)
        return value
    except (KeyError,TypeError,IndexError,AttributeError,RuntimeError) as exc:raise ValueError('malformed native WiLoR prediction population') from exc


def _device(scope):
    import torch
    if scope=='cpu_engineering':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU native evaluation requires CUDA hidden at startup')
        return dict(accelerator='cpu',visible_device='',cuda_initialized=False)
    if torch.version.cuda is None or not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise ValueError('actual single CUDA BF16 evaluation required')
    free,total=torch.cuda.mem_get_info(0)
    if free<16*1024**3:raise ValueError('actual16GiB free reserve required for original B18 frozen inference')
    return dict(accelerator='gpu',visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'),name=torch.cuda.get_device_name(0),
        cuda_initialized=True,free_memory_before=free,total_memory=total,cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version())


def _devices(value):
    gpu=value['scope']!='cpu_engineering';name='cuda:0' if gpu else 'cpu'
    expected=dict(device_type='cuda' if gpu else 'cpu',input_device=name,module_device=name,prediction_device=name,cuda_initialized=gpu)
    if (value['forward_devices']!=[expected]*len(value['statistics']['batch_sizes']) or
        value['post_pass_module_device']!=('cuda:0' if gpu and value['kind']=='replay' else 'cpu')):
        raise ValueError('actual per-forward or post-teardown native WiLoR device differs')


def _execution_header(value,fit_finished_ns):
    try:
        process=value['process'];start=value['started_ns'];finish=value['finished_ns']
        if (type(process) is not dict or set(process)!={'pid','start_ticks','uid'} or
            type(process['pid']) is not int or process['pid']<=0 or
            type(process['uid']) is not int or process['uid']<0 or
            type(process['start_ticks']) is not str or not process['start_ticks'].isdigit() or
            type(start) is not int or type(finish) is not int or start<=fit_finished_ns or finish<=start):
            raise ValueError('actual native WiLoR process identity and ordered execution times required')
        _devices(value);device=value['execution_device']
        if value['scope']=='cpu_engineering':
            if device!=dict(accelerator='cpu',visible_device='',cuda_initialized=False):
                raise ValueError('actual CUDA-hidden CPU inference receipt differs')
        elif (device['accelerator']!='gpu' or device['cuda_initialized'] is not True or
            type(device['visible_device']) is not str or not device['visible_device'].startswith('GPU-')):
            raise ValueError('actual UUID-selected CUDA inference receipt required')
    except (KeyError,TypeError,AttributeError) as exc:raise ValueError('malformed native WiLoR execution receipt') from exc


class _Evidence:
    """One source/input guarded transaction; no cross-call result cache."""
    def __init__(self):
        self.guard,self.sources=_capture_sources();self.fits={};self.passes={};self.profiles={};self.active=set()
    def fit(self,path):
        path=_path(path)
        if path not in self.fits:
            result,identity,last,cohort,assets,guard=train._read_fit(path,require_full=False)
            self.guard=train._guard_merge(self.guard,guard);del last
            import torch
            from .vision_literal_runtime import vision_state_sha256
            selected=path/'checkpoints'/result['selected_best']['file'];self.guard.check()
            checkpoint=torch.load(str(selected),map_location='cpu',weights_only=False,mmap=True)
            if vision_state_sha256(checkpoint['state_dict'])!=result['selected_best']['model_sha256']:
                raise ValueError('actual selected raw native WiLoR model differs')
            self.guard.check();self.fits[path]=(result,identity,cohort,assets,checkpoint)
        return self.fits[path]


def _read_pass(path,context):
    path=_path(path)
    if path in context.passes:return context.passes[path]
    key=('pass',path)
    if key in context.active:raise ValueError('cyclic native WiLoR prediction evidence')
    context.active.add(key)
    try:
        import torch
        for n in ('pass.json','predictions.pt','native-metrics.json'):context.guard.add(path/n)
        v=_json(path/'pass.json');_check(v,FORMAT)
        if v['sources_sha256']!=context.sources or v['agreement_policy']!=_policy():raise ValueError('native WiLoR pass source/agreement policy differs')
        if v['role'] not in ('val','test') or v['kind'] not in ('native','replay'):raise ValueError('explicit native WiLoR role/kind required')
        fit,identity,cohort,assets,checkpoint=context.fit(v['fit_path']);role=v['role']
        layout=_evaluation_layout(fit['scope'],role,identity['cohort']['counts'][role])
        if (v['fit_result_sha256']!=fit['sha256'] or v['selected_best']!=fit['selected_best'] or v['cohort_sha256']!=cohort.identity['sha256'] or
            v['scope']!=fit['scope'] or v['full_training_terminated'] is not fit['full_training_terminated'] or v['layout']!=layout or
            v['model_before_sha256']!=fit['selected_best']['model_sha256'] or v['model_after_sha256']!=v['model_before_sha256'] or
            v['runtime_versions']!=identity['construction']['runtime_versions'] or v['started_ns']<=fit['finished_ns'] or
            type(v['finished_ns']) is not int or v['finished_ns']<=v['started_ns']):raise ValueError('native WiLoR pass differs from actual selected fit/runtime/population')
        if _file_hash(path/'predictions.pt')!=v['payload_sha256']:raise ValueError('actual native WiLoR prediction file changed')
        payload=torch.load(str(path/'predictions.pt'),map_location='cpu',weights_only=True)
        if type(payload) is not dict or set(payload)!={'parts','population'}:raise ValueError('complete typed WiLoR prediction payload required')
        if _digest(payload['parts'])!=v['prediction_sha256']:raise ValueError('native WiLoR typed payload digest differs')
        from .vision_literal_eval import vision_population
        population=vision_population(cohort.common)
        if payload['population']!=population or _hash(population)!=v['population_sha256'] or v['population']!=population['public']:
            raise ValueError('actual native WiLoR population metadata differs')
        statistics=wilor_statistics(payload['parts'],samples=layout['samples'],batch_size=layout['batch_size'],role=role,
            source_samples=layout['source_samples'],source_indices=layout['source_indices'])
        if statistics!=v['statistics'] or _json(path/'native-metrics.json')!=v['native_metrics']:raise ValueError('raw WiLoR statistics/native metrics sidecar differs')
        _execution_header(v,fit['finished_ns'])
        if v['scope']!='cpu_engineering' and v['execution_device']['visible_device']!=identity['runtime_device']['visible_device']:
            raise ValueError('actual evaluation GPU differs from selected native fit')
        if v['kind']=='native':_close_native(statistics,v['native_metrics'],role)
        elif v['native_metrics']!={}:raise ValueError('independent WiLoR replay cannot invent native metrics')
        if role=='test':
            profile=_read_profile(v['profile_path'],context)
            if (v['profile_sha256']!=profile['sha256'] or profile['fit_result_sha256']!=fit['sha256'] or v['started_ns']<=profile['finished_ns']):
                raise ValueError('test did not follow its frozen independent validation profile')
        elif v['profile_path'] is not None or v['profile_sha256'] is not None:raise ValueError('validation cannot consume a test profile')
        if v['kind']=='replay':
            reference=_read_pass(v['reference_path'],context)
            if (reference['kind']!='native' or reference['role']!=role or reference['fit_result_sha256']!=fit['sha256'] or
                reference['sha256']!=v['reference_sha256'] or reference['prediction_sha256']!=v['prediction_sha256'] or
                reference['process']==v['process'] or v['started_ns']<=reference['finished_ns']):raise ValueError('actual fresh exact WiLoR prediction replay differs')
        elif v['reference_path'] is not None or v['reference_sha256'] is not None:raise ValueError('native WiLoR pass has no replay reference')
        context.guard.check();context.passes[path]=v;return v
    finally:context.active.remove(key)


def _close_native(statistics,metrics,role):
    value=metrics.get(role+'_mae');p=_policy()
    if type(value) is not float or not math.isfinite(value) or not math.isclose(value,statistics['native_batch_weighted_mae_rad'],
        abs_tol=p['native_mae_atol_rad'],rel_tol=p['native_mae_rtol']):raise ValueError('native sample-weighted MAE differs under fixed WiLoR policy')


def read_wilor_pass(output):
    context=_Evidence();value=_read_pass(output,context);context.guard.check();return value


def _read_profile(path,context):
    path=_path(path)
    if path in context.profiles:return context.profiles[path]
    key=('profile',path)
    if key in context.active:raise ValueError('cyclic native WiLoR pretest profile')
    context.active.add(key)
    try:
        context.guard.add(path/'profile.json');v=_json(path/'profile.json');_check(v,PROFILE)
        a=_read_pass(v['native_path'],context);b=_read_pass(v['replay_path'],context)
        if (a['role']!='val' or a['kind']!='native' or b['role']!='val' or b['kind']!='replay' or b['reference_sha256']!=a['sha256'] or
            a['sha256']!=v['native_sha256'] or b['sha256']!=v['replay_sha256'] or v['fit_result_sha256']!=a['fit_result_sha256'] or
            v['agreement_policy']!=_policy() or v['finished_ns']<=max(a['finished_ns'],b['finished_ns'])):
            raise ValueError('actual native/replayed WiLoR pretest freeze differs')
        context.guard.check();context.profiles[path]=v;return v
    finally:context.active.remove(key)


def freeze_wilor_validation(native,replay,output):
    context=_Evidence();a=_read_pass(native,context);b=_read_pass(replay,context)
    if a['role']!='val' or a['kind']!='native' or b['role']!='val' or b['kind']!='replay' or b['reference_sha256']!=a['sha256']:
        raise ValueError('actual matching native validation and independent exact replay required')
    output=_safe(output,exists=False)
    from .classic_emg_train import _space
    _space(output.parent);output.mkdir(mode=0o700,exist_ok=False)
    v=_sealed(dict(format=PROFILE,native_path=str(_path(native)),replay_path=str(_path(replay)),native_sha256=a['sha256'],
        replay_sha256=b['sha256'],fit_result_sha256=a['fit_result_sha256'],agreement_policy=_policy(),
        finished_ns=time.time_ns(),scientific_eligible=False,full_paper_complete=False))
    context.guard.check();_write(output/'profile.json',v);return _read_profile(output,context)


def _loader(cohort,layout):
    from egoemg.datamodule import make_data_module
    from torch.utils.data import Subset
    dm=make_data_module(cohort.config);dm.num_workers=dm.eval_num_workers=0;dm.batch_size=layout['batch_size']
    dataset=cohort.datasets[layout['role']]
    if layout['samples']!=len(dataset):dataset=Subset(dataset,layout['source_indices'])
    loader=dm._make_loader(dataset,shuffle=False)
    if loader.batch_size!=layout['batch_size'] or loader.num_workers!=0 or loader.generator is not None or loader.drop_last:
        raise ValueError('original native WiLoR evaluation loader differs')
    return loader


def _capture(batch,out,module,cohort,layout,parts,devices):
    import torch
    position=sum(len(p['index']) for p in parts);n=len(batch['vision_sample_index']);expected=layout['source_indices'][position:position+n]
    if (n!=min(layout['batch_size'],layout['samples']-position) or batch['vision_sample_index'].dtype!=torch.int64 or
        batch['vision_sample_index'].detach().cpu().tolist()!=expected):raise ValueError('actual original WiLoR evaluation source-index coverage differs')
    if type(out) is not tuple or len(out)!=3:raise ValueError('original native WiLoR tuple required')
    pred,target,mask=out;device=next(module.parameters()).device
    if (pred.shape!=(n,22,1) or target.shape!=pred.shape or pred.dtype not in (torch.float32,torch.bfloat16) or
        target.dtype!=torch.float32 or mask.shape!=(n,1) or mask.dtype!=torch.bool or batch['vision_valid_mask'].shape!=(n,1) or
        batch['vision_valid_mask'].dtype!=torch.bool or batch['vision_img'].shape!=(n,3,256,256) or batch['vision_img'].dtype!=torch.float32 or
        not bool(torch.isfinite(pred).all() and torch.isfinite(target).all() and torch.isfinite(batch['vision_img']).all()) or
        not torch.equal(target,batch['joint_angles']) or not torch.equal(mask,batch['label_valid_mask']) or
        pred.device!=device or batch['vision_img'].device!=device):raise ValueError('actual native WiLoR tuple/input/mask/forward device differs')
    ds=cohort.datasets[layout['role']];anchors=len(ds.windows);hashes=[]
    for j,index in enumerate(expected):
        unique=index%(2*anchors);repeat=index//(2*anchors);hand,window=divmod(unique,anchors);w=ds.windows[window]
        fields=dict(vision_unique_sample_index=unique,vision_repeat_index=repeat,vision_window_index=window,
            target_hand_index=hand,center=w['center'],start=w['start'],end=w['end'],episode=w['episode'],jitter_offset=0)
        if (any(int(batch[k][j])!=v for k,v in fields.items()) or batch['role'][j]!=layout['role'] or batch['split'][j]!=w['split'] or
            batch['hand'][j]!=('left','right')[hand] or batch['dataset_name'][j]!='egoemg' or batch['frame_mapping_field'][j]!=ds.frame_mapping_field):
            raise ValueError('native WiLoR original evaluation point provenance differs')
        hashes.append(_digest(dict(image=batch['vision_img'][j],vision=batch['vision_valid_mask'][j],target=target[j],mask=mask[j],
            source_unique=unique,window=window,episode=w['episode'],center=w['center'],hand=hand)))
    cpu=lambda t:t.detach().cpu().clone()
    src=cpu(batch['vision_sample_index']);idx=torch.arange(position,position+n,dtype=torch.int64)
    parts.append(dict(pred=cpu(pred),target=cpu(target),mask=cpu(mask),vision=cpu(batch['vision_valid_mask']),
        index=idx,unique=idx%(layout['samples']//2),repeat=idx//(layout['samples']//2),source_index=src,
        source_unique=src%(layout['source_samples']//2),source_repeat=src//(layout['source_samples']//2),source_window=src%anchors,
        hand=list(batch['hand']),split=list(batch['split']),user=[str(x) for x in batch['subject']],
        center=cpu(batch['center']),episode=cpu(batch['episode']),input_sha256=_hash(hashes),sample_input_sha256=hashes))
    devices.append(dict(device_type=device.type,input_device=str(batch['vision_img'].device),module_device=str(device),
        prediction_device=str(pred.device),cuda_initialized=torch.cuda.is_initialized()))


def run_wilor_pass(fit,output,*,role,kind,reference=None,profile=None,require_full=True):
    if type(require_full) is not bool:raise ValueError('full prediction requirement must be boolean')
    if role not in ('val','test') or kind not in ('native','replay'):raise ValueError('explicit native/replay and val/test required')
    if role=='test' and profile is None:raise ValueError('freeze independent validation profile before test')
    if role=='val' and profile is not None:raise ValueError('validation cannot consume a test profile')
    if kind=='replay' and reference is None:raise ValueError('matching native reference required before replay')
    if kind=='native' and reference is not None:raise ValueError('native pass has no replay reference')
    output=_safe(output,exists=False)
    if output.exists():raise FileExistsError('native WiLoR evaluation output already exists')
    context=_Evidence();fit_path=_path(fit);result,identity,cohort,assets,checkpoint=context.fit(fit_path)
    if require_full and not result['full_training_terminated']:raise ValueError('engineering WiLoR is not full native training')
    ref=_read_pass(reference,context) if kind=='replay' else None
    frozen=_read_profile(profile,context) if role=='test' else None
    process=train._process()
    if ref is not None and (ref['kind']!='native' or ref['role']!=role or ref['fit_result_sha256']!=result['sha256'] or ref['process']==process):
        raise ValueError('matching native reference and a fresh replay process required')
    if frozen is not None and frozen['fit_result_sha256']!=result['sha256']:raise ValueError('pretest freeze belongs to another WiLoR fit')
    from .classic_emg_train import _space
    from .vision_literal_runtime import vision_state_sha256
    from .wilor_literal_runtime import build_wilor_runtime
    from .vision_literal_eval import vision_population
    import torch
    _space(output.parent,budget=16*1024**2);device=_device(result['scope'])
    prepared=build_wilor_runtime(cohort.upstream,cohort.config,assets=assets,state=checkpoint['state_dict'],expected_initialization=identity['construction'])
    layout=_evaluation_layout(result['scope'],role,identity['cohort']['counts'][role]);loader=_loader(cohort,layout)
    population=vision_population(cohort.common);before=vision_state_sha256(prepared.module.state_dict());parts=[];devices=[];started_ns=time.time_ns()
    output.mkdir(mode=0o700,exist_ok=False);original=prepared.module.forward
    def observed(batch):
        context.guard.check();cohort.guard.check();prepared._source_guard.check();out=original(batch)
        _capture(batch,out,prepared.module,cohort,layout,parts,devices);return out
    prepared.module.forward=observed;native={}
    try:
        if kind=='native':
            from omegaconf import OmegaConf
            from .classic_emg_train import pl
            kwargs=OmegaConf.to_container(cohort.config.trainer,resolve=True)
            kwargs.update(accelerator=device['accelerator'],devices=1,precision=identity['policy']['precision'],logger=False,
                enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0)
            trainer=pl.Trainer(**kwargs)
            if getattr(trainer.precision_plugin,'scaler',None) is not None:raise ValueError('original WiLoR BF16/CPU32 evaluation has no FP16 scaler')
            method=trainer.validate if role=='val' else trainer.test
            metrics=method(prepared.module,dataloaders=loader,verbose=False);native={k:float(v) for k,v in metrics[0].items()}
            if not all(math.isfinite(v) for v in native.values()):raise ValueError('nonfinite original native WiLoR metrics')
        else:
            from lightning_fabric.utilities.apply_func import move_data_to_device
            target='cpu' if device['accelerator']=='cpu' else 'cuda:0';prepared.module.to(target).eval()
            with torch.inference_mode():
                for batch in loader:
                    batch=move_data_to_device(batch,target)
                    with (nullcontext() if target=='cpu' else torch.autocast('cuda',dtype=torch.bfloat16)):prepared.module.forward(batch)
        statistics=wilor_statistics(parts,samples=layout['samples'],batch_size=layout['batch_size'],role=role,
            source_samples=layout['source_samples'],source_indices=layout['source_indices'])
        if kind=='native':_close_native(statistics,native,role)
        digest=_digest(parts)
        if ref is not None and digest!=ref['prediction_sha256']:raise ValueError('actual fixed exact typed WiLoR prediction replay failed')
        context.guard.check();cohort.check(require_full=result['scope']!='cpu_engineering');prepared.check()
        after=vision_state_sha256(prepared.module.state_dict())
        if before!=after or before!=result['selected_best']['model_sha256']:raise ValueError('actual selected WiLoR prediction model changed')
        payload=dict(parts=parts,population=population)
        from .classic_emg_train import _atomic
        _atomic(output/'predictions.pt',lambda f:torch.save(payload,f),16*1024**2,replace=False)
        _write(output/'native-metrics.json',native)
        v=_sealed(dict(format=FORMAT,role=role,kind=kind,scope=result['scope'],fit_path=str(fit_path),fit_result_sha256=result['sha256'],
            full_training_terminated=result['full_training_terminated'],cohort_sha256=cohort.identity['sha256'],selected_best=result['selected_best'],
            sources_sha256=context.sources,runtime_versions=identity['construction']['runtime_versions'],agreement_policy=_policy(),layout=layout,
            prediction_sha256=digest,payload_sha256=_file_hash(output/'predictions.pt'),statistics=statistics,
            population_sha256=_hash(population),population=population['public'],native_metrics=native,model_before_sha256=before,model_after_sha256=after,
            forward_devices=devices,post_pass_module_device=str(next(prepared.module.parameters()).device),execution_device=device,process=process,
            started_ns=started_ns,finished_ns=time.time_ns(),reference_path=str(_path(reference)) if ref is not None else None,
            reference_sha256=ref['sha256'] if ref is not None else None,profile_path=str(_path(profile)) if frozen is not None else None,
            profile_sha256=frozen['sha256'] if frozen is not None else None,scientific_eligible=False,full_paper_complete=False))
        _execution_header(v,result['finished_ns']);context.guard.check();_write(output/'pass.json',v)
    finally:prepared.module.forward=original
    return _read_pass(output,context)


def _alive(value):
    try:
        f=Path('/proc',str(value['pid']),'stat').read_text().rsplit(')',1)[1].split()
        return f[19]==str(value['start_ticks']) and f[0]!='Z'
    except FileNotFoundError:return False


def _owned(root,context,process=None):
    root=_path(root)
    for n in ('started.json','stage-result.json'):context.guard.add(root/n)
    start=_json(root/'started.json');result=_json(root/'stage-result.json')
    if (result.get('format')!='egoemg_owned_stage_v1' or result.get('status')!='success' or
        result.get('worker_exit_code')!=0 or result.get('all_owned_consumers_exited') is not True or
        any(result[k]!=start[k] or _alive(start[k]) for k in ('worker','guardian'))):
        raise ValueError('actual completed owned native stage/exit required for CUDA qualification')
    if process is not None and any(process[k]!=start['worker'][k] for k in ('pid','start_ticks','uid')):
        raise ValueError('owned CUDA producer does not match actual native process')
    return dict(root=str(root),started_sha256=_file_hash(root/'started.json'),result_sha256=_file_hash(root/'stage-result.json'),
        worker={k:start['worker'][k] for k in ('pid','start_ticks','uid')},actual_exit_code=0)


def _qualification_dependencies(paths,fit,identity,native,replay,context):
    if type(paths) is not dict or set(paths)!={'input_admission','fit','native','replay'}:
        raise ValueError('raw input admission and three owned GPU producer artifact paths required')
    if (fit['scope']!='cuda_engineering' or fit['completed_epochs']!=2 or fit['optimizer_executions']!=4 or
        identity['policy']['precision']!='bf16-mixed' or identity['policy']['batch_size_by_role']!={r:18 for r in train.ROLES} or
        identity['cohort']['registered_full_legacy'] is not True or identity['cohort']['counts']!=train.FULL_COUNTS or
        fit['train_samples_per_epoch']!=[36,36] or fit['val_samples_per_epoch']!=[3152,3152] or
        native['scope']!='cuda_engineering' or native['role']!='val' or native['kind']!='native' or
        replay['scope']!='cuda_engineering' or replay['role']!='val' or replay['kind']!='replay' or
        replay['reference_sha256']!=native['sha256'] or native['layout']['samples']!=3152 or
        native['layout']['batch_size']!=18 or replay['layout']!=native['layout']):
        raise ValueError('actual original B18/BF16 two-epoch complete3152 CUDA qualification required')
    if any(v['execution_device']['visible_device']!=identity['runtime_device']['visible_device'] for v in (native,replay)):
        raise ValueError('actual native/replay GPU visibility differs from qualified fit')
    proofs={name:_owned(paths[name],context,process) for name,process in
        (('fit',fit['execution']['process']),('native',native['process']),('replay',replay['process']))}
    data=_path(paths['input_admission']);private=data/'private'
    for n in ('execution-state.json','execution-result.json','admit-a-result.json','admit-b-result.json','stage-receipt.json'):
        context.guard.add(private/n)
    state=_json(private/'execution-state.json');actual=_json(private/'execution-result.json')
    if (state['status']!='complete' or state['actual_exit_code']!=0 or any(_alive(state[k]) for k in ('wrapper','child')) or
        actual['format']!='actual_native_wilor_complete_content_numeric_admission_v1'):
        raise ValueError('actual independently completed input admission required before CUDA')
    from datetime import datetime
    ended_ns=int(datetime.fromisoformat(state['ended_utc']).timestamp()*1e9)
    if ended_ns>=fit['execution']['started_ns']:raise ValueError('independent full input admission did not precede CUDA fit')
    receipts=[_json(private/('admit-'+name+'-result.json')) for name in ('a','b')]
    if actual['identity']!=identity['cohort'] or any(v['identity']!=identity['cohort'] for v in receipts):
        raise ValueError('actual numeric input admission differs from qualified CUDA cohort')
    admission_stages={name:_owned(private/('owned-admit-'+name),context) for name in ('a','b')}
    for v in receipts:
        if (v['actual_rng_unchanged'] is not True or v['actual_cuda_initialized'] is not False or
            v['models_constructed']!=0 or v['optimizer_updates']!=0):raise ValueError('data-only admission cannot invent model qualification')
        for name,saved in v['private_admitted_stamps'].items():
            context.guard.add(name)
            if tuple(saved)!=context.guard.paths[Path(name)]:raise ValueError('actual admitted source/input generation changed')
    proofs['input_admission']=dict(root=str(data),identity_sha256=identity['cohort']['sha256'],
        execution_sha256=_file_hash(private/'execution-result.json'),stages=admission_stages)
    context.guard.check();return proofs


def qualify_wilor(fit,native,replay,output):
    context=_Evidence();fit_path=_path(fit);result,identity,cohort,assets,checkpoint=context.fit(fit_path)
    a=_read_pass(native,context);b=_read_pass(replay,context)
    if a['fit_result_sha256']!=result['sha256'] or b['fit_result_sha256']!=result['sha256']:
        raise ValueError('matching raw CUDA qualification fit and validation required')
    source=os.environ.get('EGOEMG_WILOR_QUALIFICATION_CONTEXT')
    if not source:raise ValueError('raw owned CUDA qualification context artifact required')
    source=_path(source);context.guard.add(source);paths=_json(source)
    proofs=_qualification_dependencies(paths,result,identity,a,b,context)
    output=_safe(output,exists=False)
    from .classic_emg_train import _space
    _space(output.parent);output.mkdir(mode=0o700,exist_ok=False)
    v=_sealed(dict(format=QUALIFICATION,fit_path=str(fit_path),native_path=str(_path(native)),replay_path=str(_path(replay)),
        raw_context_path=str(source),raw_context_sha256=_file_hash(source),raw_producer_proofs=proofs,
        fit_result_sha256=result['sha256'],native_sha256=a['sha256'],replay_sha256=b['sha256'],
        training_sources_sha256=train._sources(),evaluation_sources_sha256=context.sources,cohort_sha256=cohort.identity['sha256'],
        wilor_protocol_sha256=identity['construction']['protocol_sha256'],construction_identity_sha256=identity['construction']['sha256'],
        runtime_versions=identity['construction']['runtime_versions'],agreement_policy=_policy(),finished_ns=time.time_ns(),
        scientific_eligible=False,full_paper_complete=False))
    context.guard.check();_write(output/'qualification.json',v);return read_wilor_qualification(output)


def read_wilor_qualification(output):
    if not isinstance(output,(str,os.PathLike)):raise ValueError('actual native WiLoR CUDA qualification artifact required')
    try:
        context=_Evidence();path=_path(output)
        if not path.is_dir():raise ValueError('raw CUDA qualification artifact directory required')
        context.guard.add(path/'qualification.json');v=_json(path/'qualification.json');_check(v,QUALIFICATION)
        result,identity,cohort,assets,checkpoint=context.fit(v['fit_path'])
        a=_read_pass(v['native_path'],context);b=_read_pass(v['replay_path'],context)
        source=_path(v['raw_context_path']);context.guard.add(source)
        if _file_hash(source)!=v['raw_context_sha256']:raise ValueError('raw CUDA qualification context changed')
        proofs=_qualification_dependencies(_json(source),result,identity,a,b,context)
        if (v['fit_result_sha256']!=result['sha256'] or v['native_sha256']!=a['sha256'] or v['replay_sha256']!=b['sha256'] or
            v['raw_producer_proofs']!=proofs or v['training_sources_sha256']!=train._sources() or v['evaluation_sources_sha256']!=context.sources or
            v['cohort_sha256']!=cohort.identity['sha256'] or v['wilor_protocol_sha256']!=identity['construction']['protocol_sha256'] or
            v['construction_identity_sha256']!=identity['construction']['sha256'] or v['runtime_versions']!=identity['construction']['runtime_versions'] or
            v['agreement_policy']!=_policy() or v['finished_ns']<=max(a['finished_ns'],b['finished_ns'])):
            raise ValueError('actual raw native WiLoR CUDA qualification source/state/runtime differs')
        context.guard.check();return v
    except (OSError,KeyError,TypeError) as exc:raise ValueError('complete raw native WiLoR CUDA qualification artifact required') from exc


PUBLIC_FILES=('metrics.json','metrics.md','curve-data.json','training-curves.png','training-curves.svg')


def _require_full(result,require_full):
    if type(require_full) is not bool:raise ValueError('full report requirement must be boolean')
    if require_full and not result['full_training_terminated']:raise ValueError('engineering WiLoR cannot be a full scientific report')


def _curve_data(fit,result):
    from .vision_literal_train import _read_epochs
    path=_path(fit)/'epochs.jsonl'
    if _file_hash(path)!=result['journal_sha256']:raise ValueError('actual native WiLoR training journal changed')
    rows,_=_read_epochs(path,result['identity_sha256']);best=result['selected_best'];i=best['epoch']
    if (len(rows)!=result['completed_epochs'] or not rows or type(i) is not int or not 0<=i<len(rows) or
        rows[-1]['optimizer_executions']!=result['optimizer_executions'] or
        rows[i]['validation']['val_mae']!=best['score'] or best['score']!=min(r['validation']['val_mae'] for r in rows)):
        raise ValueError('actual native WiLoR curve budget or selected validation best differs')
    return dict(format='native_wilor_curve_data_v1',scope=result['scope'],fit_result_sha256=result['sha256'],
        journal_sha256=result['journal_sha256'],selected_epoch_zero_based=i,selected_best=copy.deepcopy(best),epochs=rows,
        training_unit='native sample-weighted composite loss; effective fingertip weight0.02',
        validation_unit='native sample-weighted val_mae, radians; checkpoint selection',scientific_eligible=False,full_paper_complete=False)


def _plot_runtime():
    import matplotlib
    from matplotlib import font_manager,ft2font
    font=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans'))
    return dict(matplotlib_version=matplotlib.__version__,freetype_version=ft2font.__freetype_version__,font_sha256=_file_hash(font),
        backend='Agg',svg_hashsalt='egoemg-native-wilor-v1')


def _curve_images(curves):
    import io
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'svg.hashsalt':'egoemg-native-wilor-v1','font.family':'DejaVu Sans'}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        try:
            rows=curves['epochs'];x=[r['epoch']+1 for r in rows]
            axes[0].plot(x,[r['train_loss'] for r in rows]);axes[0].set(title='Training',ylabel='Native composite loss')
            axes[1].plot(x,[r['validation']['val_mae'] for r in rows]);axes[1].set(title='Held-out validation',ylabel='Native validation MAE (radians)')
            best=curves['selected_best'];axes[1].scatter([best['epoch']+1],[best['score']],color='#b45309',label='Selected best',zorder=3);axes[1].legend()
            for ax in axes:ax.set_xlabel('Completed epoch');ax.grid(alpha=.2)
            fig.suptitle('WiLoR — '+('full original run' if curves['scope']=='full' else 'ENGINEERING ONLY; no scientific result'))
            images={}
            for ext,metadata in (('png',{'Software':'EgoEMG-Reproduction'}),('svg',{'Date':None,'Creator':'EgoEMG-Reproduction'})):
                stream=io.BytesIO();fig.savefig(stream,format=ext,dpi=150,metadata=metadata);images['training-curves.'+ext]=stream.getvalue()
            return images
        finally:plt.close(fig)


def _build_report(replay,*,require_full):
    if type(require_full) is not bool:raise ValueError('full report requirement must be boolean')
    from .classic_emg_report import PUBLIC_NATIVE_METRICS
    context=_Evidence();right=_read_pass(replay,context)
    if right['role']!='test' or right['kind']!='replay':raise ValueError('actual native test and independent exact test replay required')
    result,identity,cohort,assets,checkpoint=context.fit(right['fit_path']);_require_full(result,require_full)
    left=_read_pass(right['reference_path'],context)
    report=_sealed(dict(format='native_wilor_completed_report_v1',variant='wilor',scope=result['scope'],
        scientific_eligible=result['full_training_terminated'],full_paper_complete=False,
        training_sources_sha256=identity['training_sources_sha256'],evaluation_sources_sha256=context.sources,
        fit_result_sha256=result['sha256'],native_test_sha256=left['sha256'],test_replay_sha256=right['sha256'],
        validation_profile_sha256=right['profile_sha256'],cohort_sha256=cohort.identity['sha256'],
        counts=identity['counts'],underlying_counts=identity['cohort']['counts'],evaluation_population=right['layout'],population=right['population'],
        selected_best=result['selected_best'],completed_epochs=result['completed_epochs'],optimizer_executions=result['optimizer_executions'],
        full_training_terminated=result['full_training_terminated'],policy=identity['policy'],
        construction_identity_sha256=identity['construction']['sha256'],runtime_versions=identity['construction']['runtime_versions'],
        native_metrics={k:v for k,v in left['native_metrics'].items() if k in PUBLIC_NATIVE_METRICS},
        native_metrics_units='original composite loss; MAE radians; landmarks meters; derivatives separate',
        agreement_policy=_policy(),statistics=copy.deepcopy(right['statistics']),curve_data=_curve_data(right['fit_path'],result),plot_runtime=_plot_runtime(),
        limitations=['Legacy release, not demonstrated equivalent to the PDF or unavailable merged data.',
            'Prospective WiLoR recipe follows original module and official-code S lineage; no author standalone WiLoR YAML.',
            'One CUDA rank/original batch18 differs from declared six-rank BatchNorm statistics.',
            'Original cosine floor equals initial LR1e-5, so LR remains constant; effective fingertip loss weight0.02.',
            'Missing crops stay in the all-sample population under the original generated label mask.',
            'CPU engineering selects only first window per hand and both repeats; it has no scientific score.',
            'One complete setting does not complete the whole paper.']))
    context.guard.check();return report,context


def _render(report):
    scope='完整训练实验' if report['scientific_eligible'] else '工程验证，不能作为科学复现成绩'
    s=report['statistics'];lines=['# WiLoR 原生评价报告','',scope,'',
        f"实际完成 {report['completed_epochs']} 轮、{report['optimizer_executions']} 次更新；最佳轮次 {report['selected_best']['epoch']+1}，按原生 val_mae 选择。",'',
        f"测试消费 {s['consumed_samples']} 项，其中独立手样本 {s['unique_samples']} 项，原生重复消费 {s['duplicate_consumptions']} 项。",'',
        '| 统计范围 | 全部样本 MAE (°) | 有有效图像 MAE (°) |','|---|---:|---:|']
    for label,key in (('原生全部消费','consumed'),('每手样本第一次消费','unique_first_repeat')):
        cells=[s[key][domain]['overall']['mae_deg'] for domain in ('all','vision_valid')]
        lines.append('| '+label+' | '+' | '.join('无有效角' if v is None else f'{v:.8f}' for v in cells)+' |')
    lines+=['','按原始22角及生成标签掩码统计，误差总和除以有效角总数；缺图样本保留。',
        'native MAE 保留官方按批次及样本数聚合的弧度值，与汇总有效角后计算的角度指标分别报告。',
        '训练曲线为复合损失；验证曲线为 val_mae（弧度）。原生重复消费与唯一手样本分别统计。',
        '旧版数据、单卡以及官方代码配方的差异已披露；本报告不表示整篇论文已经复现。','']
    return '\n'.join(lines)


def _completion(output,report):
    return _sealed(dict(format='native_wilor_report_completion_v1',report_sha256=report['sha256'],
        files_sha256={n:_file_hash(_path(output/n)) for n in PUBLIC_FILES},scientific_eligible=report['scientific_eligible'],full_paper_complete=False))


def publish_wilor_report(fit,native,replay,output,*,require_full=True):
    if type(require_full) is not bool:raise ValueError('full report requirement must be boolean')
    output=_safe(output,exists=False)
    if output.exists():raise FileExistsError('native WiLoR public report output exists')
    report,context=_build_report(replay,require_full=require_full);right=context.passes[_path(replay)]
    if _path(fit)!=_path(right['fit_path']) or _path(native)!=_path(right['reference_path']):
        raise ValueError('report requires exact fit/native-test/independent-replay paths')
    from .classic_emg_train import _space
    _space(output.parent);images=_curve_images(report['curve_data']);context.guard.check()
    output.mkdir(mode=0o700,exist_ok=False);_write(output/'metrics.json',report);_write(output/'curve-data.json',report['curve_data'])
    for name,raw in dict(images,**{'metrics.md':_render(report).encode()}).items():
        with (output/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    context.guard.check();_write(output/'completion.json',_completion(output,report));context.guard.check();return report


def verify_wilor_report(replay,output,*,require_full=True):
    if type(require_full) is not bool:raise ValueError('full report requirement must be boolean')
    output=_path(output);report,context=_build_report(replay,require_full=require_full)
    for n in (*PUBLIC_FILES,'completion.json'):context.guard.add(output/n)
    if (_json(output/'metrics.json')!=report or _json(output/'curve-data.json')!=report['curve_data'] or
        (output/'metrics.md').read_text()!=_render(report) or
        any((output/n).read_bytes()!=raw for n,raw in _curve_images(report['curve_data']).items()) or
        _json(output/'completion.json')!=_completion(output,report)):
        raise ValueError('actual native WiLoR report/curves/last completion differs from reopened raw evidence')
    context.guard.check();return report
