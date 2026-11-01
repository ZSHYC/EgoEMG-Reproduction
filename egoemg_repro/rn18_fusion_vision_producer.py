"""Read the known completed legacy RN18 LAST, then bind current fusion inputs.

This vision-only admission does not grant a complete fusion initializer permit.
The designated RN18-compatible cotrain-S validation BEST is separately required.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import random
import sys
import types
import weakref

import numpy as np

from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard,_json
from .data import file_sha256
from .fusion_initializer_handoff import verify_initializer_footprints
from .vision_literal_runtime import _schema,vision_state_sha256

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
LEGACY_SOURCES={
 'egoemg_repro/__init__.py':'6103cfcd4a90339b804308dcb82a86448749c728fd093ebfab546ff9bfdb3e96',
 'egoemg_repro/crops.py':'1aa0871d6ede77d93988d455d0930e1f42470c1a92a47cfc8e49252bbfc0695e',
 'egoemg_repro/data.py':'de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e',
 'egoemg_repro/download.py':'5063fc0a5a0dbb1929c054a9cac86d3f90c01b375171660b6d75e2707e684e75',
 'egoemg_repro/metrics.py':'12d6eb170997ae78ddbf5602d1252d05a778f1357f17b28aa3b96d06a618798e',
 'egoemg_repro/partitions.py':'ec816483e5d567cec45df77bd6823e97c201d3fd20f45b0d8004827565fdf678',
 'egoemg_repro/train.py':'f40f375d11bd535bd54edb4a6d98f399e3c74ea6c9f846ba94210a203e808418',
 'egoemg_repro/visual.py':'588a18cfd0bbd106c6843fbbe6e436635784cd2b0ea7b59f3db33878ab1f084b'}
LEGACY_FILES={
 'last.pt':'e9bf7c7fa2abee7997c2c7c50f750137ded4ae8dad1cba998f52392516db5fc0',
 'provenance.json':'ae260b37ea588f188d26619c54ca2123a5c7d01de652b56b4d5057c02e4bd871',
 'config.json':'2ab2c4ab20e58fdb71db7ac1606b6b2b127ee507d872dd42782b8cde38f9bac8',
 'normalization.json':'62734cd4766804b1e405ad0d64c172b9b6a9fe367988c6310ab5d4652b341f49',
 'official_module.json':'31a1509528e9370f948ac81d1e7ee76d6d5282ed1b67b142a201103f21c668ce',
 'run.json':'2f81eab3d0ce73a804583ca8f854ffc26e80d779678046e551838aeb7baffa17',
 'summary.json':'8014c41af5dcf7598549f52d1bdfaefa5a09ee528a61b3eb8821e74ca16cd568',
 'environment.json':'093227449f34c2e572f87aa9efa2d15a265d1ca6d0e3bacef974f6acae3a6bdb',
 'partitions.json':'534a20d8710b0706362a6b26e02c7c4e06bf854e00fb9e073a8179ecc048d896',
 'progress.jsonl':'de43369906460ebf5ad2c4aca3acd4ddba3de01b48135b34d41b6098ee7e7736',
 'recipe_source.json':'ad3d43fd4757408f8357cb15769fe314ec1ef41913e03fdc3c1566b5a2ebf33c'}
READER_SOURCES={**LEGACY_SOURCES,'egoemg_repro/audit.py':'86cf0c74086f2581f430ee88d8725df10c3389379535ef9036e2d7a892b27cdd'}
_READER_SERIAL=itertools.count()
_READERS=weakref.WeakKeyDictionary()
_VISION=weakref.WeakKeyDictionary()
FULL_FUTURE='44e9ebc8a2f63a79699c5a976bf2717b0fe7a376427c7c46f538d0745c2b4bd0'


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported RN18 handoff source changed')
    result={n:_file_hash(ROOT/n)for n in LEGACY_SOURCES}
    if any(result[n]!=h for n,h in LEGACY_SOURCES.items()if n not in('egoemg_repro/train.py','egoemg_repro/visual.py')):
        raise ValueError('common frozen legacy input providers changed')
    for n in ('rn18_fusion_vision_producer','rn18_fusion_cohort','rn18_fusion_data','rn18_fusion_literal',
              'fusion_initializer_handoff','audit','vision_literal_runtime'):
        result['egoemg_repro/'+n+'.py']=_file_hash(ROOT/'egoemg_repro'/(n+'.py'))
    return result


@contextmanager
def _frozen_reader(root):
    """Execute only whitelisted approved bytes in an isolated relative package.

    Current train/visual adapters have advanced. Neither is replaced or used to
    pretend the historical producer ran current code. Compile approved bytes
    directly, avoiding previously loaded modules and timestamp-based pyc caches.
    """
    code={}
    for n,h in READER_SOURCES.items():
        raw=(root/n).read_bytes()
        if __import__('hashlib').sha256(raw).hexdigest()!=h:raise ValueError('known frozen reader source differs')
        code[n]=compile(raw,str(root/n),'exec')
    name='_egoemg_rn18_frozen_'+str(next(_READER_SERIAL));loaded=[]
    if name in sys.modules:raise ValueError('frozen reader namespace already occupied')
    package=types.ModuleType(name);package.__package__=name;package.__path__=[]
    package.__file__=str(root/'egoemg_repro/__init__.py');sys.modules[name]=package;loaded.append(name)
    try:
        exec(code['egoemg_repro/__init__.py'],package.__dict__)
        for child in('download','data','crops','metrics','partitions','visual','train','audit'):
            full=name+'.'+child;module=types.ModuleType(full);module.__package__=name
            module.__file__=str(root/'egoemg_repro'/(child+'.py'))
            sys.modules[full]=module;loaded.append(full);setattr(package,child,module)
            exec(code['egoemg_repro/'+child+'.py'],module.__dict__)
        yield package.train,package.audit
    finally:
        for full in reversed(loaded):sys.modules.pop(full,None)


def _rn18_schema():
    """Names/shapes of the unchanged original torchvision RN18+512/22 head."""
    result={}
    def weight(name,shape):result[name]=dict(shape=shape,dtype='torch.float32')
    def bn(name,n):
        for suffix in('weight','bias','running_mean','running_var'):weight(name+'.'+suffix,[n])
        result[name+'.num_batches_tracked']=dict(shape=[],dtype='torch.int64')
    weight('backbone.0.weight',[64,3,7,7]);bn('backbone.1',64);channels=64
    for layer,planes in enumerate((64,128,256,512),4):
        for block in range(2):
            name=f'backbone.{layer}.{block}'
            weight(name+'.conv1.weight',[planes,channels,3,3]);bn(name+'.bn1',planes)
            weight(name+'.conv2.weight',[planes,planes,3,3]);bn(name+'.bn2',planes)
            if channels!=planes:
                weight(name+'.downsample.0.weight',[planes,channels,1,1]);bn(name+'.downsample.1',planes)
            channels=planes
    for n,s in [('0.weight',[512,512]),('0.bias',[512]),('3.weight',[22,512]),('3.bias',[22])]:weight('head.'+n,s)
    return result


def _validate_legacy_domains(packet,config,*,require_complete=True):
    """Validate actual saved model/AdamW/cosine/loops/RNG without constructors."""
    import torch
    try:
        if (_hash(config)!='aa46eaac0b603200f9a1b7da909ab421ddc6f8aa7578757ffbdef8a9230084f2'or
            type(packet['epoch'])is not int or not 1<=packet['epoch']<=150 or
            require_complete and packet['epoch']!=150 or packet['next_batch']!=0 or
            packet['global_step']!=packet['epoch']*71 or packet['validation_bounded']is not False):
            raise ValueError('legacy complete loop/config differs')
        state=packet['model'];schema=_rn18_schema()
        if _schema(state)!=schema:raise ValueError('complete native RN18 typed tensor state required')
        if any(int(t)!=packet['global_step']for n,t in state.items()if n.endswith('num_batches_tracked')):
            raise ValueError('native RN18 BatchNorm counters differ from actual updates')
        names=[n for n in schema if not n.endswith(('running_mean','running_var','num_batches_tracked'))]
        opt=packet['optimizer'];groups=opt['param_groups']
        if len(names)!=64 or len(groups)!=1:raise ValueError('original complete64parameter AdamW required')
        g=groups[0];ids=g['params']
        if len(ids)!=64 or len(set(ids))!=64 or set(opt['state'])!=set(ids):raise ValueError('legacy AdamW coverage differs')
        rate=1e-5+(.0001-1e-5)*(1+math.cos(math.pi*packet['epoch']/150))/2
        if (not math.isclose(g['lr'],rate,rel_tol=1e-10,abs_tol=1e-14)or g['initial_lr']!=.0001 or
            g['weight_decay']!=.01 or tuple(g['betas'])!=(.9,.999)or g['eps']!=1e-8 or
            any(g[k]is not False for k in('amsgrad','maximize','capturable','differentiable'))or
            g['foreach']is not None or g['fused']is not None):raise ValueError('legacy AdamW options differ')
        for n,i in zip(names,ids):
            a=opt['state'][i]
            if set(a)!= {'step','exp_avg','exp_avg_sq'}or a['step'].shape!=torch.Size([])or a['step'].dtype!=torch.float32 or float(a['step'])!=packet['global_step']:
                raise ValueError('legacy AdamW actual steps differ')
            for k in('exp_avg','exp_avg_sq'):
                t=a[k]
                if list(t.shape)!=schema[n]['shape']or t.dtype!=torch.float32 or not torch.isfinite(t).all():raise ValueError('legacy AdamW moment state differs')
            if (a['exp_avg_sq']<0).any():raise ValueError('negative AdamW squared moments')
        s=packet['scheduler']
        if (s['T_max']!=150 or s['eta_min']!=1e-5 or s['base_lrs']!=[.0001]or
            s['last_epoch']!=packet['epoch']or s['_step_count']!=packet['epoch']+1 or
            len(s['_last_lr'])!=1 or not math.isclose(s['_last_lr'][0],rate,rel_tol=1e-10,abs_tol=1e-14)):
            raise ValueError('legacy full cosine state differs')
        rng=packet['rng'];random.Random(0).setstate(rng['python']);np.random.RandomState(0).set_state(rng['numpy'])
        t=rng['torch'];c=rng['cuda']
        if t.dtype!=torch.uint8 or list(t.shape)!=[5056]or c.dtype!=torch.uint8 or list(c.shape)!=[16]:raise ValueError('legacy CPU/CUDA RNG state differs')
        torch.Generator(device='cpu').set_state(t)
    except (KeyError,TypeError,AttributeError,RuntimeError,OverflowError)as exc:raise ValueError('incomplete native legacy training domains')from exc
    return dict(native_state_tensors=124,native_optimizer_parameters=64,completed_epochs=packet['epoch'],optimizer_executions=packet['global_step'])


def _legacy_history(events,packet,config):
    try:
        counts={k:0 for k in('start','train_step','validation','final_test')};epoch=batch=step=0;best=math.inf;selected=None
        for event in events:
            kind=event['event']
            if kind not in counts:raise ValueError('unexpected historical event')
            counts[kind]+=1
            if kind=='start':
                if counts[kind]!=1 or step or event['step']!=0 or event['bounded_integration']is not False or event['config']!=config or event['train_samples']!=63172 or event['validation_samples']!=1052 or event['test_samples']!=2792:
                    raise ValueError('legacy full history start differs')
            elif kind=='train_step':
                if counts['start']!=1 or epoch>=150 or event['epoch']!=epoch or event['batch']!=batch or event['step']!=step+1 or not math.isfinite(event['loss'])or event['valid_angle_count']<=0:
                    raise ValueError('legacy update history incomplete/duplicate/out of order')
                rate=1e-5+(.0001-1e-5)*(1+math.cos(math.pi*epoch/150))/2
                if not math.isclose(event['lr'],rate,rel_tol=1e-10,abs_tol=1e-14):raise ValueError('legacy update learning rate differs')
                batch+=1;step+=1
            elif kind=='validation':
                metrics=event['metrics'];score=metrics['sampling']['center']['overall']['mae_deg'];improved=score<best
                if (batch!=71 or event['epoch']!=epoch or event['step']!=step or metrics['bounded']is not False or metrics['evaluated_batches']!=2 or
                    metrics['sampling']['center']['overall']['valid_angle_count']!=22064 or not math.isfinite(score)or
                    event['improved']is not improved or event['best_val_mae_deg']!=min(best,score)):
                    raise ValueError('legacy full validation/selection history differs')
                if improved:best=score;selected=step
                epoch+=1;batch=0
            else:
                if epoch!=150 or step!=10650 or event['status']!='training_complete'or event['global_step']!=step or event['selected_checkpoint_step']!=selected or event['test_evaluated']is not True or event['test']['bounded']is not False:
                    raise ValueError('legacy terminal history differs')
        if counts!=dict(start=1,train_step=10650,validation=150,final_test=1)or epoch!=150 or batch or step!=packet['global_step']or best!=packet['best_val_mae_deg']or selected!=packet['best']['global_step']:
            raise ValueError('complete legacy150 budget and selected history required')
        return counts
    except (KeyError,TypeError,AttributeError)as exc:raise ValueError('incomplete native legacy history')from exc


@dataclass(eq=False)
class LegacyRn18Producer:
    identity:dict
    partitions:dict
    state:dict
    provenance:dict
    guard:object

    def check(self):
        r=_READERS.get(self)
        if r is None:raise ValueError('registered original legacy producer required')
        self.guard.check()
        if (self.identity.get('sha256')!=r['identity']or _hash({k:v for k,v in self.identity.items()if k!='sha256'})!=r['identity']or
            _hash(self.partitions)!=r['partitions']or _hash(self.provenance)!=r['provenance']or
            vision_state_sha256(self.state)!=r['state']or _sources()!=r['sources']):raise ValueError('legacy producer identity/source/tensors changed')


def read_legacy_rn18_producer(run,*,source_root,upstream):
    """Read only the exact already published completed run, selecting its LAST."""
    try:
        run=_safe(run);source_root=_safe(source_root);sources=_sources()
        guard=_StatGuard([*[run/n for n in LEGACY_FILES],*[source_root/n for n in READER_SOURCES],*[ROOT/n for n in sources]])
        if any(file_sha256(run/n)!=h for n,h in LEGACY_FILES.items()):raise ValueError('known completed RN18 run bytes differ')
        import torch
        packet=torch.load(str(run/'last.pt'),map_location='cpu',weights_only=False,mmap=True)
        with _frozen_reader(source_root)as(train,audit):
            official=train.validate_upstream(upstream)
            if official['sha256']!='46446093fff601875f3b7f76f9fe9ef3d898c555f1e5cc10a3bf0df6ddd3ff69':raise ValueError('known official legacy model source differs')
            documents=audit.validate_artifacts(run,packet,device='cuda:0') # Saved precision policy only; never initializes CUDA.
        config=documents['config'];details=_validate_legacy_domains(packet,config)
        _validate_legacy_domains(documents['best'],config,require_complete=False)
        events=[__import__('json').loads(line)for line in(run/'progress.jsonl').read_text().splitlines()]
        counts=_legacy_history(events,packet,config);guard.check()
        if file_sha256(run/'last.pt')!=LEGACY_FILES['last.pt']:raise ValueError('legacy checkpoint bytes changed while reading')
        state=packet['model'];inventory={n:dict(shape=list(t.shape),dtype=str(t.dtype),sha256=vision_state_sha256({n:t}))for n,t in state.items()}
        identity=dict(format='known_completed_legacy_rn18_last_reader_v1',producer_family='legacy-resnet18-150-0.01-nojitter',
            checkpoint_selection='LAST',source_checkpoint_sha256=LEGACY_FILES['last.pt'],source_run_files_sha256=dict(LEGACY_FILES),
            source_reproduction_sha256=packet['provenance']['reproduction']['sha256'],source_runtime_files_sha256=dict(LEGACY_SOURCES),
            reader_sources_sha256=dict(READER_SOURCES),historical_reader_isolated=True,
            sources_sha256=sources,source_recipe={k:config[k]for k in('window_length','train_stride','eval_stride','val_fraction','epochs','batch_size','num_workers','optimizer_class','lr','weight_decay','eta_min','fingertip_weight','augmentation','precision','seed')},
            source_partition_content_sha256=_hash(documents['manifest']),source_normalization_sha256=packet['provenance']['normalization_sha256'],
            source_validation_selected_step=documents['best']['global_step'],actual_history_counts=counts,
            tensor_inventory=inventory,model_state_sha256=vision_state_sha256(state),**details,
            producer_training_verified=True,current_raw_input_bytes_revalidated=False,fusion_initializers_complete=False,
            source_adaptations=dict(loss='legacy single fingertip0.01, not pending standalone native0.02',
                normalization='train-only per-hand/channel, not future fusion published vectors',
                masks='legacy separate wrist-valid/finite masking and nan_to_num',jitter='disabled in legacy producer'),
            scientific_eligible=False,full_paper_complete=False)
        identity['sha256']=_hash(identity)
        result=LegacyRn18Producer(identity,documents['manifest'],state,packet['provenance'],guard)
        _READERS[result]=dict(identity=identity['sha256'],partitions=_hash(result.partitions),provenance=_hash(result.provenance),state=identity['model_state_sha256'],sources=sources)
        result.check();return result
    except (OSError,KeyError,TypeError,AttributeError)as exc:raise ValueError('complete known legacy RN18 producer unavailable')from exc


def _vision_binding(cohort,producer):
    """Reconstruct actual old windows and match every consumed current payload."""
    from .partitions import build_partitions
    producer.check();cohort.check();store=cohort.datasets['train'].store
    original=build_partitions(store)
    if _hash(original)!=producer.identity['source_partition_content_sha256']or original!=producer.partitions:
        raise ValueError('legacy training partition differs from independent raw reconstruction')
    future=cohort.partitions
    if _hash(future)!=FULL_FUTURE:raise ValueError('exact RN18 fusion future holdouts required')
    footprints={'windows':[dict(w,footprint_start=w['start'],footprint_end=w['end'])for w in original['windows']]}
    overlap=verify_initializer_footprints(footprints,future)
    data=producer.provenance['data'];visual=producer.provenance['visual_data']
    shared={}
    for name,h in cohort.identity['payload_files_sha256'].items():
        path=cohort.input_paths[name]
        if name.startswith('metadata:'):old=data['metadata_sha256'].get(path.name)
        elif name.startswith('field:'):old=data['verified_transfer_records'].get(path.name,{}).get('sha256')
        else:
            episode=int(name.split(':')[1]);old=visual['payloads'].get(f'episode_{episode:06d}.data.mdb',{}).get('sha256')
        if old!=h:raise ValueError('legacy producer and actual fusion payload bytes differ')
        shared[name]=h
    extras={}
    for hand in('left','right'):
        name=f'mocap_{hand}_wrist_angles_valid';path=_safe(store.root/store.fields[name]['filename']);producer.guard.add(path)
        h=file_sha256(path)
        if h!=data['verified_transfer_records'].get(path.name,{}).get('sha256'):raise ValueError('legacy wrist-valid payload differs')
        extras[name]=dict(sha256=h,bytes=path.stat().st_size)
    producer.check();cohort.check()
    return dict(reconstructed_legacy_partition_sha256=_hash(original),future_fusion_partition_content_sha256=FULL_FUTURE,
        source_counts={r:original['coverage'][r]['hand_samples']for r in('train','val','test')},
        current_fusion_counts=copy.deepcopy(cohort.identity['counts']),shared_actual_payload_files=len(shared),
        shared_actual_payloads_sha256=_hash(shared),legacy_only_wrist_valid_payloads=extras,footprints=overlap,
        current_fusion_cohort_sha256=cohort.identity['sha256'],all_consumed_raw_bytes_match=True)


@dataclass(eq=False)
class Rn18FusionVisionLast:
    identity:dict
    checkpoint:Path
    _producer:object
    _cohort:object
    _guard:object

    def check(self):
        r=_VISION.get(self)
        if r is None:raise ValueError('registered RN18 trained vision-only admission required')
        self._guard.check();self._producer.check();self._cohort.check()
        if (self.identity.get('sha256')!=r['identity']or _hash({k:v for k,v in self.identity.items()if k!='sha256'})!=r['identity']or
            self.checkpoint!=r['path']or _sources()!=r['sources']or
            file_sha256(self.checkpoint)!=self.identity['export_file_sha256']or
            file_sha256(self.checkpoint.parent/'vision-last-admission.json')!=r['receipt_sha256']):raise ValueError('RN18 vision-only admission changed')


def _vision_identity(producer,binding,dest):
    identity=dict(format='rn18_fusion_completed_legacy_vision_last_admission_v1',scope='completed vision branch only',
        producer=copy.deepcopy(producer.identity),current_input_binding=binding,checkpoint_selection='LAST',
        export_file_sha256=file_sha256(dest),tensor_inventory=copy.deepcopy(producer.identity['tensor_inventory']),
        model_state_sha256=producer.identity['model_state_sha256'],sources_sha256=_sources(),
        producer_training_verified=True,current_raw_input_bytes_revalidated=True,
        fusion_initializers_complete=False,emg_initializer_admitted=False,scientific_eligible=False,full_paper_complete=False)
    identity['sha256']=_hash(identity);return identity


def _register_vision(identity,dest,producer,cohort):
    receipt=dest.parent/'vision-last-admission.json';guard=_StatGuard([dest,receipt])
    result=Rn18FusionVisionLast(identity,dest,producer,cohort,guard)
    _VISION[result]=dict(identity=identity['sha256'],path=dest,sources=_sources(),receipt_sha256=file_sha256(receipt))
    result.check();return result


def admit_rn18_fusion_vision_last(cohort,vision_fit,output,*,vision_source_root):
    """Export exact completed LAST only after registered complete current IO."""
    from .rn18_fusion_cohort import Rn18FusionCohort
    if type(cohort)is not Rn18FusionCohort:raise ValueError('registered complete RN18 fusion cohort required')
    cohort.check();producer=read_legacy_rn18_producer(vision_fit,source_root=vision_source_root,upstream=cohort.upstream)
    binding=_vision_binding(cohort,producer);output=Path(output).expanduser().absolute()
    if output.exists():raise ValueError('fresh vision admission output required; inspect existing evidence')
    _safe(output.parent);output.mkdir(mode=0o700)
    import json,os,torch
    dest=output/'legacy-rn18-last.pt'
    torch.save(dict(state_dict=producer.state,source_checkpoint_sha256=LEGACY_FILES['last.pt'],producer_identity_sha256=producer.identity['sha256']),dest)
    packet=torch.load(str(dest),map_location='cpu',weights_only=False,mmap=True)
    if _schema(packet['state_dict'])!=_rn18_schema()or vision_state_sha256(packet['state_dict'])!=producer.identity['model_state_sha256']:
        raise ValueError('native RN18 LAST export differs from every original tensor')
    producer.check();cohort.check()
    identity=_vision_identity(producer,binding,dest);receipt=output/'vision-last-admission.json'
    with receipt.open('x')as f:json.dump(identity,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return _register_vision(identity,dest,producer,cohort)


def read_rn18_fusion_vision_last(cohort,vision_fit,output,*,vision_source_root):
    """Independently reopen the historical producer, future IO and all124 tensors."""
    from .rn18_fusion_cohort import Rn18FusionCohort
    if type(cohort)is not Rn18FusionCohort:raise ValueError('registered complete RN18 fusion cohort required')
    cohort.check();producer=read_legacy_rn18_producer(vision_fit,source_root=vision_source_root,upstream=cohort.upstream)
    binding=_vision_binding(cohort,producer)
    try:
        output=_safe(output);dest=_safe(output/'legacy-rn18-last.pt');receipt=_safe(output/'vision-last-admission.json')
        guard=_StatGuard([dest,receipt]);saved=_json(receipt);expected=_vision_identity(producer,binding,dest)
        if saved!=expected:raise ValueError('vision LAST receipt differs from independent full producer/input reconstruction')
        import torch
        packet=torch.load(str(dest),map_location='cpu',weights_only=False,mmap=True)
        if (set(packet)!= {'state_dict','source_checkpoint_sha256','producer_identity_sha256'}or
            packet['source_checkpoint_sha256']!=LEGACY_FILES['last.pt']or packet['producer_identity_sha256']!=producer.identity['sha256']or
            _schema(packet['state_dict'])!=_rn18_schema()or vision_state_sha256(packet['state_dict'])!=producer.identity['model_state_sha256']):
            raise ValueError('complete exported RN18 LAST native tensor bytes or lineage differ')
        producer.check();cohort.check();guard.check()
        return _register_vision(saved,dest,producer,cohort)
    except (OSError,KeyError,TypeError,AttributeError)as exc:raise ValueError('complete trained RN18 LAST export unavailable')from exc
