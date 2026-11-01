"""Source-bound RN18 LAST + exact fixed-holdout cotrain validation-BEST handoff.

Engineering producer reads retain their true two-epoch scope. Only independently
completed150 producers and the registered complete fusion cohort grant the
combined initializer object. No learned module or optimizer is constructed.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import weakref

from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _safe,_StatGuard,_json
from .fusion_literal_runtime import _admit,_check_assets
from .fusion_initializer_handoff import _producer_cohort_binding,verify_initializer_footprints
from .rn18_fusion_literal import rn18_fusion_protocol

ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
LOCAL=('rn18_fusion_trained_handoff','rn18_fusion_vision_producer','rn18_fusion_literal','rn18_fusion_runtime',
       'rn18_fusion_cohort','rn18_fusion_data','rn18_initializer_train','rn18_initializer_data',
       'rn18_initializer_cohort','fusion_initializer_handoff','fusion_literal_runtime')
_EMG=weakref.WeakKeyDictionary()
_COMBINED=weakref.WeakKeyDictionary()


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported trained RN18 handoff source changed')
    return {'egoemg_repro/'+n+'.py':_file_hash(ROOT/'egoemg_repro'/(n+'.py'))for n in LOCAL}


def _seal(value):
    value['sha256']=_hash(value);return value


def _read_emg_fit(output,*,upstream,source_root,require_full):
    if type(require_full)is not bool:raise ValueError('explicit full producer scope required')
    from . import rn18_initializer_train as fit
    from .rn18_initializer_cohort import _sources as cohort_sources
    try:
        output=_safe(output);source_root=_safe(source_root);upstream=_safe(upstream)
        guard=_StatGuard([output/'identity.json',output/'fit-result.json',output/'epochs.jsonl',output/'checkpoints/last.ckpt'])
        identity=_json(output/'identity.json')
        # Independently bind every recorded scientific reader/factory source;
        # a source directory is never permission to execute arbitrary Python.
        expected=fit._sources();paths=cohort_sources(upstream)
        if identity['fit_sources_sha256']!=expected or identity['cohort']['source_files_sha256']!={n:_file_hash(p)for n,p in paths.items()}:
            raise ValueError('exact cotrain producer source inventory differs')
        for n,h in expected.items():
            p=_safe(source_root/n);guard.add(p)
            if _file_hash(p)!=h:raise ValueError('cotrain producer source root differs')
            guard.add(ROOT/n)
        for n,p in paths.items():
            guard.add(p)
            if n.startswith('egoemg_repro/'):
                original=_safe(source_root/n);guard.add(original)
                if _file_hash(original)!=identity['cohort']['source_files_sha256'][n]:raise ValueError('cotrain input provider source differs')
        for p in fit._loop_paths().values():guard.add(p)
        binding=identity['cohort_receipt']
        if binding is not None:guard.add(binding['path'])
        result=fit.read_rn18_initializer_fit(output,require_full=require_full)
        if not result['stage_complete']:raise ValueError('complete producer stage required before branch handoff')
        best=output/'checkpoints'/result['selected_best']['file'];guard.add(best);guard.check()
        return output,identity,result,guard
    except (OSError,KeyError,TypeError,AttributeError)as exc:
        raise ValueError('complete exact RN18 cotrain source/fit unavailable')from exc


def _read_emg_producer(output,export,*,upstream,source_root,require_full=True):
    """Check actual histories, six saved domains, BEST and all110 exported tensors."""
    from .rn18_initializer_train import read_rn18_initializer_export
    output,identity,result,guard=_read_emg_fit(output,upstream=upstream,source_root=source_root,require_full=require_full)
    try:
        export=_safe(export);guard.add(export)
        receipt=read_rn18_initializer_export(export,source_fit=output,require_full=require_full)
        if (len(receipt['tensor_inventory'])!=110 or receipt['source_fit_identity_sha256']!=identity['sha256']or
            receipt['future_fusion_partition_content_sha256']!=identity['cohort']['future_fusion_partition_content_sha256']or
            receipt['source_checkpoint_sha256']!=result['selected_best']['checkpoint_sha256']):
            raise ValueError('complete fixed-holdout native validation BEST lineage required')
        full=result['full_training_terminated']
        value=_seal(dict(format='rn18_fusion_exact_cotrain_best_admission_v1',
            producer_family='rn18-fixed-holdout-cotrain-small-150-0.01',scope='full'if full else'engineering',
            full_initializer=full,completed_epochs=result['completed_epochs'],optimizer_executions=result['optimizer_executions'],
            checkpoint_selection='native validation BEST',export_file_sha256=_file_hash(export),
            source_fit_identity_sha256=identity['sha256'],source_fit_result_file_sha256=_file_hash(output/'fit-result.json'),
            source_cohort_sha256=identity['cohort']['sha256'],source_selected_checkpoint_sha256=receipt['source_checkpoint_sha256'],
            selected_epoch=receipt['selected_epoch'],selected_score=receipt['selected_score'],export_lineage_sha256=receipt['sha256'],
            model_state_sha256=receipt['state_sha256'],tensor_inventory=receipt['tensor_inventory'],
            future_fusion_partition_content_sha256=receipt['future_fusion_partition_content_sha256'],
            producer_reader_sources_sha256=identity['fit_sources_sha256'],scientific_eligible=False,full_paper_complete=False))
        guard.check();return export,value,identity,guard
    except (OSError,KeyError,TypeError,AttributeError)as exc:
        raise ValueError('complete exact RN18 cotrain BEST export unavailable')from exc


def _input_binding(cohort,producer,guard):
    from .rn18_initializer_data import build_rn18_initializer_partitions
    from .emg_initializer_data import _schema
    store=cohort.datasets['train'].store;partitions=build_rn18_initializer_partitions(store)
    old=producer['cohort'];required={'field:emg_left_filtered','field:emg_right_filtered'}
    if not required<=old['payload_files_sha256'].keys()or not required<=cohort.identity['payload_files_sha256'].keys():
        raise ValueError('both actual paired eight-channel EMG payloads required')
    binding=_producer_cohort_binding(cohort,old,partitions)
    binding['footprints']=verify_initializer_footprints(partitions,cohort.partitions)
    if (partitions['provenance']['future_fusion_partition_content_sha256']!=cohort.identity['partition_content_sha256']or
        old['future_fusion_partition_content_sha256']!=cohort.identity['partition_content_sha256']or
        old['future_holdout_overlap_rows']!=dict(val=0,test=0)):
        raise ValueError('cotrain producer reserved another future fusion fold')
    # Gesture/wrist fields consumed only by the producer are independently read;
    # the common-payload comparison alone cannot attest these optional inputs.
    fields=_schema(store);extras={}
    for name in binding['producer_only_payload_files']:
        field=name.removeprefix('field:')
        if not name.startswith('field:')or field not in fields:raise ValueError('producer-only payload absent from actual store')
        path=fields[field];guard.add(path);digest=_file_hash(path)
        if digest!=old['payload_files_sha256'][name]:raise ValueError('producer-only payload bytes differ')
        extras[name]=digest
    binding['producer_only_payload_sha256']=extras
    return binding


@dataclass(eq=False)
class Rn18FusionEmgBest:
    identity:dict
    checkpoint:Path
    source_fit:Path
    source_root:Path
    _cohort:object
    _guard:object

    def check(self,*,require_full=True):
        if type(require_full)is not bool:raise ValueError('explicit full producer scope required')
        r=_EMG.get(self)
        if r is None:raise ValueError('registered exact RN18 cotrain BEST required')
        if require_full and (self.identity.get('scope')!='full'or self.identity.get('full_initializer')is not True):
            raise ValueError('engineering producer cannot initialize full RN18 fusion')
        self._guard.check();self._cohort.check(require_full=require_full)
        if (self.identity.get('sha256')!=r['identity']or _hash({k:v for k,v in self.identity.items()if k!='sha256'})!=r['identity']or
            (self.checkpoint,self.source_fit,self.source_root)!=(r['checkpoint'],r['fit'],r['source_root'])or
            self._cohort is not r['cohort']or _sources()!=r['sources']or
            self._cohort.identity['sha256']!=r['cohort_identity']or _file_hash(self.checkpoint)!=self.identity['export_file_sha256']):
            raise ValueError('registered RN18 cotrain producer/input/source binding changed')
        self._guard.check()


def read_rn18_fusion_emg_best(cohort,emg_fit,emg_export,*,source_root,require_full=True):
    from .rn18_fusion_cohort import Rn18FusionCohort
    if type(cohort)is not Rn18FusionCohort:raise ValueError('registered RN18 fusion cohort required')
    cohort.check(require_full=require_full)
    checkpoint,value,producer,guard=_read_emg_producer(emg_fit,emg_export,upstream=cohort.upstream,source_root=source_root,require_full=require_full)
    value=copy.deepcopy(value);value.pop('sha256');value['input_binding']=_input_binding(cohort,producer,guard)
    value['fusion_cohort_sha256']=cohort.identity['sha256'];value['sources_sha256']=_sources();_seal(value)
    for n in _sources():guard.add(ROOT/n)
    result=Rn18FusionEmgBest(value,checkpoint,_safe(emg_fit),_safe(source_root),cohort,guard)
    _EMG[result]=dict(identity=value['sha256'],checkpoint=checkpoint,fit=result.source_fit,source_root=result.source_root,
        cohort=cohort,cohort_identity=cohort.identity['sha256'],sources=_sources())
    result.check(require_full=require_full);return result


@dataclass(eq=False)
class TrainedRn18FusionInitializers:
    identity:dict
    upstream:Path
    config:object
    _assets:tuple
    producer_sources:dict
    _vision:object
    _emg:object
    _cohort:object
    _guard:object

    def check(self,*,require_full=True):
        if type(require_full)is not bool:raise ValueError('explicit combined handoff scope required')
        r=_COMBINED.get(self)
        if r is None:raise ValueError('registered complete trained RN18 initializers required')
        self._guard.check();self._vision.check();self._emg.check(require_full=True);self._cohort.check(require_full=True)
        protocol=rn18_fusion_protocol(self.upstream,self.config)
        if (self.identity.get('sha256')!=r['identity']or _hash({k:v for k,v in self.identity.items()if k!='sha256'})!=r['identity']or
            _sources()!=r['sources']or _hash(self.producer_sources)!=r['locators']or self._assets!=r['assets']or
            (self._vision,self._emg,self._cohort)!=(r['vision'],r['emg'],r['cohort'])or
            protocol['sha256']!=self.identity['protocol_sha256']or self.upstream!=r['upstream']or
            self.identity['fusion_cohort_sha256']!=self._cohort.identity['sha256']):
            raise ValueError('registered combined RN18 initializer binding changed')
        _check_assets(self._assets)


def _assemble(cohort,vision,emg,output,locators,*,saved=None):
    vision.check();emg.check(require_full=True);cohort.check(require_full=True)
    assets=tuple(_admit(branch.checkpoint,branch.identity['export_file_sha256'])for branch in(vision,emg))
    value=_seal(dict(format='rn18_fusion_complete_trained_initializers_v1',scope='full',trained_branches_admitted=True,
        producer_training_verified=True,scientific_eligible=False,full_paper_complete=False,
        fusion_cohort_sha256=cohort.identity['sha256'],fusion_partition_content_sha256=cohort.identity['partition_content_sha256'],
        protocol_sha256=rn18_fusion_protocol(cohort.upstream,cohort.config)['sha256'],sources_sha256=_sources(),
        files={role:dict(sha256=branch.identity['export_file_sha256'],bytes=branch.checkpoint.stat().st_size)for role,branch in zip(('vision','emg'),(vision,emg))},
        vision=copy.deepcopy(vision.identity),emg=copy.deepcopy(emg.identity)))
    receipt=output/'initializers-admission.json'
    if saved is None:
        from .vision_literal_train import _write
        _write(receipt,value)
    elif saved!=value:raise ValueError('saved combined handoff differs from complete independent reconstruction')
    guard=_StatGuard([receipt,*[ROOT/n for n in _sources()]])
    result=TrainedRn18FusionInitializers(value,cohort.upstream,copy.deepcopy(cohort.config),assets,copy.deepcopy(locators),vision,emg,cohort,guard)
    _COMBINED[result]=dict(identity=value['sha256'],sources=_sources(),locators=_hash(locators),assets=assets,
        vision=vision,emg=emg,cohort=cohort,upstream=cohort.upstream)
    result.check();return result


def _locators(vision_fit,emg_fit,output,vision_source_root,emg_source_root):
    return {n:str(_safe(p))for n,p in dict(vision_fit=vision_fit,emg_fit=emg_fit,output=output,
        vision_source_root=vision_source_root,emg_source_root=emg_source_root).items()}


def admit_rn18_fusion_initializers(cohort,vision_fit,emg_fit,output,*,vision_source_root,emg_source_root):
    """Export only fully verified producers; reject before making any output."""
    from .rn18_fusion_cohort import Rn18FusionCohort
    from .rn18_initializer_train import export_rn18_initializer_best
    from .rn18_fusion_vision_producer import admit_rn18_fusion_vision_last
    if type(cohort)is not Rn18FusionCohort:raise ValueError('registered complete RN18 fusion cohort required')
    cohort.check(require_full=True)
    _read_emg_fit(emg_fit,upstream=cohort.upstream,source_root=emg_source_root,require_full=True)
    output=_safe(output,exists=False)
    if output.exists():raise ValueError('fresh combined initializer output required; reopen existing evidence')
    output.mkdir(mode=0o700)
    export=output/'emg-validation-best.pt';export_rn18_initializer_best(emg_fit,export,require_full=True)
    emg=read_rn18_fusion_emg_best(cohort,emg_fit,export,source_root=emg_source_root)
    vision=admit_rn18_fusion_vision_last(cohort,vision_fit,output/'vision',vision_source_root=vision_source_root)
    return _assemble(cohort,vision,emg,output,_locators(vision_fit,emg_fit,output,vision_source_root,emg_source_root))


def read_rn18_fusion_initializers(cohort,vision_fit,emg_fit,output,*,vision_source_root,emg_source_root):
    """Reopen both complete actual producers and every saved branch tensor."""
    from .rn18_fusion_vision_producer import read_rn18_fusion_vision_last
    output=_safe(output);receipt=output/'initializers-admission.json';guard=_StatGuard([receipt]);saved=_json(receipt)
    emg=read_rn18_fusion_emg_best(cohort,emg_fit,output/'emg-validation-best.pt',source_root=emg_source_root)
    vision=read_rn18_fusion_vision_last(cohort,vision_fit,output/'vision',vision_source_root=vision_source_root)
    result=_assemble(cohort,vision,emg,output,_locators(vision_fit,emg_fit,output,vision_source_root,emg_source_root),saved=saved)
    guard.check();return result
