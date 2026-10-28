"""Completed, holdout-compatible branch handoff for the original ViT-S fusion.

Handoff is not a model result. Historical vision validation executes only a
known frozen reader, without replacing the current producer or its reports.
All checkpoint inputs are trusted local task-owned artifacts.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass,field
import importlib.util
from pathlib import Path
import sys

from .classic_emg import _file_hash,_hash
from .classic_emg_cohort import _StatGuard,_safe,_json
from .data import union_intervals
from .fusion_literal_runtime import _admit


ROOT=Path(__file__).resolve().parents[1]
IMPORTED_SHA=_file_hash(__file__)
FORMAT='original_fusion_initializer_handoff_v1'
LOCAL_SOURCES=('egoemg_repro/fusion_initializer_handoff.py','egoemg_repro/fusion_literal_runtime.py',
    'egoemg_repro/fusion_literal_data.py','egoemg_repro/vision_literal_data.py',
    'egoemg_repro/emg_initializer_data.py','egoemg_repro/emg_initializer_train.py')
HISTORICAL_VISION_SOURCES={
    'egoemg_repro/vision_literal_train.py':'e44dc947111bec3aa858fb5fc36493650cda24f2093915bcd133a2d090d070da',
    'egoemg_repro/vision_literal_cohort.py':'997cb2f7975f49b45b6313294b0dc3f20150f775bbfe65aa1c02857ace89b977',
    'egoemg_repro/classic_emg_train.py':'52b1ce655f3c4e107e3d809e403cd507fbcfd655f7ac9ea7bc0b9a592b18b0d8',
    'egoemg_repro/vision_literal_check.py':'9601034285aaea5cb35abed793576ab5ecf901146b55a4c3901f7b242234b227'}
_REQUIRED_PAYLOAD={'metadata:manifest','metadata:episodes','field:frame_split_id','field:generated_label_valid',
    *('field:'+name for hand in ('left','right') for name in
      (f'generated_joint_angles_{hand}',f'mocap_{hand}_wrist_pitch',f'mocap_{hand}_wrist_yaw'))}
_EXTRA_INITIALIZER_PAYLOAD={'field:label_gesture_class','field:label_gesture_active',
    'field:mocap_left_wrist_angles_valid','field:mocap_right_wrist_angles_valid'}


def _sources():
    if _file_hash(__file__)!=IMPORTED_SHA:raise ValueError('imported initializer handoff source changed')
    return {name:_file_hash(ROOT/name) for name in LOCAL_SOURCES}


def _bounds(a,b):
    if type(a) is not int or type(b) is not int or a<0 or b<=a:
        raise ValueError('complete positive half-open integer footprint/interval required')
    return a,b


def _intersection(left,right):
    i=j=total=0
    while i<len(left) and j<len(right):
        a,b=left[i];c,d=right[j];total+=max(0,min(b,d)-max(a,c))
        if b<d:i+=1
        else:j+=1
    return total


def verify_initializer_footprints(producer,fusion):
    """Reject training overlap with whole future validation intervals or test IO."""
    try:
        train=union_intervals(_bounds(w['footprint_start'],w['footprint_end'])
            for w in producer['windows'] if w['role']=='train')
        holdouts=dict(val=union_intervals(_bounds(v['start'],v['end']) for v in fusion['validation_intervals']),
            test=union_intervals(_bounds(w['footprint_start'],w['footprint_end'])
                for w in fusion['windows'] if w['role']=='test'))
    except (KeyError,TypeError) as exc:raise ValueError('complete initializer and future holdout footprints required') from exc
    if not train or any(not spans for spans in holdouts.values()):
        raise ValueError('nonempty initializer train and both holdout footprints required')
    overlaps={r:_intersection(train,spans) for r,spans in holdouts.items()}
    if any(overlaps.values()):raise ValueError('initializer training input overlaps future fusion validation/test footprint')
    return dict(train_union_rows=sum(b-a for a,b in train),
        holdout_union_rows={r:sum(b-a for a,b in spans) for r,spans in holdouts.items()},overlap_rows=overlaps)


def _producer_cohort_binding(cohort,producer,partitions):
    """Compare reconstructed partition and every common actual payload digest."""
    if _hash(partitions)!=producer.get('partition_content_sha256'):
        raise ValueError('producer cohort partition differs from independent reconstruction')
    old=producer['payload_files_sha256'];current=cohort.identity['payload_files_sha256']
    if not _REQUIRED_PAYLOAD<=old.keys() or not _REQUIRED_PAYLOAD<=current.keys():
        raise ValueError('producer cohort lacks required paired metadata/label payload')
    if any(name not in current and name not in _EXTRA_INITIALIZER_PAYLOAD for name in old):
        raise ValueError('producer cohort payload is absent from current paired inputs')
    shared=old.keys()&current.keys()
    if any(old[name]!=current[name] for name in shared):
        raise ValueError('producer cohort metadata or paired payload bytes differ')
    for key in ('manifest_sha256','metadata_sha256','frame_split_sha256'):
        if partitions['provenance'][key]!=cohort.partitions['provenance'][key]:
            raise ValueError('producer partition metadata provenance differs')
    return dict(partition_content_sha256=_hash(partitions),cohort_sha256=producer['sha256'],
        shared_payload_files=len(shared),shared_payload_sha256=_hash({name:old[name] for name in sorted(shared)}),
        producer_only_payload_files=sorted(old.keys()-current.keys()))


def _historical_vision_reader(source_root,identity):
    """Load only the already published reader; never run an arbitrary source."""
    if identity.get('fit_sources_sha256')!=HISTORICAL_VISION_SOURCES:
        raise ValueError('historical vision source inventory is not the known frozen reader')
    try:
        source_root=_safe(source_root)
        paths=[source_root/name for name in HISTORICAL_VISION_SOURCES]
        paths.extend(ROOT/name for name in HISTORICAL_VISION_SOURCES if not name.endswith('/vision_literal_train.py'))
        guard=_StatGuard(paths)
        if any(_file_hash(source_root/name)!=sha for name,sha in HISTORICAL_VISION_SOURCES.items()):
            raise ValueError('frozen historical vision source bytes differ')
        if any(_file_hash(ROOT/name)!=sha for name,sha in HISTORICAL_VISION_SOURCES.items()
            if not name.endswith('/vision_literal_train.py')):
            raise ValueError('current common dependencies differ from frozen historical source')
        name='egoemg_repro._fusion_historical_vision_reader'
        spec=importlib.util.spec_from_file_location(name,source_root/'egoemg_repro/vision_literal_train.py')
        if spec is None or spec.loader is None:raise ValueError('frozen historical vision source cannot be loaded')
        module=importlib.util.module_from_spec(spec);prior=sys.modules.get(name);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        finally:
            if prior is None:sys.modules.pop(name,None)
            else:sys.modules[name]=prior
        guard.check()
        if module._sources()!=HISTORICAL_VISION_SOURCES:raise ValueError('loaded historical reader source differs')
        return module,guard
    except OSError as exc:raise ValueError('frozen historical vision source unavailable') from exc


def _documents(output):
    try:
        output=_safe(output);guard=_StatGuard([output/'identity.json',output/'fit-result.json',output/'epochs.jsonl'])
        identity=_json(output/'identity.json');result=_json(output/'fit-result.json')
        if identity['policy']['scope']!='full':raise ValueError('engineering producer cannot initialize a full fusion fit')
        best=result['selected_best']['file']
        if type(best) is not str or Path(best).name!=best or best=='last.ckpt':raise ValueError('native selected best filename required')
        for path in (output/'checkpoints/last.ckpt',output/'checkpoints'/best):guard.add(path)
        binding=identity['cohort_receipt']
        if not isinstance(binding,dict):raise ValueError('full producer requires frozen cohort receipt')
        guard.add(binding['path']);guard.check()
        return output,identity,result,guard
    except (OSError,KeyError,TypeError) as exc:raise ValueError('complete full producer fit documents unavailable') from exc


def _vision_producer(output,source_root):
    """Reopen a completed full ViT-S fit and choose native LAST, not its best."""
    output,identity,_,guard=_documents(output)
    from . import vision_literal_train as current
    if identity['construction'].get('variant')!='vit_small' or identity['policy'].get('variant')!='vit_small':
        raise ValueError('full native ViT-S vision producer required')
    if identity['fit_sources_sha256']==current._sources():
        source_root=_safe(source_root)
        if any(_file_hash(source_root/name)!=sha for name,sha in identity['fit_sources_sha256'].items()):
            raise ValueError('current vision source root differs from producer')
        reader=current
        for name in identity['fit_sources_sha256']:guard.add(source_root/name);guard.add(ROOT/name)
    else:
        reader,historical=_historical_vision_reader(source_root,identity)
        for path in historical.paths:guard.add(path)
    for path in reader._loop_paths().values():guard.add(path)
    result=reader.read_vision_fit(output,require_full=True);guard.check()
    path=output/'checkpoints/last.ckpt';digest=result['last_checkpoint_sha256']
    _admit(path,digest);guard.check()
    summary=dict(checkpoint_selection='completed native vision last; not validation best',
        checkpoint_sha256=digest,source_fit_identity_sha256=identity['sha256'],
        source_fit_result_sha256=_file_hash(output/'fit-result.json'),source_fit_identity_file_sha256=_file_hash(output/'identity.json'),
        source_cohort_sha256=identity['cohort']['sha256'],completed_epochs=result['completed_epochs'],
        optimizer_executions=result['optimizer_executions'],model_state_sha256=result['final_model_state_sha256'],
        producer_reader_sources_sha256=identity['fit_sources_sha256'])
    guard.check();return path,summary,identity,guard


def _emg_producer(output,export):
    """Reopen the full compatible150 fit and its complete validation-best export."""
    output,identity,_,guard=_documents(output);export=_safe(export);guard.add(export)
    from .emg_initializer_train import read_emg_initializer_fit,read_emg_initializer_export,_sources as fit_sources,_loop_paths
    for name in fit_sources():guard.add(ROOT/name)
    for path in _loop_paths().values():guard.add(path)
    result=read_emg_initializer_fit(output,require_full=True)
    receipt=read_emg_initializer_export(export,source_fit=output,require_full=True);guard.check()
    digest=_file_hash(export);_admit(export,digest);guard.check()
    summary=dict(checkpoint_selection='completed native EMG validation best; complete source-fit-bound export',
        checkpoint_sha256=digest,source_fit_identity_sha256=identity['sha256'],
        source_fit_result_sha256=_file_hash(output/'fit-result.json'),source_cohort_sha256=identity['cohort']['sha256'],
        completed_epochs=result['completed_epochs'],optimizer_executions=result['optimizer_executions'],
        source_selected_checkpoint_sha256=result['selected_best']['checkpoint_sha256'],selected_epoch=receipt['selected_epoch'],
        selected_score=receipt['selected_score'],export_lineage_sha256=receipt['sha256'],
        model_state_sha256=receipt['state_sha256'],complete_source_tensors=len(receipt['tensor_inventory']),
        future_fusion_partition_content_sha256=receipt['future_fusion_partition_content_sha256'])
    guard.check();return export,summary,identity,guard


def _runtime_seal(value):return _hash({k:str(v) if isinstance(v,Path) else v for k,v in value.items()})


@dataclass(frozen=True)
class FusionInitializers:
    identity: dict
    runtime_kwargs: dict
    _cohort: object=field(repr=False)
    _guard: object=field(repr=False)
    _seal: str=field(repr=False)
    _runtime_binding: str=field(repr=False)
    producer_sources: dict|None=field(default=None,repr=False)
    _producer_binding: str=field(default='',repr=False)

    def check(self,*,require_full=False):
        if type(require_full) is not bool:raise ValueError('explicit full handoff requirement must be boolean')
        self._guard.check();self._cohort.check()
        if (self.identity.get('sha256')!=self._seal or _hash({k:v for k,v in self.identity.items() if k!='sha256'})!=self._seal or
            self.identity['sources_sha256']!=_sources() or _runtime_seal(self.runtime_kwargs)!=self._runtime_binding or
            _hash(self.producer_sources)!=self._producer_binding or
            self.identity['fusion_cohort_sha256']!=self._cohort.identity['sha256'] or
            self.identity['fusion_partition_content_sha256']!=self._cohort.identity['partition_content_sha256']):
            raise ValueError('fusion initializer identity/source/runtime binding changed')
        if require_full and (self.identity['scope']!='full' or self.identity['producer_training_verified'] is not True):
            raise ValueError('engineering handoff cannot initialize full fusion training')


def _assemble(cohort,kwargs,vision,emg,*,scope,guards=(),producer_sources=None):
    guard=_StatGuard([ROOT/name for name in LOCAL_SOURCES])
    for child in guards:
        for path in child.paths:guard.add(path)
    for role in ('vision','emg'):
        path,_=_admit(kwargs[role+'_checkpoint'],kwargs[role+'_sha256']);guard.add(path)
    identity=dict(format=FORMAT,scope=scope,producer_training_verified=scope=='full',scientific_eligible=False,
        full_paper_complete=False,fusion_cohort_sha256=cohort.identity['sha256'],
        fusion_partition_content_sha256=cohort.identity['partition_content_sha256'],sources_sha256=_sources(),
        vision=copy.deepcopy(vision),emg=copy.deepcopy(emg))
    identity['sha256']=_hash(identity)
    result=FusionInitializers(identity,copy.copy(kwargs),cohort,guard,identity['sha256'],_runtime_seal(kwargs),
        copy.deepcopy(producer_sources),_hash(producer_sources))
    result.check(require_full=scope=='full');return result


def engineering_fusion_initializers(cohort,*,vision_checkpoint,vision_sha256,emg_checkpoint,emg_sha256):
    """Explicit bounded engineering only; never asserts that files were trained."""
    cohort.check();kwargs=dict(vision_checkpoint=Path(vision_checkpoint).absolute(),vision_sha256=vision_sha256,
        emg_checkpoint=Path(emg_checkpoint).absolute(),emg_sha256=emg_sha256)
    return _assemble(cohort,kwargs,dict(checkpoint_selection='explicit trusted engineering file',checkpoint_sha256=vision_sha256),
        dict(checkpoint_selection='explicit trusted engineering file',checkpoint_sha256=emg_sha256),scope='engineering')


def admit_fusion_initializers(cohort,*,vision_fit,vision_source_root,emg_fit,emg_export):
    """Full-only branch admission against complete current fusion input bytes."""
    from .fusion_literal_cohort import FusionCohort,FULL_COUNTS
    if type(cohort) is not FusionCohort:raise ValueError('actual complete cohort object required')
    cohort.check()
    if (cohort.identity['metadata_matches_registered_legacy_release'] is not True or
        cohort.identity['counts']!=FULL_COUNTS or cohort.identity['full_payload_content_hashed'] is not True):
        raise ValueError('full initializer handoff requires complete registered fusion cohort')
    vision,vs,vi,vg=_vision_producer(vision_fit,vision_source_root)
    emg,es,ei,eg=_emg_producer(emg_fit,emg_export)
    from .vision_literal_data import build_vision_partitions
    from .emg_initializer_data import build_initializer_partitions
    store=cohort.datasets['train'].store
    vp=build_vision_partitions(store);ep=build_initializer_partitions(store)
    vs['cohort_binding']=_producer_cohort_binding(cohort,vi['cohort'],vp)
    es['cohort_binding']=_producer_cohort_binding(cohort,ei['cohort'],ep)
    vs['footprints']=verify_initializer_footprints(vp,cohort.partitions)
    es['footprints']=verify_initializer_footprints(ep,cohort.partitions)
    if (es['future_fusion_partition_content_sha256']!=cohort.identity['partition_content_sha256'] or
        ei['cohort']['future_fusion_partition_content_sha256']!=cohort.identity['partition_content_sha256'] or
        ep['provenance']['fusion_partition_content_sha256']!=cohort.identity['partition_content_sha256']):
        raise ValueError('EMG producer did not reserve this exact future fusion partition')
    cohort.check();vg.check();eg.check()
    kwargs=dict(vision_checkpoint=vision,vision_sha256=vs['checkpoint_sha256'],emg_checkpoint=emg,emg_sha256=es['checkpoint_sha256'])
    sources={name:str(_safe(path)) for name,path in dict(vision_fit=vision_fit,vision_source_root=vision_source_root,
        emg_fit=emg_fit,emg_export=emg_export).items()}
    return _assemble(cohort,kwargs,vs,es,scope='full',guards=(vg,eg),producer_sources=sources)
