"""Selected original NeuroPose: native evaluation and exact new-process replay.

Only bounded typed receipts and sufficient statistics are retained. Training
continuation equivalence is not claimed by this fresh-run execution profile.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import math
import os
from pathlib import Path
import time

from . import neuropose_native_train as train

ROOT=train.ROOT
IMPORTED_SHA=train._sha(__file__)
FORMAT='native_neuropose_selected_pass_v1'
PROFILE='native_neuropose_pretest_profile_v1'
POLICY=dict(prediction='exact_typed_prediction_replay_v1',native_loss_atol=1e-7,native_loss_rtol=2e-6,
    native_loss_aggregation='sample-count weighted original per-batch composite loss',
    pooled_aggregation='float64 absolute-error sum / exact valid-angle count; degrees',
    trajectory_length=7790,center_index=3895,original_angles=22)


def _sources():
    if train._sha(__file__)!=IMPORTED_SHA:raise ValueError('imported native evaluation source changed')
    names=('egoemg_repro/neuropose_native_eval.py','egoemg_repro/neuropose_native_train.py',
        'egoemg_repro/classic_emg_report.py','egoemg_repro/classic_emg_evaluate.py','egoemg_repro/metrics.py')
    return {n:train._sha(ROOT/n) for n in names}


def _safe(path,*,exists=True):
    from .classic_emg_cohort import _safe as check
    return check(path,exists=exists)


def _process():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=fields[19],uid=os.getuid())


def _merge(guard,other):
    other.check()
    for path,stamp in other.paths.items():
        if path in guard.paths and guard.paths[path]!=stamp:raise ValueError('native evidence input generation changed')
        guard.paths[path]=stamp
    guard.parents.update(other.parents);guard.check()


class _Evidence:
    """Read raw dependencies once in one guarded transaction; no global cache."""
    def __init__(self):
        from .classic_emg_cohort import _StatGuard
        self.guard=_StatGuard([ROOT/n for n in _sources()]);self.fits={};self.scalers={};self.passes={};self.profiles={};self.active=set()

    def fit(self,path):
        path=_safe(path)
        if path not in self.fits:
            result,identity,last,guard=train._read_fit(path,require_full=False)
            _merge(self.guard,guard);del last
            from . import classic_emg_train as storage
            scaler={}
            if identity['policy']['scope']!='cpu_engineering':
                checkpoint=storage._runtime().load(str(path/'checkpoints'/result['selected_best']['file']),map_location='cpu',weights_only=False)
                scaler=checkpoint['MixedPrecision'];del checkpoint
            self.scalers[path]=storage.state_digest(scaler)
            self.fits[path]=(result,identity)
        return self.fits[path]


def _require_full(value,require_full):
    if type(require_full) is not bool:raise ValueError('explicit full requirement must be boolean')
    if require_full and value['scope']!='full':raise ValueError('engineering prediction cannot qualify full native training')


def _read_pass(path,context):
    from . import classic_emg_report as report
    path=_safe(path)
    if path in context.passes:return context.passes[path]
    key=('pass',path)
    if key in context.active:raise ValueError('cyclic native prediction evidence')
    context.active.add(key)
    try:
        for name in ('pass.json','coverage.json','statistics.json','native-metrics.json'):context.guard.add(path/name)
        v=train._json(path/'pass.json');train._check(v)
        if (v['format']!=FORMAT or v['sources']!=_sources() or v['agreement_policy']!=POLICY or
            v['role'] not in ('val','test') or v['kind'] not in ('native','replay')):
            raise ValueError('native pass source/policy/role differs')
        fit,identity=context.fit(v['fit_path']);role=v['role'];scope=fit['scope']
        if (v['fit_result_sha256']!=fit['sha256'] or v['selected_best']!=fit['selected_best'] or
            v['scope']!=scope or v['runtime']!=identity['runtime'] or v['counts']!=identity['counts'] or
            v['batch_size']!=identity['policy']['batch_size'] or
            v['model_before_sha256']!=fit['selected_best']['model_sha256'] or
            v['model_after_sha256']!=v['model_before_sha256'] or v['scaler_before_sha256']!=v['scaler_after_sha256'] or
            v['scaler_before_sha256']!=context.scalers[_safe(v['fit_path'])] or
            v['device_type']!=('cpu' if scope=='cpu_engineering' else 'cuda') or
            v['cuda_initialized'] is not (scope!='cpu_engineering') or v['started_ns']<=fit['finished_ns'] or
            v['scientific_eligible'] is not False or v['full_paper_complete'] is not False):
            raise ValueError('native pass differs from raw selected fit/state/runtime')
        for name,keyname in (('coverage.json','coverage'),('statistics.json','statistics'),('native-metrics.json','native_metrics')):
            if train._json(path/name)!=v[keyname]:raise ValueError('native raw prediction sidecar differs')
        report._validate_saved_coverage(v['coverage'],dict(variant='neuropose',role=role,
            samples=identity['counts'][role],batch_size=v['batch_size']))
        if role=='val':
            if v['statistics'] is not None or v['selected_validation_digest']!=fit['selected_best']['validation_prediction_sha256']:
                raise ValueError('selected native validation digest differs')
            if v['profile_path'] is not None or v['profile_sha256'] is not None:raise ValueError('validation cannot consume a test profile')
        else:
            if v['selected_validation_digest'] is not None:raise ValueError('test cannot invent a validation digest')
            _check_statistics(v['statistics'],identity['counts']['test'])
            p=_read_profile(v['profile_path'],context)
            if (v['profile_sha256']!=p['sha256'] or p['fit_result_sha256']!=fit['sha256'] or v['started_ns']<=p['finished_ns']):
                raise ValueError('test requires its earlier frozen validation profile')
        if v['kind']=='native':
            metrics=v['native_metrics'];loss=metrics.get(role+'_loss')
            if (not metrics or any(type(x) not in (float,int) or not math.isfinite(x) for x in metrics.values()) or
                not isinstance(loss,(float,int)) or not math.isfinite(v['sample_weighted_native_loss']) or
                not math.isclose(loss,v['sample_weighted_native_loss'],rel_tol=2e-6,abs_tol=1e-7) or
                v['reference_path'] is not None or v['reference_sha256'] is not None):
                raise ValueError('actual native sample-weighted metric differs')
            if role=='val' and not math.isclose(loss,fit['selected_best']['score'],rel_tol=2e-6,abs_tol=1e-7):
                raise ValueError('native selected validation score differs')
        else:
            if v['native_metrics']!={} or v['sample_weighted_native_loss'] is not None:raise ValueError('replay cannot invent native metrics')
            ref=_read_pass(v['reference_path'],context)
            if (ref['kind']!='native' or ref['role']!=role or ref['fit_result_sha256']!=fit['sha256'] or
                ref['sha256']!=v['reference_sha256'] or ref['process']==v['process'] or
                v['started_ns']<=ref['finished_ns'] or ref['profile_sha256']!=v['profile_sha256'] or
                ref['statistics']!=v['statistics']):raise ValueError('replay requires an earlier exact native pass in a different process')
            report.compare_prediction_replay(ref['coverage'],v['coverage'])
        if (type(v['started_ns']) is not int or type(v['finished_ns']) is not int or
            v['finished_ns']<v['started_ns'] or set(v['process'])!={'pid','start_ticks','uid'} or
            type(v['process']['pid']) is not int or v['process']['pid']<=0 or type(v['process']['uid']) is not int or
            type(v['process']['start_ticks']) is not str or not v['process']['start_ticks'].isdigit()):
            raise ValueError('actual native prediction process/time required')
        context.guard.check();context.passes[path]=v;return v
    finally:context.active.remove(key)


def _check_statistics(value,samples):
    """The same full schema/conservation checks, with the admitted population."""
    from . import classic_emg_report as report
    if samples==4306:return report.validate_test_statistics(value,variant='neuropose')
    if type(value) is not dict or value.get('samples')!=samples or value.get('alignment')!=report.alignment('neuropose'):
        raise ValueError('actual engineering statistic population/alignment differs')
    # The strict full checker fixes4306. Scale the *missing* population as invalid
    # only in this private validation copy; no added predictions or public counts.
    checked=copy.deepcopy(value);checked['samples']=4306
    for mode,width in (('trajectory',7790),('center',1)):
        checked[mode]['invalid_angle_count']+=(4306-samples)*width*22
        # Tiny engineering populations need not contain every held-out split.
        zero=report._record(0.,0)
        for hand in ('left','right'):
            checked[mode]['by_hand'].setdefault(hand,copy.deepcopy(zero))
            checked[mode]['by_hand_split'].setdefault(hand,{})
            for split in ('user','gesture','both'):checked[mode]['by_hand_split'][hand].setdefault(split,copy.deepcopy(zero))
        for split in ('user','gesture','both'):checked[mode]['by_split'].setdefault(split,copy.deepcopy(zero))
    report.validate_test_statistics(checked,variant='neuropose')


def _read_profile(path,context):
    from .classic_emg_report import compare_prediction_replay
    path=_safe(path)
    if path in context.profiles:return context.profiles[path]
    key=('profile',path)
    if key in context.active:raise ValueError('cyclic native validation profile')
    context.active.add(key)
    try:
        context.guard.add(path/'profile.json');v=train._json(path/'profile.json');train._check(v)
        left=_read_pass(v['native_path'],context);right=_read_pass(v['replay_path'],context)
        if (v['format']!=PROFILE or v['sources']!=_sources() or left['role']!='val' or right['kind']!='replay' or
            _safe(right['reference_path'])!=_safe(v['native_path']) or left['sha256']!=v['native_sha256'] or
            right['sha256']!=v['replay_sha256'] or v['fit_result_sha256']!=left['fit_result_sha256'] or
            v['scope']!=left['scope'] or v['agreement']!=compare_prediction_replay(left['coverage'],right['coverage']) or
            v['finished_ns']<=max(left['finished_ns'],right['finished_ns']) or v['scientific_eligible'] is not False):
            raise ValueError('raw frozen native validation profile differs')
        context.guard.check();context.profiles[path]=v;return v
    finally:context.active.remove(key)


def read_neuropose_pass(output):
    context=_Evidence();v=_read_pass(output,context);context.guard.check();return v


def freeze_neuropose_validation(native,replay,output):
    from .classic_emg_report import compare_prediction_replay
    context=_Evidence();left=_read_pass(native,context);right=_read_pass(replay,context)
    if (left['role']!='val' or right['role']!='val' or right['kind']!='replay' or
        _safe(right['reference_path'])!=_safe(native)):raise ValueError('actual native validation and exact replay required')
    output=_safe(output,exists=False)
    if output.exists():raise FileExistsError('frozen validation profile exists')
    v=train._sealed(dict(format=PROFILE,native_path=str(_safe(native)),replay_path=str(_safe(replay)),
        native_sha256=left['sha256'],replay_sha256=right['sha256'],fit_result_sha256=left['fit_result_sha256'],
        scope=left['scope'],sources=_sources(),agreement=compare_prediction_replay(left['coverage'],right['coverage']),
        finished_ns=time.time_ns(),scientific_eligible=False))
    context.guard.check();output.mkdir(mode=0o700);train._write(output/'profile.json',v);context.guard.check();return v


def run_neuropose_pass(fit,output,*,role,kind,reference=None,profile=None,require_full=True):
    from . import classic_emg_train as storage,classic_emg_formal as formal,classic_emg_report as report
    if role not in ('val','test') or kind not in ('native','replay'):raise ValueError('explicit native role/kind required')
    context=_Evidence();result,identity=context.fit(fit);_require_full(result,require_full)
    if role=='test' and profile is None:raise ValueError('test requires earlier frozen validation profile')
    frozen=_read_profile(profile,context) if role=='test' else None
    if (role=='val' and profile is not None) or (frozen and frozen['fit_result_sha256']!=result['sha256']):
        raise ValueError('validation profile belongs to another native fit/role')
    ref=_read_pass(reference,context) if kind=='replay' and reference is not None else None
    if ((kind=='native' and reference is not None) or (kind=='replay' and
        (ref is None or ref['kind']!='native' or ref['role']!=role or ref['fit_result_sha256']!=result['sha256'] or
         ref['process']==_process() or ref['profile_sha256']!=(frozen['sha256'] if frozen else None)))):
        raise ValueError('replay requires actual reference from a different process and same fit/profile')
    output=_safe(output,exists=False)
    if output.exists():raise FileExistsError('native prediction output exists')
    storage._space(output.parent);started=time.time_ns();prepared=train.reopen_neuropose_best(fit,require_full=require_full)
    torch=storage._runtime();dataset=prepared.datasets[role]
    subjects={w['episode']:str(dataset.store.episodes[w['episode']]['subject']) for w in dataset.windows}
    ledger=report.ReportInputLedger(dataset.windows,variant='neuropose',batch_size=prepared.policy['batch_size'],role=role,subjects=subjects)
    statistics=report.ClassicAngleReport('neuropose') if role=='test' else None
    pending=[];parts=[];losses=[]
    from torch.utils.data._utils.collate import default_collate
    def collate(items):
        batch=default_collate(items);receipt=ledger.prepare(batch);pending.append((batch,receipt));return batch
    loader=torch.utils.data.DataLoader(formal._Dataset(dataset,prepared,len(dataset)),batch_size=prepared.policy['batch_size'],
        shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(46 if role=='val' else 45),collate_fn=collate)
    module=prepared.module;original_forward=module.forward
    def observed(batch):
        prepared.check()
        if not pending:raise ValueError('unprepared native forward')
        original,receipt=pending.pop(0)
        if report._evaluation()._tensor_digest(batch['emg'])!=receipt['emg_sha256']:raise ValueError('actual transferred EMG differs')
        output_tuple=original_forward(batch);ledger.commit(original,receipt,output_tuple)
        if role=='val':parts.append(train._part(batch,output_tuple))
        else:
            prediction,target,mask=(x.detach().cpu().numpy() for x in output_tuple)
            for i in range(len(prediction)):
                statistics.update(prediction[i],target[i],mask[i],hand=original['hand'][i],split=original['split'][i],user=original['subject'][i])
        return output_tuple
    module.forward=observed
    scaler_before=storage.state_digest(prepared.precision.state_dict() if prepared.precision else {})
    model_before=storage.state_digest(module.state_dict());metrics={}
    try:
        if kind=='native':
            class LossWitness(storage.pl.Callback):
                def take(self,outputs,batch):
                    loss=float(outputs.detach().cpu())
                    if not math.isfinite(loss):raise ValueError('nonfinite original native loss')
                    losses.append((len(batch['emg']),loss))
                def on_validation_batch_end(self,trainer,module,outputs,batch,batch_idx,dataloader_idx=0):self.take(outputs,batch)
                def on_test_batch_end(self,trainer,module,outputs,batch,batch_idx,dataloader_idx=0):self.take(outputs,batch)
            kwargs=prepared.policy['trainer'].copy();plugins=[]
            if prepared.precision is not None:kwargs.pop('precision');plugins=[prepared.precision]
            native=storage.pl.Trainer(**kwargs,callbacks=[LossWitness()],plugins=plugins,logger=False,enable_checkpointing=False,
                enable_progress_bar=False,enable_model_summary=False)
            getattr(native,'validate' if role=='val' else 'test')(module,dataloaders=loader,verbose=False)
            metrics={k:float(v.detach().cpu()) for k,v in native.callback_metrics.items()}
        else:
            device='cpu' if prepared.policy['scope']=='cpu_engineering' else 'cuda:0';module.to(device).eval()
            for original in loader:
                batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in original.items()}
                with torch.inference_mode(),prepared.precision.test_step_context() if prepared.precision else nullcontext():module.forward(batch)
    finally:module.forward=original_forward
    coverage=ledger.finish();stats=statistics.report() if statistics else None
    after=storage.state_digest(module.state_dict());scaler_after=storage.state_digest(prepared.precision.state_dict() if prepared.precision else {})
    if model_before!=after or scaler_before!=scaler_after or any(p.grad is not None for p in module.parameters()):
        raise ValueError('native prediction mutated parameters/buffers/scaler or created gradients')
    validation_digest=train._hash(parts) if role=='val' else None
    if role=='val' and validation_digest!=result['selected_best']['validation_prediction_sha256']:
        raise ValueError('strict complete selected validation predictions differ')
    if ref is not None:
        report.compare_prediction_replay(ref['coverage'],coverage)
        if ref['statistics']!=stats:raise ValueError('exact test sufficient statistics differ')
    weighted=math.fsum(n*x for n,x in losses)/sum(n for n,_ in losses) if losses else None
    if kind=='native' and (sum(n for n,_ in losses)!=len(dataset) or
        not math.isclose(metrics[role+'_loss'],weighted,rel_tol=2e-6,abs_tol=1e-7)):
        raise ValueError('native complete sample-weighted loss differs')
    if kind=='native' and role=='val' and not math.isclose(metrics['val_loss'],result['selected_best']['score'],rel_tol=2e-6,abs_tol=1e-7):
        raise ValueError('native complete selected validation loss differs')
    v=train._sealed(dict(format=FORMAT,fit_path=str(_safe(fit)),fit_result_sha256=result['sha256'],scope=result['scope'],
        selected_best=result['selected_best'],sources=_sources(),agreement_policy=POLICY,runtime=identity['runtime'],counts=identity['counts'],
        role=role,kind=kind,batch_size=prepared.policy['batch_size'],process=_process(),started_ns=started,finished_ns=time.time_ns(),
        reference_path=str(_safe(reference)) if ref else None,reference_sha256=ref['sha256'] if ref else None,
        profile_path=str(_safe(profile)) if frozen else None,profile_sha256=frozen['sha256'] if frozen else None,
        coverage=coverage,statistics=stats,native_metrics=metrics,sample_weighted_native_loss=weighted,
        selected_validation_digest=validation_digest,model_before_sha256=model_before,model_after_sha256=after,
        scaler_before_sha256=scaler_before,scaler_after_sha256=scaler_after,scientific_eligible=False,full_paper_complete=False))
    v.pop('sha256');v=train._sealed(dict(v,device_type=next(module.parameters()).device.type,cuda_initialized=torch.cuda.is_initialized()))
    if v['device_type']!=('cpu' if result['scope']=='cpu_engineering' else 'cuda') or v['cuda_initialized'] is not (result['scope']!='cpu_engineering'):
        raise ValueError('actual native prediction device/CUDA initialization differs')
    if stats is not None:_check_statistics(stats,len(dataset))
    prepared.check();context.guard.check();output.mkdir(mode=0o700)
    for name,key in (('coverage.json','coverage'),('statistics.json','statistics'),('native-metrics.json','native_metrics')):train._write(output/name,v[key])
    prepared.check();context.guard.check();train._write(output/'pass.json',v)
    return v


QUALIFICATION='native_neuropose_cuda_frozen_model_qualification_v1'


def _qualified_runtime(identity):
    raw=train._json(Path(identity['preconditions']['numeric_prefix'])/'identity.json')['runtime']
    if (raw!=identity['runtime'] or raw.get('torch_build')!='2.2.2+cu118' or
        not raw.get('visible_uuid','').startswith('GPU-') or not raw.get('device_name')):
        raise ValueError('actual numeric/CUDA engineering runtime binding differs')
    return raw


def _exited(process):
    try:fields=Path('/proc',str(process['pid']),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return True
    return fields[19]!=str(process['start_ticks']) or fields[0]=='Z'


def _qualification_evidence(fit,native,replay,context):
    from .classic_emg_report import compare_prediction_replay
    result,identity=context.fit(fit)
    if (result['scope']!='cuda_engineering' or identity['counts']!=dict(train=512,val=1576,test=4306) or
        identity['underlying_counts']!=train.COUNTS or identity['policy']['batch_size']!=256 or
        identity['policy']['trainer']['precision']!='16-mixed' or result['completed_epochs']!=2 or
        result['attempts']!=4 or not 0<result['optimizer_executions']<=4):
        raise ValueError('actual CUDA engineering B256/two-epoch/full-validation fit required')
    runtime=_qualified_runtime(identity)
    left=_read_pass(native,context);right=_read_pass(replay,context)
    if (left['kind']!='native' or left['role']!='val' or right['role']!='val' or right['kind']!='replay' or
        left['fit_result_sha256']!=result['sha256'] or right['fit_result_sha256']!=result['sha256'] or
        _safe(right['reference_path'])!=_safe(native)):
        raise ValueError('actual completed CUDA fit and full frozen validation/native replay required')
    owners=[result['execution']['process'],left['process'],right['process']]
    if not all(_exited(p) for p in owners):raise ValueError('actual fit/native/replay owners must have exited before qualification')
    context.guard.check()
    return dict(format=QUALIFICATION,fit_path=str(_safe(fit)),native_path=str(_safe(native)),replay_path=str(_safe(replay)),
        fit_result_sha256=result['sha256'],native_sha256=left['sha256'],replay_sha256=right['sha256'],
        training_sources=identity['sources'],evaluation_sources=_sources(),cohort_sha256=identity['input_identity']['sha256'],
        precondition_files_sha256=identity['precondition_files_sha256'],runtime=runtime,owners=owners,
        agreement=compare_prediction_replay(left['coverage'],right['coverage']),scientific_eligible=False,full_paper_complete=False)


def qualify_neuropose(fit,native,replay,output):
    context=_Evidence();value=_qualification_evidence(fit,native,replay,context)
    output=_safe(output,exists=False)
    if output.exists():raise FileExistsError('native CUDA qualification exists')
    value=train._sealed(dict(value,finished_ns=time.time_ns()))
    context.guard.check();output.mkdir(mode=0o700);train._write(output/'qualification.json',value);context.guard.check();return value


def read_neuropose_qualification(output):
    if not isinstance(output,(str,os.PathLike)):raise ValueError('actual raw CUDA qualification directory required')
    output=_safe(output);context=_Evidence();context.guard.add(output/'qualification.json')
    v=train._json(output/'qualification.json');train._check(v)
    expected=_qualification_evidence(v['fit_path'],v['native_path'],v['replay_path'],context)
    if (set(v)!=set(expected)|{'sha256','finished_ns'} or any(v.get(k)!=x for k,x in expected.items()) or
        type(v['finished_ns']) is not int or v['finished_ns']<=max(context.passes[_safe(v[k])]['finished_ns'] for k in ('native_path','replay_path'))):
        raise ValueError('qualification differs from raw actual CUDA fit/replay/runtime evidence')
    context.guard.check();return v


def _population(identity):
    from .data import LegacyStore
    store=LegacyStore(identity['locations']['data']);parts=train._json(identity['locations']['partitions'])
    users={role:set() for role in ('train','val','test')};splits={s:set() for s in ('user','gesture','both')}
    for w in parts['windows']:
        user=str(store.episodes[w['episode']]['subject']);users[w['role']].add(user)
        if w['role']=='test':splits[w['split']].add(user)
    return dict(episode_records=len(store.episodes),unique_subjects_by_role={r:len(v) for r,v in users.items()},
        cross_role_user_intersection_counts={a+'_'+b:len(users[a]&users[b]) for a,b in (('train','val'),('train','test'),('val','test'))},
        train_test_user_intersection_by_split={s:len(users['train']&v) for s,v in splits.items()},
        counts_describe='released metadata subject keys, not independently verified natural-person identities')


def _curve_data(fit,result):
    rows=train._read_epochs(_safe(fit))
    return dict(format='native_neuropose_curve_data_v1',scope=result['scope'],fit_result_sha256=result['sha256'],
        journal_sha256=result['journal_sha256'],selected_epoch_zero_based=result['selected_best']['epoch'],
        selected_best=result['selected_best'],training_unit='native sample-weighted composite loss',
        validation_unit='native sample-weighted composite loss (val_loss); checkpoint selection',epochs=rows)


def _plot_runtime():
    import matplotlib
    from matplotlib import font_manager,ft2font
    font=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans'))
    return dict(matplotlib_version=matplotlib.__version__,freetype_version=ft2font.__freetype_version__,
        font_sha256=train._sha(font),backend='Agg',svg_hashsalt='egoemg-native-neuropose-v1')


def _curve_images(curves):
    import io
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'svg.hashsalt':'egoemg-native-neuropose-v1','font.family':'DejaVu Sans'}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        try:
            rows=curves['epochs'];x=[r['epoch']+1 for r in rows]
            for ax,values,title in ((axes[0],[r['train_loss'] for r in rows],'Training'),
                (axes[1],[r['validation']['val_loss'] for r in rows],'Held-out validation')):
                ax.plot(x,values);ax.set(xlabel='Completed epoch',ylabel='Native composite loss',title=title);ax.grid(alpha=.2)
            best=curves['selected_best'];axes[1].scatter([best['epoch']+1],[best['score']],color='#b45309',label='Selected best',zorder=3);axes[1].legend()
            fig.suptitle('NeuroPose — '+('full original run' if curves['scope']=='full' else 'ENGINEERING ONLY; no scientific result'))
            images={}
            for ext,metadata in (('png',{'Software':'EgoEMG-Reproduction'}),('svg',{'Date':None,'Creator':'EgoEMG-Reproduction'})):
                stream=io.BytesIO();fig.savefig(stream,format=ext,dpi=150,metadata=metadata);images['training-curves.'+ext]=stream.getvalue()
            return images
        finally:plt.close(fig)


def _build_report(replay,*,require_full):
    from .classic_emg_report import PUBLIC_NATIVE_METRICS
    context=_Evidence();right=_read_pass(replay,context)
    if right['role']!='test' or right['kind']!='replay':raise ValueError('actual complete native test and new-process replay required')
    result,identity=context.fit(right['fit_path']);_require_full(result,require_full)
    left=_read_pass(right['reference_path'],context);stats=copy.deepcopy(right['statistics'])
    aliases={user:'user_'+str(i+1).zfill(3) for i,user in enumerate(sorted(stats['center']['by_user']))}
    for mode in ('trajectory','center'):stats[mode]['by_user']={aliases[k]:v for k,v in stats[mode]['by_user'].items()}
    v=train._sealed(dict(format='native_neuropose_completed_report_v1',variant='neuropose',scope=result['scope'],
        scientific_eligible=result['scope']=='full',full_paper_complete=False,training_sources=identity['sources'],evaluation_sources=_sources(),
        fit_result_sha256=result['sha256'],native_test_sha256=left['sha256'],test_replay_sha256=right['sha256'],
        validation_profile_sha256=right['profile_sha256'],counts=identity['counts'],underlying_counts=identity['underlying_counts'],
        population=_population(identity),selected_best=result['selected_best'],completed_epochs=result['completed_epochs'],
        optimizer_executions=result['optimizer_executions'],stop_reason=result['stop_reason'],policy=identity['policy'],
        native_metrics={k:v for k,v in left['native_metrics'].items() if k in PUBLIC_NATIVE_METRICS},
        native_metrics_units='original native composite loss; MAE radians; landmarks meters; native derivatives separate',
        agreement_policy=POLICY,statistics=stats,curve_data=_curve_data(right['fit_path'],result),plot_runtime=_plot_runtime(),
        limitations=['Legacy release, not demonstrated equivalent to the PDF or unavailable merged data.',
            'One CUDA rank/global batch256 differs from the declared six-rank BatchNorm statistics.',
            'No checkpoint continuation or stochastic training-trajectory equivalence claim.',
            'CPU/CUDA engineering has no scientific score; the complete paper remains unfinished.']))
    context.guard.check();return v,context


def _render(report):
    scope='完整训练实验' if report['scientific_eligible'] else '工程验证，不能作为科学复现成绩'
    lines=['# NeuroPose 原生连续训练报告','',scope,'',
        f"实际完成 {report['completed_epochs']} 轮；最佳轮次 {report['selected_best']['epoch']+1}，以原生 val_loss 选择。",'',
        '| 统计范围 | 全轨迹 MAE (°) | 中心 MAE (°) |','|---|---:|---:|']
    for label,key in (('全部','overall'),):
        values=[report['statistics'][mode][key]['mae_deg'] for mode in ('trajectory','center')]
        lines.append('| '+label+' | '+' | '.join('无有效角' if v is None else f'{v:.8f}' for v in values)+' |')
    lines+=['','原始22角、7790点轨迹，中心3895；由生成mask决定有效角，按有效角总数加权。',
        'native指标保持官方批聚合与单位，曲线为复合损失，不能把val_loss当角度MAE。','',
        '旧版数据、单rank/全局B256与原六rank的差异已披露；本报告不表示整篇论文已复现。','']
    return '\n'.join(lines)


PUBLIC_FILES=('metrics.json','metrics.md','curve-data.json','training-curves.png','training-curves.svg')


def _completion(output,report):
    return train._sealed(dict(format='native_neuropose_report_completion_v1',report_sha256=report['sha256'],
        files_sha256={n:train._sha(_safe(output/n)) for n in PUBLIC_FILES},scientific_eligible=report['scientific_eligible'],full_paper_complete=False))


def publish_neuropose_report(fit,native,replay,output,*,require_full=True):
    output=_safe(output,exists=False)
    if output.exists():raise FileExistsError('native public report output exists')
    report,context=_build_report(replay,require_full=require_full);right=context.passes[_safe(replay)]
    if _safe(fit)!=_safe(right['fit_path']) or _safe(native)!=_safe(right['reference_path']):
        raise ValueError('report requires exact fit/native-test/replay paths')
    from . import classic_emg_train as storage
    storage._space(output.parent);images=_curve_images(report['curve_data']);context.guard.check()
    output.mkdir(mode=0o700);train._write(output/'metrics.json',report);train._write(output/'curve-data.json',report['curve_data'])
    for name,raw in dict(images,**{'metrics.md':_render(report).encode()}).items():
        with (output/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    context.guard.check();train._write(output/'completion.json',_completion(output,report));context.guard.check();return report


def verify_neuropose_report(replay,output,*,require_full=True):
    output=_safe(output);report,context=_build_report(replay,require_full=require_full)
    for n in (*PUBLIC_FILES,'completion.json'):context.guard.add(output/n)
    if (train._json(output/'metrics.json')!=report or train._json(output/'curve-data.json')!=report['curve_data'] or
        (output/'metrics.md').read_text()!=_render(report) or
        any((output/n).read_bytes()!=raw for n,raw in _curve_images(report['curve_data']).items()) or
        train._json(output/'completion.json')!=_completion(output,report)):
        raise ValueError('native report/actual curves/last completion differs from reopened raw evidence')
    context.guard.check();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('pass')
    for n in ('fit','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--role',choices=('val','test'),required=True);p.add_argument('--kind',choices=('native','replay'),required=True)
    for n in ('reference','profile'):p.add_argument('--'+n,type=Path)
    p.add_argument('--engineering',action='store_true')
    args=vars(parser.parse_args());args.pop('action');args['require_full']=not args.pop('engineering')
    import json
    print(json.dumps(run_neuropose_pass(**args),sort_keys=True))


if __name__=='__main__':main()
