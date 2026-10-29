"""Explicit recovered WiLoR evaluation using the unchanged native operators.

The isolated function namespace replaces only artifact/evidence boundaries. Its
runtime view is explicitly unsealed: the original sealed identity stays intact,
while evaluation must use the actual continuation device. No native module is
monkeypatched and no historical artifact is rewritten.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import FunctionType

from . import wilor_native_eval as kernel, wilor_recovery as recovery
from .classic_emg import _file_hash
from .classic_emg_cohort import _StatGuard

ROOT=Path(__file__).resolve().parents[1]
FORMAT='explicit_recovered_wilor_selected_pass_v1'
PROFILE='explicit_recovered_wilor_pretest_profile_v1'
_FILES=('egoemg_repro/wilor_recovery.py','egoemg_repro/wilor_recovery_eval.py')
_IMPORTED={n:_file_hash(ROOT/n) for n in _FILES}


def _sources():
    actual={n:_file_hash(ROOT/n) for n in _FILES}
    if actual!=_IMPORTED:raise ValueError('imported recovered WiLoR evaluation source changed')
    return dict(kernel._sources(),**actual)


def _capture_sources():
    guard,_=kernel._capture_sources()
    guard=kernel.train._guard_merge(guard,_StatGuard([ROOT/n for n in _FILES]))
    value=_sources();guard.check();return guard,value


class _Evidence:
    """One independently reopened recovery/source/input guarded transaction."""
    def __init__(self):
        self.guard,self.sources=_capture_sources()
        self.fits={};self.passes={};self.profiles={};self.active=set();self.recoveries={}

    def fit(self,path):
        path=kernel._path(path)
        if path not in self.fits:
            import torch
            from .vision_literal_runtime import vision_state_sha256
            result,identity,last,cohort,assets,guard=recovery._read_recovery(path,require_full=False)
            self.guard=kernel.train._guard_merge(self.guard,guard);del last
            self.guard.add(path/'recovery-identity.json');continuation=kernel._json(path/'recovery-identity.json')
            if (continuation['sha256']!=result['continuation_identity_sha256'] or
                    identity['sha256']!=result['identity_sha256']):
                raise ValueError('reopened recovery lineage differs from selected result')
            self.guard.check();selected=path/'checkpoints'/result['selected_best']['file'];self.guard.add(selected)
            checkpoint=torch.load(str(selected),map_location='cpu',weights_only=False,mmap=True)
            if vision_state_sha256(checkpoint['state_dict'])!=result['selected_best']['model_sha256']:
                raise ValueError('actual selected recovered WiLoR model differs')
            # This is a runtime view, not a resealed or substituted fit identity.
            view=copy.deepcopy(identity);view.pop('sha256')
            view.update(format='explicit_wilor_recovery_evaluation_runtime_view_v1',
                historical_identity_sha256=identity['sha256'],runtime_device=copy.deepcopy(continuation['actual_runtime_device']))
            self.recoveries[path]=dict(continuation_identity_sha256=continuation['sha256'],historical_identity_sha256=identity['sha256'],
                parent_checkpoint_sha256=continuation['parent_admission']['parent_checkpoint_sha256'],
                prefix_epochs=result['prefix_epochs'],prefix_optimizer_executions=result['prefix_optimizer_executions'],
                segment_optimizer_executions=result['segment_optimizer_executions'],uninterrupted_execution=False,
                historical_runtime_device=copy.deepcopy(identity['runtime_device']),actual_runtime_device=copy.deepcopy(continuation['actual_runtime_device']),
                device_admission=copy.deepcopy(continuation['device_admission']),new_process=copy.deepcopy(continuation['new_process']))
            self.guard.check();self.fits[path]=(result,view,cohort,assets,checkpoint)
        return self.fits[path]


def _curve_data(fit,result):
    curves=kernel._curve_data(fit,result);n=result['prefix_epochs'];prefix=result['prefix_optimizer_executions'];new=result['segment_optimizer_executions']
    if (type(n)is not int or not 0<n<len(curves['epochs']) or type(prefix)is not int or prefix<=0 or
            curves['epochs'][n-1]['optimizer_executions']!=prefix or type(new)is not int or new<=0 or
            prefix+new!=result['optimizer_executions']):
        raise ValueError('recovered WiLoR curve interruption/update partition differs')
    curves.update(format='explicit_recovered_wilor_curve_data_v1',continuation_identity_sha256=result['continuation_identity_sha256'],
        resume_boundary=dict(after_complete_epoch=n,parent_optimizer_executions=prefix,new_optimizer_executions=new))
    return curves


def _curve_images(curves):
    import io
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'svg.hashsalt':'egoemg-recovered-wilor-v1','font.family':'DejaVu Sans'}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        try:
            rows=curves['epochs'];x=[r['epoch']+1 for r in rows]
            axes[0].plot(x,[r['train_loss']for r in rows]);axes[0].set(title='Training',ylabel='Native composite loss')
            axes[1].plot(x,[r['validation']['val_mae']for r in rows]);axes[1].set(title='Held-out validation',ylabel='Native validation MAE (radians)')
            best=curves['selected_best'];axes[1].scatter([best['epoch']+1],[best['score']],color='#b45309',label='Selected best',zorder=3)
            for ax in axes:
                ax.axvline(curves['resume_boundary']['after_complete_epoch']+.5,color='#64748b',linestyle='--',label='Resume boundary')
                ax.set_xlabel('Completed epoch');ax.grid(alpha=.2);ax.legend()
            fig.suptitle('WiLoR — '+('full continuation'if curves['scope']=='full'else'ENGINEERING ONLY; no scientific result'))
            images={}
            for ext,metadata in(('png',{'Software':'EgoEMG-Reproduction'}),('svg',{'Date':None,'Creator':'EgoEMG-Reproduction'})):
                stream=io.BytesIO();fig.savefig(stream,format=ext,dpi=150,metadata=metadata);images['training-curves.'+ext]=stream.getvalue()
            return images
        finally:plt.close(fig)


def _plot_runtime():
    return dict(kernel._plot_runtime(),svg_hashsalt='egoemg-recovered-wilor-v1')


def _completion(output,report):
    value=kernel._completion(output,report);value.pop('sha256')
    value['format']='explicit_recovered_wilor_report_completion_v1';return kernel._sealed(value)


_namespace=dict(kernel.__dict__,__name__=__name__,__file__=__file__,FORMAT=FORMAT,PROFILE=PROFILE,
    _Evidence=_Evidence,_sources=_sources,_capture_sources=_capture_sources,_curve_data=_curve_data,_curve_images=_curve_images,
    _plot_runtime=_plot_runtime)


def _bind(function):
    value=FunctionType(function.__code__,_namespace,name=function.__name__,argdefs=function.__defaults__,closure=function.__closure__)
    value.__kwdefaults__=function.__kwdefaults__;value.__annotations__=dict(function.__annotations__);return value


for _name in('_read_pass','_read_profile','read_wilor_pass','freeze_wilor_validation','run_wilor_pass',
        '_build_report','_render','_completion','publish_wilor_report','verify_wilor_report'):
    _namespace[_name]=_bind(getattr(kernel,_name))
_build_base_report=_namespace['_build_report'];_render_base=_namespace['_render']


def _build_report(replay,*,require_full):
    report,context=_build_base_report(replay,require_full=require_full)
    path=kernel._path(context.passes[kernel._path(replay)]['fit_path'])
    report.pop('sha256');report.update(format='explicit_recovered_wilor_completed_report_v1',recovery=copy.deepcopy(context.recoveries[path]))
    report['limitations'].append('Training was interrupted and explicitly continued; historical identity, original prefix and actual device migration are recorded separately.')
    context.guard.check();return kernel._sealed(report),context


def _render(report):
    r=report['recovery']
    return _render_base(report)+('\n显式中断续训：原前缀'+str(r['prefix_epochs'])+'轮、'+str(r['prefix_optimizer_executions'])+
        '次更新；新进程执行'+str(r['segment_optimizer_executions'])+'次更新。曲线标明恢复边界，历史身份与实际设备分别记录。\n')


_namespace.update(_build_report=_build_report,_render=_render,_completion=_completion)
_sealed=kernel._sealed
run_wilor_recovery_pass=_namespace['run_wilor_pass']
read_wilor_recovery_pass=_namespace['read_wilor_pass']
freeze_wilor_recovery_validation=_namespace['freeze_wilor_validation']
publish_wilor_recovery_report=_namespace['publish_wilor_report']
verify_wilor_recovery_report=_namespace['verify_wilor_report']
