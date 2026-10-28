"""Wait for the existing original source, then run exactly three full models.

Private config contains source/controller paths and separately frozen model
qualifications. There is no downloader, engineering mode or automatic retry.
"""
from pathlib import Path,PurePosixPath
import argparse
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from egoemg_repro import emg2pose_direct_experiment as driver
from egoemg_repro.emg2pose_direct_source import open_source
from egoemg_repro.emg2pose_memory_process import run_owned_stage

SIZES=('small','middle','large')
FORMAT='egoemg_full_emg2pose_queue_v1'
SCRIPT_SHA=driver.storage._sha256(__file__)

def read(p):return driver.storage._json(driver.storage._safe_path(p,file=True))
def sha(p):return driver.storage._sha256(p)
def alive(p):
    if type(p) is not dict or type(p.get('pid')) is not int or type(p.get('start_ticks')) is not str:
        raise ValueError('original source process identity required')
    try:
        f=Path('/proc',str(p['pid']),'stat').read_text().rsplit(')',1)[1].split()
        return f[19]==p['start_ticks'] and f[0]!='Z'
    except FileNotFoundError:return False

def source_handoff(config):
    snapshot=config['source_stage'];root=driver.storage._safe_path(snapshot['root'])
    for name,h in snapshot['files_sha256'].items():
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:raise ValueError('unsafe source producer path')
        driver._expected(root/name,h)
    state=read(root/'private/execution-state.json')
    if any(state[k]!=config['source_launch']['state'][k] for k in ('wrapper','child')):
        raise ValueError('original source owner identity changed')
    current={k:alive(state[k]) for k in ('wrapper','child')}
    if state['status']=='failed':raise RuntimeError('original source controller failed; no restart')
    if state['status']=='running':
        if not any(current.values()):raise RuntimeError('original source controller vanished without a terminal result')
        return None
    if state['status']!='complete' or state.get('actual_exit_code')!=0:raise ValueError('successful original source exit required')
    if any(current.values()):return None
    value=read(root/'private/full-source-result.json');pointer=read(root/'private/source-output.json')
    if value.get('cohort')!='full' or value.get('full_corpus_complete') is not True:raise ValueError('actual full source required')
    source=driver.storage._safe_path(value['source_path'])
    if str(source)!=pointer['source']:raise ValueError('completed source handoff path changed')
    frozen=value['source_pipeline_completion_sha256'];driver._expected(source/'completion.json',frozen)
    completion=read(source/'completion.json')
    if completion.get('cohort')!='full' or completion.get('full_corpus_complete') is not True or completion.get('pipeline_complete') is not True:
        raise ValueError('engineering or partial pipeline cannot supply full experiments')
    # The producer returns raw numeric identity. Archive provenance is added
    # only by open_source after deep admission; main applies the unchanged
    # verified-source/archive gate there before any GPU or model work.
    native=value['native_data_identity']
    if (native.get('format')!='emg2pose_runtime_data_identity_v1' or native.get('cohort')!='full'
            or native.get('full_corpus_complete') is not True or native.get('engineering_only') is not False
            or native.get('file_count')!=25253
            or native.get('counts',{}).get('split')!={'train':17136,'val':1950,'test':6167}):
        raise ValueError('complete native25253original sessions and roles required at source handoff')
    expected={n:h for n,h in snapshot['files_sha256'].items() if n.startswith('egoemg_repro/')}
    if value.get('producer_files_sha256')!=expected:raise ValueError('full source producer binding differs')
    owned=value['actual_owned_stage'];actual=read(root/'private/owned-stage/stage-result.json')
    if (owned!=actual or owned.get('worker_exit_code')!=0 or owned.get('status')!='success'
            or owned.get('all_owned_consumers_exited') is not True or owned.get('descriptor_bindings')!={}
            or any(alive(owned[k]) for k in ('worker','guardian'))):
        raise ValueError('actual full source consumers have not completed')
    return dict(source_root=str(source),metadata_csv=str(root/'private/emg2pose_metadata.csv'),completion_sha256=frozen,
        source_result_sha256=sha(root/'private/full-source-result.json'),source_owned_sha256=sha(root/'private/owned-stage/stage-result.json'))

def read_config(path,expected):
    driver._expected(path,expected);q=read(path)
    if (q.get('format')!=FORMAT or q.get('models')!=list(SIZES)
            or set(q.get('training_qualifications',{}))!=set(SIZES)
            or set(q.get('evaluation_qualifications',{}))!=set(SIZES)
            or type(q.get('gpu_uuid')) is not str or not q['gpu_uuid'].startswith('GPU-')):
        raise ValueError('fixed full S/M/L queue configuration required')
    for group in ('training_qualifications','evaluation_qualifications'):
        for v in q[group].values():
            if set(v)!={'directory','sha256'} or not driver._HASH.fullmatch(v['sha256']):raise ValueError('frozen per-model qualification required')
    v=q['controller_qualification']
    if set(v)!={'directory','sha256'} or not driver._HASH.fullmatch(v['sha256']):raise ValueError('frozen shared controller qualification required')
    if set(q.get('source_stage',{}))!={'root','files_sha256'} or not q['source_stage']['files_sha256']:
        raise ValueError('frozen original source stage required')
    if not {'wrapper','child'}<=set(q.get('source_launch',{}).get('state',{})):raise ValueError('original source owner launch required')
    return q

def model_command(config,handoff,size,output):
    if size not in SIZES:raise ValueError('one of three original full models required')
    a=config['training_qualifications'][size];b=config['evaluation_qualifications'][size];c=config['controller_qualification']
    return [sys.executable,'-m','egoemg_repro.emg2pose_direct_experiment','run','--upstream',config['upstream'],
        '--source-root',handoff['source_root'],'--metadata-csv',handoff['metadata_csv'],
        '--expected-source-completion-sha256',handoff['completion_sha256'],'--size',size,'--scope','full',
        '--gpu-uuid',config['gpu_uuid'],'--output',str(output),
        '--qualification',a['directory'],'--expected-qualification-sha256',a['sha256'],
        '--evaluation-qualification',b['directory'],'--expected-evaluation-qualification-sha256',b['sha256'],
        '--controller-qualification',c['directory'],'--expected-controller-qualification-sha256',c['sha256']]

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--expected-config-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    if 'torch' in sys.modules or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('queue parent must be NoTorch with CUDA hidden')
    q=read_config(a.config,a.expected_config_sha256);output=driver.storage._safe_path(a.output)
    if output.exists():raise FileExistsError('full queue output already exists')
    protected=[q['upstream'],q['source_stage']['root'],a.config.parent,
        *[v['directory'] for g in ('training_qualifications','evaluation_qualifications') for v in q[g].values()],q['controller_qualification']['directory']]
    for path in protected:
        path=driver.storage._safe_path(path)
        if output==path or output in path.parents or path in output.parents:raise ValueError('queue output overlaps immutable inputs')
    driver.storage._space(output.parent);output.mkdir(mode=0o700);start=time.monotonic();records=[]
    def guard():
        driver._expected(a.config,a.expected_config_sha256)
        if sha(__file__)!=SCRIPT_SHA:raise ValueError('operational queue code changed')
        driver._sources();driver.storage._space(output)
        if driver.memory.available_memory_bytes()<driver.memory.RESERVE_BYTES:raise RuntimeError('observed256GiB host reserve lost')
    def status(phase,**more):
        driver._write(output/'queue-status.json',dict(status=phase,completed_models=len(records),full_paper_complete=False,**more),replace=True)
        print(json.dumps(dict(phase=phase,completed_models=len(records),**more)),flush=True)
    try:
        last=0
        while True:
            guard();handoff=source_handoff(q)
            if handoff is not None:break
            if time.monotonic()-last>60:status('waiting-existing-full-source');last=time.monotonic()
            if time.monotonic()-start>7*24*3600:raise RuntimeError('bounded original source wait expired; no restart')
            time.sleep(10)
        with open_source(handoff['source_root'],expected_completion_sha256=handoff['completion_sha256'],upstream=q['upstream'],metadata_csv=handoff['metadata_csv']) as source:
            driver._full_source(source.data_identity)
            def current():
                guard();source.assert_current()
                if driver._gpu(q['gpu_uuid'])['free_bytes']<8*1024**3:raise RuntimeError('observed8GiB GPU reserve lost')
            driver._write(output/'source-handoff.json',handoff)
            for size in SIZES:
                while driver._gpu(q['gpu_uuid'])['free_bytes']<64*1024**3:
                    current();status('waiting-selected-gpu',size=size);time.sleep(30)
                current();status('full-experiment-running',size=size)
                model=output/size
                owned=run_owned_stage(model_command(q,handoff,size,model),output/(size+'-controller-owned'),descriptors={},guard=current,
                    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=''),poll_seconds=2.,terminate_grace=10.)
                if owned['worker_exit_code']!=0 or not owned['all_owned_consumers_exited']:raise RuntimeError('full controller failed; no retry')
                frozen=sha(model/'driver-completion.json');status('independent-full-verification',size=size)
                verified=run_owned_stage([sys.executable,'-m','egoemg_repro.emg2pose_direct_experiment','verify','--output',str(model),
                    '--expected-completion-sha256',frozen],output/(size+'-verify-owned'),descriptors={},guard=current,
                    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=''),poll_seconds=2.,terminate_grace=10.)
                if verified['worker_exit_code']!=0 or not verified['all_owned_consumers_exited']:raise RuntimeError('full experiment verification failed')
                records.append(dict(size=size,driver_completion_sha256=frozen,report_completion_sha256=sha(model/'report/completion.json'),
                    controller_owned_sha256=sha(output/(size+'-controller-owned/stage-result.json')),
                    verification_owned_sha256=sha(output/(size+'-verify-owned/stage-result.json'))))
                driver._write(output/(size+'-verified.json'),records[-1]);status('one-full-experiment-verified',size=size)
            current()
        if 'torch' in sys.modules:raise ValueError('queue parent imported Torch')
        driver._write(output/'queue-completion.json',dict(format=FORMAT,config_sha256=a.expected_config_sha256,source=handoff,
            models=records,all_three_full_experiments_independently_verified=True,parent_torch_unimported=True,full_paper_complete=False))
        status('complete')
    except BaseException as error:
        (output/'queue-completion.json').unlink(missing_ok=True)
        status('failed',error_type=type(error).__name__,error=str(error));raise

if __name__=='__main__':main()
