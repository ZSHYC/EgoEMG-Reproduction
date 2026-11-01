"""Staged EMG2Pose-S native continuation; safe when DataLoader imports its main.

Stage this file as private/native-resume.py beside the accepted resume-request.
Only the guarded parent may admit data/GPU or resume the unchanged frozen fit.
"""

def main():
    from pathlib import Path
    import datetime,hashlib,json,os,subprocess,sys,time
    R=Path(__file__).resolve().parents[1];q=json.loads((R/'private/resume-request.json').read_text())
    a=q['admission'];S=Path(a['frozen_source_root']);F=Path(a['fit']);sys.path.insert(0,str(S))
    def sha(p):
     h=hashlib.sha256()
     with Path(p).open('rb') as f:
      for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
     return h.hexdigest()
    def write(phase,**values):
     v=dict(phase=phase,kind=a['kind'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      parent_complete_epochs=a['complete_epochs'],parent_optimizer_executions=a['optimizer_executions'],
      full_training_complete=False,full_paper_complete=False,**values)
     p=R/'private/native-progress.json';tmp=p.with_suffix('.writing')
     with tmp.open('w') as f:json.dump(v,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
     tmp.replace(p);print(json.dumps(v),flush=True)
    def original_parent():
     assert sha(F/'checkpoints/last.ckpt')==a['checkpoint_sha256']
     assert sha(F/'identity.json')==a['identity_file_sha256']
     assert sha(F/'epochs.jsonl')==a['journal_sha256']
     assert sha(S/'private/execution-state.json')==a['original_failed_state_sha256']
    def gpu_gate():
     write('waiting-for-original-64GiB-start-reserve',cuda_initialized=False)
     while True:
      row=subprocess.check_output(['nvidia-smi','-i',a['gpu_uuid'],'--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
      assert row[0].strip()==a['gpu_uuid'] and row[1].strip()=='NVIDIA A800-SXM4-80GB'
      free=int(row[2].strip())*1024**2
      if free>64*1024**3:break
      time.sleep(5)
     original_parent()
     v=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_uuid=a['gpu_uuid'],actual_free_bytes=free,
      required_start_free_bytes=64*1024**3,external_consumers_preserved=True)
     with (R/'private/gpu-admission.json').open('x') as f:json.dump(v,f,sort_keys=True,indent=2)
     write('native-resume-starting',gpu_admission=v,actual_six_state_restore_yet_serialized=False)
    original_parent()
    for n,d in a['original_frozen_source_sha256'].items():assert sha(S/n)==d,n
    import torch
    assert not torch.cuda.is_initialized()
    write('full-source-admission',cuda_initialized=False,actual_full_gpu_fit_started=False)
    assert a['kind']=='small','this entry only continues the original small experiment'
    from egoemg_repro.emg2pose_direct_source import open_source
    from egoemg_repro.emg2pose_direct_train import fit_source,read_fit
    n=q['native_request']
    with open_source(n['source_root'],expected_completion_sha256=n['expected_source_completion_sha256'],
      upstream=n['upstream'],metadata_csv=n['metadata_csv']) as source:
     assert source.data_identity['full_corpus_complete'] and source.data_identity['file_count']==25253
     assert source.data_identity['counts']['split']==dict(train=17136,val=1950,test=6167)
     assert not torch.cuda.is_initialized();gpu_gate();source.assert_current()
     result=fit_source(n['upstream'],source,n['metadata_csv'],'small',F,scope='full',resume=True,
      expected_last_sha256=a['checkpoint_sha256'],qualification=n['qualification'],
      expected_qualification_sha256=n['expected_qualification_sha256'])
    assert result['completed_epochs']==250 and result['optimizer_executions']==250*1584
    matches=result['resume_matches'];assert set(matches)=={'model','optimizer','scheduler','callbacks','loops','rng'} and all(matches.values())
    reopened=read_fit(F,require_full=True);assert reopened==result

    for n,d in a['original_frozen_source_sha256'].items():assert sha(S/n)==d,n
    assert sha(S/'private/execution-state.json')==a['original_failed_state_sha256']
    v=dict(format='actual_native_resumed_full_fit_completion_v1',kind=a['kind'],parent_complete_epochs=a['complete_epochs'],
     parent_optimizer_executions=a['optimizer_executions'],fit_result_sha256=sha(F/'fit-result.json'),result=result,
     six_state_restoration_verified=True,native_policy_unchanged=True,evaluation_and_scientific_report_pending=True,
     bitwise_uninterrupted_trajectory_claimed=False,full_paper_complete=False)
    with (R/'private/resumed-fit-completion.json').open('x') as f:json.dump(v,f,sort_keys=True,indent=2)
    write('native-full-fit-complete-evaluation-pending',actual_final_complete_epochs=result['completed_epochs'])


if __name__=='__main__':
    main()
