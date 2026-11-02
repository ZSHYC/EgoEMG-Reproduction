
from pathlib import Path
import datetime,hashlib,json,os,re,shutil,sys,time
q=json.load(sys.stdin);R=Path(q['root']);started=time.monotonic();os.nice(10)
assert shutil.disk_usage('/data/shihao').free>225*1024**3
assert next(int(x.split()[1])for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))*1024>128*1024**3
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def closed(h):
 p=Path('/proc',str(h['pid']),'stat')
 if not p.exists():return True
 f=p.read_text().rsplit(')',1)[1].split();return f[19]!=str(h['start_ticks'])or f[0]in('Z','X')
def stamp(p):
 p=Path(p);assert p.is_file()and not p.is_symlink();s=p.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
state=json.loads((R/'private/execution-state.json').read_text());assert state['status']=='complete'and state['actual_exit_code']==0 and state['wrapper']==q['actual_controller']and closed(state['wrapper'])
assert len(state['passes'])==2
for stage in state['passes']:assert stage['status']=='complete'and stage['actual_exit_code']==0 and closed(stage['worker'])
assert all(sha(R/n)==h for n,h in q['all_frozen_members_sha256'].items())
config=json.loads((R/'config.json').read_text());assert config['source_commit']==q['source_commit']
completion=json.loads((R/'private/completed.json').read_text());assert completion['complete_reference_input_validation']and not completion['model_evaluation_complete']
actual_files={};passes=[];raw=[]
for stage in state['passes']:
 P=R/'private'/stage['label'];result=json.loads((P/'result.json').read_text());records=json.loads((P/'input-digests.json').read_text());assert result==completion['passes'][len(passes)]
 assert result['actor']['pid']==stage['worker']['pid']and result['actor']['start_ticks']==stage['worker']['start_ticks']
 assert sha(P/'input-digests.json')==result['input_digests_file_sha256']
 assert len(records)==4282 and [r['index']for r in records]==list(range(4282))
 assert all(set(r)=={'index','emg_sha256','target_mask_sha256','identity_sha256'}and all(re.fullmatch('[0-9a-f]{64}',r[k])for k in r if k!='index')for r in records)
 counts=result['counts'];assert counts==dict(windows=2141,hand_requests=4282,by_split=dict(gesture=2314,user=1646,both=322),valid_label_requests=4154,invalid_label_requests=128,valid_angles=91388,visible_requests=4112,missing_image_requests=170)
 assert result['same_actual_visual_target_and_mask_bytes_for_all4282']and result['all_inputs_finite']and result['all_source_bytes_and_guarded_generations_unchanged']
 assert result['input_contract_equal_across_nativeSML']and result['construction_attempts']==dict(learned_module=0,optimizer=0)and not result['cuda_initialized']and result['new_neural_forwards']==result['new_optimizer_updates']==0
 passes.append(result);raw.append(records)
 for n in('result.json','input-digests.json'):actual_files[str((P/n).relative_to(R))]=dict(sha256=sha(P/n),bytes=(P/n).stat().st_size)
assert raw[0]==raw[1]
for k in('grid_sha256','source_files_sha256','input_stream_sha256','target_mask_stream_sha256','private_identity_stream_sha256','input_digests_file_sha256'):assert passes[0][k]==passes[1][k]
assert sha(config['original_fit_identity_path'])==config['original_fit_identity_sha256']
D=Path(config['data']);manifest=json.loads((D/'manifest.json').read_text());sources={};generations={}
for name,h in passes[0]['source_files_sha256'].items():
 if name=='metadata:manifest':p=D/'manifest.json'
 elif name=='metadata:episodes':p=D/'metadata.npz'
 else:
  key=name.removeprefix('field:');p=D/manifest['fields'][key].get('filename',key+'.dat')
 before=stamp(p);assert sha(p)==h and stamp(p)==before
 generations[str(p)]=before;sources[name]=dict(sha256=h,bytes=p.stat().st_size)
assert len(sources)==17
assert all(stamp(p)==v for p,v in generations.items())
P=Path(config['unit_package']);unit=json.loads((P/'stage-receipt.json').read_text());assert all(sha(P/n)==h for n,h in unit['files_sha256'].items())
assert all(sha(Path(config['upstream'])/n)==h for n,h in config['official_sources_sha256'].items())
assert sha(config['vision_payload'])==config['vision_payload_sha256']
for n in('private/execution-state.json','private/completed.json','stage-receipt.json','config.json','worker.py','controller.py'):actual_files[n]=dict(sha256=sha(R/n),bytes=(R/n).stat().st_size)
assert 'torch'not in sys.modules and'numpy'not in sys.modules and'pytorch_lightning'not in sys.modules
f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
print(json.dumps(dict(format='actual_independent_closed_two_native_center_reference_input_passes_v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reader=dict(pid=os.getpid(),start_ticks=f[19]),actual_seconds=time.monotonic()-started,source_commit=config['source_commit'],actual_state=state,all_three_input_stage_actors_physically_closed=True,both_full_4282_typed_input_digest_lists_exact=True,all_17_input_source_files_independently_rehashed_after_exit=sources,total_input_source_bytes=sum(v['bytes']for v in sources.values()),unit_package_all_frozen_members_unchanged=True,official_all_registered_sources_unchanged=True,actual_artifact_files=actual_files,complete_reference_input_validation=True,full_model_evaluation_complete=False,passes=passes,reader_torch_numpy_lightning_unimported=True,new_neural_forwards=0,new_optimizer_updates=0,scientific_eligible=False,full_paper_complete=False),sort_keys=True))
