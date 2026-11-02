from pathlib import Path
import datetime,hashlib,json,os,pickle,random,subprocess,sys
P=Path('/data/shihao/egoemg-rn18-fusion-reviewed-original-full-cli-e53ecbef4581-20261102');U=Path('/home/shihao/projects/EgoEMG-Reproduction/upstream');T=Path('/data/shihao/egoemg-native-trajectory-shared-lineage-queue-14a9a1e4486d-20261102')
sys.path.insert(0,str(P));sys.path.extend(['/home/shihao/projects/EgoEMG-Classic-Canonical-20261026/.deps','/home/shihao/projects/EgoEMG-WiLoR-20261026/.deps'])
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
torch.set_num_threads(2)
counters=dict(neural_module=0,optimizer=0)
def deny_module(*a,**kw):counters['neural_module']+=1;raise RuntimeError('source audit cannot construct neural modules')
def deny_optimizer(*a,**kw):counters['optimizer']+=1;raise RuntimeError('source audit cannot construct optimizers')
torch.nn.Module.__init__=deny_module;torch.optim.Optimizer.__init__=deny_optimizer
from hydra import compose,initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from egoemg_repro.emgformer_literal import compose_emgformer,emgformer_protocol
from egoemg_repro.classic_emg import _activate
_activate(U)
from egoemg.transforms import Compose

def sha(p):
 with Path(p).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def h(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def nine(p):
 s=p.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]
def sealed(v):return h({k:x for k,x in v.items()if k!='sha256'})
def clean():
 assert subprocess.check_output(['git','-C',str(U),'rev-parse','HEAD'],text=True).strip()=='f7cbd048b7092139852270491d3859cd472e0311'
 assert not subprocess.check_output(['git','-C',str(U),'status','--porcelain','--untracked-files=all'],text=True).strip()
clean();request=json.loads((T/'private/request.json').read_text());inputs={};stamps={};bindings={};recipes={};catalog=None
for size in('small','middle','large'):
 cfg=compose_emgformer(U,size);protocol=emgformer_protocol(U,cfg);raw=OmegaConf.to_container(cfg,resolve=True);fit=Path(request['fits'][size]['path'])
 for name in('identity.json','fit-result.json'):
  path=fit/name;stamps[str(path)]=nine(path);inputs[str(path)]=sha(path)
 identity=json.loads((fit/'identity.json').read_text());result=json.loads((fit/'fit-result.json').read_text())
 assert identity['sha256']==sealed(identity)and result['sha256']==sealed(result)==request['fits'][size]['fit_result_sha256']
 assert result['completed_epochs']==200 and result['optimizer_executions']==25000 and result['full_training_terminated']and result['stage_complete']
 assert identity['policy']['protocol_sha256']==identity['construction']['protocol_sha256']==identity['cohort']['protocol_sha256']==protocol['sha256']
 assert identity['construction']['batch_augmentation']is None and protocol['batch_augmentation']is None
 assert all(identity['construction']['sources_sha256']['upstream/'+name]==digest for name,digest in protocol['sources_sha256'].items())
 assert protocol['transforms']['train']==[{'_target_':'egoemg.transforms.ExtractToTensor','field':'emg'},{'_target_':'egoemg.transforms.RotationAugmentation'}]
 assert protocol['transforms']['val']==protocol['transforms']['test']==[{'_target_':'egoemg.transforms.ExtractToTensor','field':'emg'}]
 entry=dict(experiment=protocol['upstream_experiment'],transforms=protocol['transforms'],batch_augmentation=protocol['batch_augmentation'],window=protocol['window'],
  dataset_jitter_by_role={role:[bool(d.get('jitter',False))for d in raw['dataset'][role]]for role in('train','val','test')},
  actual_completed_fit_bound=True,fit_identity_sha256=identity['sha256'],fit_result_sha256=result['sha256'],selected_best=result['selected_best'],parameters=identity['construction']['parameter_count'],
  protocol_sha256=protocol['sha256'],all_recorded_upstream_config_and_native_sources_match=True,recorded_upstream_source_catalog_count=len(protocol['sources_sha256']),recorded_source_catalog_sha256=h(protocol['sources_sha256']))
 recipes['EMGFormer-'+size]=entry;bindings[size]=dict(fit_path=str(fit),identity_file_sha256=inputs[str(fit/'identity.json')],fit_result_file_sha256=inputs[str(fit/'fit-result.json')],fit_result_sha256=result['sha256'],protocol_sha256=protocol['sha256'])
 if catalog is None:catalog=protocol['sources_sha256']
 else:assert catalog==protocol['sources_sha256']
# The other named recipes are real original configurations, not a claim that
# their complete datasets/training have been re-admitted by this audit.
for name,experiment in [('VEMG-default','emg2pose/regression_vemg2pose_egoemg'),('VEMG-with-aug','emg2pose/regression_vemg2pose_egoemg_with_aug'),('NeuroPose-default','emg2pose/regression_neuropose_egoemg'),('NeuroPose-with-aug','emg2pose/regression_neuropose_egoemg_with_aug'),('RN18-fusion','fusion/fusion_rn18_s_center_8ch'),('ViT-S-fusion','fusion/fusion_vits_s_center_16ch')]:
 with initialize_config_dir(config_dir=str(U/'config'),version_base='1.1'):cfg=compose(config_name='base',overrides=['experiment='+experiment])
 OmegaConf.resolve(cfg);raw=OmegaConf.to_container(cfg,resolve=True)
 recipes[name]=dict(experiment=experiment,transforms=raw['transforms'],batch_augmentation=raw['batch_augmentation'],actual_completed_fit_bound=False,scope='original resolved configuration only; no new full input/model/training admission')
# RN18 current full identity is immutable; inspect config metadata, never its
# live checkpoints or in-progress journal/model state.
path=P/'private/experiment/full-fit/identity.json';stamps[str(path)]=nine(path);inputs[str(path)]=sha(path);rn=json.loads(path.read_text());assert rn['sha256']==sealed(rn)
assert rn['saved_config']['transforms']==recipes['RN18-fusion']['transforms']and rn['saved_config']['batch_augmentation']==recipes['RN18-fusion']['batch_augmentation']
recipes['RN18-fusion']['immutable_current_formal_fit_config_bound']=True;recipes['RN18-fusion']['current_formal_identity_sha256']=rn['sha256'];recipes['RN18-fusion']['formal_training_completion_not_inferred']=True
best=OmegaConf.load(U/'config/transforms/emgformer_regression_aug_best.yaml');OmegaConf.resolve(best);best=OmegaConf.to_container(best,resolve=True)
# Three deterministic branch observations of the unchanged actual baseline
# chain on explicit synthetic data. They do not read the real datasets.
np_before=np.random.get_state();py_before=random.getstate();torch_before=torch.get_rng_state().clone()
def rng():return hashlib.sha256(pickle.dumps(random.getstate(),protocol=5)+pickle.dumps(np.random.get_state(),protocol=5)+torch.get_rng_state().numpy().tobytes()).hexdigest()
rng_before=rng();cfg=compose_emgformer(U,'small');chain=Compose([instantiate(t)for t in cfg.transforms.train]);valchain=Compose([instantiate(t)for t in cfg.transforms.val]);probes=[]
x=np.tile(np.arange(1,9,dtype=np.float32),(32,1));found=set()
for seed in range(32):
 expected=int(np.random.RandomState(seed).choice([-1,0,1]));
 if expected in found:continue
 np.random.seed(seed);actual=chain({'emg':x.copy()});wanted=np.roll(x,expected,axis=-1)
 assert torch.is_tensor(actual)and actual.dtype==torch.float32 and np.array_equal(actual.numpy(),wanted)
 assert not np.array_equal(actual.numpy(),x[:,::-1])
 probes.append(dict(seed=seed,actual_selected_shift=expected,input_shape=list(x.shape),output_shape=list(actual.shape),dtype=str(actual.dtype),expected_circular_roll_exact=True,is_left_right_mirror=False,output_array_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest()));found.add(expected)
assert found=={-1,0,1}and len(probes)==3
assert np.array_equal(valchain({'emg':x.copy()}).numpy(),x)
np.random.set_state(np_before);random.setstate(py_before);torch.set_rng_state(torch_before);assert rng()==rng_before
# Snapshot every actual original source used to draw the semantics. The full
# catalogue is already bound to the three completed construction receipts.
source_names=['config/base.yaml','config/transforms/rotation_augmentation.yaml','config/transforms/emgformer_regression_aug_best.yaml','config/transforms/emgformer_regression_aug_extended.yaml','config/transforms/emgformer_regression_no_aug.yaml','config/augmentation/batch_aug_best_v2_no_mixup.yaml',
 'egoemg/transforms.py','egoemg/transforms_batch.py','egoemg/datasets/egoemg_memmap_dataset.py','egoemg/datamodule.py','egoemg/lightning.py','egoemg/train.py','egoemg/_train_utils.py']
source_names += ['config/experiment/'+r['experiment']+'.yaml'for r in recipes.values()]
source_names=sorted(set(source_names));source_hashes={name:sha(U/name)for name in source_names};assert all(catalog[name]==digest for name,digest in source_hashes.items()if name in catalog)
assert all(nine(Path(path))==stamp and sha(Path(path))==inputs[path]for path,stamp in stamps.items());clean()
assert counters==dict(neural_module=0,optimizer=0)and not torch.cuda.is_initialized()
f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
result=dict(format='actual_source_and_completed_fit_emg_augmentation_audit_v1',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actor=dict(pid=os.getpid(),start_ticks=f[19],uid=os.getuid()),
 source_package_root=str(P),upstream_revision='f7cbd048b7092139852270491d3859cd472e0311',upstream_clean_before_and_after=True,upstream_selected_source_files_sha256=source_hashes,
 all_three_completed_200_fit_construction_source_catalogs_exact=True,actual_completed_fit_bindings=bindings,actual_metadata_files_sha256=inputs,actual_metadata_nine_attributes=stamps,
 resolved_original_recipes=recipes,paper_like_optional_cpu_preset=best,paper_like_optional_preset_not_selected_by_three_main_baselines=True,
 deterministic_actual_baseline_chain_probes=probes,actual_validation_chain_is_identity_on_values=True,probe_input_scope='explicit32x8synthetic float32 signal; no real data or learned forward',
 all_python_numpy_torch_rng_restored_after_synthetic_probes=True,construction_attempts=counters,cuda_initialized=False,new_neural_forwards=0,new_optimizer_updates=0,new_training_or_scientific_settings=0,
 physical_left_hand_column_orientation_not_proven=True,full_paper_complete=False)
print(json.dumps(result))
