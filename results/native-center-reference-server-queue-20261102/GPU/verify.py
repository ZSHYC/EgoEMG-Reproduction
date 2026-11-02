
from pathlib import Path
import hashlib,json,os,sys
R=Path(sys.argv[1]);size=sys.argv[2];q=json.loads((R/'config.json').read_text());P=Path(q['package']);sys.path.insert(0,str(P))
import torch
torch.set_num_threads(2);attempts={'learned_module':0,'optimizer':0}
def no_model(*a,**k):attempts['learned_module']+=1;raise AssertionError('GPU qualification verifier cannot construct a model')
def no_optimizer(*a,**k):attempts['optimizer']+=1;raise AssertionError('GPU qualification verifier cannot construct an optimizer')
torch.nn.Module.__init__=no_model;torch.optim.Optimizer.__init__=no_optimizer
from egoemg_repro.center_reference_eval import read_reference_pass
a=read_reference_pass(R/'private'/size/'native');b=read_reference_pass(R/'private'/size/'replay')
assert a['kind']=='native'and b['kind']=='replay'and a['scope']==b['scope']=='engineering'and a['samples']==b['samples']==4
assert a['prediction_sha256']==b['prediction_sha256']and a['model_before_sha256']==a['model_after_sha256']==b['model_before_sha256']==b['model_after_sha256']
assert a['raw_prediction_steps']==b['raw_prediction_steps']==[q['expected_raw_steps'][size]]
assert attempts=={'learned_module':0,'optimizer':0}and not torch.cuda.is_initialized()
assert a['runtime_policy']==b['runtime_policy']and a['runtime_policy']['device']=='cuda:0'and a['runtime_policy']['gpu_uuid']==q['gpu_uuid']
def summary(v):return{k:v[k]for k in('size','kind','scope','samples','source_indices','fit_result_sha256','input_admission_sha256','grid_sha256','model_before_sha256','model_after_sha256','prediction_sha256','raw_prediction_steps','runtime_policy','neural_forwards','optimizer_updates','optimizer_construction_attempts','native_runtime_construction_calls','process','started_ns','finished_ns','sha256')}
f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
value=dict(format='actual_selected_native_reference_GPU_four_request_qualification_v1',size=size,source_commit=q['source_commit'],native=summary(a),replay=summary(b),both_original_producers_physically_closed=True,strict_named_state_restore_exact=True,typed_prediction_target_mask_and_identity_exact=True,model_free_reader_construction_attempts=attempts,reader=dict(pid=os.getpid(),start_ticks=f[19]),reader_cuda_initialized=False,engineering_only=True,full_reference_model_evaluation_complete=False,full_paper_complete=False)
with(R/'private'/size/'qualification.json').open('x')as stream:stream.write(json.dumps(value,sort_keys=True,indent=2)+'\n')
print(json.dumps(value,sort_keys=True))
