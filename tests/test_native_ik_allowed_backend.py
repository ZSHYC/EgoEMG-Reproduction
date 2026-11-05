"""Allowed-device policy positives grant no native qualification authority."""
import copy
import json

import pytest

from scripts.paper import run_native_mano_ik_continuation as workflow
from test_native_ik_workflow import request
from test_native_ik_results import inputs
from test_native_ik_continuation import prefix_source,open_prefix
from egoemg_repro import native_ik_results as results
from egoemg_repro.data import file_sha256

GPU6='GPU-6b41c390-53fc-6ccc-ea07-0648926ae5cb'
GPU7='GPU-83b7beb5-2729-4e3b-ed78-b949192152fb'
PROTECTED=['GPU-bb3b892e-454f-e10e-c305-5ab94db2c384','GPU-8b2cacdf-4470-21f2-01c5-e15475dbdfae']


def full_request(tmp_path):
    cfg=request(tmp_path);cfg.update(engineering=False,batch_size=10096,device='cuda:0',
        gpu_uuid=GPU6,qualification=str(tmp_path/'new-qualification'),minimum_free_bytes=225*1024**3)
    return cfg


@pytest.mark.parametrize('gpu',PROTECTED+['GPU-unrecognized'])
@pytest.mark.parametrize('engineering',[False,True])
def test_protected_or_unknown_GPU_never_opens_sources_or_CUDA(tmp_path,gpu,engineering):
    cfg=full_request(tmp_path);cfg.update(gpu_uuid=gpu,engineering=engineering)
    if engineering:cfg.update(qualification=None,batch_size=3)
    with pytest.raises(ValueError):workflow.validate_request(cfg)


def test_allowed_full_request_validation_is_policy_only(tmp_path):
    cfg=full_request(tmp_path);assert workflow.validate_request(cfg)==cfg
    assert not(tmp_path/'new-qualification').exists()


@pytest.mark.parametrize('uuid,index',[(GPU6,0),(GPU6,1),(GPU6,7),(GPU6,8),(PROTECTED[0],0),(PROTECTED[1],1),('GPU-unrecognized',6),(GPU6,True)])
def test_current_physical_mapping_must_match_an_allowed_card(uuid,index):
    with pytest.raises(ValueError):workflow.require_allowed_physical_gpu(uuid,index)


def test_allowed_UUID_uses_its_actual_physical_index():
    workflow.require_allowed_physical_gpu(GPU6,6)
    workflow.require_allowed_physical_gpu(GPU7,7)


def test_formal_operational_handoff_preserves_scientific_request(tmp_path):
    old=full_request(tmp_path);old.update(gpu_uuid=PROTECTED[1],qualification=str(tmp_path/'old-qualification'),minimum_free_bytes=200*1024**3)
    new=dict(old,gpu_uuid=GPU6,qualification=str(tmp_path/'new-qualification'),minimum_free_bytes=225*1024**3)
    workflow.require_scientific_request(new,old)
    assert old['gpu_uuid']==PROTECTED[1]and old['qualification'].endswith('old-qualification')


@pytest.mark.parametrize('field',['data','geometry','upstream','manotorch','chumpy','mano_assets','engineering','batch_size','device','unknown_key','missing_key'])
def test_handoff_does_not_allow_scientific_or_source_changes(tmp_path,field):
    old=full_request(tmp_path);new=copy.deepcopy(old)
    if field=='unknown_key':new['new_math']=True
    elif field=='missing_key':del new['geometry']
    elif field=='engineering':new[field]=True
    elif field=='batch_size':new[field]=10095
    elif field=='device':new[field]='cpu'
    else:new[field]+='-different'
    with pytest.raises(ValueError):workflow.require_scientific_request(new,old)


def test_CPU_engineering_cannot_switch_hardware_or_qualification(tmp_path):
    old=request(tmp_path);workflow.require_scientific_request(dict(old,minimum_free_bytes=20*1024**3),old)
    for changes in [dict(gpu_uuid=GPU6),dict(qualification=str(tmp_path/'fake'))]:
        with pytest.raises(ValueError):workflow.require_scientific_request(dict(old,**changes),old)


@pytest.mark.parametrize('field',['uuid','driver_version','pci_bus_id','name','total_memory_mib'])
def test_new_physical_descriptor_must_match_the_new_qualification(tmp_path,field):
    cfg=full_request(tmp_path);physical=dict(uuid=GPU6,driver_version='595.91.07',pci_bus_id='00000000:C6:00.0',name='NVIDIA A800-SXM4-80GB',total_memory_mib=81920)
    changed=dict(physical);changed[field]='changed'
    with pytest.raises(ValueError):workflow.require_qualified_physical_backend(cfg,dict(physical_device=physical),changed)
    workflow.require_qualified_physical_backend(cfg,dict(physical_device=physical),physical)


def test_CPU_carrier_backend_keeps_false_qualification_and_old_provenance(prefix_source,tmp_path):
    inputs,_,_=prefix_source;prefix=open_prefix(prefix_source);cfg=request(tmp_path);cfg['batch_size']=3
    record,binding=workflow.admit_current_backend(cfg,inputs,None,prefix,None,None)
    assert record['engineering']and not record['original_default_batch10096_qualified']
    assert record['physical_device']is None and record['original_prefix_physical_device']is None
    assert record['qualification_file_sha256']is None and record['physical_index']is None
    assert not record['scientific_eligible']and not record['full_paper_complete']
    binding.check(rehash=True)


@pytest.mark.parametrize('permit',['missing','forged','engineering'])
def test_carrier_or_forged_permit_cannot_admit_a_formal_backend(prefix_source,tmp_path,permit):
    inputs,_,_=prefix_source;prefix=open_prefix(prefix_source);cfg=full_request(tmp_path)
    if permit!='missing':
        p=tmp_path/'new-qualification';p.mkdir();(p/'qualification.json').write_text(json.dumps(dict(original_default_batch10096_qualified=permit=='forged',engineering=permit=='engineering')))
    physical=dict(uuid=GPU6,driver_version='595.91.07',pci_bus_id='00000000:C6:00.0',name='NVIDIA A800-SXM4-80GB',total_memory_mib=81920)
    with pytest.raises(ValueError):workflow.admit_current_backend(cfg,inputs,None,prefix,physical,6)


@pytest.fixture
def qualification_configuration(tmp_path):
    """Configuration I/O only; never a native qualification or full permit."""
    cfg=full_request(tmp_path);root=tmp_path/'new-qualification';root.mkdir()
    physical=dict(uuid=GPU6,driver_version='595.91.07',pci_bus_id='00000000:C6:00.0',name='NVIDIA A800-SXM4-80GB',total_memory_mib=81920)
    doc=dict(action='qualify',request=dict(cfg,qualification=None),physical_device=physical,
        input_identity_sha256='a'*64,source_identity_sha256='b'*64,sources_sha256={'structural_IO_only':'c'*64})
    path=root/'configuration.json';results.write_json(path,results.seal(doc))
    permit=dict(format='structural_configuration_binding_only_NOT_a_qualification',physical_device=physical,
        configuration_sha256=file_sha256(path),input_identity_sha256='a'*64,source_identity_sha256='b'*64,sources_sha256=doc['sources_sha256'])
    return cfg,path,doc,permit,physical


def test_configuration_binding_is_live_IO_and_cannot_grant_qualification(qualification_configuration):
    cfg,path,_,permit,physical=qualification_configuration;binding=results.IKArtifactBindings()
    value=workflow.bind_current_qualification_configuration(cfg,permit,physical,binding)
    assert value==file_sha256(path)and type(value)is str
    binding.check(rehash=True);path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):binding.check()


@pytest.mark.parametrize('fault',['old_GPU','action','source_path','batch','reused_qualification','input','source_identity','source_snapshot','physical_driver','configuration_sha'])
def test_resealed_configuration_cannot_relabel_an_old_or_different_qualification(qualification_configuration,fault):
    cfg,path,doc,permit,physical=qualification_configuration;doc=copy.deepcopy(doc)
    if fault=='old_GPU':doc['request']['gpu_uuid']=PROTECTED[1]
    elif fault=='action':doc['action']='workflow'
    elif fault=='source_path':doc['request']['geometry']+='-different'
    elif fault=='batch':doc['request']['batch_size']=10095
    elif fault=='reused_qualification':doc['request']['qualification']='old-permit'
    elif fault=='input':doc['input_identity_sha256']='d'*64
    elif fault=='source_identity':doc['source_identity_sha256']='d'*64
    elif fault=='source_snapshot':doc['sources_sha256']={'different':'d'*64}
    elif fault=='physical_driver':doc['physical_device']['driver_version']='changed'
    else:permit['configuration_sha256']='d'*64
    path.write_text(json.dumps(results.seal(doc)))
    if fault!='configuration_sha':permit['configuration_sha256']=file_sha256(path)
    with pytest.raises(ValueError):workflow.bind_current_qualification_configuration(cfg,permit,physical,results.IKArtifactBindings())
