"""Real engineering handoff; full-trained positives require actual producer runs."""
import copy
import importlib
import json
from pathlib import Path
import shutil

import pytest

from test_fusion_literal_cohort import files,admit
from test_fusion_literal_runtime import branch_files,native_packet,sha
from test_vision_literal_train import native_runs as vision_runs
from test_emg_initializer_train import native_runs as emg_runs


def api():
    try:return importlib.import_module('egoemg_repro.fusion_initializer_handoff')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.fusion_initializer_handoff':pytest.fail('verified fusion initializer handoff missing')
        raise


def test_real_engineering_files_are_bound_but_never_claim_full_lineage(files,branch_files):
    c=admit(files);value=api().engineering_fusion_initializers(c,**branch_files)
    assert value.runtime_kwargs==branch_files
    identity=value.identity
    assert identity['scope']=='engineering' and identity['producer_training_verified'] is False
    assert identity['vision']['checkpoint_selection']=='explicit trusted engineering file'
    assert identity['emg']['checkpoint_selection']=='explicit trusted engineering file'
    assert identity['fusion_partition_content_sha256']==c.identity['partition_content_sha256']
    assert identity['scientific_eligible'] is identity['full_paper_complete'] is False
    assert '/home/' not in json.dumps(identity) and '/tmp/' not in json.dumps(identity)
    value.check()
    with pytest.raises(ValueError,match='engineering|full'):value.check(require_full=True)


@pytest.mark.parametrize('damage',['branch_file','receipt','cohort','runtime_path'])
def test_engineering_handoff_checks_actual_files_and_both_identities(files,branch_files,tmp_path,damage):
    c=admit(files);local=dict(branch_files)
    if damage=='branch_file':
        dest=tmp_path/'emg.pt';shutil.copyfile(local['emg_checkpoint'],dest);local['emg_checkpoint']=dest
    handoff=api().engineering_fusion_initializers(c,**local)
    if damage=='branch_file':
        with local['emg_checkpoint'].open('ab') as f:f.write(b'changed')
    elif damage=='receipt':handoff.identity['producer_training_verified']=True
    elif damage=='cohort':c.partitions['windows'][0]['center']+=1
    elif damage=='runtime_path':handoff.runtime_kwargs['vision_checkpoint']=local['emg_checkpoint']
    with pytest.raises(ValueError,match='identity|changed|binding|source|initializer'):handoff.check()


def test_unregistered_cohort_cannot_enter_full_handoff(files,branch_files,tmp_path):
    c=admit(files)
    with pytest.raises(ValueError,match='registered|complete cohort'):
        api().admit_fusion_initializers(c,vision_fit=tmp_path/'missing',vision_source_root=tmp_path,
            emg_fit=tmp_path/'missing-emg',emg_export=branch_files['emg_checkpoint'])


def test_private_producer_locators_are_sealed_separately_from_safe_lineage(files,branch_files):
    from dataclasses import replace
    c=admit(files);b=api().engineering_fusion_initializers(c,**branch_files)
    assert b.producer_sources is None and 'producer_sources' not in b.identity
    altered=replace(b,producer_sources={'emg_fit':'different'})
    with pytest.raises(ValueError,match='binding|source|identity'):altered.check()


def partition(*,train=(),val=(),test=()):
    return dict(windows=[dict(role=role,footprint_start=a,footprint_end=b) for role,spans in
        [('train',train),('val',val),('test',test)] for a,b in spans],
        validation_intervals=[dict(start=200,end=240)])


def test_full_possible_footprints_merge_duplicates_and_accept_touching_boundaries():
    producer=partition(train=[(0,100),(50,160),(240,300),(400,500)])
    fusion=partition(val=[(210,220)],test=[(300,400),(330,360)])
    observed=api().verify_initializer_footprints(producer,fusion)
    assert observed==dict(train_union_rows=320,holdout_union_rows=dict(val=40,test=100),
        overlap_rows=dict(val=0,test=0))


@pytest.mark.parametrize('role,train',[('val',[(150,201)]),('test',[(150,301)]),('val',[(220,221)])])
def test_jitter_only_and_validation_interval_overlap_refuse(role,train):
    producer=partition(train=train);fusion=partition(val=[(210,215)],test=[(300,400)])
    with pytest.raises(ValueError,match='overlap'):
        api().verify_initializer_footprints(producer,fusion)


@pytest.mark.parametrize('bad',[True,1.5,-1,'1',None])
def test_invalid_footprint_bounds_are_not_silently_cast(bad):
    producer=partition(train=[(0,100)]);producer['windows'][0]['footprint_end']=bad
    with pytest.raises(ValueError,match='footprint|interval'):api().verify_initializer_footprints(producer,partition(test=[(300,400)]))


def test_missing_footprint_is_not_replaced_by_smaller_nominal_window():
    producer=partition(train=[(0,100)]);del producer['windows'][0]['footprint_end']
    producer['windows'][0]['end']=100
    with pytest.raises(ValueError,match='footprint'):api().verify_initializer_footprints(producer,partition(test=[(300,400)]))


def test_changed_historical_reader_is_rejected_before_any_of_its_code_executes(tmp_path):
    root=tmp_path/'frozen';p=root/'egoemg_repro/vision_literal_train.py';p.parent.mkdir(parents=True)
    sentinel=tmp_path/'executed'
    p.write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').write_text("executed")\n')
    project=Path(__file__).parents[1]
    published=json.loads((project/'results/original-vision/vit-small/publication-manifest.json').read_text())
    names=('vision_literal_train','vision_literal_cohort','classic_emg_train','vision_literal_check')
    sources={f'egoemg_repro/{name}.py':published['source_files_sha256'][f'egoemg_repro/{name}.py'] for name in names}
    for name in sources:
        if name!='egoemg_repro/vision_literal_train.py':shutil.copyfile(project/name,root/name)
    with pytest.raises(ValueError,match='frozen|historical|source'):
        api()._historical_vision_reader(root,{'fit_sources_sha256':sources})
    assert not sentinel.exists()


@pytest.mark.parametrize('damage',['missing_key','changed_shared_payload','partition','metadata'])
def test_producer_cohort_binding_rejects_different_payload_or_reconstructed_partition(files,damage):
    from egoemg_repro.classic_emg import _hash
    c=admit(files);p=copy.deepcopy(c.partitions);old=copy.deepcopy(c.identity)
    if damage=='missing_key':del old['payload_files_sha256']['field:frame_split_id']
    elif damage=='changed_shared_payload':old['payload_files_sha256']['field:generated_joint_angles_left']='0'*64
    elif damage=='partition':old['partition_content_sha256']='0'*64
    elif damage=='metadata':old['payload_files_sha256']['metadata:episodes']='0'*64
    old['sha256']=_hash({k:v for k,v in old.items() if k!='sha256'})
    with pytest.raises(ValueError,match='payload|partition|metadata|cohort'):
        api()._producer_cohort_binding(c,old,p)


def test_real_engineering_vision_fit_cannot_supply_a_full_fusion_initializer(vision_runs):
    root,_,_,_=vision_runs
    with pytest.raises(ValueError,match='engineering|full'):api()._vision_producer(root/'resumed',Path(__file__).parents[1])


def test_real_engineering_emg_best_export_cannot_supply_full_fusion_initializer(emg_runs,tmp_path):
    from egoemg_repro.emg_initializer_train import export_emg_initializer_best
    _,out,_,_,_=emg_runs;dest=tmp_path/'best.pt'
    export_emg_initializer_best(out,dest,require_full=False)
    with pytest.raises(ValueError,match='engineering|full'):api()._emg_producer(out,dest)
