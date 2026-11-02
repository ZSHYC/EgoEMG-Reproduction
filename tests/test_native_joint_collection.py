"""Real typed raw engineering IO and closed-source admission, no trained model."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_emgformer_literal import UPSTREAM


def api():
    path=Path(__file__).resolve().parents[1]/'scripts/paper/collect_native_joint_analysis.py'
    assert path.exists(), 'native raw joint collector is missing'
    spec=importlib.util.spec_from_file_location('native_joint_collection_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(value):
    return dict(value,sha256=digest(value))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    return path


def nine(path):
    s=path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]


@pytest.fixture(scope='module')
def closed_actors():
    code='import json,os; from pathlib import Path; p=os.getpid(); f=Path("/proc",str(p),"stat").read_text().rsplit(")",1)[1].split(); print(json.dumps(dict(pid=p,start_ticks=f[19],uid=os.getuid(),sid=os.getsid(0),pgrp=os.getpgrp())))'
    actors=[]
    for _ in range(18):
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,start_new_session=True)
        assert result.returncode==0
        actors.append(json.loads(result.stdout))
    return actors


@pytest.fixture
def source_factory(tmp_path,closed_actors):
    def factory(damage=None):
        from egoemg_repro.emgformer_trajectory_results import _alignment,write_trajectory_packet,trajectory_statistics
        torch.set_num_threads(1)
        root=tmp_path/'collector';parent=tmp_path/'original';flow=parent/'private/middle-flow';size='middle'
        def stage(worker,guardian):
            return dict(status='success',worker_exit_code=0,all_owned_consumers_exited=True,
                        worker=closed_actors[worker],guardian=closed_actors[guardian])
        outer=stage(0,1);extra=stage(16,17)
        if damage=='live_owner':
            f=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
            outer['worker']=dict(pid=os.getpid(),start_ticks=f[19],uid=os.getuid(),sid=os.getsid(0),pgrp=os.getpgrp())
        if damage=='pending_stage':extra['status']='started'
        alignment=_alignment(12000-13,2,3);selected=dict(checkpoint_sha256='a'*64,score=.5,validated=True,file='engineering.ckpt')
        if damage=='unvalidated_selection':selected['validated']=False
        passes={};raw_hashes={};all_paths=[];baseline=None
        for kind in ('native','replay'):
            directory=flow/('test-'+kind);(directory/'raw').mkdir(parents=True)
            records=[];packets=[]
            for ordinal,start in enumerate(range(0,6,2)):
                indexes=list(range(start,start+2));windows=[i%3 for i in indexes]
                source=torch.zeros(2,22,12000);valid=torch.zeros(2,12000,dtype=torch.bool)
                for j,w in enumerate(windows):valid[j,12000-13:12000-13+(1,3,5)[w]]=True
                values=torch.tensor([i+1 for i in indexes],dtype=torch.bfloat16)
                arrays=dict(source_target=source,source_mask=valid,target=source[...,12000-13:-2].clone(),
                    mask=valid[...,12000-13:-2].clone(),prediction=values[:,None,None].expand(2,22,11).clone(),
                    raw_prediction=values[:,None,None].expand(2,22,3).clone())
                metadata=dict(index=indexes,unique=indexes.copy(),repeat=[0,0],start=[w*12000 for w in windows],
                    end=[(w+1)*12000 for w in windows],center=[w*12000+6000 for w in windows],episode=[0,0],
                    hand=['left'if i<3 else'right'for i in indexes],split=[('user','gesture','both')[w]for w in windows],
                    user=['PRIVATE_SUBJECT_SENTINEL_'+str(w)for w in windows])
                fingerprints=[f'{i+1:064x}'for i in indexes]
                record=write_trajectory_packet(directory/'raw',ordinal,arrays=arrays,alignment=alignment,
                    metadata=metadata,input_sha256='c'*64,sample_input_sha256=fingerprints,upstream=UPSTREAM)
                packets.append(dict(arrays=arrays,alignment=alignment,metadata=copy.deepcopy(metadata),
                    input_sha256='c'*64,sample_input_sha256=fingerprints.copy()))
                if kind=='native'and ordinal==0:
                    if damage=='packet_gap':record['metadata']['index'][0]=1
                    elif damage=='repeat':record['metadata']['repeat'][0]=1
                    elif damage=='hand':record['metadata']['hand'][0]='right'
                    elif damage=='split':record['metadata']['split'][0]='train'
                if kind=='native'and ordinal==2 and damage=='paired_window':record['metadata']['user'][-1]='different_subject'
                if kind=='replay'and ordinal==0 and damage=='replay_input':record['sample_input_sha256'][0]='d'*64
                record=seal({k:v for k,v in record.items()if k!='sha256'});records.append(record)
                path=directory/'raw'/record['file'];all_paths.append(path);raw_hashes[str(path)]=sha(path)
            stats=trajectory_statistics(packets,samples=6,batch_size=2,role='test',upstream=UPSTREAM)
            if damage=='baseline_joint_order':stats['trajectory']['by_joint'][0]['index']=21
            if damage=='baseline_hand_split':
                for hand,split,delta in [('left','user',1.),('right','user',-1.),('left','gesture',-1.),('right','gesture',1.)]:
                    cell=stats['trajectory']['by_hand_split'][hand][split]
                    cell['absolute_error_sum_rad']+=delta
                    cell['mae_deg']=cell['absolute_error_sum_rad']/cell['valid_angle_count']*180/__import__('math').pi
            if baseline is None:baseline=stats
            journal=directory/'packets.jsonl';journal.write_text(''.join(json.dumps(z,sort_keys=True)+'\n'for z in records));all_paths.append(journal)
            value=seal(dict(format='original_emgformer_full_trajectory_pass_v1',directory=str(directory),
                role='test',kind=kind,samples=6,batch_size=2,workers=0,precision='32-true',training_scope='cpu_engineering',
                formal_evaluation=False,require_full=False,fit_result_sha256='b'*64,selected_best=selected,
                actor=closed_actors[2 if kind=='native'else 4],loader_workers=[],all_loader_workers_exited=True,
                actual_optimizer_updates=0,statistics=stats,packet_journal_sha256=sha(journal),
                packet_content_sha256=digest([digest({k:v for k,v in z.items()if k not in('sha256','file_sha256')})for z in records])))
            passfile=write(directory/'pass.json',value);all_paths.append(passfile);passes['test-'+kind]=value
        baseline=copy.deepcopy(baseline)
        users=sorted(baseline['trajectory']['by_user'])
        aliases={user:'user_'+str(index).zfill(3)for index,user in enumerate(users)}
        for name in ('trajectory','aligned_center'):
            baseline[name]['by_user']={aliases[user]:cell for user,cell in baseline[name]['by_user'].items()}
        if damage=='public_numeric':baseline['trajectory']['overall']['absolute_error_sum_rad']+=1.
        elif damage=='public_center_user':baseline['aligned_center']['by_user']['user_000']['valid_angle_count']+=1
        elif damage=='public_alias':baseline['trajectory']['by_user']['user_999']=baseline['trajectory']['by_user'].pop('user_000')
        report=seal(dict(format='original_emgformer_full_trajectory_report_v1',size=size,fit_result_sha256='b'*64,
            training=dict(scope='cpu_engineering',completed_epochs=2,optimizer_executions=4,full_training_terminated=False,selected_best=selected),
            test=baseline,scientific_eligible=False,trajectory_pipeline_complete=True,paper_numerical_equivalence_established=False,full_paper_complete=False))
        reportfile=write(flow/'report/metrics.json',report);all_paths.append(reportfile)
        source=parent/'frozen.py';source.write_text('# fixed engineering source\n');all_paths.append(source)
        original_request=write(parent/'private/request.json',dict(upstream=str(UPSTREAM)))
        original_receipt=dict(source_commit='1'*40,files_sha256={'frozen.py':sha(source),'private/request.json':sha(original_request)})
        original_receipt['bundle_sha256']=digest(original_receipt);write(parent/'private/stage-receipt.json',original_receipt)
        request=write(root/'private/request.json',dict(root=str(parent),source_commit='1'*40,bundle_sha256=original_receipt['bundle_sha256']))
        reader=root/'private/worker.py';reader.write_text('# completed engineering reader source\n')
        collector_receipt=dict(source_commit='2'*40,files_sha256={'private/request.json':sha(request),'private/worker.py':sha(reader)})
        collector_receipt['bundle_sha256']=digest(collector_receipt);write(root/'private/stage-receipt.json',collector_receipt)
        verification=seal(dict(format='independent_raw_trajectory_verification_v1',graph_generations={str(p):nine(p)for p in all_paths},full_paper_complete=False))
        if damage=='conflicting_generation':
            verification['graph_generations'][str(all_paths[0])][4]-=1
            verification=seal({k:v for k,v in verification.items()if k!='sha256'})
        verificationfile=write(flow/'verified/verification.json',verification);all_paths.append(verificationfile)
        exports={}
        for name in ('metrics.json','metrics.pending.json','metrics.md','completion.json','verification.json','workflow-result.json'):
            path=root/'private/export/middle'/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(reportfile.read_bytes()if name=='metrics.json'else verificationfile.read_bytes()if name=='verification.json'else b'{}\n')
            exports[name]=sha(path)
        readback=dict(format='actual_native_emgformer_complete_trajectory_model_free_postexit_v1',size=size,actual_report=report,
            actor=closed_actors[2 if damage=='reader_actor_mismatch'else 16],
            original_source_commit='1'*40,original_bundle_sha256=original_receipt['bundle_sha256'],original_outer_size_stage=outer,
            all16_original_size_actors_closed=[outer['worker'],outer['guardian'],*closed_actors[2:16]],all24_original_loader_workers_closed=True,
            all_frozen_source_files_exact=True,construction_attempts=dict(learned_module=0,optimizer=0),cuda_initialized=False,
            new_learned_forwards=0,new_optimizer_updates=0,scientific_eligible=False,exported_report_files_sha256=exports,
            non_tensor_artifact_files_after_exit_sha256={str(p):sha(p)for p in all_paths if p.suffix!='.pt'},
            raw_and_other_artifact_nine_attributes={str(p):nine(p)for p in all_paths},
            raw_tensor_payload_sha256_physically_verified_by_existing_native_reader=raw_hashes)
        readbackfile=write(root/'private/middle-postexit-readback.json',readback)
        complete=dict(size=size,actual_owned_readback_stage=extra,readback_file_sha256=sha(readbackfile),exported_report_files_sha256=exports,scientific_eligible=False)
        write(root/'private/middle-actual-postexit-complete.json',complete)
        write(root/'private/owned-middle-postexit-readback/stage-result.json',extra)
        proof=dict(format='actual_native_emgformer_complete_size_and_closed_independent_postexit_acceptance_v1',size=size,
            original_source_commit='1'*40,original_bundle_sha256=original_receipt['bundle_sha256'],collector_source_commit='2'*40,
            collector_bundle_sha256=collector_receipt['bundle_sha256'],collector_files_sha256=collector_receipt['files_sha256'],
            original_size_outer_stage=outer,all16_original_size_actors_closed=readback['all16_original_size_actors_closed'],
            all24_original_loader_workers_closed=True,additional_independent_reader_stage=extra,all2_independent_reader_actors_closed=closed_actors[16:18],
            actual_training_completed_epochs=2,actual_durable_training_updates=4,actual_fit_result_sha256='b'*64,
            actual_report_sha256=report['sha256'],selected_best=selected,readback_file_sha256=sha(readbackfile),original_report_files_sha256=exports,
            scientific_eligible=False,raw_tensor_payload_sha256_physically_verified_by_existing_native_reader=raw_hashes)
        public_metrics=write(tmp_path/'metrics.json',report);proofpath=write(tmp_path/'proof.json',proof)
        arguments=dict(postexit_root=root,size=size,metrics=public_metrics,proof=proofpath,expected_proof_sha256=sha(proofpath),
            upstream=UPSTREAM,output=tmp_path/'result',scope='cpu_engineering')
        return arguments,dict(root=root,flow=flow,proof=proofpath,metrics=public_metrics,reader=reader,raw=flow/'test-native/raw/000000.pt')
    return factory


def test_true_raw_bf16_collection_keeps_all_requests_and_remains_pending(source_factory,monkeypatch):
    module=api();args,paths=source_factory()
    def deny(*a,**k):raise AssertionError('learned model or optimizer construction forbidden')
    monkeypatch.setattr(torch.nn.Module,'__init__',deny);monkeypatch.setattr(torch.optim.Optimizer,'__init__',deny)
    result=module.collect_native_joint_analysis(**args)
    assert result['views']['n_requested']==6 and result['views']['n_with_any_valid_angle']==6
    assert result['native_replay_records_equal']and result['baseline_counts_and_metrics_match']
    assert result['source_run_scientific_eligible']is False and result['analysis_reader_complete']is False
    assert result['new_learned_forwards']==result['new_optimizer_updates']==0 and not result['cuda_initialized']
    private=json.loads((args['output']/'private-records.json').read_text())
    assert len(private['records'])==6 and 'PRIVATE_SUBJECT_SENTINEL'in json.dumps(private)
    population=[{k:r[k]for k in('key','hand','split','private_metadata','sample_input_sha256','trajectory_angle_valid_counts')}for r in private['records']]
    assert result['request_population_sha256']==digest(population)
    public=(args['output']/'analysis.pending.json').read_text()
    assert 'PRIVATE_SUBJECT_SENTINEL'not in public and str(paths['root'])not in public
    assert not (args['output']/'completion.json').exists()


def test_public_report_user_aliases_match_private_pass_statistics(source_factory):
    module=api();args,paths=source_factory()
    report=json.loads(paths['metrics'].read_text())
    private=json.loads((paths['flow']/'test-native/pass.json').read_text())
    for name in ('trajectory','aligned_center'):
        assert set(report['test'][name]['by_user'])=={'user_000','user_001','user_002'}
        assert all(user.startswith('PRIVATE_SUBJECT_SENTINEL_')for user in private['statistics'][name]['by_user'])
    result=module.collect_native_joint_analysis(**args)
    assert result['baseline_counts_and_metrics_match']and result['native_replay_records_equal']


@pytest.mark.parametrize('damage',['public_numeric','public_center_user','public_alias'])
def test_public_alias_projection_preserves_numeric_and_user_identity_checks(source_factory,monkeypatch,damage):
    module=api();args,_=source_factory(damage)
    calls=[]
    def deny(*a,**k):calls.append(True);raise AssertionError('tensor loading preceded public statistics check')
    monkeypatch.setattr(torch,'load',deny)
    with pytest.raises(ValueError,match='role/policy/fit'):module.collect_native_joint_analysis(**args)
    assert not calls and not args['output'].exists()


@pytest.mark.parametrize('damage',['proof_hash','report_bytes','raw_bytes','raw_generation','reader_source','live_owner','pending_stage','missing_complete','engineering_full','conflicting_generation','unvalidated_selection','reader_actor_mismatch'])
def test_bad_source_fails_before_tensor_loading(source_factory,monkeypatch,damage):
    module=api();args,paths=source_factory(damage)
    if damage=='proof_hash':args['expected_proof_sha256']='0'*64
    elif damage=='report_bytes':paths['metrics'].write_text('{}\n')
    elif damage=='raw_bytes':
        with paths['raw'].open('ab')as stream:stream.write(b'drift')
    elif damage=='raw_generation':paths['raw'].write_bytes(paths['raw'].read_bytes())
    elif damage=='reader_source':paths['reader'].write_text('# changed\n')
    elif damage=='missing_complete':(paths['root']/'private/middle-actual-postexit-complete.json').unlink()
    elif damage=='engineering_full':args['scope']='full'
    calls=[]
    def deny(*a,**k):calls.append(True);raise AssertionError('tensor loading preceded source admission')
    monkeypatch.setattr(torch,'load',deny)
    with pytest.raises(ValueError):module.collect_native_joint_analysis(**args)
    assert not calls and not args['output'].exists()


@pytest.mark.parametrize('damage',['packet_gap','repeat','hand','split','paired_window','replay_input','baseline_joint_order','baseline_hand_split'])
def test_self_consistent_but_semantically_wrong_raw_graph_is_rejected(source_factory,damage):
    module=api();args,_=source_factory(damage)
    with pytest.raises(ValueError):module.collect_native_joint_analysis(**args)
    assert not args['output'].exists()


def test_fresh_output_required(source_factory):
    module=api();args,_=source_factory();args['output'].mkdir()
    with pytest.raises(ValueError):module.collect_native_joint_analysis(**args)


def test_upstream_argument_must_match_frozen_source_request_before_raw_decoder(source_factory,monkeypatch):
    module=api();args,_=source_factory();args['upstream']=args['output'].parent
    def deny(*a,**k):raise AssertionError('raw decoding preceded frozen upstream admission')
    monkeypatch.setattr(module,'_reduce',deny)
    with pytest.raises(ValueError):module.collect_native_joint_analysis(**args)


def produce_in_child(module,args):
    argv=[sys.executable,'-B',str(Path(module.__file__))]
    for key,value in args.items():argv.extend(['--'+key.replace('_','-'),str(value)])
    process=subprocess.run(argv,capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    return args['output']/'private-records.json',args['output']/'analysis.pending.json'


def verify_arguments(private,pending,output):
    return dict(private_records=str(private),pending=str(pending),expected_private_sha256=sha(private),
                expected_pending_sha256=sha(pending),output=str(output))


def run_verifier_cli(module,arguments):
    argv=[sys.executable,'-B',str(Path(module.__file__)),'verify']
    for key,value in arguments.items():argv.extend(['--'+key.replace('_','-'),str(value)])
    return subprocess.run(argv,capture_output=True,text=True)


def test_fresh_stdlib_verifier_recomputes_without_primary_weighting_function(source_factory):
    module=api();args,_=source_factory();private,pending=produce_in_child(module,args)
    arguments=verify_arguments(private,pending,args['output'].parent/'verification.json')
    code='''import json,runpy,sys
m=runpy.run_path(sys.argv[1],run_name='independent_test')
def deny(*a,**k):raise AssertionError('primary weighting function reused')
m['numeric'].native_joint_group_views=deny
v=m['verify_pending_native_joint_analysis'](**json.loads(sys.argv[2]))
assert 'torch'not in sys.modules and'numpy'not in sys.modules
assert v['all_cells_independently_recomputed']and v['producer_actor_physically_closed']
'''
    process=subprocess.run([sys.executable,'-B','-c',code,str(Path(module.__file__)),json.dumps(arguments)],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    result=json.loads(Path(arguments['output']).read_text())
    assert result['stdlib_only']and result['analysis_reader_complete']is False


def test_independent_verifier_rejects_resealed_wrong_weighted_value(source_factory):
    module=api();args,_=source_factory();private,pending=produce_in_child(module,args)
    value=json.loads(pending.read_text());value['views']['caption_interpretation_split_sample_weighted']['groups']['thumb']['mae_deg']+=.1
    write(pending,seal({k:v for k,v in value.items()if k!='sha256'}))
    result=run_verifier_cli(module,verify_arguments(private,pending,args['output'].parent/'verification.json'))
    assert result.returncode!=0 and'independent'in result.stderr.lower()


def test_independent_verifier_rejects_resealed_private_field_in_public_view(source_factory):
    module=api();args,_=source_factory();private,pending=produce_in_child(module,args)
    value=json.loads(pending.read_text());value['views']['by_split']['user']['private_subject']='PRIVATE_SUBJECT_SENTINEL'
    write(pending,seal({k:v for k,v in value.items()if k!='sha256'}))
    result=run_verifier_cli(module,verify_arguments(private,pending,args['output'].parent/'verification.json'))
    assert result.returncode!=0 and'public schema'in result.stderr.lower()


def test_independent_verifier_waits_for_actual_producer_exit(source_factory):
    module=api();args,_=source_factory();module.collect_native_joint_analysis(**args)
    result=run_verifier_cli(module,verify_arguments(args['output']/'private-records.json',args['output']/'analysis.pending.json',args['output'].parent/'verification.json'))
    assert result.returncode!=0 and'physically exited'in result.stderr.lower()
