"""Read accepted closed native trajectories; never train or run a model.

Outputs are pending analysis documents. The calling owned workflow must verify
them independently and observe reader exit before issuing a completion proof.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from egoemg_repro import native_joint_analysis as numeric
from egoemg_repro.paper_analysis import verify_metric


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


IMPORTED_SHA=_sha(__file__)


def _safe(path,*,exists=True):
    path=Path(path).expanduser()
    if '..'in path.parts:raise ValueError('parent traversal forbidden')
    if not path.is_absolute():path=Path.cwd()/path
    if any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('source/output symlink forbidden')
    try:return path.resolve(strict=exists)
    except OSError as error:raise ValueError('source/artifact unavailable')from error


def _nine(path):
    s=_safe(path).stat()
    if not stat.S_ISREG(s.st_mode):raise ValueError('regular source/artifact required')
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink]


def _sealed(value):
    if type(value)is not dict or value.get('sha256')!=_hash({k:v for k,v in value.items()if k!='sha256'}):
        raise ValueError('exact logical document seal required')
    return value


class Guard:
    def __init__(self):self.files={}
    def add(self,path,*,stamp=None,digest=None):
        path=_safe(path);now=_nine(path)
        if (stamp is not None and now!=stamp)or(path in self.files and self.files[path]!=now):
            raise ValueError('accepted source generation changed')
        if digest is not None and(re.fullmatch('[a-f0-9]{64}',str(digest))is None or _sha(path)!=digest):
            raise ValueError('accepted source file SHA differs')
        if _nine(path)!=now:raise ValueError('source changed while reading')
        self.files[path]=now;return path
    def json(self,path,*,digest=None,sealed=False):
        path=self.add(path,digest=digest)
        try:value=json.loads(path.read_text(),parse_constant=lambda _: (_ for _ in()).throw(ValueError('nonfinite JSON')))
        except(OSError,UnicodeError,json.JSONDecodeError)as error:raise ValueError('source JSON unavailable')from error
        self.add(path)
        if type(value)is not dict:raise ValueError('source document required')
        return _sealed(value)if sealed else value
    def check(self):
        if any(_nine(p)!=stamp for p,stamp in self.files.items()):raise ValueError('source generation changed during collection')


def _actor_key(actor):
    try:return tuple(str(actor[k])for k in('pid','start_ticks','uid','sid','pgrp'))
    except(KeyError,TypeError)as error:raise ValueError('complete source process identity required')from error


def _identity():
    pid=os.getpid();fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=fields[19],uid=os.getuid(),sid=os.getsid(0),pgrp=os.getpgrp())


def _closed(actor):
    _actor_key(actor)
    if type(actor['pid'])is not int or actor['pid']<=0 or not str(actor['start_ticks']).isdigit():
        raise ValueError('exact source process handle required')
    path=Path('/proc',str(actor['pid']),'stat')
    try:fields=path.read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return
    if fields[19]==str(actor['start_ticks']):raise ValueError('source process has not physically exited')


def _stage(value):
    if(value.get('status')!='success'or value.get('worker_exit_code')!=0
       or value.get('all_owned_consumers_exited')is not True):raise ValueError('source stage not complete0/closed')
    _closed(value['worker']);_closed(value['guardian'])


def _receipt(guard,root,source_commit,bundle):
    value=guard.json(root/'private/stage-receipt.json')
    if(value.get('bundle_sha256')!=bundle or value.get('source_commit')!=source_commit
       or _hash({k:v for k,v in value.items()if k!='bundle_sha256'})!=bundle):
        raise ValueError('source capsule receipt differs')
    for name,digest in value['files_sha256'].items():
        if Path(name).is_absolute():raise ValueError('relative capsule member required')
        guard.add(root/name,digest=digest)
    return value


def _admit(guard,postexit_root,size,metrics,proof,expected_proof_sha256,scope,upstream):
    if size not in('small','middle','large')or scope not in('full','cpu_engineering'):
        raise ValueError('explicit native size and scope required')
    full=scope=='full';root=_safe(postexit_root)
    p=guard.json(proof,digest=expected_proof_sha256)
    if(p.get('format')!='actual_native_emgformer_complete_size_and_closed_independent_postexit_acceptance_v1'
       or p.get('size')!=size or p.get('scientific_eligible')is not full):raise ValueError('accepted source proof/scope differs')
    report=guard.json(metrics,digest=p['original_report_files_sha256']['metrics.json'],sealed=True)
    training=report['training']
    if(report.get('format')!='original_emgformer_full_trajectory_report_v1'or report.get('size')!=size
       or report.get('scientific_eligible')is not full or report.get('trajectory_pipeline_complete')is not True
       or report.get('full_paper_complete')is not False or report.get('paper_numerical_equivalence_established')is not False
       or report['sha256']!=p['actual_report_sha256']or report['fit_result_sha256']!=p['actual_fit_result_sha256']
       or training['scope']!=scope or training['selected_best']!=p['selected_best']
       or training['completed_epochs']!=p['actual_training_completed_epochs']
       or training['optimizer_executions']!=p['actual_durable_training_updates']):raise ValueError('source selected fit/report differs')
    if(training['selected_best'].get('validated')is not True
       or re.fullmatch('[a-f0-9]{64}',str(training['selected_best'].get('checkpoint_sha256')))is None):
        raise ValueError('actual validation-selected complete checkpoint required')
    if full and(training['completed_epochs']!=200 or training['optimizer_executions']!=25000
       or training['full_training_terminated']is not True or p.get('full_input_payload_files')!=16
       or p.get('full_input_payload_bytes')!=16540457757):raise ValueError('complete original200/full input source required')
    capsule=_receipt(guard,root,p['collector_source_commit'],p['collector_bundle_sha256'])
    if capsule['files_sha256']!=p['collector_files_sha256']:raise ValueError('accepted reader capsule members differ')
    request=guard.json(root/'private/request.json');parent=_safe(request['root']);flow=parent/('private/'+size+'-flow')
    if request['source_commit']!=p['original_source_commit']or request['bundle_sha256']!=p['original_bundle_sha256']:
        raise ValueError('original queue source differs')
    _receipt(guard,parent,p['original_source_commit'],p['original_bundle_sha256'])
    original_request=guard.json(parent/'private/request.json')
    if _safe(original_request['upstream'])!=_safe(upstream):raise ValueError('upstream differs from frozen original source request')
    complete=guard.json(root/('private/'+size+'-actual-postexit-complete.json'))
    extra=guard.json(root/('private/owned-'+size+'-postexit-readback/stage-result.json'))
    if(complete['size']!=size or complete['actual_owned_readback_stage']!=extra
       or extra!=p['additional_independent_reader_stage']or complete['readback_file_sha256']!=p['readback_file_sha256']
       or complete['exported_report_files_sha256']!=p['original_report_files_sha256']
       or complete['scientific_eligible']is not full):raise ValueError('actual completed reader receipt differs')
    _stage(extra);_stage(p['original_size_outer_stage'])
    actors=p['all16_original_size_actors_closed']+p['all2_independent_reader_actors_closed']
    if(len(p['all16_original_size_actors_closed'])!=16 or len(p['all2_independent_reader_actors_closed'])!=2
       or len({_actor_key(a)for a in actors})!=18 or p['all24_original_loader_workers_closed']is not True
       or [_actor_key(a)for a in p['all2_independent_reader_actors_closed']]!=[_actor_key(extra[k])for k in('worker','guardian')]):
        raise ValueError('complete distinct original and reader closure required')
    for actor in actors:_closed(actor)
    r=guard.json(root/('private/'+size+'-postexit-readback.json'),digest=p['readback_file_sha256'])
    if(r.get('format')!='actual_native_emgformer_complete_trajectory_model_free_postexit_v1'or r['size']!=size
       or r['actual_report']!=report or r['scientific_eligible']is not full
       or r['original_source_commit']!=p['original_source_commit']or r['original_bundle_sha256']!=p['original_bundle_sha256']
       or r['original_outer_size_stage']!=p['original_size_outer_stage']
       or r['all16_original_size_actors_closed']!=p['all16_original_size_actors_closed']
       or r['exported_report_files_sha256']!=p['original_report_files_sha256']
       or r['construction_attempts']!={'learned_module':0,'optimizer':0}or r['cuda_initialized']is not False
       or r['new_learned_forwards']!=0 or r['new_optimizer_updates']!=0
       or r['all24_original_loader_workers_closed']is not True or r['all_frozen_source_files_exact']is not True):
        raise ValueError('accepted completed raw readback differs')
    if _actor_key(r['actor'])!=_actor_key(extra['worker']):raise ValueError('actual independent source reader actor differs')
    raw=r['raw_tensor_payload_sha256_physically_verified_by_existing_native_reader']
    if raw!=p['raw_tensor_payload_sha256_physically_verified_by_existing_native_reader']or(full and len(raw)!=18):
        raise ValueError('accepted complete raw hash catalog differs')
    for name,digest in p['original_report_files_sha256'].items():guard.add(root/('private/export/'+size)/name,digest=digest)
    verification=guard.json(flow/'verified/verification.json',digest=p['original_report_files_sha256']['verification.json'],sealed=True)
    for catalog in(verification['graph_generations'],r['raw_and_other_artifact_nine_attributes']):
        for path,stamp in catalog.items():guard.add(path,stamp=stamp)
    for path,digest in r['non_tensor_artifact_files_after_exit_sha256'].items():guard.add(path,digest=digest)
    guard.check()
    return dict(root=root,flow=flow,proof=p,report=report,readback=r,raw_hashes=raw,scope=scope,full=full)


def _reduce(guard,source,kind,upstream):
    import numpy as np
    from egoemg_repro.emgformer_trajectory_results import read_trajectory_packet
    directory=source['flow']/('test-'+kind);value=guard.json(directory/'pass.json',sealed=True)
    n=source['report']['test']['consumed_samples'];full=source['full'];batch=500 if full else 2
    public_statistics=copy.deepcopy(value['statistics'])
    users=sorted(public_statistics['trajectory']['by_user'])
    aliases={user:'user_'+str(index).zfill(3)for index,user in enumerate(users)}
    for name in ('trajectory','aligned_center'):
        public_statistics[name]['by_user']={aliases[user]:cell for user,cell in public_statistics[name]['by_user'].items()}
    if(value.get('directory')!=str(directory)or value['role']!='test'or value['kind']!=kind
       or value['samples']!=n or type(n)is not int or n<=0 or n%2 or(full and n!=2792)
       or value['batch_size']!=batch or value['workers']!=(6 if full else 0)
       or value['precision']!=('bf16-mixed'if full else'32-true')or value['training_scope']!=source['scope']
       or value['require_full']is not full or value['formal_evaluation']is not full
       or value['fit_result_sha256']!=source['report']['fit_result_sha256']
       or value['selected_best']!=source['report']['training']['selected_best']
       or value['actual_optimizer_updates']!=0 or value['all_loader_workers_exited']is not True
       or len(value['loader_workers'])!=value['workers']or public_statistics!=source['report']['test']):
        raise ValueError('original complete test role/policy/fit differs')
    _closed(value['actor'])
    for actor in value['loader_workers']:_closed(actor)
    journal=guard.add(directory/'packets.jsonl',digest=value['packet_journal_sha256'])
    records=[];digests=[];position=0;paired={};left_windows=set()
    for ordinal,line in enumerate(journal.read_text().splitlines()):
        record=_sealed(json.loads(line));path=directory/'raw'/record['file']
        if(record['ordinal']!=ordinal or source['raw_hashes'].get(str(path))!=record['file_sha256']):
            raise ValueError('ordered accepted raw packet required')
        guard.check();packet=read_trajectory_packet(directory/'raw',record,upstream=upstream)
        arrays=packet['arrays'];metadata=packet['metadata'];points=len(metadata['index'])
        if(points!=min(batch,n-position)or metadata['index']!=list(range(position,position+points))
           or metadata['unique']!=metadata['index']or any(metadata['repeat'])
           or packet['alignment']!=source['report']['test']['alignment']):raise ValueError('raw complete index/repeat/alignment differs')
        pred=arrays['prediction'].float().numpy();target=arrays['target'].numpy();mask=arrays['mask'].numpy()
        for j,index in enumerate(metadata['index']):
            hand=metadata['hand'][j];split=metadata['split'][j]
            if hand!=('left'if index<n//2 else'right')or split not in numeric.SPLITS:
                raise ValueError('raw test hand/split differs')
            coordinates=tuple(metadata[key][j]for key in('start','end','center','episode','split','user'))
            key=index%(n//2)
            if index<n//2:
                if coordinates[:4]in left_windows:raise ValueError('duplicate requested source window')
                left_windows.add(coordinates[:4]);paired[key]=coordinates
            elif paired.get(key)!=coordinates:raise ValueError('left/right source window pair differs')
            errors=np.abs(pred[j].astype(np.float64)-target[j].astype(np.float64))[:,mask[j]]
            records.append(dict(key=str(index),hand=hand,split=split,
                trajectory_angle_error_sum_rad=errors.sum(axis=1).tolist(),trajectory_angle_valid_counts=[int(mask[j].sum())]*22,
                private_metadata={key:items[j]for key,items in metadata.items()},sample_input_sha256=packet['sample_input_sha256'][j]))
        position+=points;digests.append(_hash({k:v for k,v in record.items()if k not in('sha256','file_sha256')}));guard.check()
    if position!=n or _hash(digests)!=value['packet_content_sha256']:raise ValueError('complete raw packet content differs')
    return records,value


def _baseline(views,report):
    baseline=report['test']['trajectory'];pooled=views['pooled_valid_angle_time']
    verify_metric(pooled['overall'],baseline['overall'])
    if [row.get('index')for row in baseline['by_joint']]!=list(range(22)):
        raise ValueError('exact original22joint baseline order required')
    for current,old in zip(pooled['angles'],baseline['by_joint']):verify_metric(current,old)
    if len(baseline['by_joint'])!=22:raise ValueError('all22baseline joints required')
    for name,current in pooled['groups'].items():verify_metric(current,baseline['by_joint_group'][name])
    for view,group in(('by_hand','by_hand'),('by_split','by_split')):
        for name,current in views[view].items():
            old=baseline[group].get(name,dict(absolute_error_sum_rad=0.,valid_angle_count=0,mae_deg=None))
            verify_metric(current['statistics']['overall'],old)
    for hand,cells in views['by_hand_split'].items():
        for split,current in cells.items():
            old=baseline['by_hand_split'].get(hand,{}).get(split,dict(absolute_error_sum_rad=0.,valid_angle_count=0,mae_deg=None))
            verify_metric(current['statistics']['overall'],old)


def _population(records):
    fields=('key','hand','split','private_metadata','sample_input_sha256','trajectory_angle_valid_counts')
    return [{key:row[key]for key in fields}for row in records]


def _write(path,value):
    with path.open('x')as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    path.chmod(0o600)


def _schema(value,keys):
    if type(value)is not dict or set(value)!=set(keys):raise ValueError('exact aggregate public schema required')


def _independent_float(actual,expected):
    if expected is None:
        if actual is not None:raise ValueError('independent empty aggregate differs')
    elif(type(actual)not in(float,int)or not math.isfinite(actual)
         or not math.isclose(actual,expected,rel_tol=2e-12,abs_tol=1e-9)):
        raise ValueError('independent joint aggregate differs')


def _independent_views(rows,views):
    """Separate stdlib oracle: sum observations directly inside each group."""
    if type(rows)is not list:raise ValueError('private sample rows required')
    keys=set()
    for row in rows:
        if(type(row)is not dict or type(row.get('key'))is not str or not row['key']or row['key']in keys
           or row.get('hand')not in numeric.HANDS or row.get('split')not in numeric.SPLITS):
            raise ValueError('independent private sample identity differs')
        keys.add(row['key']);errors=row['trajectory_angle_error_sum_rad'];counts=row['trajectory_angle_valid_counts']
        if type(errors)is not list or type(counts)is not list or len(errors)!=22 or len(counts)!=22:
            raise ValueError('independent22angle statistics required')
        if any(type(e)not in(float,int)or not math.isfinite(e)or e<0 or type(c)is not int or c<0 or(c==0 and e!=0)
               for e,c in zip(errors,counts)):raise ValueError('independent finite sample statistics required')
    def metric(selected,indices):
        indices=list(indices)
        error=math.fsum(row['trajectory_angle_error_sum_rad'][j]for row in selected for j in indices)
        count=sum(row['trajectory_angle_valid_counts'][j]for row in selected for j in indices)
        return dict(absolute_error_sum_rad=error,valid_angle_count=count,mae_deg=error/count*180/math.pi if count else None)
    def check_metric(actual,expected,extra=()):
        _schema(actual,('absolute_error_sum_rad','valid_angle_count','mae_deg',*extra))
        if type(actual['valid_angle_count'])is not int or actual['valid_angle_count']!=expected['valid_angle_count']:
            raise ValueError('independent joint count differs')
        _independent_float(actual['absolute_error_sum_rad'],expected['absolute_error_sum_rad'])
        _independent_float(actual['mae_deg'],expected['mae_deg'])
    def check_table(selected,table):
        _schema(table,('overall','angles','groups','aggregation'))
        if table['aggregation']!='sum radians absolute errors / valid angle-time count; overlapping groups are not additive':
            raise ValueError('exact aggregate public schema description required')
        check_metric(table['overall'],metric(selected,range(22)))
        if type(table['angles'])is not list or len(table['angles'])!=22:raise ValueError('public schema requires22angles')
        for j,angle in enumerate(table['angles']):
            check_metric(angle,metric(selected,[j]),('index',))
            if type(angle['index'])is not int or angle['index']!=j:raise ValueError('independent angle order differs')
        _schema(table['groups'],numeric.JOINT_GROUPS)
        for name,indices in numeric.JOINT_GROUPS.items():
            value=table['groups'][name];check_metric(value,metric(selected,indices),('indices',))
            if value['indices']!=indices:raise ValueError('independent group mapping differs')
    def check_cell(selected,cell):
        _schema(cell,('n_requested','n_with_any_valid_angle','statistics'))
        if(type(cell['n_requested'])is not int or cell['n_requested']!=len(selected)
           or type(cell['n_with_any_valid_angle'])is not int
           or cell['n_with_any_valid_angle']!=sum(any(row['trajectory_angle_valid_counts'])for row in selected)):
            raise ValueError('independent requested window counts differ')
        check_table(selected,cell['statistics'])
    _schema(views,('format','n_requested','n_with_any_valid_angle','pooled_valid_angle_time','by_hand','by_split','by_hand_split',
                  'caption_interpretation_split_sample_weighted','source_or_training_completion_certified','paper_numerical_equivalence_established','full_paper_complete'))
    if(views['format']!='native_joint_group_distinct_weight_views_v1'
       or views['source_or_training_completion_certified']is not False
       or views['paper_numerical_equivalence_established']is not False or views['full_paper_complete']is not False):
        raise ValueError('exact aggregate public schema semantics required')
    check_cell(rows,dict(n_requested=views['n_requested'],n_with_any_valid_angle=views['n_with_any_valid_angle'],statistics=views['pooled_valid_angle_time']))
    _schema(views['by_hand'],numeric.HANDS);_schema(views['by_split'],numeric.SPLITS);_schema(views['by_hand_split'],numeric.HANDS)
    for hand in numeric.HANDS:
        check_cell([row for row in rows if row['hand']==hand],views['by_hand'][hand]);_schema(views['by_hand_split'][hand],numeric.SPLITS)
        for split in numeric.SPLITS:
            check_cell([row for row in rows if row['hand']==hand and row['split']==split],views['by_hand_split'][hand][split])
    split_rows={split:[row for row in rows if row['split']==split]for split in numeric.SPLITS}
    for split,selected in split_rows.items():check_cell(selected,views['by_split'][split])
    caption=views['caption_interpretation_split_sample_weighted'];_schema(caption,('definition','sample_weight_denominator','groups'))
    if(caption['definition']!='sum(split requested hand-windows * split pooled group MAE) / all requested hand-windows; not a verified author Table12 script'
       or type(caption['sample_weight_denominator'])is not int or caption['sample_weight_denominator']!=len(rows)):
        raise ValueError('exact aggregate public schema weighting definition required')
    _schema(caption['groups'],numeric.JOINT_GROUPS)
    for name,indices in numeric.JOINT_GROUPS.items():
        value=caption['groups'][name];_schema(value,('weighted_mae_sum_deg_samples','mae_deg','missing_nonempty_splits'))
        means={split:metric(selected,indices)['mae_deg']for split,selected in split_rows.items()if selected}
        missing=[split for split,mean in means.items()if mean is None]
        numerator=None if missing else math.fsum(len(split_rows[split])*mean for split,mean in means.items())
        expected=numerator/len(rows)if rows and not missing else None
        if value['missing_nonempty_splits']!=missing:raise ValueError('independent undefined split groups differ')
        _independent_float(value['weighted_mae_sum_deg_samples'],numerator);_independent_float(value['mae_deg'],expected)


def verify_pending_native_joint_analysis(*,private_records,pending,expected_private_sha256,expected_pending_sha256,output):
    if'torch'in sys.modules or'numpy'in sys.modules:raise ValueError('fresh standard-library-only verifier required')
    output=_safe(output,exists=False)
    if output.exists():raise ValueError('fresh verification output required')
    guard=Guard();guard.add(__file__,digest=IMPORTED_SHA);guard.add(numeric.__file__);guard.add(ROOT/'egoemg_repro/paper_analysis.py')
    private=guard.json(private_records,digest=expected_private_sha256)
    public=guard.json(pending,digest=expected_pending_sha256,sealed=True)
    _schema(public,('format','size','scope','views','source_run_scientific_eligible','analysis_reader_complete',
        'native_replay_records_equal','baseline_counts_and_metrics_match','fit_result_sha256','selected_checkpoint_sha256',
        'source_report_sha256','accepted_source_proof_sha256','source_generation_count','source_generation_signature_sha256',
        'new_learned_forwards','new_optimizer_updates','construction_attempts','cuda_initialized','paper_numerical_equivalence_established',
        'full_paper_complete','private_records_file_sha256','request_population_sha256','sha256'))
    if(private['format']!='private_native_joint_sample_statistics_v1'or public['format']!='native_joint_group_raw_analysis_pending_v1'
       or public['size']!=private['size']or public['size']not in('small','middle','large')
       or public['scope']!=private['scope']or public['scope']not in('full','cpu_engineering')
       or public['source_run_scientific_eligible']is not(public['scope']=='full')or public['analysis_reader_complete']is not False
       or public['native_replay_records_equal']is not True or public['baseline_counts_and_metrics_match']is not True
       or public['private_records_file_sha256']!=expected_private_sha256
       or public['accepted_source_proof_sha256']!=private['expected_source_proof_sha256']
       or public['source_generation_count']!=len(private['source_generations'])
       or public['source_generation_signature_sha256']!=_hash(private['source_generations'])
       or public['construction_attempts']!={'learned_module':0,'optimizer':0}
       or public['new_learned_forwards']!=0 or public['new_optimizer_updates']!=0 or public['cuda_initialized']is not False
       or public['paper_numerical_equivalence_established']is not False or public['full_paper_complete']is not False):
        raise ValueError('independent pending analysis identity differs')
    for name in('fit_result_sha256','selected_checkpoint_sha256','source_report_sha256','accepted_source_proof_sha256','request_population_sha256'):
        if re.fullmatch('[a-f0-9]{64}',str(public[name]))is None:raise ValueError('public schema requires actual source hashes')
    _closed(private['producer_actor'])
    for path,stamp in private['source_generations'].items():guard.add(path,stamp=stamp)
    _independent_views(private['records'],public['views'])
    if public['request_population_sha256']!=_hash(_population(private['records'])):
        raise ValueError('independent request population signature differs')
    guard.check()
    value=dict(format='independent_stdlib_native_joint_verification_v1',size=public['size'],scope=public['scope'],
        pending_file_sha256=expected_pending_sha256,private_records_file_sha256=expected_private_sha256,
        all_cells_independently_recomputed=True,producer_actor_physically_closed=True,source_generations_unchanged=True,
        actual_verifier_actor=_identity(),stdlib_only='torch'not in sys.modules and'numpy'not in sys.modules,
        analysis_reader_complete=False,new_learned_forwards=0,new_optimizer_updates=0,full_paper_complete=False)
    value['sha256']=_hash(value);output.parent.mkdir(mode=0o700,parents=True,exist_ok=True);_write(output,value);guard.check()
    return value


def collect_native_joint_analysis(*,postexit_root,size,metrics,proof,expected_proof_sha256,upstream,output,scope='full'):
    output=_safe(output,exists=False)
    if output.exists():raise ValueError('fresh analysis output required')
    guard=Guard();guard.add(__file__,digest=IMPORTED_SHA);guard.add(numeric.__file__);guard.add(ROOT/'egoemg_repro/paper_analysis.py')
    try:source=_admit(guard,postexit_root,size,metrics,proof,expected_proof_sha256,scope,upstream)
    except(KeyError,TypeError,OSError,IndexError)as error:raise ValueError('complete accepted source admission required')from error
    import torch
    if torch.cuda.is_initialized():raise ValueError('CPU reader requires an uninitialized CUDA runtime')
    torch.set_num_threads(2);attempts=dict(learned_module=0,optimizer=0)
    original_module=torch.nn.Module.__init__;original_optimizer=torch.optim.Optimizer.__init__
    def deny_module(*args,**kwargs):attempts['learned_module']+=1;raise RuntimeError('joint reader cannot construct learned models')
    def deny_optimizer(*args,**kwargs):attempts['optimizer']+=1;raise RuntimeError('joint reader cannot construct optimizers')
    torch.nn.Module.__init__=deny_module;torch.optim.Optimizer.__init__=deny_optimizer
    try:
        first,a=_reduce(guard,source,'native',upstream);second,b=_reduce(guard,source,'replay',upstream)
        if(first!=second or a['packet_content_sha256']!=b['packet_content_sha256']
           or _actor_key(a['actor'])==_actor_key(b['actor'])):raise ValueError('independent native/replay raw rows differ')
        views=numeric.native_joint_group_views(first);_baseline(views,source['report'])
        guard.check()
        if attempts!={'learned_module':0,'optimizer':0}or torch.cuda.is_initialized():raise ValueError('model-free CPU reader violated')
        private=dict(format='private_native_joint_sample_statistics_v1',size=size,scope=scope,records=first,
            source_generations={str(path):stamp for path,stamp in guard.files.items()},expected_source_proof_sha256=expected_proof_sha256,
            producer_actor=_identity())
        result=dict(format='native_joint_group_raw_analysis_pending_v1',size=size,scope=scope,views=views,
            source_run_scientific_eligible=source['full'],analysis_reader_complete=False,
            native_replay_records_equal=True,baseline_counts_and_metrics_match=True,
            fit_result_sha256=source['report']['fit_result_sha256'],selected_checkpoint_sha256=source['report']['training']['selected_best']['checkpoint_sha256'],
            source_report_sha256=source['report']['sha256'],accepted_source_proof_sha256=expected_proof_sha256,
            source_generation_count=len(guard.files),source_generation_signature_sha256=_hash(private['source_generations']),
            request_population_sha256=_hash(_population(first)),
            new_learned_forwards=0,new_optimizer_updates=0,construction_attempts=attempts,cuda_initialized=False,
            paper_numerical_equivalence_established=False,full_paper_complete=False)
        output.mkdir(mode=0o700,parents=True)
        _write(output/'private-records.json',private);result['private_records_file_sha256']=_sha(output/'private-records.json')
        result['sha256']=_hash(result);_write(output/'analysis.pending.json',result);guard.check()
        return result
    finally:
        torch.nn.Module.__init__=original_module;torch.optim.Optimizer.__init__=original_optimizer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    if len(sys.argv)>1 and sys.argv[1]=='verify':
        for name in('private-records','pending','expected-private-sha256','expected-pending-sha256','output'):parser.add_argument('--'+name,required=True)
        value=verify_pending_native_joint_analysis(**vars(parser.parse_args(sys.argv[2:])))
        print(json.dumps(dict(size=value['size'],stdlib_only=value['stdlib_only'],analysis_reader_complete=False)))
    else:
        for name in('postexit-root','size','metrics','proof','expected-proof-sha256','upstream','output'):parser.add_argument('--'+name,required=True)
        parser.add_argument('--scope',choices=['full','cpu_engineering'],default='full')
        value=collect_native_joint_analysis(**vars(parser.parse_args()))
        print(json.dumps(dict(size=value['size'],scope=value['scope'],samples=value['views']['n_requested'],analysis_reader_complete=False,new_learned_forwards=0,new_optimizer_updates=0)))
