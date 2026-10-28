"""Read-only, complete selected-checkpoint joint/gesture disaggregation.

Private records retain sample metadata and both error definitions. Public
output contains aggregates/identities only. Existing legacy training is frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from egoemg_repro import audit,train
from egoemg_repro.data import file_sha256
from egoemg_repro.metrics import AngleAccumulator
from egoemg_repro.paper_analysis import joint_group_table,verify_metric

RECIPES={
    'emg-s':('emg','emgformer/egoemg_emgformer_small'),
    'emg-m':('emg','emgformer/egoemg_emgformer_middle'),
    'emg-l':('emg','emgformer/egoemg_emgformer_large'),
    'rn18':('vision','fusion/vision_resnet18'),
    'fusion-rn18-s':('fusion','fusion/fusion_rn18_s_center_8ch'),
}


def _json(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def _model_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest(value):
    digest=hashlib.sha256()
    digest.update(str(value.dtype).encode());digest.update(str(list(value.shape)).encode())
    digest.update(np.ascontiguousarray(value).tobytes());return digest.hexdigest()


def _trajectory_agreement(current,baseline):
    if current['mode']!='trajectory' or baseline['mode']!='trajectory':
        raise ValueError('actual whole-trajectory baseline required')
    verify_metric(current['overall'],baseline['overall'])
    for field in ('by_hand','by_split','by_component','by_user','by_hand_split'):
        if set(current[field])!=set(baseline[field]):raise ValueError('trajectory grouping changed')
        for key,value in current[field].items():
            if field=='by_hand_split':
                if set(value)!=set(baseline[field][key]):raise ValueError('trajectory hand/split changed')
                for split,metric in value.items():verify_metric(metric,baseline[field][key][split])
            else:verify_metric(value,baseline[field][key])


def run(args):
    import torch
    from torch.utils.data import DataLoader
    from egoemg_repro.crops import CropStore
    run_path=Path(args.run).resolve();output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('immutable collection output already exists')
    for name in ('run','data','upstream','crops'):
        protected=Path(getattr(args,name)).resolve()
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError('analysis output overlaps immutable input')
    audit.require_completed_summary(train.load_config(run_path/'summary.json'))
    checkpoint_path=run_path/'last.pt';checkpoint_stat=audit._checkpoint_identity(checkpoint_path)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    evidence=audit.validate_artifacts(run_path,checkpoint,device=args.device)
    config=evidence['config'];kind=train.model_kind(config);provenance=evidence['provenance']
    if RECIPES.get(args.label)!=(kind,config['upstream_experiment']):
        raise ValueError('model label does not match the actual frozen recipe')
    current={'source':train.validate_upstream(args.upstream),'data':train.data_identity(args.data)}
    crop_identity=train.visual_data_identity(args.data,args.crops,evidence['manifest'],allow_missing_episodes=False)
    if kind!='emg':current['visual_data']=crop_identity
    train.assert_compatible_provenance(provenance,current)
    checkpoint_sha=file_sha256(checkpoint_path);summary_sha=file_sha256(run_path/'summary.json')
    identities=audit.public_identities(provenance,checkpoint_sha256=checkpoint_sha,
        summary_sha256=summary_sha,crop_identity=crop_identity)
    producers={name:file_sha256(ROOT/name) for name in ('scripts/paper/collect_joint_gesture_errors.py',
        'egoemg_repro/paper_analysis.py','egoemg_repro/audit.py')}
    identities['collection_runtime_sha256']=audit._hash(producers)
    train.seed_everything(config['seed']);torch.set_float32_matmul_precision(evidence['matmul_precision'])
    model=train.build_training_model(args.upstream,config,resume_state=evidence['best']['model'])
    model.load_state_dict(evidence['best']['model'],strict=True);model=model.to(args.device).eval()
    model_digest=_model_digest(model);del checkpoint
    datasets=train.build_datasets(args.data,evidence['manifest'],evidence['norm'],config,crops=args.crops)
    if not all(len(datasets[role]) for role in ('train','val','test')):
        raise ValueError('all three retained roles must be nonempty')
    dataset=datasets['test'];store=dataset.store
    for field,dtype in (('label_gesture_class',np.dtype('<i4')),('label_gesture_active',np.dtype(bool))):
        specification=store.fields[field]
        if tuple(specification['shape'])!=(store.total_rows,) or np.dtype(specification['dtype'])!=dtype:
            raise ValueError('actual gesture field schema differs')
    gesture_classes=store.field('label_gesture_class');gesture_active=store.field('label_gesture_active')
    # Bind every retained data file's actual inode/size/modification identity;
    # producer transfer digests remain separately bound in the old provenance.
    data_files=[path for path in Path(args.data).iterdir() if path.is_file()]
    data_stamps={path:audit._checkpoint_identity(path) for path in data_files}
    def unchanged():
        if audit._checkpoint_identity(checkpoint_path)!=checkpoint_stat:
            raise ValueError('selected-checkpoint source changed during collection')
        if any(audit._checkpoint_identity(path)!=stamp for path,stamp in data_stamps.items()):
            raise ValueError('retained data changed during collection')
        if any(file_sha256(ROOT/name)!=digest for name,digest in producers.items()):
            raise ValueError('analysis producer changed during collection')
    all_centers=audit.GroupedCenterAccumulator();crop_centers=audit.GroupedCenterAccumulator()
    populations=audit.CenterPopulation(provenance['partition_sha256']);trajectory=AngleAccumulator('trajectory') if kind=='emg' else None
    reader=CropStore(args.crops) if kind=='emg' else None
    loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,num_workers=config['num_workers'],
        pin_memory=str(args.device).startswith('cuda'),generator=torch.Generator().manual_seed(config['seed']),drop_last=False)
    records=[];offset=0;batches=0
    with torch.no_grad():
        for batch in loader:
            unchanged();batch=train.move_batch(batch,args.device)
            with train.precision_context(args.device,config):
                if kind=='emg':
                    raw=model(batch);start=model.left_context;stop=None if not model.right_context else -model.right_context
                    target=batch['joint_angles'][...,start:stop];valid=batch['angle_valid_mask'][...,start:stop].bool()
                    prediction=model.align_predictions(raw,target.shape[-1]);center=batch['emg'].shape[-1]//2-start
                    literal_prediction=raw[:,:,raw.shape[-1]//2]
                else:
                    prediction,target,valid=train.aligned_training_batch(model,batch,config);center=0
                    literal_prediction=prediction[:,:,0]
            if prediction.shape!=target.shape or valid.shape!=target.shape or not 0<=center<prediction.shape[-1]:
                raise ValueError('actual original prediction/target/mask alignment differs')
            literal=(literal_prediction.float()-target[:,:,center].float()).abs().mean(dim=1).cpu().tolist()
            prediction,target,valid=(value.float().cpu().numpy() if value.dtype!=torch.bool else value.cpu().numpy()
                for value in (prediction,target,valid))
            for i in range(len(prediction)):
                window=dataset.windows[offset//2];hand=audit.HANDS[offset%2]
                if hand!=batch['hand'][i] or window['split']!=batch['split'][i]:
                    raise ValueError('collection differs from frozen paired center ordering')
                subject=batch['subject'][i];user=str(subject.item() if hasattr(subject,'item') else subject)
                metadata=dict(hand=hand,split=window['split'],user=user)
                p=prediction[i,:,center:center+1];t=target[i,:,center:center+1];mask=valid[i,:,center:center+1]
                present=(audit.crop_present_at_center(store,window,reader,hand) if kind=='emg' else
                    bool(batch['vision_valid_mask'][i].bool().any().item()))
                all_centers.update(p,t,mask,**metadata)
                if present:crop_centers.update(p,t,mask,**metadata)
                populations.update(center=window['center'],hand=hand,split=window['split'],crop_present=present)
                error=np.abs(p.astype(np.float64)-t.astype(np.float64))
                if not np.isfinite(error[mask]).all() or not np.isfinite(literal[i]):
                    raise ValueError('nonfinite selected-model errors')
                gid=int(gesture_classes[window['center']])
                if not -1<=gid<60:raise ValueError('unknown actual gesture label')
                record=dict(key=audit._hash(dict(partition_sha256=provenance['partition_sha256'],
                    center=int(window['center']),hand=hand)),hand=hand,split=window['split'],
                    gesture_class=gid,active=bool(gesture_active[window['center']]),crop_present=present,
                    target_sha256=_tensor_digest(t),center_angle_error_sum_rad=np.where(mask,error,0.).sum(axis=1).tolist(),
                    center_angle_valid_counts=mask.sum(axis=1).tolist(),official_unmasked_mean_error_rad=float(literal[i]),
                    private_subject=user,private_center_row=int(window['center']))
                if trajectory is not None:
                    trajectory.update(prediction[i],target[i],valid[i],**metadata)
                    full_error=np.abs(prediction[i].astype(np.float64)-target[i].astype(np.float64))
                    if not np.isfinite(full_error[valid[i]]).all():raise ValueError('nonfinite valid trajectory errors')
                    record.update(trajectory_angle_error_sum_rad=np.where(valid[i],full_error,0.).sum(axis=1).tolist(),
                        trajectory_angle_valid_counts=valid[i].sum(axis=1).tolist())
                records.append(record);offset+=1
            batches+=1
            print(json.dumps(dict(phase='selected-checkpoint-disaggregation',model=args.label,completed_samples=offset)),flush=True)
    if offset!=len(dataset) or batches!=len(loader) or batches!=evidence['summary']['test']['evaluated_batches']:
        raise ValueError('did not collect the entire frozen test population')
    metrics=all_centers.report();crop_metrics=crop_centers.report();saved=evidence['summary']['test']
    agreement=audit.verify_baseline(metrics,saved['sampling']['center'])
    if kind!='emg':
        audit.verify_baseline(crop_metrics,saved['paired_population']['vision_valid'])
        counts=populations.report()['overall']
        if saved['paired_population']['samples']!=dict(all=counts['requested_centers'],
                vision_valid=counts['crop_present_centers'],missing_crop=counts['missing_crop_centers']):
            raise ValueError('saved crop-present population differs')
    joint=joint_group_table(records) if trajectory is not None else None
    if trajectory is not None:
        _trajectory_agreement(trajectory.report(),saved['sampling']['trajectory'])
        verify_metric(joint['overall'],saved['sampling']['trajectory']['overall'])
    unchanged()
    if _model_digest(model)!=model_digest or file_sha256(checkpoint_path)!=checkpoint_sha or file_sha256(run_path/'summary.json')!=summary_sha:
        raise ValueError('model or completed source artifacts changed during collection')
    train.assert_compatible_provenance(provenance,{'data':train.data_identity(args.data)})
    result=dict(format='egoemg_retained_joint_gesture_collection_v1',model_label=args.label,model_kind=kind,
        identities=identities,selected_checkpoint_step=evidence['best']['global_step'],
        baseline_verification=agreement,trajectory_baseline_verified=True if trajectory is not None else None,
        selected_model_state_sha256=model_digest,model_state_unchanged=True,checkpoint_unchanged=True,
        complete_test_pass=True,samples=offset,evaluated_batches=batches,populations=populations.report(),
        all_center=metrics['overall'],crop_present_center=crop_metrics['overall'],joint_groups=joint,
        precision=evidence['precision'],matmul_precision=evidence['matmul_precision'],batch_size=config['batch_size'],
        protocol='retained declared legacy; official unmasked raw-token midpoint formula separately recorded',
        training_performed=False,full_paper_complete=False)
    output.mkdir(mode=0o700,parents=True)
    _json(output/'records.json',dict(format='private_retained_joint_gesture_records_v1',model_label=args.label,records=records))
    _json(output/'public-result.json',result)
    _json(output/'completion.json',dict(format='egoemg_retained_joint_gesture_collection_completion_v1',
        records_sha256=file_sha256(output/'records.json'),public_result_sha256=file_sha256(output/'public-result.json'),
        selected_checkpoint_sha256=checkpoint_sha,complete_test_pass=True,full_paper_complete=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','upstream','crops','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--label',choices=tuple(RECIPES),required=True);parser.add_argument('--device',default='cuda:0')
    value=run(parser.parse_args());print(json.dumps(dict(model=value['model_label'],samples=value['samples'],baseline_agreement=True)))


if __name__=='__main__':main()
