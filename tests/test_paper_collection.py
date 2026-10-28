"""Actual original-model collection, with raw-token and frozen-baseline checks."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('kind,force_reduction_roundoff',[
    ('emg',False),('emg',True),('vision',False),('fusion',False)])
def test_original_collection_uses_frozen_centers_and_actual_output_time_semantics(
        tmp_path,kind,force_reduction_roundoff):
    torch=pytest.importorskip('torch');pytest.importorskip('hydra');pytest.importorskip('lmdb')
    upstream=os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:pytest.skip('set EGOEMG_UPSTREAM for original-model collection')
    path=Path(__file__).resolve().parents[1]/'scripts/paper/collect_joint_gesture_errors.py'
    assert path.exists(),'selected-checkpoint collection is missing'
    spec=importlib.util.spec_from_file_location('paper_collection',path)
    collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
    from test_audit import completed_artifacts
    from test_data import make_store,window
    from test_crops import add_video_indices,make_crops
    from egoemg_repro import train
    from egoemg_repro.data import LegacyStore,compute_normalization,file_sha256
    from torch.utils.data import DataLoader
    state=(torch.get_num_threads(),torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
           torch.get_float32_matmul_precision(),random.getstate(),np.random.get_state(),torch.get_rng_state())
    try:
        torch.set_num_threads(2)
        data,crops,run=(tmp_path/name for name in ('data','crops','run'))
        data.mkdir();run.mkdir()
        checkpoint,artifacts=completed_artifacts(run,kind)
        make_store(data,lengths=(36000,));add_video_indices(data,np.full(36000,17));make_crops(crops)
        manifest=json.loads((data/'manifest.json').read_text())
        for name,array in {'label_gesture_class':np.zeros(36000,dtype='<i4'),
                           'label_gesture_active':np.ones(36000,dtype=bool)}.items():
            array.tofile(data/(name+'.dat'));manifest['fields'][name]=dict(filename=name+'.dat',
                dtype=str(array.dtype),shape=list(array.shape))
        (data/'manifest.json').write_text(json.dumps(manifest))
        receipt={'files':{p.name:dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in crops.glob('*.lmdb')}}
        (crops/'acquisition_ledger.json').write_text(json.dumps(receipt))
        partition={'windows':[dict(window(0,12000),split='train'),
            dict(window(12000,24000,role='val'),split='train'),
            dict(window(24000,36000,role='test'),split='user')]}
        norm=compute_normalization(LegacyStore(data),partition['windows'])
        provenance=checkpoint['provenance'];provenance.update(source=train.validate_upstream(upstream),
            data=train.data_identity(data),partition_sha256=train.canonical_hash(partition),
            normalization_sha256=train.canonical_hash(norm))
        config=artifacts['config.json']
        if kind=='fusion':
            initializers={}
            for branch,recipe in (('vision','resnet18'),('emg','emgformer_s')):
                branch_config=json.loads((Path(__file__).resolve().parents[1]/f'configs/{recipe}.json').read_text())
                branch_config['vision_pretrained']=False
                branch_model=train.build_training_model(upstream,branch_config)
                initializer=tmp_path/(branch+'.pt');torch.save({'model':branch_model.state_dict()},initializer)
                initializers[branch+'_checkpoint']=initializer
            model=train.build_training_model(upstream,config,**initializers).eval()
            for initializer in initializers.values():initializer.unlink()
        else:model=train.build_training_model(upstream,config).eval()
        if kind!='emg':provenance['visual_data']=train.visual_data_identity(data,crops,partition)
        checkpoint['model']=model.state_dict();checkpoint['best']['model']=model.state_dict()
        checkpoint['best']['provenance']=copy.deepcopy(provenance)
        dataset=train.build_datasets(data,partition,norm,config,crops=crops)['test']
        saved=train.evaluate(model,DataLoader(dataset,batch_size=2),'cpu',config)
        if force_reduction_roundoff:
            overall=saved['sampling']['trajectory']['overall']
            for key in ('absolute_error_sum_rad','mae_deg'):
                for _ in range(8):overall[key]=math.nextafter(overall[key],math.inf)
        summary=artifacts['summary.json'];summary.update(provenance=provenance,test=saved)
        for name,value in [('provenance.json',provenance),('summary.json',summary),
                           ('partitions.json',partition),('normalization.json',norm)]:
            (run/name).write_text(json.dumps(value))
        torch.save(checkpoint,run/'last.pt')
        with torch.no_grad():
            batch=next(iter(DataLoader(dataset,batch_size=2)));raw=model(batch)
            if isinstance(raw,tuple):raw=raw[0]
            assert raw.shape==(2,22,230 if kind=='emg' else 1)
            # The actual12k original EMG model has230tokens; code midpoint115.
            token,center=(115,6000) if kind=='emg' else (0,0)
            wanted=(raw[:,:,token].float()-batch['joint_angles'][:,:,center]).abs().mean(dim=1).tolist()
        output=tmp_path/'collected'
        args=SimpleNamespace(run=str(run),data=str(data),upstream=upstream,crops=str(crops),
            output=str(output),device='cpu',label={'emg':'emg-s','vision':'rn18','fusion':'fusion-rn18-s'}[kind])
        result=collector.run(args)
        assert result['baseline_verification']['agreement']
        if kind=='emg':
            assert result['trajectory_baseline_verified']
            actual=result['joint_groups']['overall'];expected=saved['sampling']['trajectory']['overall']
            assert set(actual)==set(expected)
            assert actual['valid_angle_count']==expected['valid_angle_count']
            # Per-angle NumPy and pooled Torch sums associate the same float64
            # terms differently. Limit this fixture allowance to roundoff; the
            # production count-exact/1e-4-degree baseline gate stays unchanged.
            for metric in ('absolute_error_sum_rad','mae_deg'):
                assert actual[metric]==pytest.approx(expected[metric],rel=0,abs=64*math.ulp(expected[metric]))
        else:assert result['trajectory_baseline_verified'] is None and result['joint_groups'] is None
        rows=json.loads((output/'records.json').read_text())['records']
        assert len(rows)==2 and [r['crop_present'] for r in rows]==[True,False]
        assert [r['official_unmasked_mean_error_rad'] for r in rows]==pytest.approx(wanted,abs=1e-7)
        if kind=='emg':
            assert rows[0]['trajectory_angle_valid_counts']==[11490]*22
            assert result['joint_groups']['overall']['valid_angle_count']==505560
        else:assert 'trajectory_angle_valid_counts' not in rows[0]
        assert result['populations']['overall']['requested_centers']==2
        assert not any(secret in json.dumps(result) for secret in (str(tmp_path),'subject_0','episode_0'))
        summary['test']['sampling']['center']['overall']['mae_deg']+=1
        (run/'summary.json').write_text(json.dumps(summary));args.output=str(tmp_path/'refused')
        with pytest.raises(ValueError):collector.run(args)
        assert not (Path(args.output)/'completion.json').exists()
    finally:
        torch.set_num_threads(state[0]);torch.backends.cudnn.deterministic=state[1];torch.backends.cudnn.benchmark=state[2]
        torch.set_float32_matmul_precision(state[3]);random.setstate(state[4]);np.random.set_state(state[5]);torch.set_rng_state(state[6])
