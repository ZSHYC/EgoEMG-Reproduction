"""Exact author S/M/L recipes, not the different cotrain initializer."""
import importlib
import os
from pathlib import Path
import numpy as np
import pytest
UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).parents[1]/'upstream')).resolve()


def api():return importlib.import_module('egoemg_repro.emgformer_literal')


@pytest.mark.parametrize('size,layers,heads,dim',[('small',3,4,256),('middle',6,8,256),('large',8,12,384)])
def test_original_resolved_recipe_and_effective_double_fingertip_loss(size,layers,heads,dim):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    cfg=api().compose_emgformer(UPSTREAM,size)
    with initialize_config_dir(config_dir=str(UPSTREAM/'config'),version_base='1.1'):
        original=compose(config_name='base',overrides=['experiment=emgformer/egoemg_emgformer_'+size])
    got=OmegaConf.to_container(cfg,resolve=True);binding=got.pop(api().CONTEXT_KEY)
    assert got==OmegaConf.to_container(original,resolve=True) and binding['size']==size
    p=api().emgformer_protocol(UPSTREAM,cfg)
    assert p['effective_loss_weights']=={'mae':1.,'landmark/fingertip':.02}
    assert p['raw_loss_weights']=={'mae':1,'landmark/fingertip':.01,'fingertip_distance':.01}
    assert p['batch_size']==500 and p['trainer']['max_epochs']==200 and p['optimizer']['lr']==.00005
    assert p['trainer']['precision']=='bf16-mixed' and p['trainer']['gradient_clip_val']==1
    assert p['lr_scheduler']['scheduler']['T_max']==200 and p['lr_scheduler']['scheduler']['eta_min']==1e-5
    assert p['module']['decoder']['num_layers']==layers and p['module']['decoder']['num_heads']==heads
    assert p['module']['decoder']['model_dim']==dim and p['module']['out_channels']==22
    assert p['window']==dict(train_length=12000,eval_length=12000,train_stride=1200,eval_stride=12000)
    assert p['dataset_admitted'] is p['scientific_eligible'] is p['production_ready'] is False


@pytest.mark.parametrize('path,value',[('batch_size',260),('trainer.max_epochs',150),('optimizer.lr',.0005),
    ('loss_weights.fingertip_distance',0.),('module.decoder.causal',True),('seed',7)])
def test_mutating_any_executed_recipe_is_rejected(path,value):
    from omegaconf import OmegaConf
    cfg=api().compose_emgformer(UPSTREAM,'small');OmegaConf.update(cfg,path,value)
    with pytest.raises(ValueError):api().emgformer_protocol(UPSTREAM,cfg)


def test_unknown_size_rejects_before_source_access():
    with pytest.raises(ValueError):api().compose_emgformer(Path('/does-not-exist'),'medium')


def test_native_per_hand_eight_float32_normalization_matches_original_loader():
    from types import SimpleNamespace
    cfg=api().compose_emgformer(UPSTREAM,'small');p=api().emgformer_normalization(UPSTREAM,cfg)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    for hand in ('left','right'):
        view=SimpleNamespace(norm_mode='per-dataset',norm_stats_path=str(UPSTREAM/api().NORM_ASSET),
            dataset_name='egoemg',emg_field_preference='filtered_paper',emg_layout='target_hand',target_hand=hand)
        EgoEmgMemmapDataset._load_norm_stats(view)
        assert np.asarray(p['hands'][hand]['mean'],dtype='f4').tobytes()==view._emg_mean.tobytes()
        assert np.asarray(p['hands'][hand]['std'],dtype='f4').tobytes()==view._emg_std.tobytes()
        assert len(p['hands'][hand]['mean'])==8
    assert p['hands']['left']['std']!=p['hands']['right']['std']
    assert p['estimated_from_project_train_rows'] is False and p['dataset_admitted'] is False
