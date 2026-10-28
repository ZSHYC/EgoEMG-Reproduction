"""Exact author initializer recipe; no model or scientific training claim."""
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',Path(__file__).parents[1]/'upstream')).resolve()
EXPERIMENT='emgformer/regression_egoemg_incre_cotrain_wl12000_small'


def api():
    try:return importlib.import_module('egoemg_repro.emg_initializer_literal')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.emg_initializer_literal':pytest.fail('official compatible EMG initializer recipe missing')
        raise


def config():
    pytest.importorskip('hydra');return api().compose_emg_initializer(UPSTREAM)


def test_composition_is_exact_original_recipe_with_only_private_source_binding():
    pytest.importorskip('hydra')
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    m=api();cfg=config()
    with initialize_config_dir(config_dir=str(UPSTREAM/'config'),version_base='1.1'):
        original=compose(config_name='base',overrides=['experiment='+EXPERIMENT])
    OmegaConf.resolve(original);got=OmegaConf.to_container(cfg,resolve=True)
    binding=got.pop(m.CONTEXT_KEY)
    assert got==OmegaConf.to_container(original,resolve=True)
    assert binding['experiment']==EXPERIMENT and binding['upstream']==str(UPSTREAM)


@pytest.mark.parametrize('key,value',[('batch_size',200),('seed',7),('module.decoder.causal',True),
    ('trainer.max_epochs',2),('datamodule.stride',400),('loss_weights.fingertip_distance',.02),
    ('batch_augmentation.random_gain.min_gain',.1)])
def test_any_executed_recipe_mutation_rejects(key,value):
    pytest.importorskip('hydra')
    from omegaconf import OmegaConf
    cfg=config();OmegaConf.update(cfg,key,value)
    with pytest.raises(ValueError,match='config|recipe|binding'):api().emg_initializer_protocol(UPSTREAM,cfg)


def test_actual_native_defaults_loss_aliases_and_complete_safe_source_contract():
    pytest.importorskip('torch');m=api();cfg=config();p=m.emg_initializer_protocol(UPSTREAM,cfg)
    from egoemg.datamodule import make_data_module
    original=make_data_module(cfg)
    assert p['batch_size_by_role']==dict(train=260,val=260,test=260)
    assert p['dataset_repeat']==original.dataset_repeat==1
    assert p['window']==dict(train_length=12000,eval_length=12000,train_stride=1200,eval_stride=12000)
    assert p['effective_loss_weights']=={'mae':1.,'landmark/fingertip':.01}
    assert p['trainer']['max_epochs']==150 and p['trainer']['precision']=='bf16-mixed'
    assert p['optimizer']['lr']==.0005 and p['lr_scheduler']['scheduler']['T_max']==150
    assert len(p['callbacks'])==1 and p['callbacks'][0]['monitor']=='val_mae'
    assert p['callbacks'][0]['mode']=='min' and p['callbacks'][0]['save_top_k']==3
    assert p['module']['featurizer']['conv_blocks'][0]['in_channels']==8
    assert p['module']['decoder']['causal'] is False and p['module']['provide_initial_pos'] is False
    assert 'egoemg/datasets/egoemg_memmap_dataset.py' in p['sources_sha256']
    assert 'config/augmentation/batch_aug.yaml' in p['sources_sha256']
    assert p['scientific_eligible'] is p['production_ready'] is p['full_paper_complete'] is False
    assert '/home/' not in json.dumps(p) and '/data/' not in json.dumps(p)


def test_native_normalization_selects_each_hands_actual_float32_channel_vectors():
    pytest.importorskip('torch');m=api();config();got=m.emg_initializer_normalization(UPSTREAM)
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    for hand in ('left','right'):
        view=SimpleNamespace(norm_mode='per-dataset',norm_stats_path=str(UPSTREAM/m.NORM_ASSET),
            dataset_name='egoemg',emg_field_preference='filtered_paper',emg_layout='target_hand',target_hand=hand)
        EgoEmgMemmapDataset._load_norm_stats(view)
        assert np.asarray(got['hands'][hand]['mean'],np.float32).tobytes()==view._emg_mean.tobytes()
        assert np.asarray(got['hands'][hand]['std'],np.float32).tobytes()==view._emg_std.tobytes()
        assert len(got['hands'][hand]['mean'])==len(got['hands'][hand]['std'])==8
    assert got['hands']['left']['std']!=got['hands']['right']['std']
    assert got['asset_sha256']=='d4b73febf386ec7b2028639eed8bbd7363d13cbe3fe489149915bcef071ee96a'
    assert got['estimated_from_project_train_rows'] is False and got['epsilon']==1e-6
    assert '/home/' not in json.dumps(got) and got['scientific_eligible'] is False
