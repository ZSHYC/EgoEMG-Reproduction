"""Native factories, immutable source binding and real model behavior."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('pytorch_lightning')
pytest.importorskip('hydra')
pytest.importorskip('h5py')
import numpy as np
from omegaconf import OmegaConf

from test_emg2pose_runtime import corpus, UPSTREAM
from test_emg2pose_memory_data import make_tar, sealed


def api():
    return importlib.import_module('egoemg_repro.emg2pose_memory_runtime')


@pytest.fixture(scope='module')
def memory(corpus):
    from egoemg_repro import emg2pose_memory_index as index, emg2pose_memory_data as data
    from egoemg_repro import emg2pose_data as storage
    raw = make_tar([('corpus/'+p.name,tarfile.REGTYPE,p.read_bytes())
                    for p in sorted(corpus.root.glob('*.hdf5'))])
    location = corpus.root/'memory-index'
    with sealed(raw) as fd:
        inv = data.inspect_archive(fd,expected_bytes=len(raw),expected_sha256=hashlib.sha256(raw).hexdigest(),
                                   metadata_csv=corpus.metadata_csv,upstream=UPSTREAM,cohort='engineering')
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=64*1024**3))
            receipt = index.prepare_index(fd,inv,location,upstream=UPSTREAM,metadata_csv=corpus.metadata_csv)
        yield SimpleNamespace(fd=fd,raw=raw,index=location,receipt=receipt,corpus=corpus)


@pytest.fixture
def source(memory):
    from egoemg_repro.emg2pose_memory_index import open_source
    with open_source(memory.fd,memory.index,expected_completion_sha256=memory.receipt['completion_sha256'],
                     upstream=UPSTREAM,metadata_csv=memory.corpus.metadata_csv) as handle:
        yield handle


def config(memory,source,size='small'):
    return api().compose_config(UPSTREAM,size,memory.index,source=source,metadata_csv=memory.corpus.metadata_csv)


@pytest.mark.parametrize(('size','parameters'),[('small',3481636),('middle',6638884),('large',16342308)])
def test_actual_native_factories_models_windows_and_seed_match_original(memory,source,size,parameters):
    from egoemg_repro import emg2pose_runtime as old
    from egoemg_repro.emg2pose_check import _rng_state,_restore_rng
    from test_emg2pose_memory_dataset import equal
    cfg = config(memory,source,size)
    module,dm = api().build_runtime(UPSTREAM,cfg,source=source)
    original_cfg = old.compose_config(UPSTREAM,size,memory.corpus.data,
                                     source_receipt=memory.corpus.source_receipt)
    reference,gold = old.build_runtime(UPSTREAM,original_cfg)
    try:
        assert type(module).__module__ == 'egoemg.lightning'
        assert sum(p.numel() for p in module.model.parameters()) == parameters
        assert (module.model.left_context,module.model.right_context,module.model.out_channels) == (510,0,20)
        assert all(torch.equal(v,reference.state_dict()[k]) for k,v in module.state_dict().items())
        assert OmegaConf.to_container(cfg.loss_weights) == OmegaConf.to_container(original_cfg.loss_weights)
        dm.setup();gold.setup()
        for role in ('train','val','test'):
            new,original = getattr(dm,role+'_dataset'),getattr(gold,role+'_dataset')
            assert type(new).__name__ == 'SealedTarDataset'
            assert len(new) == len(original) > 0
            for i in range(len(original)):
                before = _rng_state(torch.device('cpu'))
                expected = original[i]
                after = _rng_state(torch.device('cpu'))
                _restore_rng(before,torch.device('cpu'))
                equal(new[i],expected)
                actual = _rng_state(torch.device('cpu'))
                assert actual['python'] == after['python']
                assert np.array_equal(actual['numpy'][1],after['numpy'][1])
                assert actual['numpy'][2:] == after['numpy'][2:]
                assert torch.equal(actual['torch'],after['torch'])
        batch = dm._collate_fn([dm.val_dataset[0]])
        module.eval();reference.eval()
        with torch.inference_mode():
            actual,expected = module(batch),reference(batch)
        assert [tuple(v.shape) for v in actual] == [(1,20,7280),(1,20,7280),(1,7280)]
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))
        optimizer = module.configure_optimizers()
        assert type(optimizer['optimizer']) is torch.optim.AdamW
        assert type(optimizer['lr_scheduler']['scheduler']) is torch.optim.lr_scheduler.CosineAnnealingLR
        assert optimizer['lr_scheduler']['scheduler'].T_max == 250
    finally:
        api().close_data_module(dm)


@pytest.mark.parametrize('fault',['model','loss','optimizer','seed','checkpoint','target','descriptor',
                                  'proof','secondary_collection','location','norm'])
def test_changed_recipe_or_concrete_descriptor_is_rejected_before_rng_changes(memory,source,fault):
    cfg = config(memory,source)
    if fault == 'model':cfg.module.out_channels = 22
    elif fault == 'loss':cfg.loss_weights.mae = .5
    elif fault == 'optimizer':cfg.optimizer.lr = .125
    elif fault == 'seed':cfg.seed = 7
    elif fault == 'checkpoint':cfg.pretrained_checkpoint = '/private/other.pt'
    elif fault == 'target':cfg.dataset.val[0]._target_ = 'egoemg.datasets.emg2pose_dataset.Emg2PoseDataset'
    elif fault == 'descriptor':cfg.dataset.val[0].archive_fd += 1
    elif fault == 'proof':cfg.dataset.val[0].archive_handle_identity.inode += 1
    elif fault == 'secondary_collection':cfg.datamodule.dataset_conf.val[0].archive_fd += 1
    elif fault == 'location':cfg.dataset.val[0].memmap_dir = '/private/other-index'
    else:cfg.datamodule.per_dataset_norm_stats_path = '/private/unbound-norm.json'
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):api().build_runtime(UPSTREAM,cfg,source=source)
    assert torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('fault',['bare_descriptor','closed','wrong_index','csv'])
def test_composition_requires_actual_live_verified_source_and_bound_controls(memory,source,tmp_path,fault):
    candidate,index,csv = source,memory.index,memory.corpus.metadata_csv
    if fault == 'bare_descriptor':candidate = source.fileno()
    elif fault == 'closed':source.close()
    elif fault == 'wrong_index':
        index = tmp_path/'copy';shutil.copytree(memory.index,index)
    else:
        csv = tmp_path/'different.csv';csv.write_bytes(memory.corpus.metadata_csv.read_bytes()+b'\n')
    with pytest.raises(ValueError):
        api().compose_config(UPSTREAM,'small',index,source=candidate,metadata_csv=csv)


def test_private_index_attribute_cannot_redirect_an_already_verified_handle(memory,source,tmp_path):
    replacement = tmp_path/'replacement';shutil.copytree(memory.index,replacement)
    source.index = replacement
    with pytest.raises(ValueError):
        api().compose_config(UPSTREAM,'small',replacement,source=source,metadata_csv=memory.corpus.metadata_csv)


def test_identity_and_runtime_do_not_rehash_whole_immutable_source(memory,source,monkeypatch):
    cfg = config(memory,source)
    from egoemg_repro import emg2pose_memory_data
    def forbidden(*_):raise AssertionError('large source must be hashed once at admission')
    monkeypatch.setattr(emg2pose_memory_data,'_digest',forbidden)
    first = api().runtime_identity(UPSTREAM,cfg,source=source)
    assert first == api().runtime_identity(UPSTREAM,cfg,source=source)
    module,dm = api().build_runtime(UPSTREAM,cfg,source=source)
    api().close_data_module(dm)
    text = json.dumps(first)
    assert 'private-runtime-user' not in text and str(memory.corpus.root) not in text
    assert 'device' not in first['data'] and 'inode' not in first['data']
    assert first['data']['full_corpus_complete'] is False


def test_identical_relocated_rebuilt_source_preserves_runtime_identity_and_weight_reload(memory,source,tmp_path):
    from egoemg_repro.emg2pose_memory_index import open_source
    cfg = config(memory,source)
    identity = api().runtime_identity(UPSTREAM,cfg,source=source)
    module,dm = api().build_runtime(UPSTREAM,cfg,source=source)
    dm.setup('validate');batch = dm._collate_fn([dm.val_dataset[0]])
    module.eval()
    with torch.inference_mode():expected = module(batch)[0].clone()
    state = {k:v.clone() for k,v in module.state_dict().items()}
    api().close_data_module(dm)
    location = tmp_path/'relocated';shutil.copytree(memory.index,location)
    csv = tmp_path/'metadata.csv';csv.write_bytes(memory.corpus.metadata_csv.read_bytes())
    with sealed(memory.raw) as fd:
        with open_source(fd,location,expected_completion_sha256=memory.receipt['completion_sha256'],
                         upstream=UPSTREAM,metadata_csv=csv) as rebuilt:
            new_cfg = api().compose_config(UPSTREAM,'small',location,source=rebuilt,metadata_csv=csv)
            assert api().runtime_identity(UPSTREAM,new_cfg,source=rebuilt) == identity
            restored,new_dm = api().reload_runtime(UPSTREAM,new_cfg,state,identity=identity,source=rebuilt)
            try:
                restored.eval()
                with torch.inference_mode():assert torch.equal(expected,restored(batch)[0])
            finally:api().close_data_module(new_dm)


@pytest.mark.parametrize('damage',['missing','shape','dtype','identity'])
def test_strict_weight_reload_rejects_partial_state_or_wrong_identity(memory,source,damage):
    cfg = config(memory,source)
    module,dm = api().build_runtime(UPSTREAM,cfg,source=source)
    state = {k:v.clone() for k,v in module.state_dict().items()}
    identity = api().runtime_identity(UPSTREAM,cfg,source=source)
    if damage == 'missing':del state[next(iter(state))]
    elif damage == 'shape':
        key = next(k for k,v in state.items() if v.shape == (20,512));state[key] = torch.zeros(22,512)
    elif damage == 'dtype':
        key = next(k for k,v in state.items() if v.dtype == torch.float32);state[key] = state[key].double()
    else:identity['config_sha256'] = '0'*64
    try:
        with pytest.raises((ValueError,RuntimeError)):
            api().reload_runtime(UPSTREAM,cfg,state,identity=identity,source=source)
    finally:api().close_data_module(dm)


def test_dataset_late_setup_refuses_changed_csv_and_norm_asset_is_bound(memory,source,tmp_path):
    from egoemg_repro.emg2pose_memory_index import open_source
    csv = tmp_path/'bound.csv';csv.write_bytes(memory.corpus.metadata_csv.read_bytes())
    with open_source(memory.fd,memory.index,expected_completion_sha256=memory.receipt['completion_sha256'],
                     upstream=UPSTREAM,metadata_csv=csv) as alternate:
        cfg = api().compose_config(UPSTREAM,'small',memory.index,source=alternate,metadata_csv=csv)
        norm = str(UPSTREAM/'assets/per_dataset_norm_stats.json')
        assert norm in cfg.dataset.val[0].archive_handle_identity.binding_stamps
        _,dm = api().build_runtime(UPSTREAM,cfg,source=alternate)
        csv.write_bytes(csv.read_bytes())
        try:
            with pytest.raises(Exception,match='binding changed'):dm.setup('validate')
        finally:api().close_data_module(dm)


def test_unapproved_normalization_bytes_are_rejected(memory,source,monkeypatch):
    from egoemg_repro import emg2pose_data
    original = emg2pose_data._sha256
    monkeypatch.setattr(emg2pose_data,'_sha256',lambda p:'0'*64 if Path(p)==UPSTREAM/'assets/per_dataset_norm_stats.json' else original(p))
    with pytest.raises(ValueError,match='normalization'):
        config(memory,source)


def test_closing_shared_nested_datasets_releases_each_owned_descriptor(memory,source):
    from torch.utils.data import ConcatDataset,Subset
    _,dm = api().build_runtime(UPSTREAM,config(memory,source),source=source)
    dm.setup()
    leaves = [dm.train_dataset,dm.val_dataset,dm.test_dataset]
    descriptors = [leaf._archive_descriptor for leaf in leaves]
    dm.train_dataset = ConcatDataset([leaves[0],Subset(leaves[1],[0]),leaves[0]])
    api().close_data_module(dm);api().close_data_module(dm)
    for fd in descriptors:
        with pytest.raises(OSError):os.fstat(fd)
    assert source.fileno() >= 0
