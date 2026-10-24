"""Locked-checkpoint verification, conditional user statistics and privacy."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('egoemg_repro.audit') is not None, 'locked-checkpoint audit is missing'
    from egoemg_repro import audit
    return audit


def update(meter, value, *, user='private-name', hand='left', split='user', fingers=1, wrist=0):
    prediction = np.full((22, 1), value, dtype=np.float32)
    mask = np.zeros((22, 1), bool)
    mask[:fingers] = True
    mask[20:20+wrist] = True
    meter.update(prediction, np.zeros_like(prediction), mask, hand=hand, split=split, user=user)


def test_unequal_users_have_pooled_and_conditional_unweighted_statistics():
    meter = api().GroupedCenterAccumulator()
    update(meter, 1, user='Alice', fingers=3)
    update(meter, 5, user='Bob')
    update(meter, 9, user='Alice', split='both')
    update(meter, 11, user='Bob', hand='right')
    report = meter.report()
    left_user = report['by_hand_split']['left']['user']
    assert left_user['overall']['valid_angle_count'] == 4
    assert left_user['overall']['mae_deg'] == pytest.approx(2 * 180/math.pi)
    assert left_user['user_summary']['mean_mae_deg'] == pytest.approx(3 * 180/math.pi)
    assert left_user['user_summary']['std_mae_deg'] == pytest.approx(2 * 180/math.pi)
    # Per-user means over BOTH hands in this split: Alice=1; Bob=(5+11)/2=8.
    assert report['by_split']['user']['user_summary']['mean_mae_deg'] == pytest.approx(4.5 * 180/math.pi)
    assert report['by_split']['user']['user_summary']['std_mae_deg'] == pytest.approx(3.5 * 180/math.pi)
    assert set(left_user['by_user']) == {'user-001', 'user-002'}
    assert not any(name in json.dumps(report) for name in ('Alice', 'Bob', 'private-name'))


def test_masks_keep_wrist_independent_and_convert_radians_once():
    meter = api().GroupedCenterAccumulator()
    update(meter, math.pi/2, fingers=0, wrist=2)
    report = meter.report()
    assert report['overall']['mae_deg'] == pytest.approx(90, abs=1e-5)
    assert report['by_component']['fingers']['mae_deg'] is None
    assert report['by_component']['wrist']['valid_angle_count'] == 2
    assert report['user_summary_by_component']['fingers']['user_count'] == 0
    assert report['user_summary_by_component']['wrist']['std_mae_deg'] == 0
    bad = np.full((22, 1), np.nan)
    meter.update(bad, bad, np.zeros((22, 1), bool), hand='left', split='both', user='a')
    with pytest.raises(ValueError, match='nonfinite'):
        meter.update(bad, bad, np.ones((22, 1), bool), hand='left', split='both', user='a')


def test_population_digest_is_order_independent_and_binds_hand_center_partition():
    audit = api()
    first, reverse = audit.CenterPopulation('a'*64), audit.CenterPopulation('a'*64)
    samples = [(10, 'left', 'user', True), (10, 'right', 'user', False), (20, 'left', 'both', True)]
    for center, hand, split, present in samples:
        first.update(center=center, hand=hand, split=split, crop_present=present)
    for center, hand, split, present in reversed(samples):
        reverse.update(center=center, hand=hand, split=split, crop_present=present)
    assert first.report() == reverse.report()
    report = first.report()
    assert report['overall']['requested_centers'] == 3
    assert report['overall']['crop_present_centers'] == 2
    assert report['by_hand_split']['right']['user']['missing_crop_centers'] == 1
    changed = audit.CenterPopulation('b'*64)
    for center, hand, split, present in samples:
        changed.update(center=center, hand=hand, split=split, crop_present=present)
    assert report['overall']['all_sha256'] != changed.report()['overall']['all_sha256']
    changed_center = audit.CenterPopulation('a'*64)
    changed_hand = audit.CenterPopulation('a'*64)
    for center, hand, split, present in samples:
        changed_center.update(center=center+1,hand=hand,split=split,crop_present=present)
        changed_hand.update(center=center,hand='right' if hand=='left' else 'left',split=split,crop_present=present)
    assert report['overall']['all_sha256'] != changed_center.report()['overall']['all_sha256']
    assert report['overall']['crop_present_sha256'] != changed_hand.report()['overall']['crop_present_sha256']
    assert '10' not in json.dumps({k:v for k,v in report['overall'].items() if not k.endswith('sha256')})
    with pytest.raises(ValueError, match='hand|split'):
        first.update(center=10, hand='/private/path', split='user', crop_present=True)


def test_baseline_verification_checks_counts_sums_groups_and_fixed_tolerance():
    audit = api()
    from egoemg_repro.metrics import AngleAccumulator
    old, new = AngleAccumulator('center'), audit.GroupedCenterAccumulator()
    prediction = np.ones((22, 1), np.float32)
    metadata = dict(hand='left', split='user', user='private-name')
    old.update(prediction, np.zeros_like(prediction), np.ones_like(prediction, bool), center_index=0, **metadata)
    new.update(prediction, np.zeros_like(prediction), np.ones_like(prediction, bool), **metadata)
    baseline = old.report()
    assert audit.verify_baseline(new.report(), baseline)['agreement'] is True
    changed = copy.deepcopy(baseline)
    changed['overall']['valid_angle_count'] += 1
    with pytest.raises(ValueError, match='count'):
        audit.verify_baseline(new.report(), changed)
    changed = copy.deepcopy(baseline)
    changed['by_hand']['left']['absolute_error_sum_rad'] += 1
    with pytest.raises(ValueError, match='sum|MAE'):
        audit.verify_baseline(new.report(), changed)
    changed = copy.deepcopy(baseline)
    changed['by_hand_split']['left']['user']['mae_deg'] += .001
    with pytest.raises(ValueError, match='MAE'):
        audit.verify_baseline(new.report(), changed)
    changed = copy.deepcopy(baseline)
    changed['by_split'] = {}
    with pytest.raises(ValueError, match='groups'):
        audit.verify_baseline(new.report(), changed)
    changed = copy.deepcopy(baseline)
    changed['mode'] = 'trajectory'
    with pytest.raises(ValueError, match='center'):
        audit.verify_baseline(new.report(), changed)


def test_empty_crop_population_matches_frozen_empty_metric_schema():
    from egoemg_repro.metrics import AngleAccumulator
    audit = api()
    assert audit.verify_baseline(audit.GroupedCenterAccumulator().report(), AngleAccumulator('center').report())['agreement']


def test_baseline_tolerance_is_predeclared_and_applies_to_consistent_metrics():
    from egoemg_repro.metrics import AngleAccumulator
    audit = api()
    meter, old = audit.GroupedCenterAccumulator(), AngleAccumulator('center')
    prediction, target, valid = np.ones((22,1)), np.zeros((22,1)), np.ones((22,1),bool)
    metadata=dict(hand='left',split='both',user='someone')
    meter.update(prediction,target,valid,**metadata)
    old.update(prediction,target,valid,center_index=0,**metadata)
    original=old.report()
    for delta, permitted in ((.5e-4,True),(2e-4,False)):
        changed=copy.deepcopy(original)
        changed['overall']['mae_deg']+=delta
        changed['overall']['absolute_error_sum_rad']+=math.radians(delta)*22
        if permitted:
            assert audit.verify_baseline(meter.report(),changed)['agreement']
        else:
            with pytest.raises(ValueError,match='sum / MAE'):
                audit.verify_baseline(meter.report(),changed)


@pytest.mark.parametrize('status,test_evaluated,bounded', [
    ('training_complete', False, False), ('bounded_integration_complete', False, True),
    ('training_complete', True, True), ('running', False, False)])
def test_completion_gate_rejects_incomplete_or_bounded(status, test_evaluated, bounded):
    summary = dict(status=status, test_evaluated=test_evaluated, test=dict(bounded=bounded))
    with pytest.raises(ValueError, match='complete|bounded'):
        api().require_completed_summary(summary)


def test_public_identity_whitelist_never_exports_provenance_paths_or_names():
    audit = api()
    provenance = dict(config_sha256='1'*64, partition_sha256='2'*64, normalization_sha256='3'*64,
                      source={'sha256':'4'*64, 'private':'/home/secret'},
                      data={'sha256':'5'*64, 'participant':'Alice'},
                      reproduction={'sha256':'6'*64, 'selected_recipe_sha256':'7'*64},
                      initialization={'policy':'random', 'secret':'/private/weights.pt'})
    result = audit.public_identities(provenance, checkpoint_sha256='8'*64, summary_sha256='9'*64,
                                     crop_identity={'sha256':'a'*64, 'payloads':{'Alice.lmdb':'secret'}})
    rendered = json.dumps(result)
    assert not any(secret in rendered for secret in ('Alice', '/home/', '/private/', 'payloads', 'secret'))
    assert result['partition_sha256'] == '2'*64
    assert result['checkpoint_sha256'] == '8'*64
    assert len(result['initialization_sha256']) == 64


def completed_artifacts(root, kind='emg'):
    """Synthetic evidence for gate tests, never an experimental result."""
    audit = api()
    from egoemg_repro.train import canonical_hash, checkpoint_with_best, reproduction_identity
    recipe_name = {'emg':'emgformer_s', 'vision':'resnet18', 'fusion':'fusion_rn18_s'}[kind]
    config = json.loads(Path(f'configs/{recipe_name}.json').read_text())
    config.update(epochs=1, batch_size=2, num_workers=0, augmentation='none')
    if kind == 'vision':
        config['vision_pretrained'] = False
    recipe = json.dumps(config).encode()
    manifest, norm = {'windows': []}, {'left': {}, 'right': {}}
    provenance = dict(source={'sha256': 'a'*64}, data={'sha256': 'b'*64}, model_kind=kind,
                      reproduction=reproduction_identity(model_kind=kind, recipe_bytes=recipe),
                      recipe={'source_utf8': recipe.decode(), 'effective_config': config},
                      config_sha256=canonical_hash(config), partition_sha256=canonical_hash(manifest),
                      normalization_sha256=canonical_hash(norm), initialization={'policy':'random', 'seed_before_model':config['seed']})
    if kind == 'fusion':
        for branch in ('vision','emg'):
            provenance['initialization'][branch] = dict(sha256='c'*64, model_kind=branch, selection='complete independent validation')
    state = dict(model={'weight':np.zeros(1)}, optimizer={}, scheduler={}, rng={}, epoch=1,
                 next_batch=0, global_step=3, best_val_mae_deg=4., validation_bounded=False,
                 run_id='private-run', provenance=provenance)
    checkpoint = checkpoint_with_best(state)
    summary = dict(status='training_complete', test_evaluated=True, global_step=3, selected_checkpoint_step=3,
                   best_val_mae_deg=4., test={'bounded':False, 'sampling':{'center':{}}}, provenance=provenance)
    artifacts = {'config.json':config, 'partitions.json':manifest, 'normalization.json':norm,
                 'provenance.json':provenance, 'run.json':{'run_id':'private-run'}, 'summary.json':summary,
                 'environment.json':{'precision':'float32', 'matmul_precision':config.get('matmul_precision','high'),
                                     'initialization':provenance['initialization']}}
    for name, value in artifacts.items():
        (root/name).write_text(json.dumps(value))
    (root/'recipe_source.json').write_bytes(recipe)
    return checkpoint, artifacts


def test_saved_artifact_gate_checks_completed_best_original_config_and_initialization(tmp_path):
    audit = api()
    checkpoint, artifacts = completed_artifacts(tmp_path)
    evidence = audit.validate_artifacts(tmp_path, checkpoint, device='cpu')
    assert evidence['best']['global_step'] == 3
    assert evidence['config']['epochs'] == 1
    changed = copy.deepcopy(checkpoint)
    changed['epoch'] = 0
    with pytest.raises(ValueError, match='budget|epoch'):
        audit.validate_artifacts(tmp_path, changed, device='cpu')
    changed = copy.deepcopy(checkpoint)
    changed['checkpoint_schema'] = 1
    with pytest.raises(ValueError, match='self-contained'):
        audit.validate_artifacts(tmp_path, changed, device='cpu')
    (tmp_path/'recipe_source.json').write_text('{}')
    with pytest.raises(ValueError, match='recipe'):
        audit.validate_artifacts(tmp_path, checkpoint, device='cpu')


@pytest.mark.parametrize('artifact,mutate,match', [
    ('config.json', lambda v:v.update(batch_size=1), 'config'),
    ('partitions.json', lambda v:v.update(extra=True), 'partition'),
    ('normalization.json', lambda v:v.update(extra=True), 'normalization'),
    ('run.json', lambda v:v.update(run_id='another'), 'run'),
    ('environment.json', lambda v:v.update(initialization={}), 'initialization'),
    ('environment.json', lambda v:v.update(precision='bfloat16'), 'precision'),
    ('summary.json', lambda v:v.update(selected_checkpoint_step=2), 'selected'),
    ('summary.json', lambda v:v['provenance']['data'].update(sha256='c'*64), 'data'),
])
def test_saved_artifact_gate_rejects_conflicting_evidence(tmp_path, artifact, mutate, match):
    checkpoint, artifacts = completed_artifacts(tmp_path)
    changed = copy.deepcopy(artifacts[artifact])
    mutate(changed)
    (tmp_path/artifact).write_text(json.dumps(changed))
    with pytest.raises(ValueError, match=match):
        api().validate_artifacts(tmp_path, checkpoint, device='cpu')


def test_emg_crop_presence_uses_released_video_index_and_missing_keys(tmp_path):
    pytest.importorskip('lmdb')
    from test_data import make_store, window
    from test_crops import add_video_indices, make_crops
    from egoemg_repro.data import LegacyStore
    from egoemg_repro.crops import CropStore
    data, crops = tmp_path/'data', tmp_path/'crops'
    data.mkdir()
    make_store(data)
    add_video_indices(data, np.arange(47)+15)
    make_crops(crops)
    store, reader = LegacyStore(data), CropStore(crops)
    assert api().crop_present_at_center(store, window(), reader, 'left') is True
    assert api().crop_present_at_center(store, window(), reader, 'right') is False
    with pytest.raises(FileNotFoundError):
        api().crop_present_at_center(store, window(40,44,episode=1), reader, 'left')


@pytest.mark.parametrize('kind', ['emg', 'vision', 'fusion'])
def test_actual_official_fixture_audit_matches_frozen_center_metrics(tmp_path, kind, monkeypatch):
    """Actual official predictions on synthetic arrays; no training claims."""
    import os
    from argparse import Namespace
    torch = pytest.importorskip('torch')
    pytest.importorskip('hydra')
    pytest.importorskip('lmdb')
    upstream = os.environ.get('EGOEMG_UPSTREAM')
    if not upstream:
        pytest.skip('set EGOEMG_UPSTREAM for the actual pinned official-model integration')
    from test_data import make_store, window
    from test_crops import add_video_indices, make_crops
    from egoemg_repro import train
    from egoemg_repro.data import LegacyStore, compute_normalization
    from torch.utils.data import DataLoader
    audit = api()
    torch.set_num_threads(2)
    data, crops, run = (tmp_path/name for name in ('data','crops','run'))
    for path in (data,run):
        path.mkdir()
    checkpoint, artifacts = completed_artifacts(run, kind)
    make_store(data, lengths=(36000,))
    add_video_indices(data, np.full(36000,17))
    make_crops(crops)
    # make_crops deliberately omits the right-hand key.
    from egoemg_repro.data import file_sha256
    receipt = {'files': {p.name:{'bytes':p.stat().st_size, 'sha256':file_sha256(p)} for p in crops.glob('*.lmdb')}}
    (crops/'acquisition_ledger.json').write_text(json.dumps(receipt))
    manifest = {'windows':[dict(window(0,12000),split='train'),
                           dict(window(12000,24000,role='val'),split='train'),
                           dict(window(24000,36000,role='test'),split='user')]}
    norm = compute_normalization(LegacyStore(data), manifest['windows'])
    provenance = checkpoint['provenance']
    provenance.update(source=train.validate_upstream(upstream), data=train.data_identity(data),
                      partition_sha256=train.canonical_hash(manifest), normalization_sha256=train.canonical_hash(norm))
    config = artifacts['config.json']
    if kind == 'fusion':
        initializers = {}
        for branch, recipe_name in (('vision','resnet18'),('emg','emgformer_s')):
            branch_config=json.loads(Path(f'configs/{recipe_name}.json').read_text())
            branch_config['vision_pretrained']=False
            branch_model=train.build_training_model(upstream,branch_config)
            path=tmp_path/(branch+'.pt')
            torch.save({'model':branch_model.state_dict()},path)
            initializers[branch+'_checkpoint']=path
        model=train.build_training_model(upstream,config,**initializers)
        # Restoring the locked fusion must need no original branch files.
        for path in initializers.values():
            path.unlink()
    else:
        model = train.build_training_model(upstream, config)
    if kind != 'emg':
        provenance['visual_data']=train.visual_data_identity(data,crops,manifest)
    checkpoint['model'] = model.state_dict()
    checkpoint['best']['model'] = model.state_dict()
    checkpoint['best']['provenance'] = copy.deepcopy(provenance)
    datasets = train.build_datasets(data, manifest, norm, config, crops=crops)
    saved = train.evaluate(model, DataLoader(datasets['test'],batch_size=2), 'cpu', config)
    summary = artifacts['summary.json']
    summary.update(provenance=provenance, test=saved)
    for name,value in [('provenance.json',provenance),('summary.json',summary),('partitions.json',manifest),('normalization.json',norm)]:
        (run/name).write_text(json.dumps(value))
    torch.save(checkpoint,run/'last.pt')
    # An archived checkpoint's real first read updates relatime access metadata.
    # That must not be mistaken for changes to model contents or file identity.
    old_ns = 946684800_000000000
    os.utime(run/'last.pt', ns=(old_ns, old_ns))
    output=tmp_path/'public-audit'
    args=Namespace(run=str(run),data=str(data),crops=str(crops),upstream=upstream,output=str(output),device='cpu')
    result=audit.run(args)
    # noatime mounts may preserve this timestamp; the identity unit test below
    # proves access-time changes are harmless independently of mount policy.
    assert result['status']=='verified'
    assert result['baseline_verification']['agreement'] is True
    assert result['all']['overall']==saved['sampling']['center']['overall']
    assert result['populations']['overall']['requested_centers']==2
    assert result['populations']['overall']['missing_crop_centers']==1
    assert result['crop_present']['overall']['valid_angle_count']==22
    rendered=(output/'audit.json').read_text()
    assert not any(private in rendered for private in (str(tmp_path), 'subject_0','episode_0','private-run','optimizer','model.pt'))
    # A second explicit verification invocation is reproducible; output is new.
    args.output=str(tmp_path/'second-audit')
    assert audit.run(args)==result
    summary['test']['sampling']['center']['overall']['mae_deg']+=1
    (run/'summary.json').write_text(json.dumps(summary))
    args.output=str(tmp_path/'mismatch')
    with pytest.raises(ValueError,match='MAE'):
        audit.run(args)
    assert not (Path(args.output)/'audit.json').exists()
    if kind == 'emg':
        summary['test']['sampling']['center']['overall']['mae_deg']-=1
        (run/'summary.json').write_text(json.dumps(summary))
        original_recompute = audit._recompute
        def mutate_checkpoint_after_inference(*positional, **keywords):
            result = original_recompute(*positional, **keywords)
            with (run/'last.pt').open('ab') as stream:
                stream.write(b'changed-during-audit')
            return result
        monkeypatch.setattr(audit, '_recompute', mutate_checkpoint_after_inference)
        args.output = str(tmp_path/'mutated-checkpoint')
        with pytest.raises(ValueError, match='artifacts changed'):
            audit.run(args)
        assert not (Path(args.output)/'audit.json').exists()


def test_checkpoint_identity_ignores_atime_on_every_filesystem():
    from types import SimpleNamespace
    from egoemg_repro import audit
    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_size=3,
                               st_mtime_ns=4, st_ctime_ns=5, st_atime_ns=6)
    path = SimpleNamespace(stat=lambda: metadata)
    before = audit._checkpoint_identity(path)
    metadata.st_atime_ns = 7
    assert audit._checkpoint_identity(path) == before
    for field in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'):
        original = getattr(metadata, field)
        setattr(metadata, field, original + 1)
        assert audit._checkpoint_identity(path) != before
        setattr(metadata, field, original)
