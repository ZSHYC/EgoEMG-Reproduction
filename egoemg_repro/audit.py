"""Verify one completed, validation-selected checkpoint and disaggregate centers.

This additional test pass is verification only. It must never select checkpoints,
update parameters or guide hyperparameters. The executed trainer stays frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

DEGREE_TOLERANCE = 1e-4  # Predeclared; intentionally no command-line override.
HANDS = ('left', 'right')
SPLITS = ('user', 'gesture', 'both')
COMPONENTS = {'overall': slice(0, 22), 'fingers': slice(0, 20), 'wrist': slice(20, 22)}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _condition(hand, split):
    if hand not in HANDS or split not in SPLITS:
        raise ValueError('audit requires a known test hand and split')


def _record(values):
    error, count = values
    return {'absolute_error_sum_rad': float(error), 'valid_angle_count': int(count),
            'mae_deg': float(error / count * (180 / math.pi)) if count else None}


def _summary(records):
    values = [value['mae_deg'] for value in records if value['valid_angle_count']]
    return {'mean_mae_deg': float(np.mean(values)) if values else None,
            'std_mae_deg': float(np.std(values, ddof=0)) if values else None,
            'user_count': len(values),
            'aggregation': 'unweighted per-user means; population standard deviation (ddof=0)'}


class GroupedCenterAccumulator:
    """Independent radians sums/counts at hand/split/user intersections."""

    def __init__(self):
        self.groups = {}
        self.invalid_angles = 0

    def update(self, prediction, target, valid, *, hand, split, user):
        _condition(hand, split)
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != (22, 1) or target.shape != prediction.shape:
            raise ValueError('audit centers require prediction and target shape (22,1)')
        valid = np.broadcast_to(np.asarray(valid, dtype=bool), prediction.shape)
        error = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
        if not np.isfinite(error[valid]).all():
            raise ValueError('nonfinite prediction or target at a valid angle')
        self.invalid_angles += int(valid.size - valid.sum())
        values = self.groups.setdefault((hand, split, str(user)), {key: [0., 0] for key in COMPONENTS})
        for name, indexes in COMPONENTS.items():
            selected = error[indexes][valid[indexes]]
            values[name][0] += float(selected.sum())
            values[name][1] += int(selected.size)

    def report(self, *, user_labels=None):
        users = sorted({user for _, _, user in self.groups})
        labels = user_labels or {user: f'user-{index+1:03d}' for index, user in enumerate(users)}

        def aggregate(hand=None, split=None):
            totals = {name: [0., 0] for name in COMPONENTS}
            per_user = {}
            for (h, s, user), components in sorted(self.groups.items()):
                if hand is not None and hand != h or split is not None and split != s:
                    continue
                selected = per_user.setdefault(labels[user], {name: [0., 0] for name in COMPONENTS})
                for name, (error, count) in components.items():
                    for destination in (totals, selected):
                        destination[name][0] += error
                        destination[name][1] += count
            user_reports = {user: {'overall': _record(values['overall']),
                                  'by_component': {name: _record(values[name]) for name in ('fingers', 'wrist')}}
                            for user, values in sorted(per_user.items())}
            summaries = {name: _summary([_record(values[name]) for values in per_user.values()])
                         for name in COMPONENTS}
            return {'overall': _record(totals['overall']),
                    'by_component': {name: _record(totals[name]) for name in ('fingers', 'wrist')},
                    'by_user': user_reports, 'user_summary': summaries['overall'],
                    'user_summary_by_component': {name: summaries[name] for name in ('fingers', 'wrist')}}

        hands = sorted({hand for hand, _, _ in self.groups})
        splits = sorted({split for _, split, _ in self.groups})
        return {'mode': 'center', 'invalid_angle_count': self.invalid_angles, **aggregate(),
                'by_hand': {hand: aggregate(hand=hand) for hand in hands},
                'by_split': {split: aggregate(split=split) for split in splits},
                'by_hand_split': {hand: {split: aggregate(hand, split) for split in splits
                                        if any(h == hand and s == split for h, s, _ in self.groups)}
                                  for hand in hands}}


class CenterPopulation:
    """Opaque multiset digests bind center row, hand and fixed partition identity."""

    def __init__(self, partition_sha256):
        self.partition_sha256 = _sha256(partition_sha256)
        self.groups = {}

    def update(self, *, center, hand, split, crop_present):
        _condition(hand, split)
        if int(center) < 0:
            raise ValueError('center must be nonnegative')
        token = _hash({'partition_sha256': self.partition_sha256, 'center': int(center), 'hand': hand})
        population = self.groups.setdefault((hand, split), {'all': [], 'crop_present': [], 'missing': []})
        population['all'].append(token)
        population['crop_present' if crop_present else 'missing'].append(token)

    def report(self):
        def aggregate(hand=None, split=None):
            tokens = {'all': [], 'crop_present': [], 'missing': []}
            for (h, s), values in self.groups.items():
                if hand is not None and h != hand or split is not None and s != split:
                    continue
                for name in tokens:
                    tokens[name].extend(values[name])
            return {'requested_centers': len(tokens['all']), 'crop_present_centers': len(tokens['crop_present']),
                    'missing_crop_centers': len(tokens['missing']),
                    **{name + '_sha256': _hash(sorted(values)) for name, values in tokens.items()}}
        hands = sorted({hand for hand, _ in self.groups})
        splits = sorted({split for _, split in self.groups})
        return {'overall': aggregate(), 'by_hand': {hand: aggregate(hand=hand) for hand in hands},
                'by_split': {split: aggregate(split=split) for split in splits},
                'by_hand_split': {hand: {split: aggregate(hand, split) for split in splits
                                        if (hand, split) in self.groups} for hand in hands},
                'digest_definition': 'SHA256 of sorted opaque center/hand/partition tokens; repeated requests retained'}


def _metric(value):
    try:
        count, error, mae = value['valid_angle_count'], value['absolute_error_sum_rad'], value['mae_deg']
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('invalid count')
        if isinstance(error, bool) or not math.isfinite(error) or error < 0:
            raise ValueError('invalid error sum')
        if count == 0:
            if error != 0 or mae is not None:
                raise ValueError('empty count requires zero sum and null MAE')
        elif mae is None or not math.isfinite(mae) or mae < 0 or abs(mae-error/count*180/math.pi) > 1e-10:
            raise ValueError('MAE does not match radians sum/count')
    except (KeyError, TypeError) as exc:
        raise ValueError('incomplete metric record') from exc
    return count, error, mae


def verify_baseline(recomputed, saved):
    """Fail closed on changed groups/counts/sums/MAEs at the fixed tolerance."""
    if recomputed.get('mode') != 'center' or saved.get('mode') != 'center':
        raise ValueError('audit baseline must use center sampling')
    checks = []

    def compare(current, original, label):
        count, error, mae = _metric(current)
        old_count, old_error, old_mae = _metric(original)
        if count != old_count:
            raise ValueError(f'baseline {label} valid angle count mismatch')
        if count and (abs(mae-old_mae) > DEGREE_TOLERANCE or
                      abs(error-old_error) > count*math.radians(DEGREE_TOLERANCE)):
            raise ValueError(f'baseline {label} error sum / MAE mismatch')
        checks.append(label)

    compare(recomputed['overall'], saved['overall'], 'overall')
    for dimension in ('by_hand', 'by_split', 'by_component', 'by_hand_split'):
        original_groups = saved[dimension]
        # The frozen accumulator creates component keys only on its first update.
        # No samples means no component observations, equivalently two empty records.
        if dimension == 'by_component' and not original_groups and not saved['by_hand']:
            original_groups = {name: _record([0., 0]) for name in ('fingers', 'wrist')}
        if set(recomputed[dimension]) != set(original_groups):
            raise ValueError(f'baseline {dimension} groups mismatch')
        for key, current in recomputed[dimension].items():
            if dimension == 'by_hand_split':
                if set(current) != set(saved[dimension][key]):
                    raise ValueError('baseline hand/split groups mismatch')
                for split, condition in current.items():
                    compare(condition['overall'], saved[dimension][key][split], f'{dimension}/{key}/{split}')
            else:
                compare(current if dimension == 'by_component' else current['overall'],
                        original_groups[key], f'{dimension}/{key}')
    return {'agreement': True, 'absolute_tolerance_deg': DEGREE_TOLERANCE,
            'count_tolerance': 0,
            'error_sum_tolerance_rad': 'valid_angle_count * radians(absolute_tolerance_deg)',
            'checked_groups': checks}


def require_completed_summary(summary):
    if summary.get('status') != 'training_complete' or summary.get('test_evaluated') is not True:
        raise ValueError('audit requires a completed full training run with final test evaluated')
    if summary.get('test', {}).get('bounded') is not False:
        raise ValueError('audit rejects bounded or undeclared test evaluation')


def _sha256(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('artifact identity must be a lowercase SHA256')
    return value


def public_identities(provenance, *, checkpoint_sha256, summary_sha256, crop_identity):
    """Explicit whitelist: never copy arbitrary metadata into a public artifact."""
    identities = {name: _sha256(provenance[name]) for name in
                  ('config_sha256', 'partition_sha256', 'normalization_sha256')}
    identities.update({name + '_sha256': _sha256(provenance[name]['sha256'])
                       for name in ('source', 'data', 'reproduction')})
    identities.update(checkpoint_sha256=_sha256(checkpoint_sha256), summary_sha256=_sha256(summary_sha256),
                      selected_recipe_sha256=_sha256(provenance['reproduction']['selected_recipe_sha256']),
                      initialization_sha256=_hash(provenance['initialization']),
                      visual_data_sha256=_sha256(crop_identity['sha256']))
    return identities


def validate_artifacts(run, checkpoint, *, device):
    """Bind completed summary, authoritative embedded best and saved run files."""
    from . import train
    run = Path(run)
    summary = train.load_config(run / 'summary.json')
    require_completed_summary(summary)
    best = train.selected_best_checkpoint(checkpoint, bounded=False)
    provenance = checkpoint['provenance']
    train.assert_compatible_provenance(train.load_config(run / 'provenance.json'), provenance)
    train.assert_compatible_provenance(summary.get('provenance', {}), provenance)
    if train.load_config(run / 'run.json').get('run_id') != checkpoint.get('run_id'):
        raise ValueError('saved run identity differs from checkpoint')
    config = train.load_config(run / 'config.json')
    recipe = provenance.get('recipe', {})
    source_bytes = (run / 'recipe_source.json').read_bytes()
    if source_bytes.decode('utf-8') != recipe.get('source_utf8'):
        raise ValueError('saved original recipe differs from checkpoint recipe')
    if hashlib.sha256(source_bytes).hexdigest() != provenance['reproduction']['selected_recipe_sha256']:
        raise ValueError('original recipe hash differs from checkpoint')
    json.loads(source_bytes)  # It must remain a JSON recipe, not opaque bytes.
    if config != recipe.get('effective_config') or train.canonical_hash(config) != provenance.get('config_sha256'):
        raise ValueError('saved effective config differs from checkpoint')
    kind = train.model_kind(config)
    if provenance.get('model_kind') != kind:
        raise ValueError('saved model kind differs from effective config')
    reproduction = train.reproduction_identity(model_kind=kind, recipe_bytes=source_bytes)
    train.assert_compatible_provenance(provenance, {'reproduction': reproduction})
    manifest = train.load_config(run / 'partitions.json')
    norm = train.load_config(run / 'normalization.json')
    for name, value in (('partition', manifest), ('normalization', norm)):
        if train.canonical_hash(value) != provenance.get(name + '_sha256'):
            raise ValueError(f'saved {name} differs from checkpoint identity')
    if checkpoint.get('epoch') != config['epochs'] or checkpoint.get('next_batch') != 0:
        raise ValueError('checkpoint has not completed the saved epoch budget')
    if summary.get('global_step') != checkpoint['global_step']:
        raise ValueError('summary and completed checkpoint global steps differ')
    if summary.get('selected_checkpoint_step') != best['global_step']:
        raise ValueError('summary selected checkpoint differs from embedded best')
    if summary.get('best_val_mae_deg') != best['best_val_mae_deg']:
        raise ValueError('summary selected validation score differs from embedded best')
    initialization = provenance.get('initialization', {})
    if not initialization.get('policy') or initialization.get('seed_before_model') != config['seed']:
        raise ValueError('checkpoint lacks original initialization identity')
    if kind == 'fusion':
        for branch in ('vision', 'emg'):
            _sha256(initialization.get(branch, {}).get('sha256'))
            if initialization[branch].get('model_kind') != branch or initialization[branch].get('selection') != 'complete independent validation':
                raise ValueError('fusion initialization lacks complete validation branch identity')
    environment = train.load_config(run / 'environment.json')
    if environment.get('initialization') != initialization:
        raise ValueError('saved environment initialization identity differs from checkpoint')
    precision = config['precision'] if str(device).startswith('cuda') else 'float32'
    if precision != environment.get('precision'):
        raise ValueError('audit device precision differs from the completed final test')
    matmul = config.get('matmul_precision', 'high')
    if matmul != environment.get('matmul_precision'):
        raise ValueError('audit matmul precision differs from saved run')
    if config['batch_size'] < 1 or config['num_workers'] < 0:
        raise ValueError('saved evaluation batch configuration is invalid')
    return dict(summary=summary, best=best, config=config, manifest=manifest, norm=norm,
                provenance=provenance, precision=precision, matmul_precision=matmul)


def crop_present_at_center(store, window, reader, hand):
    """Read released LMDB presence using the original EMG-row/video mapping."""
    from .crops import video_frame_at_center
    episode_id = store.episodes[int(window['episode'])]['id']
    return reader.read(episode_id, video_frame_at_center(store, window), hand) is not None


def _recompute(model, loader, config, device, *, partition_sha256, crops):
    import torch
    from . import train
    from .crops import CropStore
    visual = train.model_kind(config) != 'emg'
    all_centers, crop_centers = GroupedCenterAccumulator(), GroupedCenterAccumulator()
    populations = CenterPopulation(partition_sha256)
    reader = CropStore(crops) if not visual else None
    dataset = loader.dataset
    offset, batches = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = train.move_batch(batch, device)
            with train.precision_context(device, config):
                prediction, target, valid = train.aligned_training_batch(model, batch, config)
            center = 0 if visual else batch['emg'].shape[-1] // 2 - model.left_context
            if not 0 <= center < prediction.shape[-1]:
                raise ValueError('aligned center is outside the official prediction interval')
            prediction, target, valid = (value[:, :, center:center+1].float().cpu().numpy()
                                         if value.dtype != torch.bool else value[:, :, center:center+1].cpu().numpy()
                                         for value in (prediction, target, valid))
            for index in range(len(prediction)):
                window = dataset.windows[offset // 2]
                hand = HANDS[offset % 2]
                subject = batch['subject'][index]
                user = str(subject.item() if hasattr(subject, 'item') else subject)
                metadata = dict(hand=hand, split=window['split'], user=user)
                if hand != batch['hand'][index] or window['split'] != batch['split'][index]:
                    raise ValueError('evaluation order differs from the frozen paired center grid')
                present = bool(batch['vision_valid_mask'][index].bool().any().item()) if visual else \
                    crop_present_at_center(dataset.store, window, reader, hand)
                all_centers.update(prediction[index], target[index], valid[index], **metadata)
                if present:
                    crop_centers.update(prediction[index], target[index], valid[index], **metadata)
                populations.update(center=window['center'], hand=hand, split=window['split'], crop_present=present)
                offset += 1
            batches += 1
    if offset != len(dataset) or batches != len(loader):
        raise ValueError('audit did not evaluate every requested test center')
    users = sorted({user for _, _, user in all_centers.groups})
    labels = {user: f'user-{index+1:03d}' for index, user in enumerate(users)}
    return {'all': all_centers.report(user_labels=labels), 'crop_present': crop_centers.report(user_labels=labels),
            'populations': populations.report(), 'evaluated_batches': batches}


def _checkpoint_identity(path):
    """Detect mutation/replacement while allowing the audit's reads to update atime."""
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def run(args):
    """Perform a full locked-checkpoint verification; publish only on agreement."""
    from . import train
    from .data import file_sha256
    import torch
    from torch.utils.data import DataLoader
    run_path, output = Path(args.run), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('audit output must be empty; existing verification artifacts are immutable')
    # Reject incomplete runs before reading potentially large checkpoint tensors.
    require_completed_summary(train.load_config(run_path / 'summary.json'))
    checkpoint_path = run_path / 'last.pt'  # The trainer declares this authoritative.
    checkpoint_stat = _checkpoint_identity(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    evidence = validate_artifacts(run_path, checkpoint, device=args.device)
    config, provenance = evidence['config'], evidence['provenance']
    kind = train.model_kind(config)
    current = {'source': train.validate_upstream(args.upstream), 'data': train.data_identity(args.data)}
    crop_identity = train.visual_data_identity(args.data, args.crops, evidence['manifest'], allow_missing_episodes=False)
    if kind != 'emg':
        current['visual_data'] = crop_identity
    train.assert_compatible_provenance(provenance, current)
    checkpoint_sha = file_sha256(checkpoint_path)
    summary_sha = file_sha256(run_path / 'summary.json')
    identities = public_identities(provenance, checkpoint_sha256=checkpoint_sha, summary_sha256=summary_sha,
                                   crop_identity=crop_identity)
    identities['audit_runtime_sha256'] = _hash({name: file_sha256(Path(__file__).with_name(name))
                                              for name in ('audit.py', 'crops.py', 'visual.py')})
    train.seed_everything(config['seed'])
    torch.set_float32_matmul_precision(evidence['matmul_precision'])
    # resume_state prevents branch-file reads and ImageNet downloads for vision/fusion.
    model = train.build_training_model(args.upstream, config, resume_state=evidence['best']['model'])
    model.load_state_dict(evidence['best']['model'], strict=True)
    model = model.to(args.device)
    del checkpoint
    datasets = train.build_datasets(args.data, evidence['manifest'], evidence['norm'], config, crops=args.crops)
    if not all(len(dataset) for dataset in datasets.values()):
        raise ValueError('complete train/validation/test partitions must be nonempty')
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(datasets['test'], batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], pin_memory=str(args.device).startswith('cuda'),
                        generator=generator, drop_last=False)
    metrics = _recompute(model, loader, config, args.device, partition_sha256=provenance['partition_sha256'], crops=args.crops)
    saved_test = evidence['summary']['test']
    if metrics['evaluated_batches'] != saved_test.get('evaluated_batches'):
        raise ValueError('saved test batch count differs from the full audit grid')
    agreement = verify_baseline(metrics['all'], saved_test['sampling']['center'])
    if kind != 'emg':
        paired = saved_test.get('paired_population', {})
        verify_baseline(metrics['all'], paired['all'])
        agreement['saved_crop_present_agreement'] = verify_baseline(metrics['crop_present'], paired['vision_valid'])['agreement']
        counts = metrics['populations']['overall']
        if paired.get('samples') != {'all': counts['requested_centers'], 'vision_valid': counts['crop_present_centers'],
                                     'missing_crop': counts['missing_crop_centers']}:
            raise ValueError('saved missing-crop population counts differ from audit')
    if _checkpoint_identity(checkpoint_path) != checkpoint_stat or file_sha256(run_path / 'summary.json') != summary_sha:
        raise ValueError('completed run artifacts changed during the audit')
    result = {'audit_schema': 1, 'status': 'verified', 'purpose': 'locked_checkpoint_verification',
              'evaluation_policy': 'additional full test pass for verification and disaggregation only; no selection or tuning',
              'training_test_passes': 1, 'additional_verification_passes': 1,
              'model_kind': kind, 'selected_checkpoint_step': evidence['best']['global_step'],
              'best_val_mae_deg': evidence['best']['best_val_mae_deg'], 'identities': identities,
              'precision': evidence['precision'], 'matmul_precision': evidence['matmul_precision'],
              'batch_size': config['batch_size'], 'baseline_verification': agreement, **metrics,
              'population_definition': 'all requested labeled centers; crop_present uses identical released LMDB keys across modalities',
              'user_identifier_policy': 'sorted test-subject labels, anonymous within this identical data/partition identity'}
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'audit.json.tmp'
    try:
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        temporary.replace(output / 'audit.json')
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'upstream', 'crops', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(1, f'audit refused: {exc}\n')
    print(json.dumps({'status': result['status'], 'baseline_agreement': result['baseline_verification']['agreement']}))
    return result


if __name__ == '__main__':
    main()
