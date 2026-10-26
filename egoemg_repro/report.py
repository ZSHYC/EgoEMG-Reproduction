"""Independent, sanitized reporting for frozen EgoEMG run artifacts.

Status reads only config/progress. Completed export binds recorded test metrics to
saved protocol artifacts and authoritative selected-best checkpoints; it never
loads datasets, runs inference, or redistributes model weights or participant IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from . import UPSTREAM_COMMIT


class ReportError(ValueError):
    """Run evidence is incomplete, inconsistent or unsuitable for publication."""


def _reject_constant(_):
    raise ReportError('nonfinite JSON number in run evidence')


def _json(path, *, optional=False):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=_reject_constant)
    except FileNotFoundError:
        if optional:
            return None
        raise ReportError('missing required ' + Path(path).name) from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ReportError('invalid JSON in ' + Path(path).name) from None


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ReportError(label + ' must be a finite number')
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ReportError(label + ' must be finite and nonnegative')
    return value


def _count(value, label):
    if type(value) is not int or value < 0:
        raise ReportError(label + ' must be a nonnegative integer')
    return value


def _same(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)


def _record(value):
    if not isinstance(value, dict):
        raise ReportError('metric record must be an object')
    count = _count(value.get('valid_angle_count'), 'valid angle count')
    error = _finite(value.get('absolute_error_sum_rad'), 'absolute error sum', nonnegative=True)
    if count == 0 and error != 0:
        raise ReportError('zero-count metric has nonzero error sum')
    degree = error / count * 180 / math.pi if count else None
    supplied = value.get('mae_deg')
    if degree is None:
        if supplied is not None:
            raise ReportError('zero-count metric must have null MAE')
    elif supplied is None or not _same(_finite(supplied, 'MAE'), degree):
        raise ReportError('degree MAE disagrees with radians error sum and valid count')
    return {'absolute_error_sum_rad': error, 'valid_angle_count': count, 'mae_deg': degree}


def _sum_matches(records, total):
    if sum(item['valid_angle_count'] for item in records) != total['valid_angle_count']:
        raise ReportError('group valid counts do not sum to the pooled count')
    if not _same(sum(item['absolute_error_sum_rad'] for item in records), total['absolute_error_sum_rad']):
        raise ReportError('group error sums do not sum to the pooled error')


def _progress(run):
    path = Path(run) / 'progress.jsonl'
    if not path.exists():
        return [], False
    events, partial = [], False
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                partial = True
                break
            try:
                event = json.loads(line, parse_constant=_reject_constant)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ReportError('invalid complete progress line') from None
            if not isinstance(event, dict):
                raise ReportError('progress event must be an object')
            events.append(event)
    return events, partial


def _kind(config):
    kind = config.get('model_kind', 'emg')
    if kind not in ('emg', 'vision', 'fusion'):
        raise ReportError('unknown model kind')
    return kind


def _vision_backbone(config):
    if 'vision_backbone_type' not in config:
        return None
    backbone = config['vision_backbone_type']
    choices = {'fusion/vision_resnet18': ('resnet18', 'resnet50', 'resnet152'),
               'fusion/vision_vit_small': ('vit_small', 'vit_base', 'vit_large')}.get(config.get('upstream_experiment'), ())
    if _kind(config) != 'vision' or not isinstance(backbone, str) or backbone not in choices:
        raise ReportError('vision_backbone_type requires a matching supported official ResNet or ViT vision model')
    return backbone


def _model_name(config):
    backbone = _vision_backbone(config)
    if backbone is not None:
        return {'resnet18':'ResNet-18', 'resnet50':'ResNet-50', 'resnet152':'ResNet-152',
                'vit_small':'ViT-S', 'vit_base':'ViT-B', 'vit_large':'ViT-L'}[backbone]
    known = {'emgformer/egoemg_emgformer_small':'EMGFormer-S',
             'emgformer/egoemg_emgformer_middle':'EMGFormer-M',
             'emgformer/egoemg_emgformer_large':'EMGFormer-L',
             'fusion/vision_resnet18':'ResNet-18',
             'fusion/vision_vit_small':'ViT-S',
             'fusion/fusion_rn18_s_center_8ch':'ResNet-18 + EMGFormer-S'}
    return known.get(config.get('upstream_experiment'),
                     {'emg':'EMG model','vision':'Vision model','fusion':'Fusion model'}[_kind(config)])


def _safe_config(config):
    result = {}
    backbone = _vision_backbone(config)
    if backbone is not None:
        result['vision_backbone_type'] = backbone
    for key in ('seed', 'epochs', 'batch_size', 'official_batch_size', 'window_length',
                'emg_window_length', 'lr', 'weight_decay', 'eta_min', 'gradient_clip',
                'fingertip_weight', 'train_stride', 'eval_stride', 'val_fraction'):
        if key in config:
            result[key] = _finite(config[key], 'configuration number', nonnegative=True)
    allowed = {'precision': {'float32','bfloat16'}, 'optimizer_class': {'Adam','AdamW'},
               'augmentation': {'none','official_rotation'}, 'emg_layout': {'target_hand','emg2pose_sparse16'}}
    for key, choices in allowed.items():
        if config.get(key) in choices:
            result[key] = config[key]
    result['model_kind'] = _kind(config)
    return result


def _curves(events, kind):
    training, validation = [], []
    segment = -1
    skipped = 0
    latest_step = 0
    for event in events:
        name = event.get('event')
        if name == 'start':
            segment += 1
            latest_step = max(latest_step, _count(event.get('step', 0), 'start step'))
        elif name == 'skipped_batch':
            skipped += 1
        elif name == 'train_step':
            step = _count(event.get('step'), 'training step')
            epoch = _count(event.get('epoch'), 'training epoch')
            point = {'step': step, 'epoch': epoch, 'segment': max(0, segment),
                     'loss': _finite(event.get('loss'), 'training loss')}
            for key in ('angle_l1_rad','fingertip_distance','lr'):
                if key in event:
                    point[key] = _finite(event[key], 'training metric', nonnegative=True)
            training.append(point)
            latest_step = max(latest_step, step)
        elif name == 'validation':
            mode = 'trajectory' if kind == 'emg' else 'center'
            try:
                metric = _record(event['metrics']['sampling'][mode]['overall'])
                bounded = event['metrics']['bounded']
            except (KeyError, TypeError):
                raise ReportError('validation event lacks explicit sampling or bounded policy') from None
            if type(bounded) is not bool:
                raise ReportError('validation bounded policy must be boolean')
            validation.append({'step': _count(event.get('step'), 'validation step'),
                               'epoch': _count(event.get('epoch'), 'validation epoch'),
                               'segment': max(0, segment), 'mode': mode, 'bounded': bounded,
                               **metric})
    return {'training': training, 'validation': validation}, latest_step, skipped


def build_status(run):
    """Read config/progress only, with no test-summary or checkpoint dependency."""
    run = Path(run)
    if not run.is_dir():
        raise ReportError('run directory is unavailable')
    config = _json(run / 'config.json', optional=True) or {}
    events, partial = _progress(run)
    curves, step, skipped = _curves(events, _kind(config))
    return {'schema_version': 1, 'report_type': 'training_status', 'benchmark_eligible': False,
            'completion_status': 'incomplete_or_unverified', 'model_kind': _kind(config),
            'model_label': _model_name(config),
            'configuration': _safe_config(config), 'progress': {'global_step': step,
                'validation_events': len(curves['validation']), 'skipped_batches': skipped,
                'restart_segments': sum(event.get('event') == 'start' for event in events)},
            'incomplete_trailing_line_ignored': partial, 'curves': curves,
            'notice': '训练进度；未完成或未核验最终结果，不可作为测试集基准。'}


def _trajectory_points(config):
    """Pinned S/M/L tds_slim context, also enforced by the model builder."""
    if _kind(config) != 'emg':
        return None
    if config.get('upstream_experiment') not in {
            'emgformer/egoemg_emgformer_small', 'emgformer/egoemg_emgformer_middle',
            'emgformer/egoemg_emgformer_large'}:
        raise ReportError('unsupported EMG trajectory context contract')
    # All three pinned experiments have left_context=510 and right_context=0;
    # prediction interpolation targets the entire cropped label interval.
    length = _count(config.get('window_length'), 'trajectory input window length')
    if length <= 510:
        raise ReportError('trajectory window does not contain the pinned context')
    return length - 510


def _metric(metric, mode, users, samples, *, window_length, split_samples=None, paired_hands=True,
            trajectory_points=None):
    if not isinstance(metric, dict) or metric.get('mode') != mode:
        raise ReportError('metric sampling definition is inconsistent')
    total = _record(metric.get('overall'))
    invalid = _count(metric.get('invalid_angle_count'), 'invalid angle count')
    observed = total['valid_angle_count'] + invalid
    if mode == 'center':
        if observed != samples * 22:
            raise ReportError('center observations disagree with requested paired windows')
        time_points = 1
    else:
        if trajectory_points is None or observed != samples * 22 * trajectory_points:
            raise ReportError('trajectory observations disagree with the pinned aligned interval')
        time_points = trajectory_points
    public = {'mode': mode, 'overall': total, 'invalid_angle_count': invalid,
              'time_points_per_window': time_points}
    groups = {'by_hand': {'left','right'}, 'by_split': {'train','user','gesture','both'},
              'by_component': {'fingers','wrist'}, 'by_user': None}
    for group, choices in groups.items():
        raw = metric.get(group)
        if not isinstance(raw, dict) or (choices is not None and not set(raw) <= choices):
            raise ReportError('metric group keys are missing or invalid')
        if samples and (group == 'by_component' or group == 'by_hand' and paired_hands) and set(raw) != choices:
            raise ReportError('paired hand or angle-component groups are incomplete')
        checked = {key: _record(value) for key,value in raw.items()}
        _sum_matches(list(checked.values()), total)
        if group == 'by_hand' and any(value['valid_angle_count'] > (math.ceil(samples/2) if paired_hands else samples)*22*time_points for value in checked.values()):
            raise ReportError('per-hand count exceeds paired population')
        if group == 'by_component':
            for key, angles in (('fingers',20),('wrist',2)):
                if checked.get(key, {}).get('valid_angle_count',0) > samples*angles*time_points:
                    raise ReportError('angle-component count exceeds its denominator')
        if group == 'by_split' and split_samples is not None:
            for key,value in checked.items():
                if key not in split_samples or value['valid_angle_count'] > split_samples[key]*22*time_points:
                    raise ReportError('split count disagrees with official test windows')
        public[group] = {users[key] if group == 'by_user' else key: value for key,value in sorted(checked.items())}
    raw_joint = metric.get('by_hand_split')
    if not isinstance(raw_joint, dict) or not set(raw_joint) <= {'left','right'}:
        raise ReportError('hand/split groups are invalid')
    public_joint = {}
    for hand, splits in raw_joint.items():
        if not isinstance(splits, dict) or not set(splits) <= {'train','user','gesture','both'}:
            raise ReportError('hand/split keys are invalid')
        public_joint[hand] = {key:_record(value) for key,value in sorted(splits.items())}
        _sum_matches(list(public_joint[hand].values()), public['by_hand'].get(hand, _record({'absolute_error_sum_rad':0,'valid_angle_count':0,'mae_deg':None})))
    _sum_matches([item for splits in public_joint.values() for item in splits.values()],total)
    for split, expected in public['by_split'].items():
        _sum_matches([splits[split] for splits in public_joint.values() if split in splits],expected)
    public['by_hand_split'] = public_joint
    means = [item['mae_deg'] for item in public['by_user'].values() if item['mae_deg'] is not None]
    user_summary = {'mean_mae_deg': statistics.fmean(means) if means else None,
                    'std_mae_deg': statistics.pstdev(means) if means else None,
                    'user_count': len(means),
                    'aggregation': 'unweighted per-user means; population standard deviation (ddof=0)',
                    'scope': 'per-user means pooled across all reported hands and test splits; not per-condition dispersion'}
    supplied = metric.get('user_summary', {})
    if supplied.get('user_count') != len(means):
        raise ReportError('per-user count is inconsistent')
    for key in ('mean_mae_deg','std_mae_deg'):
        actual = supplied.get(key)
        expected = user_summary[key]
        if (actual is None) != (expected is None) or (actual is not None and not _same(_finite(actual,'user statistic'), expected)):
            raise ReportError('per-user statistics disagree with error sums/counts')
    public['user_summary'] = user_summary
    return public


def _user_map(test):
    keys = set()
    reports = list(test.get('sampling',{}).values())
    paired = test.get('paired_population',{})
    reports += [paired[key] for key in ('all','vision_valid') if key in paired]
    for metric in reports:
        if isinstance(metric, dict) and isinstance(metric.get('by_user'), dict):
            keys.update(metric['by_user'])
    return {name:'user_%03d' % (index+1) for index,name in enumerate(sorted(keys))}


def _evaluate_report(report, kind, samples, window_length, split_samples, users, *, stage='test',
                     trajectory_points=None):
    if report.get('bounded') is not False:
        raise ReportError('completed export requires a full test evaluation')
    required = {'center','trajectory'} if kind == 'emg' else {'center'}
    if set(report.get('sampling',{})) != required:
        raise ReportError('test sampling differs from the model target definition')
    public = {'bounded': False, 'evaluated_batches': _count(report.get('evaluated_batches'),'test batches'),
              'requested_hand_windows': samples, 'sampling': {}}
    for mode in sorted(required):
        public['sampling'][mode] = _metric(report['sampling'][mode],mode,users,samples,
                                           window_length=window_length,split_samples=split_samples,
                                           trajectory_points=trajectory_points)
    if kind != 'emg':
        paired = report.get('paired_population')
        if not isinstance(paired,dict):
            raise ReportError('visual results require explicit paired crop populations')
        counts = paired.get('samples',{})
        counts = {key:_count(counts.get(key),'crop population count') for key in ('all','vision_valid','missing_crop')}
        if counts['all'] != samples or counts['vision_valid']+counts['missing_crop'] != samples:
            raise ReportError('crop population counts do not match requested centers')
        all_metric = _metric(paired.get('all'),'center',users,samples,window_length=window_length,split_samples=split_samples)
        if all_metric != public['sampling']['center']:
            raise ReportError('all-crop population differs from the primary center population')
        matched = _metric(paired.get('vision_valid'),'center',users,counts['vision_valid'],
                          window_length=window_length,paired_hands=False)
        missing = {}
        for key, allowed in (('missing_by_hand',{'left','right'}),
                             ('missing_by_split',{'train'} if stage=='validation' else {'user','gesture','both'})):
            raw = paired.get(key,{})
            if not isinstance(raw,dict) or not set(raw)<=allowed:
                raise ReportError('missing-crop groups are invalid')
            missing[key] = {group:_count(value,'missing crop count') for group,value in sorted(raw.items())}
            if sum(missing[key].values()) != counts['missing_crop']:
                raise ReportError('missing-crop group counts disagree')
        for hand,count in missing['missing_by_hand'].items():
            if count > samples//2:
                raise ReportError('missing hand count exceeds paired population')
        for hand,value in matched['by_hand'].items():
            maximum = samples//2 - missing['missing_by_hand'].get(hand,0)
            if value['valid_angle_count'] > maximum*22:
                raise ReportError('matched hand count exceeds crop-present population')
        for split,count in missing['missing_by_split'].items():
            if count > split_samples.get(split,0):
                raise ReportError('missing split count exceeds official test population')
        for split,value in matched['by_split'].items():
            maximum = split_samples.get(split,0) - missing['missing_by_split'].get(split,0)
            if value['valid_angle_count'] > maximum*22:
                raise ReportError('matched split count exceeds crop-present population')
        if matched['overall']['valid_angle_count'] > all_metric['overall']['valid_angle_count']:

            raise ReportError('matched population exceeds all-label population')
        public['paired_population'] = {'all':all_metric,'vision_valid':matched,'samples':counts,**missing,
            'definition':'All requested labeled centers; vision_valid is the identical crop-present subset.'}
    if stage=='test':
        public['unavailable_test_splits'] = sorted({'user','gesture','both'} - set(split_samples))
    return public


def _provenance_equal(recorded, current):
    if not isinstance(recorded,dict):
        raise ReportError('checkpoint lacks provenance')
    for key,value in current.items():
        if key == 'reproduction':
            equal = isinstance(recorded.get(key),dict) and recorded[key].get('sha256') == value.get('sha256')
        else:
            equal = recorded.get(key) == value
        if not equal:
            raise ReportError('saved provenance identities disagree')


def _state_equal(left,right):
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(_state_equal(left[key],right[key]) for key in left)
    if isinstance(left,(list,tuple)):
        return isinstance(right,type(left)) and len(left)==len(right) and all(_state_equal(a,b) for a,b in zip(left,right))
    if hasattr(left,'detach'):
        import torch
        return hasattr(right,'detach') and torch.equal(left.cpu(),right.cpu())
    return left == right


def load_checkpoint(path):
    try:
        import torch
    except ImportError:
        raise ReportError('completed export requires Torch from the training environment') from None
    try:
        return torch.load(path,map_location='cpu',weights_only=False)
    except Exception:
        raise ReportError('checkpoint cannot be decoded') from None


def _checkpoint(state, provenance, run_id):
    if not isinstance(state,dict) or state.get('checkpoint_schema')!=2 or not isinstance(state.get('best'),dict):
        raise ReportError('self-contained checkpoint schema 2 is required')
    best = state['best']
    required = {'model','optimizer','scheduler','rng','epoch','next_batch','global_step','best_val_mae_deg','validation_bounded','provenance','run_id'}
    if not required <= set(state) or not required <= set(best):
        raise ReportError('checkpoint training state is incomplete')
    for item in (state,best):
        if item['validation_bounded'] is not False or item['run_id']!=run_id:
            raise ReportError('checkpoint validation policy or run identity disagrees')
        _provenance_equal(item['provenance'],provenance)
        _count(item['global_step'],'checkpoint step');_count(item['epoch'],'checkpoint epoch')
        if item['epoch'] < 1:
            raise ReportError('completed checkpoint has no completed validation epoch')
        if item['next_batch'] != 0 or not isinstance(item['model'],dict) or not item['model']:
            raise ReportError('completed checkpoint is not an epoch-complete model state')
    if best['epoch']>state['epoch']:
        raise ReportError('selected validation epoch follows the current checkpoint')
    if best['global_step']>state['global_step'] or not _same(_finite(state['best_val_mae_deg'],'best score'),_finite(best['best_val_mae_deg'],'selected score')):
        raise ReportError('selected best score/step disagrees with its checkpoint')
    return best


def _coverage(manifest, config):
    windows = manifest.get('windows')
    if not isinstance(windows,list) or not windows:
        raise ReportError('partition windows are missing')
    counts = {role:0 for role in ('train','val','test')}
    test_splits = {}
    intervals = {role:[] for role in counts}
    length = _count(config.get('window_length'),'window length')
    if length==0:raise ReportError('window length must be positive')
    for window in windows:
        role = window.get('role')
        if role not in counts:
            raise ReportError('unknown partition role')
        start,end,center = (_count(window.get(key),'partition coordinate') for key in ('start','end','center'))
        if end-start != length or center!=start+length//2:
            raise ReportError('partition center/window length is inconsistent')
        counts[role]+=2
        intervals[role].append((start,end))
        if role=='test':
            split=window.get('split')
            if split not in ('user','gesture','both'):
                raise ReportError('official test windows contain an invalid split')
            test_splits[split]=test_splits.get(split,0)+2
    def merged(items):
        result=[]
        for start,end in sorted(items):
            if result and start<=result[-1][1]:result[-1][1]=max(result[-1][1],end)
            else:result.append([start,end])
        return result
    joined={role:merged(items) for role,items in intervals.items()}
    for a,b in (('train','val'),('train','test'),('val','test')):
        i=j=0
        while i<len(joined[a]) and j<len(joined[b]):
            x,y=joined[a][i],joined[b][j]
            if max(x[0],y[0])<min(x[1],y[1]):raise ReportError('partition role footprints overlap')
            if x[1]<=y[1]:i+=1
            else:j+=1
    if any(not count for count in counts.values()):raise ReportError('a required partition is empty')
    for role,count in counts.items():
        coverage=manifest.get('coverage',{}).get(role,{})
        if coverage.get('hand_samples')!=count or coverage.get('windows')!=count//2:
            raise ReportError('saved partition coverage disagrees with window records')
    return counts,test_splits,sum(end-start for start,end in joined['train'])


def _hex(value,length):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{%d}'%length,value):
        raise ReportError('required evidence hash is invalid')
    return value


def _safe_dinov2_identity(pretrained):
    variants = {'vit_small_patch14_dinov2': 'vits14', 'vit_base_patch14_dinov2': 'vitb14',
                'vit_large_patch14_dinov2': 'vitl14'}
    name = pretrained.get('timm_model')
    if not isinstance(name, str) or name not in variants:
        raise ReportError('unknown DINOv2 pretrained model')
    tag = variants[name]
    hub_id = 'timm/' + name + '.lvd142m'
    url = f'https://dl.fbaipublicfiles.com/dinov2/dinov2_{tag}/dinov2_{tag}_pretrain.pth'
    source_type = pretrained.get('source_type')
    location = pretrained.get('source_location')
    if source_type not in ('hf-hub', 'url') or location != {'hf-hub': hub_id, 'url': url}[source_type]:
        raise ReportError('unapproved DINOv2 pretrained source')
    version = pretrained.get('timm_version', '')
    if not isinstance(version, str) or not re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,3}(?:\+[A-Za-z0-9.]+)?', version):
        raise ReportError('invalid DINOv2 timm version')
    cfg = pretrained.get('resolved_pretrained_cfg')
    if not isinstance(cfg, dict):
        raise ReportError('missing resolved DINOv2 pretrained configuration')
    # Core identity/preprocessing evidence must exist and match the pinned timm
    # cfg. Contradictory known evidence is rejected before private extras drop.
    required = {'architecture', 'tag', 'url', 'hf_hub_id', 'input_size', 'mean', 'std', 'num_classes'}
    known = {'architecture': name, 'tag': 'lvd142m', 'url': url, 'hf_hub_id': hub_id,
             'first_conv': 'patch_embed.proj', 'classifier': 'head', 'license': 'apache-2.0',
             'interpolation': 'bicubic', 'crop_mode': 'center', 'input_size': [3, 518, 518],
             'mean': [.485, .456, .406], 'std': [.229, .224, .225], 'num_classes': 0,
             'custom_load': False, 'fixed_input_size': True, 'crop_pct': 1., 'pool_size': None}
    safe_cfg = {}
    for key, expected in known.items():
        if key not in cfg:
            if key in required:
                raise ReportError('DINOv2 pretrained configuration is missing core field ' + key)
            continue
        actual = cfg[key]
        if isinstance(expected, list):
            if not isinstance(actual, (list, tuple)):
                raise ReportError('DINOv2 pretrained configuration conflicts at ' + key)
            actual = [_finite(x, 'DINOv2 configuration number', nonnegative=True) for x in actual]
        elif type(expected) in (int, float):
            actual = _finite(actual, 'DINOv2 configuration number', nonnegative=True)
        if actual != expected or (type(expected) is bool and type(actual) is not bool):
            raise ReportError('DINOv2 pretrained configuration conflicts at ' + key)
        safe_cfg[key] = expected
    return {'pretrained_family': 'DINOv2', 'pretrained_timm_model': name, 'timm_version': version,
            'pretrained_source_type': source_type, 'pretrained_source_location': location,
            'pretrained_resolved_cfg': safe_cfg}


def _safe_identities(provenance):
    if provenance.get('source',{}).get('revision')!=UPSTREAM_COMMIT:
        raise ReportError('upstream is not the approved source revision')
    result={'upstream_commit':UPSTREAM_COMMIT,
        'upstream_source_sha256':_hex(provenance['source'].get('sha256'),64),
        'runtime_sha256':_hex(provenance.get('reproduction',{}).get('sha256'),64),
        'data_sha256':_hex(provenance.get('data',{}).get('sha256'),64)}
    for key in ('config_sha256','partition_sha256','normalization_sha256'):
        result[key]=_hex(provenance.get(key),64)
    result['original_recipe_sha256']=_hex(provenance['reproduction'].get('selected_recipe_sha256'),64)
    config = provenance.get('recipe', {}).get('effective_config', {})
    selected_vit = config.get('upstream_experiment') == 'fusion/vision_vit_small'
    initialization=provenance.get('initialization',{})
    policy=initialization.get('policy')
    builder=initialization.get('official_builder',{})
    if (selected_vit or policy == 'DINOv2 backbone') and not isinstance(builder, dict):
        raise ReportError('DINOv2 policy requires pretrained source and tensor evidence')
    pretrained=builder.get('pretrained_weights')
    if selected_vit:
        if policy not in ('DINOv2 backbone', 'random', 'clean vision checkpoint'):
            raise ReportError('DINOv2 recipe initialization policy/family mismatch')
        if policy != 'DINOv2 backbone' and pretrained is not None:
            raise ReportError('DINOv2 recipe has pretrained evidence inconsistent with its policy')
        if isinstance(pretrained, dict) and any(key in pretrained for key in ('enum', 'source_url', 'torchvision_version')):
            raise ReportError('DINOv2 recipe contains conflicting ResNet initialization evidence')
        branch=initialization.get('vision')
        if policy == 'random' and branch is not None:
            raise ReportError('DINOv2 random policy conflicts with vision checkpoint evidence')
        if policy == 'clean vision checkpoint' and (
                not isinstance(branch, dict) or branch.get('model_kind') != 'vision' or
                branch.get('selection') != 'complete independent validation'):
            raise ReportError('DINOv2 clean checkpoint requires validated vision branch evidence')
    public_init={}
    if initialization.get('policy') in ('random','ImageNet1K_V1 backbone','DINOv2 backbone','clean vision checkpoint','clean branches then official residual-zero initialization'):
        public_init['policy']=initialization['policy']
    if 'seed_before_model' in initialization:public_init['seed']=_count(initialization['seed_before_model'],'initialization seed')
    for branch in ('emg','vision'):
        item=initialization.get(branch)
        if item is not None:
            public_init[branch]={'checkpoint_sha256':_hex(item.get('sha256'),64),
                'selected_step':_count(item.get('selected_step'),'initializer step'),
                'best_val_mae_deg':_finite(item.get('best_val_mae_deg'),'initializer score')}
    if initialization.get('policy') == 'DINOv2 backbone' and (
            not isinstance(builder, dict) or not isinstance(builder.get('pretrained_weights'), dict)
            or not builder['pretrained_weights']):
        raise ReportError('DINOv2 policy requires nonempty pretrained source and tensor evidence')
    if pretrained:
        if initialization.get('policy') == 'DINOv2 backbone' and pretrained.get('family') != 'DINOv2':
            raise ReportError('DINOv2 pretrained family mismatch')
        if pretrained.get('family') == 'DINOv2':
            backbone = config.get('vision_backbone_type', 'vit_small')
            expected = {'vit_small': 'vit_small_patch14_dinov2', 'vit_base': 'vit_base_patch14_dinov2',
                        'vit_large': 'vit_large_patch14_dinov2'}.get(backbone) if isinstance(backbone, str) else None
            if (config.get('upstream_experiment') != 'fusion/vision_vit_small' or
                    pretrained.get('timm_model') != expected or initialization.get('policy') != 'DINOv2 backbone'):
                raise ReportError('DINOv2 pretrained model/family does not match the selected recipe')
            public_init.update(_safe_dinov2_identity(pretrained))
        else:
            url=pretrained.get('source_url','')
            if re.fullmatch(r'https://download\.pytorch\.org/models/resnet[0-9]+-[0-9a-f]{8}\.pth',url):
                public_init['pretrained_source_url']=url
            # Exact hashes identify weights; only known enum/version strings are public.
            enum=pretrained.get('enum','')
            version=pretrained.get('torchvision_version','')
            if re.fullmatch(r'torchvision\.models\.ResNet\d+_Weights\.IMAGENET1K_V1',enum):public_init['pretrained_enum']=enum
            if re.fullmatch(r'[0-9.]+(?:\+[A-Za-z0-9.]+)?',version):public_init['torchvision_version']=version
        public_init['pretrained_backbone_tensor_sha256']=_hex(pretrained.get('backbone_tensor_sha256'),64)
    result['initialization']=public_init
    for key in ('visual_data',):
        if key in provenance:result[key+'_sha256']=_hex(provenance[key].get('sha256'),64)
    return result


def build_export(run, model_label, *, checkpoint_loader=None):
    """Reject incomplete evidence, then produce whitelisted publication content."""
    if not isinstance(model_label,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._+\-]{0,79}',model_label):
        raise ReportError('model label must be a short plain model name')
    run=Path(run)
    summary=_json(run/'summary.json')
    if summary.get('status')!='training_complete' or summary.get('test_evaluated') is not True:
        raise ReportError('only training_complete with an actual full final test may be exported')
    config=_json(run/'config.json');manifest=_json(run/'partitions.json');norm=_json(run/'normalization.json')
    provenance=_json(run/'provenance.json');run_meta=_json(run/'run.json')
    if not isinstance(summary.get('test'),dict) or summary['test'].get('bounded') is not False:
        raise ReportError('completed export requires a full final test')
    _provenance_equal(summary.get('provenance'),provenance)
    for value,key in ((config,'config_sha256'),(manifest,'partition_sha256'),(norm,'normalization_sha256')):
        if _canonical(value)!=provenance.get(key):raise ReportError('saved protocol artifact hash disagrees')
    recipe_bytes=(run/'recipe_source.json').read_bytes()
    recipe=provenance.get('recipe',{})
    if hashlib.sha256(recipe_bytes).hexdigest()!=provenance.get('reproduction',{}).get('selected_recipe_sha256'):
        raise ReportError('original recipe identity disagrees')
    if recipe.get('source_utf8','').encode('utf-8')!=recipe_bytes or _canonical(recipe.get('effective_config'))!=_canonical(config):
        raise ReportError('original/effective recipe evidence disagrees')
    kind=_kind(config)
    if ('vision_backbone_type' in config or config.get('upstream_experiment') == 'fusion/vision_vit_small') and model_label!=_model_name(config):
        raise ReportError('model label disagrees with configured vision backbone')
    trajectory_points=_trajectory_points(config)
    if _count(config.get('epochs'),'configured epochs')==0 or _count(config.get('batch_size'),'batch size')==0:
        raise ReportError('completed configuration requires positive epochs and batch size')
    if provenance.get('model_kind',kind)!=kind:raise ReportError('model target kind disagrees')
    counts,split_samples,training_rows=_coverage(manifest,config)
    for hand in ('left','right'):
        item=norm.get(hand,{})
        if item.get('count')!=training_rows or len(item.get('mean',[]))!=8 or len(item.get('std',[]))!=8:
            raise ReportError('training normalization count/shape disagrees')
        for value in item['mean']:_finite(value,'normalization mean')
        if any(_finite(value,'normalization standard deviation')<=0 for value in item['std']):
            raise ReportError('normalization standard deviation must be positive')
    loader=checkpoint_loader or load_checkpoint
    last=loader(run/'last.pt');convenience=loader(run/'best.pt')
    best=_checkpoint(last,provenance,run_meta.get('run_id'))
    other=_checkpoint(convenience,provenance,run_meta.get('run_id'))
    step=_count(summary.get('global_step'),'completed step');selected=_count(summary.get('selected_checkpoint_step'),'selected step')
    score=_finite(summary.get('best_val_mae_deg'),'best validation score',nonnegative=True)
    if step<=0 or last['global_step']!=step or last['epoch']!=config.get('epochs') or selected!=best['global_step']:
        raise ReportError('completed or selected checkpoint position disagrees')
    if convenience['global_step']!=selected or other['global_step']!=selected or convenience['epoch']!=best['epoch'] or other['epoch']!=best['epoch'] or not _state_equal(best['model'],convenience['model']) or not _state_equal(best['model'],other['model']):
        raise ReportError('selected best weights disagree with the convenience artifact')
    if not _same(score,best['best_val_mae_deg']) or not _same(score,convenience['best_val_mae_deg']):
        raise ReportError('selected best validation score disagrees')
    users=_user_map(summary['test'])
    test=_evaluate_report(summary['test'],kind,counts['test'],config['window_length'],split_samples,users,
                          trajectory_points=trajectory_points)
    expected_batches=math.ceil(counts['test']/_count(config.get('batch_size'),'batch size'))
    if test['evaluated_batches']!=expected_batches:raise ReportError('test batch coverage is incomplete')
    primary='trajectory' if kind=='emg' else 'center'
    if test['sampling'][primary]['overall']['mae_deg'] is None:raise ReportError('final test has no valid angle observations')
    events,partial=_progress(run)
    curves,latest,skipped=_curves(events,kind)
    for event in events:
        if event.get('event')=='validation' and event.get('metrics',{}).get('bounded') is False:
            val=event['metrics']
            _evaluate_report(val,kind,counts['val'],config['window_length'],{'train':counts['val']},
                             _user_map(val),stage='validation',trajectory_points=trajectory_points)
            if val['evaluated_batches']!=math.ceil(counts['val']/config['batch_size']):
                raise ReportError('full validation batch coverage is incomplete')
            if event['epoch']>=config['epochs']:
                raise ReportError('validation epoch exceeds configured training')
    full=[point for point in curves['validation'] if not point['bounded']]
    if not curves['training'] or not full or latest!=step:raise ReportError('completed training/validation progress evidence is incomplete')
    if any(point['mae_deg'] is not None and point['mae_deg']<score and not _same(point['mae_deg'],score) for point in full):
        raise ReportError('selected checkpoint is worse than a recorded full validation candidate')
    starts=[event for event in events if event.get('event')=='start']
    initial_step=starts[0].get('step',0) if starts else 0
    matched=[point for point in full if point['step']==selected and point['epoch']+1==best['epoch'] and point['mae_deg'] is not None and _same(point['mae_deg'],score)]
    if not matched and selected>initial_step:raise ReportError('selected full-validation score is absent from run history')
    for event in events:
        if event.get('event')=='final_test':
            if event.get('status')!='training_complete' or event.get('selected_checkpoint_step')!=selected or event.get('test')!=summary['test']:
                raise ReportError('final test progress record disagrees with atomic summary')
    identities=_safe_identities(provenance)
    identities['selected_checkpoint_file_sha256']=_file_hash(run/'best.pt')
    identities['authoritative_checkpoint_file_sha256']=_file_hash(run/'last.pt')
    return {'schema_version':1,'report_type':'completed_benchmark','benchmark_eligible':True,
            'completion_status':'training_complete','model_label':model_label,'model_kind':kind,
            'configuration':_safe_config(config),'identities':identities,
            'protocol':{'selection_population':'independent full validation; all-label population',
                'degrees':'recomputed once from absolute radians error sums and valid angle counts',
                'user_ids':'anonymous ordinal labels within this export; no identity mapping is published',
                'limitations':['legacy whole-episode prefilter dependence','released generated-label provenance differs from paper; no paper-equivalence claim']},
            'training':{'global_step':step,'selected_checkpoint_step':selected,'best_val_mae_deg':score,
                'hand_windows':counts,'skipped_batches':skipped,
                'validation_history_complete':initial_step==0 and {point['epoch'] for point in full}==set(range(config['epochs']))},
            'test':test,'curves':curves,'incomplete_trailing_line_ignored':partial}


def _markdown(report):
    if not report['benchmark_eligible']:
        progress=report['progress']
        return '# '+report['model_label']+'：训练进度（未完成／未核验）\n\n此文件仅描述训练和验证过程，不能作为最终测试集结果。\n\n' + \
            f"- 优化器更新：{progress['global_step']}\n- 验证记录：{progress['validation_events']}\n- 空标签跳过批次：{progress['skipped_batches']}\n\n" + \
            '![Training and validation curves](training-curves.svg)\n'
    lines=['# '+report['model_label']+'：完整运行结果','',
           '已核对完整测试、验证选择检查点和协议身份；这不是论文表格等价性的声明。','',
           '| 采样 | 人口／分组 | MAE (°) | 有效角度数 | 弧度绝对误差和 |',
           '| --- | --- | ---: | ---: | ---: |']
    def row(mode,label,value):
        degree='—' if value['mae_deg'] is None else format(value['mae_deg'],'.6f')
        lines.append(f"| {mode} | {label} | {degree} | {value['valid_angle_count']} | {value['absolute_error_sum_rad']:.10g} |")
    for mode,metric in report['test']['sampling'].items():
        row(mode,'all / pooled',metric['overall'])
        for name in ('by_hand','by_split','by_component'):
            for key,value in metric[name].items():row(mode,name.removeprefix('by_')+'/'+key,value)
        for hand,splits in metric['by_hand_split'].items():
            for split,value in splits.items():row(mode,hand+'/'+split,value)
    paired=report['test'].get('paired_population')
    if paired:
        row('center','vision_valid / pooled',paired['vision_valid']['overall'])
        lines += ['',f"缺失 crop：{paired['samples']['missing_crop']} / {paired['samples']['all']}；全标签结果保留这些样本。"]
    lines += ['', '用户均值、总体标准差及匿名逐用户统计见 metrics.json。该标准差来自跨手、跨 split 合并的逐用户均值，并非每个条件的 σ；与论文条件内标准差的比较须等待独立审计。不同采样及 crop 人口不得混算。',
              '', '![Training and validation curves](training-curves.svg)', '',
              '源代码、数据、配置、划分、归一化及所选检查点的 SHA256 身份见 metrics.json。','']
    return '\n'.join(lines)


def _plot(report, output):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise ReportError('plotting requires the optional report extra: pip install -e ".[report]"') from None
    navy,teal,amber='#132B45','#158A91','#DB9A25'
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(10.2,3.5),constrained_layout=True)
        train=report['curves']['training'];validation=report['curves']['validation']
        for segment in sorted({point['segment'] for point in train}):
            points=[point for point in train if point['segment']==segment]
            axes[0].plot([point['step'] for point in points],[point['loss'] for point in points],color=navy,linewidth=1.2)
        for bounded,color,label in ((False,teal,'Full validation'),(True,amber,'Bounded validation')):
            points=[point for point in validation if point['bounded']==bounded and point['mae_deg'] is not None]
            if points:axes[1].plot([point['step'] for point in points],[point['mae_deg'] for point in points],
                color=color,marker='o',markersize=3,linewidth=1.3,linestyle=':' if bounded else '-',label=label)
        axes[0].set(title='Training objective',xlabel='Optimizer updates',ylabel='Loss (L1 rad + weighted FK)')
        axes[1].set(title='Validation MAE',xlabel='Optimizer updates',ylabel='Angular MAE (degrees)')
        for axis in axes:
            axis.grid(alpha=.2);axis.spines[['top','right']].set_visible(False)
        if axes[1].lines:axes[1].legend(frameon=False)
        title=report.get('model_label','Training progress')
        if not report['benchmark_eligible']:title+=' — INCOMPLETE / UNVERIFIED'
        fig.suptitle(title,color=navy if report['benchmark_eligible'] else amber,weight='bold')
        fig.savefig(output/'training-curves.svg',metadata={'Creator':'EgoEMG Reproduction','Date':None})
        fig.savefig(output/'training-curves.png',dpi=180,metadata={'Software':'EgoEMG Reproduction'})
        plt.close(fig)


def write_report(report, output, *, plots=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    opposite='status' if report['benchmark_eligible'] else 'metrics'
    if any((output/(opposite+suffix)).exists() for suffix in ('.json','.md')):
        raise ReportError('use separate status and completed-benchmark output directories')
    if plots:_plot(report,output)
    name='metrics' if report['benchmark_eligible'] else 'status'
    (output/(name+'.json')).write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    markdown=_markdown(report)
    if not plots:markdown=markdown.replace('![Training and validation curves](training-curves.svg)','图表未生成（显式 --no-plots）。')
    (output/(name+'.md')).write_text(markdown,encoding='utf-8')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for command in ('status','export'):
        sub=commands.add_parser(command)
        sub.add_argument('--run',required=True);sub.add_argument('--output',required=True)
        sub.add_argument('--no-plots',action='store_true')
        if command=='export':sub.add_argument('--model-label',required=True)
    args=parser.parse_args(argv)
    try:
        report=build_status(args.run) if args.command=='status' else build_export(args.run,args.model_label)
        write_report(report,args.output,plots=not args.no_plots)
    except (ReportError,KeyError,TypeError,FileNotFoundError) as error:
        parser.error(str(error) if isinstance(error,ReportError) else 'required run evidence is missing or malformed')
    print(json.dumps({'report_type':report['report_type'],'benchmark_eligible':report['benchmark_eligible']},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
