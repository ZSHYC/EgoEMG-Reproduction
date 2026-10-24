"""Count-weighted angular MAE with explicit sampling and validity semantics."""
from __future__ import annotations

import math
import numpy as np


class AngleAccumulator:
    """Accumulate radians; convert to degrees only when serializing the report.

    Each update is one hand/window, shaped (angle,time). ``center_index`` is
    relative to the context-cropped prediction, not its original EMG window.
    """

    def __init__(self, mode: str = 'trajectory'):
        if mode not in ('trajectory', 'center'):
            raise ValueError('mode must be trajectory or center')
        self.mode = mode
        self.groups = {name: {} for name in ('by_hand', 'by_split', 'by_user', 'by_component')}
        self.hand_split = {}
        self.invalid_angles = 0
        self.total = [0., 0]

    def update(self, prediction, target, valid, *, hand, split, user, center_index=None):
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError('prediction and target must share (angle,time) shape')
        valid = np.broadcast_to(np.asarray(valid, dtype=bool), prediction.shape)
        if self.mode == 'center':
            if center_index is None or not 0 <= center_index < prediction.shape[-1]:
                raise ValueError('center mode requires an in-range center_index')
            prediction, target, valid = (x[:, center_index:center_index + 1]
                                        for x in (prediction, target, valid))
        values = np.abs(prediction[valid].astype(np.float64) - target[valid].astype(np.float64))
        if not np.isfinite(values).all():
            raise ValueError('nonfinite prediction or target at a valid angle')
        error, count = float(values.sum()), int(values.size)
        self.invalid_angles += int(valid.size - count)
        joint = self.hand_split.setdefault(str(hand), {}).setdefault(str(split), [0., 0])
        joint[0] += error
        joint[1] += count
        for name, indexes in (('fingers', slice(0, 20)), ('wrist', slice(20, 22))):
            component = self.groups['by_component'].setdefault(name, [0., 0])
            component_valid = valid[indexes]
            component[0] += float(np.abs(prediction[indexes][component_valid].astype(np.float64) -
                                         target[indexes][component_valid].astype(np.float64)).sum())
            component[1] += int(component_valid.sum())
        self.total[0] += error
        self.total[1] += count
        for name, key in (('by_hand', hand), ('by_split', split), ('by_user', user)):
            group = self.groups[name].setdefault(str(key), [0., 0])
            group[0] += error
            group[1] += count

    @staticmethod
    def _record(values):
        error, count = values
        return {'absolute_error_sum_rad': error, 'valid_angle_count': count,
                'mae_deg': error / count * (180 / math.pi) if count else None}

    def report(self):
        users = [self._record(value)['mae_deg'] for value in self.groups['by_user'].values() if value[1]]
        return {'mode': self.mode, 'overall': self._record(self.total),
                'invalid_angle_count': self.invalid_angles,
                'by_hand_split': {hand: {split: self._record(value) for split, value in sorted(splits.items())}
                                  for hand, splits in sorted(self.hand_split.items())},
                'user_summary': {'mean_mae_deg': float(np.mean(users)) if users else None,
                                 'std_mae_deg': float(np.std(users)) if users else None,
                                 'user_count': len(users),
                                 'aggregation': 'unweighted per-user means; population standard deviation (ddof=0)'},
                **{name: {key: self._record(value) for key, value in sorted(groups.items())}
                   for name, groups in self.groups.items()}}


class PairedCenterAccumulator:
    """Retain every requested center and expose the crop-present comparison set."""

    def __init__(self):
        self.all = AngleAccumulator('center')
        self.vision_valid = AngleAccumulator('center')
        self.samples = {'all': 0, 'vision_valid': 0, 'missing_crop': 0}
        self.missing_by_hand, self.missing_by_split = {}, {}

    def update(self, prediction, target, valid, *, vision_valid, hand, split, user):
        metadata = dict(hand=hand, split=split, user=user, center_index=0)
        self.all.update(prediction, target, valid, **metadata)
        self.samples['all'] += 1
        if vision_valid:
            self.vision_valid.update(prediction, target, valid, **metadata)
            self.samples['vision_valid'] += 1
        else:
            self.samples['missing_crop'] += 1
            for counts, key in ((self.missing_by_hand, str(hand)), (self.missing_by_split, str(split))):
                counts[key] = counts.get(key, 0) + 1

    def report(self):
        return {'all': self.all.report(), 'vision_valid': self.vision_valid.report(),
                'samples': dict(self.samples), 'missing_by_hand': dict(self.missing_by_hand),
                'missing_by_split': dict(self.missing_by_split),
                'definition': 'all requested labeled centers; vision_valid is the identical crop-present subset'}
