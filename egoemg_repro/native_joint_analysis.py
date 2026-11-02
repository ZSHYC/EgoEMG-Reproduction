"""Aggregate native joint statistics without granting run/source eligibility.

The official helper pools valid-angle counts. The separate caption view is an
explicit split-window-weighted interpretation, not an unpublished author script.
Only aggregates leave this module; private sample keys and fields stay private.
"""
from __future__ import annotations

import math

from .paper_analysis import JOINT_GROUPS, joint_group_table

SPLITS = ('gesture', 'user', 'both')
HANDS = ('left', 'right')


def _cell(rows):
    try:
        statistics = joint_group_table(rows)
    except OverflowError as exc:
        raise ValueError('finite joint aggregate and degree conversion required') from exc
    metrics = [statistics['overall'], *statistics['angles'], *statistics['groups'].values()]
    if any(value is not None and not math.isfinite(value)
           for metric in metrics
           for value in (metric['absolute_error_sum_rad'], metric['mae_deg'])):
        raise ValueError('finite joint aggregate and degree conversion required')
    return dict(n_requested=len(rows),
                n_with_any_valid_angle=sum(any(row['trajectory_angle_valid_counts']) for row in rows),
                statistics=statistics)


def native_joint_group_views(records: list[dict]) -> dict:
    """Retain all requested windows and report two explicitly different weights.

    Rows contain a unique private ``key``, hand/split and the existing22-angle
    sufficient statistics. Extra fields are ignored. The pooled view validates
    the statistics through the existing joint reducer; no source, trained model,
    completed process, or equivalence to the paper is certified here.
    """
    if type(records) is not list:
        raise ValueError('native joint sample rows must be a list')
    keys = set()
    for row in records:
        if (type(row) is not dict or type(row.get('key')) is not str or not row['key']
                or row['key'] in keys or row.get('hand') not in HANDS
                or row.get('split') not in SPLITS):
            raise ValueError('unique sample key and valid native test hand/split required')
        keys.add(row['key'])
    global_cell = _cell(records)
    by_hand = {hand: _cell([row for row in records if row['hand'] == hand]) for hand in HANDS}
    by_split = {split: _cell([row for row in records if row['split'] == split]) for split in SPLITS}
    by_hand_split = {
        hand: {split: _cell([row for row in records if row['hand'] == hand and row['split'] == split])
               for split in SPLITS}
        for hand in HANDS}
    caption_groups = {}
    for name in JOINT_GROUPS:
        terms = []
        missing = []
        for split, cell in by_split.items():
            if cell['n_requested'] == 0:
                continue
            mean = cell['statistics']['groups'][name]['mae_deg']
            if mean is None:
                missing.append(split)
            else:
                terms.append(cell['n_requested'] * mean)
        try:
            weighted_sum = math.fsum(terms) if not missing else None
        except OverflowError as exc:
            raise ValueError('finite sample-weighted joint aggregate required') from exc
        if weighted_sum is not None and not math.isfinite(weighted_sum):
            raise ValueError('finite sample-weighted joint aggregate required')
        caption_groups[name] = dict(
            weighted_mae_sum_deg_samples=weighted_sum,
            mae_deg=weighted_sum/len(records) if records and not missing else None,
            missing_nonempty_splits=missing)
    return dict(
        format='native_joint_group_distinct_weight_views_v1',
        n_requested=global_cell['n_requested'],
        n_with_any_valid_angle=global_cell['n_with_any_valid_angle'],
        pooled_valid_angle_time=global_cell['statistics'],
        by_hand=by_hand, by_split=by_split, by_hand_split=by_hand_split,
        caption_interpretation_split_sample_weighted=dict(
            definition='sum(split requested hand-windows * split pooled group MAE) / all requested hand-windows; not a verified author Table12 script',
            sample_weight_denominator=len(records), groups=caption_groups),
        source_or_training_completion_certified=False,
        paper_numerical_equivalence_established=False,
        full_paper_complete=False)
