"""Ordered prediction/geometry joins, never a trained-model inference permit.

The caller must separately admit actual producer weights, normalized inputs,
complete native selections and authenticated geometric journals. This unit
checks their explicit keys and transports source-native errors/flags to CSV.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Real
from pathlib import Path

from .classic_emg import _hash
from .classic_emg_cohort import _safe, _StatGuard
from .occlusion_dense_analysis import source_analysis_contract, source_csv_rows

KEY_FIELDS = ('sequence', 'split', 'hand', 'ep_idx', 'center_idx', 'window_start', 'window_end')
GEOMETRY_FIELDS = ('sequence', 'hand', 'ep_idx', 'center_idx')
ERROR_FIELDS = ('mae_vision_rad', 'mae_emg_rad', 'mae_fusion_rad')
WINDOW_LENGTHS = dict(vits_s=7790, rn18_s=12000)
IMPORTED_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _key(record, fields):
    try:
        values = tuple(record[n] for n in fields)
        if any(type(v) is not (str if n in ('split', 'hand') else int) for n, v in zip(fields, values)):
            raise ValueError('exact typed native join key required')
        return values
    except (KeyError, TypeError) as exc:
        raise ValueError('explicit complete native join key required') from exc


def _score(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('actual finite native geometry score in [0,1] required')
    return float(value)


def join_dense_predictions(expected_keys, prediction_records, geometry_rows, *, recipe, upstream):
    """Reject missing, extra, duplicated or reordered rows; never repair a join."""
    if recipe not in WINDOW_LENGTHS:
        raise ValueError('explicit native dense recipe required')
    guard = _StatGuard([_safe(__file__)])
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != IMPORTED_SHA:
        raise ValueError('imported dense prediction join source changed')
    keys, predictions, geometry = map(list, (expected_keys, prediction_records, geometry_rows))
    if not keys or len(keys) != len(predictions) or len(keys) != len(geometry):
        raise ValueError('nonempty complete one-to-one prediction and geometry population required')
    seen = set()
    for i, key in enumerate(keys):
        _key(key, KEY_FIELDS)
        if (any(type(key[n]) is not int or key[n] < 0 for n in
                ('sequence', 'ep_idx', 'center_idx', 'window_start', 'window_end'))
            or key['sequence'] != i or key['split'] not in ('user', 'gesture', 'both')
            or key['hand'] not in ('left', 'right')
            or key['window_end'] - key['window_start'] != WINDOW_LENGTHS[recipe]
            or key['center_idx'] != key['window_start'] + WINDOW_LENGTHS[recipe] // 2):
            raise ValueError('exact ordered source window/center/hand/split keys required')
        point = _key(key, ('hand', 'ep_idx', 'center_idx'))
        if point in seen:
            raise ValueError('duplicate expected native hand point')
        seen.add(point)
    raw, invalid = [], 0
    for key, prediction, row in zip(keys, predictions, geometry):
        if _key(prediction, KEY_FIELDS) != _key(key, KEY_FIELDS):
            raise ValueError('prediction source order, window or sample key differs')
        if _key(row, GEOMETRY_FIELDS) != _key(key, GEOMETRY_FIELDS):
            raise ValueError('geometry source order or exact hand point differs')
        if not isinstance(prediction, dict) or any(n not in prediction for n in ERROR_FIELDS):
            raise ValueError('all three actual prediction errors required')
        if any(type(row.get(n)) is not bool for n in ('mocap_tracked', 'image_stale', 'markers_valid')):
            raise ValueError('actual complete geometry tracking/stale/marker flags required')
        if row.get('status') == 'invalid_zero_pose':
            if (row.get('occlusion_score', 'missing') is not None
                or row.get('visible_ratio', 'missing') is not None
                or 'author_occlusion_score_round6' in row or 'author_visible_ratio_round6' in row):
                raise ValueError('native zero-pose geometry must remain explicitly unavailable')
            score = ratio = float('nan')
            invalid += 1
        elif row.get('status') == 'native_geometry':
            try:
                score, ratio = (_score(row[n]) for n in ('occlusion_score', 'visible_ratio'))
                if (round(score, 6) != row['author_occlusion_score_round6']
                    or round(ratio, 6) != row['author_visible_ratio_round6']):
                    raise ValueError('native raw and author-rounded geometry differ')
            except KeyError as exc:
                raise ValueError('actual complete raw/rounded native geometry required') from exc
        else:
            raise ValueError('unknown native geometry row status')
        raw.append(dict({n: key[n] for n in KEY_FIELDS},
            **{n: prediction[n] for n in ERROR_FIELDS},
            mocap_tracked=row['mocap_tracked'], mocap_stale=row['image_stale'],
            markers_valid=row['markers_valid'], occlusion_score=score, visible_ratio=ratio))
    csv = source_csv_rows(raw, upstream=upstream)
    contract = source_analysis_contract(upstream)
    guard.check()
    return dict(format='ordered_native_dense_prediction_geometry_join_v1', recipe=recipe,
        counts=dict(expected=len(keys), predicted=len(predictions), geometry=len(geometry), joined=len(raw),
            invalid_zero_pose_retained=invalid),
        source_key_sha256=_hash([{n: k[n] for n in KEY_FIELDS} for k in keys]),
        source_analysis_contract_sha256=contract['sha256'], adapter_source_sha256=IMPORTED_SHA,
        raw_records=raw, csv_rows=csv, trained_model_or_full_population_admitted=False,
        scientific_eligible=False, figure13_reproduced=False, full_paper_complete=False)
