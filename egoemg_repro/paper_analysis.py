"""Scientific sufficient-statistic reductions; no model, IO or training imports.

Official gesture formulas and the retained legacy masked protocol stay separate.
Only aggregate fields leave this module, never private sample metadata or keys.
"""
from __future__ import annotations

from collections import Counter
import math

DEGREES=180/math.pi
MODELS=('emg','vision','fusion')
FAMILIES=('Single-hand','Symmetric bimanual','Asymmetric bimanual')
JOINT_GROUPS={
    'proximal':[0,1,4,5,8,9,12,13,16,17],
    'mid':[2,6,10,14,18], 'distal':[3,7,11,15,19],
    'thumb':[0,1,2,3], 'index':[4,5,6,7], 'middle':[8,9,10,11],
    'ring':[12,13,14,15], 'pinky':[16,17,18,19],
    'wrist_FE':[20], 'wrist_RU':[21],
}


def _number(value):
    if type(value) not in (float,int) or not math.isfinite(value) or value<0:
        raise ValueError('finite nonnegative sufficient-statistic value required')
    return float(value)


def _stats(row,prefix):
    if type(row) is not dict:raise ValueError('sufficient-statistic row required')
    errors=row.get(prefix+'_angle_error_sum_rad');counts=row.get(prefix+'_angle_valid_counts')
    if type(errors) is not list or type(counts) is not list or len(errors)!=22 or len(counts)!=22:
        raise ValueError('exactly22 angle error sums and valid counts required')
    values=[]
    for error,count in zip(errors,counts):
        error=_number(error)
        if type(count) is not int or count<0 or (count==0 and error!=0):
            raise ValueError('nonnegative integer count with zero error when empty required')
        values.append(error)
    return values,counts


def _metric(error,count):
    error=_number(error)
    if type(count) is not int or count<0 or (count==0 and error!=0):
        raise ValueError('invalid aggregate count/error')
    return dict(absolute_error_sum_rad=error,valid_angle_count=count,
                mae_deg=error/count*DEGREES if count else None)


def joint_group_table(rows):
    """Pool actual angle-time counts; groups intentionally overlap."""
    if type(rows) is not list:raise ValueError('joint statistic rows must be a list')
    validated=[_stats(row,'trajectory') for row in rows]
    errors=[math.fsum(row[0][j] for row in validated) for j in range(22)]
    counts=[sum(row[1][j] for row in validated) for j in range(22)]
    def group(indices):
        return _metric(math.fsum(errors[j] for j in indices),sum(counts[j] for j in indices))
    return dict(overall=group(range(22)),
        angles=[dict(index=j,**group([j])) for j in range(22)],
        groups={name:dict(indices=list(indices),**group(indices)) for name,indices in JOINT_GROUPS.items()},
        aggregation='sum radians absolute errors / valid angle-time count; overlapping groups are not additive')


def _definitions(definitions):
    if type(definitions) is not list or len(definitions)!=60:
        raise ValueError('complete60-class semantic definitions required')
    result={}
    for row in definitions:
        if (type(row) is not dict or type(row.get('id')) is not int or not 0<=row['id']<60
                or row['id'] in result or type(row.get('name')) is not str or not row['name']
                or row.get('family') not in FAMILIES):
            raise ValueError('invalid/duplicate gesture definition')
        result[row['id']]={name:row[name] for name in ('id','name','family')}
    if Counter(row['family'] for row in result.values())!=dict(zip(FAMILIES,(30,18,12))):
        raise ValueError('gesture family definition counts differ')
    return result


def _paired(records):
    if type(records) is not dict or set(records)!=set(MODELS):
        raise ValueError('all three independently selected model records required')
    paired={};reference=None;metadata={}
    for model in MODELS:
        if type(records[model]) is not list:raise ValueError('model records must be lists')
        selected={}
        for row in records[model]:
            if type(row) is not dict:raise ValueError('model sample row required')
            key=row.get('key')
            if type(key) is not str or not key or key in selected:
                raise ValueError('nonempty unique sample keys required')
            gesture=row.get('gesture_class')
            if (type(gesture) is not int or not -1<=gesture<60 or row.get('hand') not in ('left','right')
                    or row.get('split') not in ('gesture','user','both')
                    or type(row.get('active')) is not bool or type(row.get('crop_present')) is not bool):
                raise ValueError('valid paired gesture/hand/split/crop/active metadata required')
            errors,counts=_stats(row,'center')
            if any(count not in (0,1) for count in counts):
                raise ValueError('one observation at most for each center-angle required')
            literal=_number(row.get('official_unmasked_mean_error_rad'))
            target=row.get('target_sha256')
            if type(target) is not str or len(target)!=64 or any(c not in '0123456789abcdef' for c in target):
                raise ValueError('actual center target tensor digest required')
            identity=tuple(row[name] for name in ('hand','split','gesture_class','active','crop_present','target_sha256'))
            if model==MODELS[0]:metadata[key]=identity
            elif key not in metadata or metadata[key]!=identity:
                raise ValueError('model sample metadata mismatch')
            selected[key]=(errors,counts,literal)
        if reference is None:reference=set(selected)
        elif set(selected)!=reference:raise ValueError('model sample key population mismatch')
        paired[model]=selected
    for key in sorted(reference):
        if any(paired[model][key][1]!=paired[MODELS[0]][key][1] for model in MODELS[1:]):
            raise ValueError('paired target valid-angle counts differ')
    return paired,metadata


def gesture_error_tables(records,definitions):
    """Pair by key, then preserve both masked pooled and official sample means.

    The latter consumes actual raw-token-midpoint/vision-center means from the
    collector; it cannot be reconstructed from masked aligned-center totals.
    """
    definitions=_definitions(definitions);paired,metadata=_paired(records)
    def aggregate(keys):
        keys=sorted(keys);masked={};literal={}
        for model in MODELS:
            values=[paired[model][key] for key in keys]
            masked[model]=_metric(math.fsum(math.fsum(x[0]) for x in values),sum(sum(x[1]) for x in values))
            total=math.fsum(x[2] for x in values)
            literal[model]=dict(sample_error_sum_rad=total,sample_count=len(keys),
                mae_deg=total/len(keys)*57.3 if keys else None)
        def gain(values):
            vision=values['vision']['mae_deg'];fusion=values['fusion']['mae_deg']
            return vision-fusion if vision is not None and fusion is not None else None
        return dict(n_requested=len(keys),n_valid=sum(bool(sum(paired['emg'][key][1])) for key in keys),
            masked_pooled=masked,official_unmasked=literal,
            masked_gain_vision_fusion_deg=gain(masked),official_gain_vision_fusion_deg=gain(literal))
    views={}
    for condition in ('all_test','gesture_only'):
        for crops in ('all_requested','crop_present'):
            selected=[key for key,meta in metadata.items()
                if (condition=='all_test' or meta[1]=='gesture') and (crops=='all_requested' or meta[4])]
            eligible=[key for key in selected if metadata[key][2] not in (-1,50)]
            rows=[dict(definitions[j],**aggregate([key for key in eligible if metadata[key][2]==j]))
                  for j in range(60) if j!=50]
            families=[dict(family=family,observed_gestures=sum(row['n_requested']>0 for row in rows if row['family']==family),
                **aggregate([key for key in eligible if definitions[metadata[key][2]]['family']==family])) for family in FAMILIES]
            views[condition+'/'+crops]=dict(
                counts=dict(requested=len(selected),raw_excluded=sum(metadata[key][2]==50 for key in selected),
                    unassigned_excluded=sum(metadata[key][2]==-1 for key in selected),eligible=len(eligible),
                    valid_samples=sum(bool(sum(paired['emg'][key][1])) for key in eligible)),
                rows=rows,families=families,overall=aggregate(eligible))
    return dict(views=views,protocols={
        'masked_pooled':'aligned center; sum errors / valid-angle counts; degrees=180/pi',
        'official_unmasked':'collector raw EMG-token midpoint / vision center; equal-sample22-angle means; degrees=57.3'},
        raw_class_excluded=50,unassigned_class_excluded=-1,
        activity_policy='recorded and pair-checked; not filtered',
        definition_count=60,analysis_definition_count=59,
        population='retained declared legacy experiments; not PDF/current release equivalence')


def verify_metric(current,baseline):
    """Keep the existing count-exact and1e-4degree baseline agreement gate."""
    def checked(value):
        if type(value) is not dict:raise ValueError('metric record required')
        expected=_metric(value.get('absolute_error_sum_rad'),value.get('valid_angle_count'))
        mae=value.get('mae_deg')
        if expected['mae_deg'] is None:
            if mae is not None:raise ValueError('empty metric must have null MAE')
        elif mae is None or abs(_number(mae)-expected['mae_deg'])>1e-10:
            raise ValueError('metric MAE does not match radians sum/count')
        return expected
    current=checked(current);baseline=checked(baseline)
    count=baseline['valid_angle_count']
    if current['valid_angle_count']!=count:raise ValueError('baseline angle count mismatch')
    if count and (abs(current['mae_deg']-baseline['mae_deg'])>1e-4 or
            abs(current['absolute_error_sum_rad']-baseline['absolute_error_sum_rad'])>count*math.radians(1e-4)):
        raise ValueError('baseline sum/MAE mismatch at fixed tolerance')
