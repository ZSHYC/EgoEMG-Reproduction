"""Hand-calculated scientific aggregates: counts, pairing and protocol separation."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('egoemg_repro.paper_analysis') is not None, 'paper analysis is missing'
    from egoemg_repro import paper_analysis
    return paper_analysis


def definitions():
    return json.loads((Path(__file__).resolve().parents[1] /
        'results/dataset-semantics/gesture-vocabulary.json').read_text())['rows']


def sample(key, gesture, error, count=1, *, split='gesture', crop=True, literal=None):
    return dict(key=key, gesture_class=gesture, hand='left', split=split, active=True,
        crop_present=crop, target_sha256='a'*64, center_angle_error_sum_rad=[error]+[0.]*21,
        center_angle_valid_counts=[1]*count+[0]*(22-count),
        official_unmasked_mean_error_rad=error if literal is None else literal,
        private_subject='PRIVATE_ID_SENTINEL', private_path='/private/source/path')


def records():
    # Eligible a/b/c: Vision pooled 16/4=4; equal-sample literal (2+9+5)/3=16/3.
    # Fusion pooled 10/4=2.5; equal-sample literal (1+6+3)/3=10/3.
    values={'vision':[2.,9.,5.,11.,13.], 'fusion':[1.,6.,3.,8.,10.], 'emg':[4.,12.,6.,15.,17.]}
    out={}
    for model, errors in values.items():
        out[model]=[
            sample('a',0,errors[0],split='gesture'),
            sample('b',0,errors[1],count=2,split='user',crop=False),
            sample('c',33,errors[2],split='both'),
            sample('raw',50,errors[3]),
            sample('unassigned',-1,errors[4]),
        ]
    return out


def test_joint_groups_pool_angle_counts_and_preserve_overlap_and_empty_groups():
    # Proximal and thumb overlap at0/1. Averaging per-angle means would give2,
    # but the actual sum/count is10/4=2.5rad.
    sums=[0.]*22;counts=[0]*22
    sums[0]=1.;counts[0]=1;sums[1]=9.;counts[1]=3
    sums[20]=2.;counts[20]=1;sums[21]=6.;counts[21]=2
    result=api().joint_group_table([dict(trajectory_angle_error_sum_rad=sums,
        trajectory_angle_valid_counts=counts)])
    assert result['groups']['proximal']['valid_angle_count']==4
    assert result['groups']['proximal']['mae_deg']==pytest.approx(2.5*180/math.pi)
    assert result['groups']['thumb']['absolute_error_sum_rad']==10.
    assert result['groups']['wrist_FE']['mae_deg']==pytest.approx(2*180/math.pi)
    assert result['groups']['wrist_RU']['mae_deg']==pytest.approx(3*180/math.pi)
    assert result['groups']['mid']['mae_deg'] is None
    assert result['overall']['absolute_error_sum_rad']==18.
    assert result['overall']['valid_angle_count']==7
    assert len(result['angles'])==22


def test_joint_sum_is_order_independent_and_does_not_fabricate_empty_errors():
    row=dict(trajectory_angle_error_sum_rad=[0.]*22,trajectory_angle_valid_counts=[0]*22)
    assert api().joint_group_table([row])['overall']['mae_deg'] is None
    assert api().joint_group_table([])['overall']['valid_angle_count']==0


@pytest.mark.parametrize('error,count',[(float('nan'),1),(float('inf'),1),(-1.,1),(1.,0),(0.,-1),(0.,1.5),(0.,True)])
def test_invalid_sufficient_statistics_cannot_be_published(error,count):
    row=dict(trajectory_angle_error_sum_rad=[error]+[0.]*21,
             trajectory_angle_valid_counts=[count]+[0]*21)
    with pytest.raises(ValueError):api().joint_group_table([row])


def test_gesture_pairing_keeps_pooled_and_literal_equal_sample_formula_separate():
    result=api().gesture_error_tables(records(),definitions())
    view=result['views']['all_test/all_requested']
    assert view['counts']==dict(requested=5,raw_excluded=1,unassigned_excluded=1,eligible=3,valid_samples=3)
    assert view['overall']['masked_pooled']['vision']['mae_deg']==pytest.approx(4*180/math.pi)
    assert view['overall']['official_unmasked']['vision']['mae_deg']==pytest.approx((16/3)*57.3)
    assert view['overall']['masked_gain_vision_fusion_deg']==pytest.approx(1.5*180/math.pi)
    assert len(view['rows'])==59
    by_id={x['id']:x for x in view['rows']}
    assert 50 not in by_id
    assert by_id[0]['n_requested']==2
    assert by_id[0]['masked_pooled']['vision']['mae_deg']==pytest.approx((11/3)*180/math.pi)
    assert by_id[1]['n_requested']==0 and by_id[1]['official_unmasked']['vision']['mae_deg'] is None
    families={x['family']:x for x in view['families']}
    assert families['Single-hand']['official_unmasked']['vision']['mae_deg']==pytest.approx(5.5*57.3)
    assert families['Asymmetric bimanual']['n_requested']==1
    assert families['Symmetric bimanual']['n_requested']==0
    assert 'PRIVATE_ID_SENTINEL' not in json.dumps(result) and '/private/source/path' not in json.dumps(result)
    assert '"key"' not in json.dumps(result)


def test_crop_and_gesture_split_views_are_filters_of_the_same_paired_population():
    result=api().gesture_error_tables(records(),definitions())['views']
    assert result['all_test/crop_present']['overall']['n_requested']==2
    assert result['all_test/crop_present']['overall']['masked_pooled']['vision']['mae_deg']==pytest.approx(3.5*180/math.pi)
    assert result['gesture_only/all_requested']['overall']['n_requested']==1
    assert result['gesture_only/all_requested']['overall']['official_unmasked']['vision']['mae_deg']==pytest.approx(2*57.3)


def test_models_pair_by_key_not_iteration_order():
    data=records();expected=api().gesture_error_tables(data,definitions())
    data['emg'].reverse();data['fusion']=[data['fusion'][i] for i in (2,4,0,3,1)]
    assert api().gesture_error_tables(data,definitions())==expected


def test_equal_masks_cannot_pair_predictions_against_different_target_tensors():
    data=records();data['fusion'][0]['target_sha256']='b'*64
    with pytest.raises(ValueError):api().gesture_error_tables(data,definitions())


@pytest.mark.parametrize('mutation',['duplicate','missing','gesture','hand','split','crop','active','valid_counts'])
def test_mismatched_pairing_or_target_validity_is_rejected(mutation):
    data=records()
    if mutation=='duplicate':data['vision'].append(copy.deepcopy(data['vision'][0]))
    elif mutation=='missing':data['vision'].pop()
    elif mutation=='gesture':data['vision'][0]['gesture_class']=1
    elif mutation=='hand':data['vision'][0]['hand']='right'
    elif mutation=='split':data['vision'][0]['split']='user'
    elif mutation=='crop':data['vision'][0]['crop_present']=False
    elif mutation=='active':data['vision'][0]['active']=False
    else:data['vision'][0]['center_angle_valid_counts'][0]=2
    with pytest.raises(ValueError):api().gesture_error_tables(data,definitions())


def test_all_invalid_center_remains_requested_but_has_null_masked_mean():
    data={m:[sample('empty',0,0.,count=0,literal=3.)] for m in ('emg','vision','fusion')}
    result=api().gesture_error_tables(data,definitions())['views']['all_test/all_requested']
    assert result['counts']['eligible']==1 and result['counts']['valid_samples']==0
    assert result['overall']['masked_pooled']['vision']['mae_deg'] is None
    assert result['overall']['official_unmasked']['vision']['mae_deg']==pytest.approx(3*57.3)


@pytest.mark.parametrize('field,value',[('gesture_class',60),('gesture_class',-2),('active',1),('official_unmasked_mean_error_rad',float('nan'))])
def test_malformed_labels_or_unmasked_means_are_not_silently_dropped(field,value):
    data=records()
    for model in data:data[model][0][field]=value
    with pytest.raises(ValueError):api().gesture_error_tables(data,definitions())


def test_incomplete_definitions_or_a_missing_model_cannot_fill_a_paper_table():
    with pytest.raises(ValueError):api().gesture_error_tables(records(),definitions()[:-1])
    data=records();data.pop('emg')
    with pytest.raises(ValueError):api().gesture_error_tables(data,definitions())


def test_baseline_verification_keeps_existing_absolute_tolerance_and_exact_counts():
    baseline=dict(absolute_error_sum_rad=2.,valid_angle_count=2,mae_deg=180/math.pi)
    api().verify_metric(dict(baseline),baseline)
    changed=dict(baseline,valid_angle_count=3,mae_deg=2/3*180/math.pi)
    with pytest.raises(ValueError):api().verify_metric(changed,baseline)
    changed=dict(absolute_error_sum_rad=2.+math.radians(.001)*2,valid_angle_count=2,mae_deg=180/math.pi+.001)
    with pytest.raises(ValueError):api().verify_metric(changed,baseline)
