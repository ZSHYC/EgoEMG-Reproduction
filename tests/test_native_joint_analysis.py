"""Native group weighting contracts, independent of model or dataset IO."""
import copy
import importlib.util
import json
import math
import subprocess
import sys

import pytest


def api():
    assert importlib.util.find_spec('egoemg_repro.native_joint_analysis') is not None, 'native joint weighting is missing'
    from egoemg_repro.native_joint_analysis import native_joint_group_views
    return native_joint_group_views


def row(key, split, hand, error, count):
    return dict(key=key, split=split, hand=hand,
                trajectory_angle_error_sum_rad=[error]*22,
                trajectory_angle_valid_counts=[count]*22,
                private_subject='PRIVATE_SUBJECT_SENTINEL',
                private_path='/PRIVATE_SOURCE_SENTINEL')


def records():
    return [row('PRIVATE_uL', 'user', 'left', 1., 1),
            row('PRIVATE_uR', 'user', 'right', 1., 1),
            row('PRIVATE_gR', 'gesture', 'right', 27., 9),
            row('PRIVATE_invalid', 'gesture', 'left', 0., 0),
            row('PRIVATE_bL', 'both', 'left', 0., 2)]


def test_unequal_valid_time_and_invalid_request_distinguish_two_weights():
    result=api()(records());degrees=180/math.pi
    assert result['n_requested']==5 and result['n_with_any_valid_angle']==4
    assert result['pooled_valid_angle_time']['overall']['valid_angle_count']==13*22
    assert result['pooled_valid_angle_time']['overall']['mae_deg']==pytest.approx(29/13*degrees)
    caption=result['caption_interpretation_split_sample_weighted']
    assert caption['sample_weight_denominator']==5
    for group, value in caption['groups'].items():
        assert value['mae_deg']==pytest.approx(1.6*degrees)
        assert value['weighted_mae_sum_deg_samples']==pytest.approx(8*degrees)
        assert value['missing_nonempty_splits']==[]
        assert result['pooled_valid_angle_time']['groups'][group]['mae_deg']==pytest.approx(29/13*degrees)
    assert result['by_split']['gesture']['n_requested']==2
    assert result['by_split']['gesture']['n_with_any_valid_angle']==1


def test_hand_split_partitions_conserve_requests_and_angle_observations():
    result=api()(records());overall=result['pooled_valid_angle_time']['overall']
    for cells in (result['by_hand'].values(), result['by_split'].values(),
                  [cell for split_cells in result['by_hand_split'].values()for cell in split_cells.values()]):
        cells=list(cells)
        assert sum(c['n_requested']for c in cells)==5
        assert sum(c['n_with_any_valid_angle']for c in cells)==4
        assert sum(c['statistics']['overall']['valid_angle_count']for c in cells)==overall['valid_angle_count']
        assert math.fsum(c['statistics']['overall']['absolute_error_sum_rad']for c in cells)==overall['absolute_error_sum_rad']
    empty=result['by_hand_split']['right']['both']
    assert empty['n_requested']==0 and empty['statistics']['overall']['mae_deg']is None


def test_unobserved_wrist_in_populated_split_is_undefined_not_dropped():
    data=records()
    for sample in data:
        if sample['split']=='user':
            sample['trajectory_angle_error_sum_rad'][21]=0.
            sample['trajectory_angle_valid_counts'][21]=0
    result=api()(data)
    caption=result['caption_interpretation_split_sample_weighted']
    assert caption['sample_weight_denominator']==5
    wrist=caption['groups']['wrist_RU']
    assert wrist['mae_deg']is None and wrist['weighted_mae_sum_deg_samples']is None
    assert wrist['missing_nonempty_splits']==['user']
    assert result['pooled_valid_angle_time']['groups']['wrist_RU']['mae_deg']==pytest.approx(27/11*180/math.pi)
    assert caption['groups']['thumb']['mae_deg']==pytest.approx(1.6*180/math.pi)


def test_an_empty_split_has_no_weight_and_empty_input_has_no_zero_mae():
    result=api()([r for r in records()if r['split']!='both'])
    caption=result['caption_interpretation_split_sample_weighted']
    assert caption['sample_weight_denominator']==4
    assert caption['groups']['thumb']['mae_deg']==pytest.approx(2*180/math.pi)
    assert caption['groups']['thumb']['missing_nonempty_splits']==[]
    empty=api()([])
    assert empty['n_requested']==empty['n_with_any_valid_angle']==0
    assert empty['pooled_valid_angle_time']['overall']['mae_deg']is None
    assert all(g['mae_deg']is None for g in empty['caption_interpretation_split_sample_weighted']['groups'].values())


def test_order_independent_input_unchanged_and_public_output_has_no_private_values():
    data=records();before=copy.deepcopy(data);result=api()(data)
    assert data==before and result==api()(list(reversed(data)))
    text=json.dumps(result)
    assert 'PRIVATE'not in text and 'private_subject'not in text and 'private_path'not in text
    assert all(r['key']not in text for r in data)


@pytest.mark.parametrize('mutation', ['duplicate', 'empty_key', 'bad_hand', 'bad_split', 'bad_row', 'bad_collection'])
def test_invalid_sample_identity_or_metadata_is_rejected(mutation):
    data=records()
    if mutation=='duplicate':data[-1]['key']=data[0]['key']
    elif mutation=='empty_key':data[0]['key']=''
    elif mutation=='bad_hand':data[0]['hand']='third'
    elif mutation=='bad_split':data[0]['split']='train'
    elif mutation=='bad_row':data[0]=None
    elif mutation=='bad_collection':data=tuple(data)
    with pytest.raises(ValueError):api()(data)


@pytest.mark.parametrize('mutation', ['nonfinite', 'negative_error', 'bad_length', 'fractional_count', 'bool_count', 'zero_count_nonzero_error'])
def test_malformed_sufficient_statistics_are_rejected(mutation):
    data=records();sample=data[0]
    if mutation=='nonfinite':sample['trajectory_angle_error_sum_rad'][0]=float('nan')
    elif mutation=='negative_error':sample['trajectory_angle_error_sum_rad'][0]=-1.
    elif mutation=='bad_length':sample['trajectory_angle_valid_counts'].pop()
    elif mutation=='fractional_count':sample['trajectory_angle_valid_counts'][0]=1.5
    elif mutation=='bool_count':sample['trajectory_angle_valid_counts'][0]=True
    elif mutation=='zero_count_nonzero_error':sample['trajectory_angle_valid_counts'][0]=0
    with pytest.raises(ValueError):api()(data)


def test_reducer_import_requires_neither_torch_nor_numpy():
    api()
    process=subprocess.run([sys.executable,'-c',
        'import sys; from egoemg_repro.native_joint_analysis import native_joint_group_views; '
        'native_joint_group_views([]); assert "torch" not in sys.modules; assert "numpy" not in sys.modules'],
        capture_output=True,text=True)
    assert process.returncode==0,process.stderr


@pytest.mark.parametrize('error', [5e306, 1e308])
def test_finite_inputs_that_overflow_aggregate_or_degree_conversion_are_rejected(error):
    with pytest.raises(ValueError):
        api()([row('overflow', 'gesture', 'left', error, 1)])
