"""Complete ordered joins; synthetic records grant no model admission."""
import copy
import importlib
import math
from pathlib import Path
import os

import pytest

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',ROOT/'upstream')).resolve()


def api():
    try:return importlib.import_module('egoemg_repro.occlusion_dense_predictions')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.occlusion_dense_predictions':pytest.fail('dense prediction join missing')
        raise


def fixture(recipe='vits_s'):
    length=7790 if recipe=='vits_s' else 12000
    keys=[dict(sequence=i,split=('user'if i<2 else'gesture'),hand=('left','right')[i%2],
        ep_idx=i//2,center_idx=10000+i//2,window_start=10000+i//2-length//2,
        window_end=10000+i//2-length//2+length)for i in range(4)]
    predictions=[dict(k,mae_vision_rad=.2+i*.1,mae_emg_rad=.1,mae_fusion_rad=.15)for i,k in enumerate(keys)]
    geometry=[dict({n:k[n]for n in('sequence','hand','ep_idx','center_idx')},status='native_geometry',
        mocap_tracked=i!=1,image_stale=i==2,markers_valid=i!=2,
        occlusion_score=.12345678,visible_ratio=.87654322,
        author_occlusion_score_round6=.123457,author_visible_ratio_round6=.876543)for i,k in enumerate(keys)]
    geometry[3].update(status='invalid_zero_pose',occlusion_score=None,visible_ratio=None)
    geometry[3].pop('author_occlusion_score_round6');geometry[3].pop('author_visible_ratio_round6')
    return keys,predictions,geometry


@pytest.mark.parametrize('recipe',['vits_s','rn18_s'])
def test_all_native_keys_both_hands_zero_rows_flags_and_real_errors_retained(recipe):
    m=api();keys,p,g=fixture(recipe)
    before=copy.deepcopy((keys,p,g));out=m.join_dense_predictions(keys,p,g,recipe=recipe,upstream=UPSTREAM)
    assert (keys,p,g)==before
    assert out['counts']==dict(expected=4,predicted=4,geometry=4,joined=4,invalid_zero_pose_retained=1)
    assert [r['sequence']for r in out['raw_records']]==list(range(4))
    assert [r['hand']for r in out['csv_rows']]==['left','right','left','right']
    assert out['csv_rows'][1]['mocap_tracked']==0 and out['csv_rows'][2]['mocap_stale']==1
    assert out['csv_rows'][2]['markers_valid']==0 and out['csv_rows'][2]['mae_vision_deg']==22.92
    assert math.isnan(out['csv_rows'][3]['occlusion_score'])and out['csv_rows'][3]['mae_vision_deg']==28.65
    assert out['csv_rows'][0]['occlusion_score']==.123457
    assert out['csv_rows'][0]['delta_fusion_deg']==2.865
    assert not out['trained_model_or_full_population_admitted']and not out['figure13_reproduced']


@pytest.mark.parametrize('damage',['missing_prediction','extra_prediction','reordered_prediction','duplicate_prediction',
    'wrong_prediction_window','wrong_prediction_split','missing_geometry','reordered_geometry','duplicate_geometry',
    'wrong_geometry_hand','wrong_geometry_center','changed_rounded_score','nonfinite_geometry',
    'invented_zero_geometry','missing_geometry_flag','missing_real_branch','float_prediction_key','float_geometry_key'])
def test_bad_or_incomplete_join_rejects_without_dropping_or_repairing_records(damage):
    m=api();keys,p,g=fixture();assert m.join_dense_predictions(keys,p,g,recipe='vits_s',upstream=UPSTREAM)['counts']['joined']==4
    if damage=='missing_prediction':p.pop()
    elif damage=='extra_prediction':p.append(dict(p[-1]))
    elif damage=='reordered_prediction':p[0],p[1]=p[1],p[0]
    elif damage=='duplicate_prediction':p[1]=dict(p[0])
    elif damage=='wrong_prediction_window':p[0]['window_end']+=1
    elif damage=='wrong_prediction_split':p[0]['split']='both'
    elif damage=='missing_geometry':g.pop()
    elif damage=='reordered_geometry':g[0],g[1]=g[1],g[0]
    elif damage=='duplicate_geometry':g[1]=dict(g[0])
    elif damage=='wrong_geometry_hand':g[0]['hand']='right'
    elif damage=='wrong_geometry_center':g[0]['center_idx']+=1
    elif damage=='changed_rounded_score':g[0]['author_occlusion_score_round6']+=.000001
    elif damage=='nonfinite_geometry':g[0]['occlusion_score']=float('nan')
    elif damage=='invented_zero_geometry':g[-1]['occlusion_score']=0.
    elif damage=='missing_geometry_flag':g[0].pop('image_stale')
    elif damage=='missing_real_branch':p[0].pop('mae_emg_rad')
    elif damage=='float_prediction_key':p[0]['sequence']=0.
    elif damage=='float_geometry_key':g[0]['center_idx']=float(g[0]['center_idx'])
    with pytest.raises(ValueError):m.join_dense_predictions(keys,p,g,recipe='vits_s',upstream=UPSTREAM)


@pytest.mark.parametrize('damage',['empty','wrong_sequence','duplicate_expected','wrong_length','wrong_center','unknown_recipe'])
def test_expected_native_selection_is_not_guessed_or_renumbered(damage):
    m=api();keys,p,g=fixture();recipe='vits_s'
    if damage=='empty':keys=[];p=[];g=[]
    elif damage=='wrong_sequence':keys[0]['sequence']=5
    elif damage=='duplicate_expected':keys[1]=dict(keys[0])
    elif damage=='wrong_length':keys[0]['window_end']+=1
    elif damage=='wrong_center':keys[0]['center_idx']+=1
    elif damage=='unknown_recipe':recipe='other'
    with pytest.raises(ValueError):m.join_dense_predictions(keys,p,g,recipe=recipe,upstream=UPSTREAM)
