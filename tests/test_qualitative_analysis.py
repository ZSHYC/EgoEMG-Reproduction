"""Original Figure7 reduction, tracking filter, stable ranking and CSV oracle."""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="native tensor checks require the optional Torch tier")

from test_occlusion_dense_data import UPSTREAM


def api():
    from egoemg_repro import qualitative_analysis
    return qualitative_analysis


def case():
    groups=[(0,'left')]*3+[(0,'right'),(1,'left'),(1,'right'),(2,'left'),(2,'right')]
    keys=[dict(sequence=i,split=('user','gesture','both')[g],hand=h,ep_idx=0,
        center_idx=i*7790+3895,window_start=i*7790,window_end=(i+1)*7790,
        native_index=i,group_index=g)for i,(g,h)in enumerate(groups)]
    target=np.zeros((8,22),np.float32);vision=np.full((8,22),.1,np.float32)
    vision[0]=.10000001;vision[7,20:]=.9
    arrays=dict(targets=target,vision_prediction=vision,fusion_prediction=target.copy(),
        emg_prediction=np.full((8,22),.2,np.float32),label_valid_mask=np.ones((8,1),bool),
        crop_valid_mask=np.ones((8,1),bool))
    arrays['label_valid_mask'][5]=False;arrays['crop_valid_mask'][5:7]=False
    arrays['emg_full_prediction']=np.repeat(arrays['emg_prediction'][:,:,None],3,axis=2)
    for role in ('vision','emg','fusion'):
        arrays['mae_'+role+'_rad']=np.array([float((torch.from_numpy(arrays[role+'_prediction'][i])-torch.from_numpy(target[i])).abs().mean())for i in range(8)],np.float64)
    flags=[dict({n:k[n]for n in ('sequence','split','hand','ep_idx','center_idx','window_start','window_end')},
        mocap_tracked=i!=2,image_stale=i==3,markers_valid=i!=4)for i,k in enumerate(keys)]
    return keys,arrays,flags


def original_oracle(keys,arrays,flags):
    """Execute unchanged source reduction statements, filter loop, sort and CSV."""
    tree=ast.parse((UPSTREAM/'scripts/eval/find_fusion_wins.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='evaluate_per_sample')
    sample=next(n for n in ast.walk(fn)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='i')
    code=compile(ast.Module(body=sample.body[:7],type_ignores=[]),'<actual-F7-per-sample-errors>','exec')
    rows=[]
    for i,k in enumerate(keys):
        env=dict(i=i,y_v_all=torch.from_numpy(arrays['vision_prediction']),
            preds_fusion=torch.from_numpy(arrays['fusion_prediction']),
            y_emg_all=torch.from_numpy(arrays['emg_prediction']),targets=torch.from_numpy(arrays['targets']))
        exec(code,env);r=env['result'];r.update(k);rows.append(r)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    filt=next(n for n in main.body if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='r'and isinstance(n.iter,ast.Name)and n.iter.id=='all_results')
    bycenter={k['center_idx']:f for k,f in zip(keys,flags)}
    class Values:
        def __init__(self,name):self.name=name
        def __getitem__(self,center):
            f=bycenter[center]
            return np.array([f['markers_valid']],bool)if self.name.startswith('mocap_')and self.name.endswith('_valid')else f[self.name]
    env=dict(all_results=rows,tracked_mm=Values('mocap_tracked'),stale_mm=Values('image_stale'),
        tracked_count=0,filtered_results=[],_load_mm=lambda p,n:Values(n),Path=Path,
        args=type('Args',(),dict(data_location='/unused'))())
    exec(compile(ast.Module(body=[filt],type_ignores=[]),'<actual-F7-filter>','exec'),env)
    env['all_results']=env['filtered_results']
    sort=next(n for n in main.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='sort')
    csv=next(n for n in main.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='csv_rows'for t in n.targets))
    exec(compile(ast.Module(body=[sort,csv],type_ignores=[]),'<actual-F7-sort-and-CSV>','exec'),env)
    return rows,env['all_results'],env['csv_rows']


def test_registered_producers_are_required_separately_from_numeric_analysis():
    keys,arrays,flags=case();v=api().rank_qualitative_predictions(keys,arrays,flags,recipe='vits_s',upstream=UPSTREAM)
    assert not v['trained_model_or_full_input_admitted']and not v['figure7_reproduced']


# Catches masking wrists/labels, crop filtering, ranking rounded errors, unstable
# ties, degrees conversion before subtracting, or a wrong tracked/stale predicate.
def test_full_unmasked22_reduction_filters_and_stable_order_equal_executed_original_source():
    keys,arrays,flags=case();before=copy.deepcopy((keys,arrays,flags));raw,ranked,csv=original_oracle(keys,arrays,flags)
    result=api().rank_qualitative_predictions(keys,arrays,flags,recipe='vits_s',upstream=UPSTREAM)
    assert [r['sequence']for r in result['ranked_rows']]==[7,0,1,5,6]
    assert result['csv_rows']==csv
    assert result['counts']==dict(selected=8,tracked_filter_pass=5,rejected=3,crop_invalid_retained=2,label_invalid_retained=1)
    for actual,expected in zip(result['raw_rows'],raw):
        assert actual['key']==keys[expected['sequence']]
        for n in ('mae_vision','mae_emg','mae_fusion','delta_mae'):assert actual[n]==expected[n]
    assert [r['sequence']for r in ranked]==[r['sequence']for r in result['ranked_rows']]
    assert result['raw_rows'][7]['mae_vision']!=float(np.mean(arrays['vision_prediction'][7,:20]))
    assert keys==before[0]and flags==before[2]
    for n,a in arrays.items():np.testing.assert_array_equal(a,before[1][n])


@pytest.mark.parametrize('damage',['missing_array','float64_prediction','nonfinite','bad_shape','mask_dtype','midpoint','saved_error','missing_flag','flag_type','flag_order','extra_flag','key_window','key_order','unknown_recipe'])
def test_incomplete_or_misaligned_raw_arrays_and_flags_cannot_be_ranked(damage):
    keys,a,f=case();recipe='vits_s'
    if damage=='missing_array':a.pop('emg_prediction')
    elif damage=='float64_prediction':a['vision_prediction']=a['vision_prediction'].astype(np.float64)
    elif damage=='nonfinite':a['targets'][0,0]=np.nan
    elif damage=='bad_shape':a['vision_prediction']=a['vision_prediction'][:,:20]
    elif damage=='mask_dtype':a['crop_valid_mask']=a['crop_valid_mask'].astype(np.int8)
    elif damage=='midpoint':a['emg_full_prediction'][0,0,1]+=1
    elif damage=='saved_error':a['mae_vision_rad'][0]+=.001
    elif damage=='missing_flag':f[0].pop('markers_valid')
    elif damage=='flag_type':f[0]['image_stale']=0
    elif damage=='flag_order':f[0],f[1]=f[1],f[0]
    elif damage=='extra_flag':f.append(copy.deepcopy(f[0]))
    elif damage=='key_window':keys[0]['window_end']+=1
    elif damage=='key_order':keys[0],keys[1]=keys[1],keys[0]
    else:recipe='vits_l'
    with pytest.raises(ValueError):api().rank_qualitative_predictions(keys,a,f,recipe=recipe,upstream=UPSTREAM)


def test_zero_tracking_population_is_explicit_and_does_not_promote_a_figure():
    keys,a,f=case()
    for r in f:r['mocap_tracked']=False
    v=api().rank_qualitative_predictions(keys,a,f,recipe='vits_s',upstream=UPSTREAM)
    assert v['counts']['selected']==8 and v['counts']['tracked_filter_pass']==0
    assert not v['ranking_available']and v['csv_rows']==[]and v['ranked_rows']==[]
