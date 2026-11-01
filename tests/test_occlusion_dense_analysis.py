"""Source-native CSV/statistics integration; synthetic numbers are not model results."""
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=Path(os.environ.get('EGOEMG_UPSTREAM',ROOT/'upstream')).resolve()


def api():
    try:return importlib.import_module('egoemg_repro.occlusion_dense_analysis')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.occlusion_dense_analysis':pytest.fail('dense source analysis missing')
        raise


def record(i=0,**kwargs):
    return dict(split='user',hand='left',ep_idx=0,center_idx=10000+i,
        mocap_tracked=True,mocap_stale=False,markers_valid=True,
        occlusion_score=.01234567,visible_ratio=.98765432,
        mae_vision_rad=1.,mae_emg_rad=.5,mae_fusion_rad=.25,**kwargs)


def frame(occlusions,vision,fusion,delta):
    pd=pytest.importorskip('pandas');n=len(occlusions)
    return pd.DataFrame(dict(split=['user']*n,hand=['left']*n,ep_idx=[0]*n,center_idx=np.arange(n),
        mocap_tracked=[1]*n,mocap_stale=[0]*n,markers_valid=[1]*n,
        occlusion_score=occlusions,visible_ratio=np.ones(n),mae_vision_deg=vision,
        mae_emg_deg=np.ones(n),mae_fusion_deg=fusion,delta_fusion_deg=delta))


def regular():
    occ=np.repeat([.0125,.0625,.1125,.1625],100)
    vis=np.repeat([6.,8.,10.,12.],100)+np.tile([-1.,1.],200)
    fus=np.repeat([5.,6.,7.,8.],100)+np.tile([-1.,1.],200)
    return frame(occ,vis,fus,vis-fus)


def analyze(df,tmp_path,name='source.csv'):
    pytest.importorskip('scipy');p=tmp_path/name;df.to_csv(p,index=False)
    return api().analyze_source_csv(p,upstream=UPSTREAM)


def test_source_metadata_and_csv_use_real_degree_rounding_and_keep_every_row():
    m=api();c=m.source_analysis_contract(UPSTREAM)
    assert c['degree_factor']==57.3 and c['error_round_decimals']==4 and c['geometry_round_decimals']==6
    assert c['min_samples_per_bin']==100 and c['minimum_final_bin']==10
    assert not c['source_error_uses_mask']and c['spline_degree']==3 and not c['figure13_reproduced']
    rows=m.source_csv_rows([record(),dict(record(1),occlusion_score=np.nan)])
    assert len(rows)==2 and rows[0]['mae_vision_deg']==57.3 and rows[0]['mae_emg_deg']==28.65
    assert rows[0]['mae_fusion_deg']==14.325 and rows[0]['delta_fusion_deg']==42.975
    assert rows[0]['occlusion_score']==.012346 and rows[0]['visible_ratio']==.987654
    assert math.isnan(rows[1]['occlusion_score'])and rows[1]['center_idx']==10001


def test_source_delta_is_rounded_from_rad_difference_not_rounded_mae_columns():
    r=dict(record(),mae_vision_rad=1.000049/57.3,mae_fusion_rad=.999951/57.3)
    out=api().source_csv_rows([r])[0]
    assert out['mae_vision_deg']==out['mae_fusion_deg']==1.
    assert out['delta_fusion_deg']==.0001


@pytest.mark.parametrize('damage',['missing_branch','negative_error','non_numeric_error','invalid_hand','duplicate_key'])
def test_missing_or_malformed_raw_prediction_records_never_turn_into_zero_scores(damage):
    rows=[record()]
    if damage=='missing_branch':rows[0].pop('mae_emg_rad')
    elif damage=='negative_error':rows[0]['mae_vision_rad']=-1.
    elif damage=='non_numeric_error':rows[0]['mae_vision_rad']='bad'
    elif damage=='invalid_hand':rows[0]['hand']='unknown'
    else:rows.append(dict(rows[0]))
    with pytest.raises(ValueError):api().source_csv_rows(rows)


def test_literal_bins_means_sample_sem_and_spearman_have_hand_derived_results(tmp_path):
    out=analyze(regular(),tmp_path)
    assert out['input_rows']==out['filtered_rows']==out['binned_rows']==400
    assert out['filtered_source_rows']==list(range(400))and out['filtered_not_binned_source_rows']==[]
    bins=out['bins'];assert [b['count']for b in bins]==[100]*4
    np.testing.assert_allclose([b['occlusion_mean']for b in bins],[.0125,.0625,.1125,.1625],atol=1e-15)
    np.testing.assert_allclose([b['vision_mean_deg']for b in bins],[6,8,10,12],atol=1e-15)
    np.testing.assert_allclose([b['fusion_mean_deg']for b in bins],[5,6,7,8],atol=1e-15)
    np.testing.assert_allclose([b['vision_sem_deg']for b in bins],[1/math.sqrt(99)]*4,atol=1e-15)
    assert [b['delta_mean_deg']for b in bins]==[1.,2.,3.,4.]and [b['delta_sem_deg']for b in bins]==[0.]*4
    assert out['spearman']['rho']==pytest.approx(1.,abs=1e-15) and out['spearman']['sample_count']==400
    assert len(out['trend']['x'])==200
    x=np.asarray(out['trend']['x']);y=np.asarray(out['trend']['y'])
    np.testing.assert_allclose(y,20*x+.75,atol=1e-12)
    assert not out['scientific_eligible']and not out['figure13_reproduced']


def test_sparse_bins_accumulate_until100_and_keep_source_order(tmp_path):
    occ=np.repeat([.001,.051,.101,.151,.201,.251],[30,40,30,100,100,100])
    v=np.repeat([6.,6.,6.,8.,10.,12.],[30,40,30,100,100,100]);f=v-1.
    out=analyze(frame(occ,v,f,v-f),tmp_path)
    assert [b['count']for b in out['bins']]==[100]*4
    np.testing.assert_allclose([b['occlusion_mean']for b in out['bins']],[.051,.151,.201,.251],atol=1e-15)
    assert out['bins'][0]['source_rows']==list(range(100))
    assert out['spearman']['rho']is None and out['spearman']['native_nonfinite']


def test_rounding_threshold_nan_filter_final_lt10_and_right_edge_are_source_exact(tmp_path):
    pd=pytest.importorskip('pandas');df=regular()
    extra=frame([.201]*9+[.25]+[np.nan,.1,.1],[5.]*10+[5.,.01,.0101],[4.]*13,[1.]*13)
    df=pd.concat([df,extra],ignore_index=True);out=analyze(df,tmp_path)
    # Native >0.01 excludes the exact threshold and NaN geometry; all other
    # filtered samples enter Spearman even when sparse/right-edge bins omit them.
    assert out['input_rows']==413 and out['filtered_rows']==411
    assert 410 not in out['filtered_source_rows']and 411 not in out['filtered_source_rows']and 412 in out['filtered_source_rows']
    assert out['binned_rows']==401 and out['filtered_not_binned_source_rows']==list(range(400,410))
    assert out['spearman']['sample_count']==411


@pytest.mark.parametrize('case',['empty','constant_occlusion','too_few_bins'])
def test_unplottable_source_populations_fail_without_inventing_a_trend(tmp_path,case):
    if case=='empty':df=frame([],[],[],[])
    elif case=='constant_occlusion':df=frame([.2]*400,[5.]*400,[4.]*400,[1.]*400)
    else:df=frame([.0125]*100+[.0625]*100,[5.]*200,[4.]*200,[1.]*200)
    with pytest.raises(ValueError):analyze(df,tmp_path)


def test_metadata_does_not_import_torch_scipy_pandas_or_plotting():
    api()
    code='''import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in('torch','scipy','pandas','matplotlib'):raise AssertionError('metadata unexpectedly imports '+name)
sys.meta_path.insert(0,Block())
from egoemg_repro.occlusion_dense_analysis import source_analysis_contract
c=source_analysis_contract(sys.argv[1]);assert not c['figure13_reproduced'];print(json.dumps({'metadata_only':True}))
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(UPSTREAM)],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr[-1500:]


def test_modified_pinned_plot_source_rejects_before_numeric_execution(tmp_path):
    m=api();copy=tmp_path/'source'
    p=subprocess.run(['git','clone','--quiet','--shared',str(UPSTREAM),str(copy)],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    path=copy/'scripts/paper/plot_occlusion_fusion.py'
    with path.open('a')as f:f.write('\n# changed source\n')
    with pytest.raises(ValueError):m.source_analysis_contract(copy)


def test_statistics_match_actual_unmodified_author_main_and_real_png(tmp_path):
    pytest.importorskip('scipy');pytest.importorskip('matplotlib');pd=pytest.importorskip('pandas')
    m=api();counts=[100,170,110,210,120,190,100]
    occ=np.concatenate([np.linspace(.002+.05*i,.042+.05*i,n)for i,n in enumerate(counts)])
    # Nonlinear unequal-count bins distinguish count weighting from sqrt/count-
    # free splines; the independent reference executes the entire author main.
    vis=5+occ*7+np.sin(np.arange(len(occ))*.13);delta=np.cos(occ*21)*.4+occ*.7
    df=frame(occ,vis,vis-delta,delta);csv=tmp_path/'nonlinear.csv';df.to_csv(csv,index=False)
    output=tmp_path/'author.png';reference=tmp_path/'author-reference.json'
    code='''import importlib.util,json,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('actual_author_plot',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
captured={}
def profile(frame,event,arg):
 if event=='return'and frame.f_code is module.main.__code__:
  for name in('bc','bv','bf','bvs','bfs','cnt','bin_delta_m','bin_delta_s','x_smooth','y_smooth','rho','pval','s_idx'):
   v=frame.f_locals[name];captured[name]=v.tolist()if isinstance(v,np.ndarray)else v
sys.argv=['author','--csv',sys.argv[2],'--output',sys.argv[3]];sys.setprofile(profile)
module.main();sys.setprofile(None)
Path(sys.argv_original_reference).write_text(json.dumps(captured))
'''
    # Keep the output path independent of argv after native argparse consumes it.
    code=code.replace("p=Path(sys.argv[1]);","sys.argv_original_reference=sys.argv[4];p=Path(sys.argv[1]);")
    env=dict(os.environ,MPLBACKEND='Agg',CUDA_VISIBLE_DEVICES='')
    p=subprocess.run([sys.executable,'-B','-c',code,str(UPSTREAM/'scripts/paper/plot_occlusion_fusion.py'),str(csv),str(output),str(reference)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stderr[-2500:]
    expected=json.loads(reference.read_text());out=m.analyze_source_csv(csv,upstream=UPSTREAM)
    for key,field in [('bc','occlusion_mean'),('bv','vision_mean_deg'),('bf','fusion_mean_deg'),('bvs','vision_sem_deg'),('bfs','fusion_sem_deg'),('cnt','count'),('bin_delta_m','delta_mean_deg'),('bin_delta_s','delta_sem_deg')]:
        np.testing.assert_array_equal([b[field]for b in out['bins']],expected[key])
    np.testing.assert_array_equal(out['trend']['x'],expected['x_smooth']);np.testing.assert_array_equal(out['trend']['y'],expected['y_smooth'])
    np.testing.assert_array_equal(out['scatter_source_rows'],expected['s_idx'])
    assert out['spearman']['rho']==expected['rho']and out['spearman']['pvalue']==expected['pval']
    assert output.is_file()and output.stat().st_size>10000
