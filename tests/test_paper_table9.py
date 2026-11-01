"""Source-exact Table9 analysis; no model or new inference is required."""
import copy
import importlib
import json
import math
from pathlib import Path
import shutil

import pytest

ROOT=Path(__file__).resolve().parents[1]


def api():
    try:return importlib.import_module('egoemg_repro.paper_table9')
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.paper_table9':pytest.fail('complete five-method Table9 reader missing')
        raise


def record(error,count):
    return dict(absolute_error_sum_rad=error,valid_angle_count=count,mae_deg=error/count*180/math.pi if count else None)


def statistics():
    cells={'left':{'gesture':record(.1,1),'user':record(.8,2),'both':record(.3,1)},
        'right':{'gesture':record(.3,3),'user':record(.4,2),'both':record(.5,1)}}
    def total(rows):return record(math.fsum(x['absolute_error_sum_rad']for x in rows),sum(x['valid_angle_count']for x in rows))
    return dict(mode='trajectory',by_hand_split=cells,by_hand={h:total(cells[h].values())for h in cells},
        by_split={s:total([cells[h][s]for h in cells])for s in ('gesture','user','both')},
        overall=total([c for hand in cells.values()for c in hand.values()]))


def test_hand_and_split_margins_conserve_primitive_angle_time_sums():
    s=statistics();r=api().trajectory_cells(s)
    assert len(r)==6
    assert math.isclose(sum(c['absolute_error_sum_rad']for c in r),2.4)
    assert sum(c['valid_angle_count']for c in r)==10
    overall=2.4/10*180/math.pi
    assert not math.isclose(sum(c['mae_deg']for c in r)/6,overall)
    assert r[0]==dict(hand='left',split='gesture',**record(.1,1))


@pytest.mark.parametrize('damage',['center','extra_hand','missing_split','bool_count','negative_count','negative_error','nan','rounded_mean','hand_count','split_sum','zero_nonzero'])
def test_wrong_mode_coverage_numeric_or_margin_cannot_be_reported(damage):
    s=statistics();c=s['by_hand_split']['left']['gesture']
    if damage=='center':s['mode']='center'
    elif damage=='extra_hand':s['by_hand_split']['third']=s['by_hand_split']['left']
    elif damage=='missing_split':s['by_hand_split']['left'].pop('both')
    elif damage=='bool_count':c['valid_angle_count']=True
    elif damage=='negative_count':c['valid_angle_count']=-1
    elif damage=='negative_error':c['absolute_error_sum_rad']=-.1
    elif damage=='nan':c['absolute_error_sum_rad']=float('nan')
    elif damage=='rounded_mean':c['mae_deg']=round(c['mae_deg'],1)
    elif damage=='hand_count':s['by_hand']['left']['valid_angle_count']+=1
    elif damage=='split_sum':s['by_split']['user']['absolute_error_sum_rad']+=.001
    else:c.update(valid_angle_count=0,mae_deg=None)
    with pytest.raises(ValueError):api().trajectory_cells(s)


def test_empty_cell_keeps_null_without_fabricating_a_value():
    s=statistics();s['by_hand_split']['left']['gesture']=record(0.,0)
    for target,rows in [(s['by_hand']['left'],list(s['by_hand_split']['left'].values())),
        (s['by_split']['gesture'],[s['by_hand_split'][h]['gesture']for h in ('left','right')]),
        (s['overall'],[c for h in s['by_hand_split'].values()for c in h.values()])]:
        target.update(record(math.fsum(x['absolute_error_sum_rad']for x in rows),sum(x['valid_angle_count']for x in rows)))
    assert api().trajectory_cells(s)[0]['mae_deg']is None


def test_actual_five_reports_keep_different_populations_and_all30_pdf_deltas(monkeypatch):
    import builtins
    original=builtins.__import__
    def no_ml(name,*a,**k):
        if name.split('.')[0]in ('torch','pytorch_lightning'):raise AssertionError('Table9 imported ML')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',no_ml)
    r=api().build_table9(ROOT)
    assert r['analysis_complete']and not r['full_paper_complete']and not r['paper_numerical_equivalence_established']
    assert not r['all_methods_same_test_population']and not r['paper_cell_aggregation_equivalence_established']
    assert len(r['cells'])==30 and len(r['hand_gaps'])==15 and len(r['models'])==5
    assert [x['requested_hand_windows']for x in r['models']]==[2792,2792,2792,4306,4306]
    assert [x['output_length']for x in r['models']]==[11490,11490,11490,6000,7790]
    assert [c['paper_mae_deg']for c in r['cells'][:3]]==[12.4,14.7,16.5]
    assert all(math.isclose(c['difference_from_paper_deg'],c['mae_deg']-c['paper_mae_deg'],abs_tol=1e-12)for c in r['cells'])
    assert all(len(m['source_report_sha256'])==64 and m['selected_weights_identity']for m in r['models'])


def copy_reports(root):
    for p in (ROOT/'results/completed').glob('emgformer-*/metrics.json'):
        target=root/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for label in ('vemg2pose','neuropose'):
        p=ROOT/'results/original-classic'/label/'metrics.json';target=root/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)


def test_resealed_source_summary_cannot_replace_the_pinned_completed_report(tmp_path):
    m=api();copy_reports(tmp_path);p=tmp_path/'results/completed/emgformer-s/metrics.json';v=json.loads(p.read_text())
    v['completion_status']='training_complete';v['test']['sampling']['trajectory']['overall']['mae_deg']+=1;p.write_text(json.dumps(v))
    with pytest.raises(ValueError):m.build_table9(tmp_path)


def export_api():
    m=api()
    if not hasattr(m,'export_table9')or not hasattr(m,'verify_table9'):pytest.fail('Table9 immutable export/verify missing')
    return m


@pytest.fixture(scope='module')
def actual_output(tmp_path_factory):
    m=export_api();root=tmp_path_factory.mktemp('actual-five-method-table9');m.export_table9(ROOT,root/'table')
    return root/'table'


def test_actual_export_has_all30_cells_and_verifies_exact_source_reduction(actual_output):
    import csv
    m=export_api();r=m.verify_table9(ROOT,actual_output)
    with(actual_output/'cells.csv').open()as f:rows=list(csv.DictReader(f))
    assert len(rows)==30 and {x['hand']for x in rows}=={'left','right'}
    assert all(int(x['valid_angle_count'])>0 for x in rows)
    assert len(list(csv.DictReader((actual_output/'hand-gaps.csv').open())))==15
    assert not r['paper_numerical_equivalence_established']and not r['full_paper_complete']
    assert all(x in(actual_output/'README.md').read_text()for x in ('2,792','4,306','Table 9','11490','6000','7790'))
    with pytest.raises((ValueError,FileExistsError)):m.export_table9(ROOT,actual_output)


@pytest.mark.parametrize('name',['metrics.json','cells.csv','hand-gaps.csv','README.md','completion.json'])
@pytest.mark.parametrize('damage',['changed','missing'])
def test_changed_output_generation_is_rejected(actual_output,name,damage):
    m=export_api();p=actual_output/name;before=p.read_bytes()
    try:
        if damage=='changed':p.write_bytes(before+b'\n')
        else:p.unlink()
        with pytest.raises((ValueError,OSError)):m.verify_table9(ROOT,actual_output)
    finally:p.write_bytes(before)


def test_source_generation_change_inside_hash_is_rejected(tmp_path,monkeypatch):
    m=api();copy_reports(tmp_path);p=tmp_path/'results/completed/emgformer-s/metrics.json';original=m._sha;done=False
    def changed(path):
        nonlocal done
        digest=original(path)
        if Path(path)==p and not done:done=True;p.write_bytes(p.read_bytes()+b' ')
        return digest
    monkeypatch.setattr(m,'_sha',changed)
    with pytest.raises(ValueError):m.build_table9(tmp_path)


def test_export_cannot_write_inside_a_frozen_model_report_directory():
    m=export_api();target=ROOT/'results/completed/emgformer-s/forbidden-table9'
    with pytest.raises(ValueError):m.export_table9(ROOT,target)
    assert not target.exists()


def test_separate_offline_cli_processes_reproduce_identical_artifact_bytes(tmp_path):
    import hashlib,os,subprocess,sys
    m=export_api();block=tmp_path/'block';block.mkdir()
    (block/'sitecustomize.py').write_text("import builtins\noriginal=builtins.__import__\ndef blocked(name,*a,**k):\n if name.split('.')[0]in ('torch','pytorch_lightning'):raise RuntimeError('Table9 cannot import ML')\n return original(name,*a,**k)\nbuiltins.__import__=blocked\n")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(block)+os.pathsep+str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for _,name,_ in m.SOURCE_REPORTS}
    for action,out in [('export',tmp_path/'one'),('export',tmp_path/'two'),('verify',tmp_path/'one')]:
        p=subprocess.run([sys.executable,'-B','-m','egoemg_repro.paper_table9',action,'--repo',str(ROOT),'--output',str(out)],env=env,cwd=ROOT,capture_output=True,text=True,timeout=30)
        assert p.returncode==0,p.stderr
    assert {p.name:p.read_bytes()for p in(tmp_path/'one').iterdir()}=={p.name:p.read_bytes()for p in(tmp_path/'two').iterdir()}
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in source.items())
