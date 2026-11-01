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
