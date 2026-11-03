"""Independent numeric examples for paired control summaries, not full runs."""
import importlib
import numpy as np
import pytest


def api():return importlib.import_module('egoemg_repro.modality_control_run')


def errors():
    f=np.array([[1,999,3],[2,888,4],[2,777,5],[4,666,8],[1.5,555,3.5],[3,444,5]],np.float64)
    v=np.array([[2,500,4]if c%2==0 else[5,500,7]for c in range(6)],np.float64)
    label=np.array([[True],[False],[True]]);crop=np.array([[False],[False],[True]])
    return f,v,label,crop


def test_paired_summary_retains_valid_labels_under_missing_vision_and_ignores_invalid_denominator():
    s=api().summarize_control_errors(*errors())
    assert (s['requests'],s['valid_label_requests'],s['invalid_label_requests'])==(3,2,1)
    assert s['natural_missing_valid_requests']==1
    np.testing.assert_allclose([r['fusion_mae_deg']for r in s['conditions']],
        [114.6,171.9,200.55,343.8,143.25,229.2],rtol=0,atol=1e-12)
    rows=s['conditions']
    assert rows[1]['valid_label_requests']==2  # All artificial vision flags would be false.
    assert rows[2]['aligned_reference_condition']==0 and rows[3]['aligned_reference_condition']==1
    assert rows[2]['penalty_vs_aligned_deg']==pytest.approx(85.95)
    assert rows[3]['penalty_vs_aligned_deg']==pytest.approx(171.9)
    assert rows[2]['aligned_better_requests']==2 and rows[2]['aligned_worse_requests']==0
    assert rows[0]['vision_gain_deg']==pytest.approx(57.3)


def test_empty_selected_group_has_null_scores_and_never_invents_zero_error():
    s=api().summarize_control_errors(*errors(),selection=np.zeros(3,bool))
    assert s['requests']==s['valid_label_requests']==0
    assert all(r['fusion_mae_deg']is None and r['penalty_vs_aligned_deg']is None for r in s['conditions'])


@pytest.mark.parametrize('damage',['hidden_invalid_nan','float_label','wrong_condition_count','bad_selection'])
def test_malformed_summary_inputs_are_rejected_before_any_filter(damage):
    f,v,label,crop=errors();options={}
    if damage=='hidden_invalid_nan':f[0,1]=np.nan
    elif damage=='float_label':label=label.astype(np.float32)
    elif damage=='wrong_condition_count':f=f[:5]
    else:options['selection']=np.array([1,0,1],np.int64)
    with pytest.raises(ValueError):api().summarize_control_errors(f,v,label,crop,**options)
