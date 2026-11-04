"""Independent arithmetic and publication gates; never synthetic model evidence."""
import copy
import hashlib
import importlib
import json
import math

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="native tensor checks require the optional Torch tier")


def api():
    try:return importlib.import_module('scripts.paper.verify_native_center_reference')
    except ModuleNotFoundError as error:
        if error.name=='scripts.paper.verify_native_center_reference':pytest.fail('independent reference verifier missing')
        raise


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def example():
    windows=[('gesture','private_A',1.,True),('gesture','private_A',1.,True),('gesture','private_B',5.,True),
             ('user','private_A',4.,False),('both','private_B',5.,True),('both','private_A',6.,True)]
    parts=[];identities=[];records=[]
    for hand in ('left','right'):
        for w,(split,user,pred,valid)in enumerate(windows):
            i=len(parts);center=10000+w*7790
            meta=dict(episode=w,start=center-6000,end=center+6000,center=center,hand=hand,split=split,subject=user)
            target=np.zeros((22,1),np.float32);mask=np.array([valid],np.bool_)
            record=dict(index=i,emg_sha256='0'*64,target_mask_sha256=hashlib.sha256(target.tobytes()+mask.tobytes()).hexdigest(),identity_sha256=digest(meta))
            record_token=digest(record)
            parts.append(dict(pred=torch.full((1,22,1),pred+(1. if hand=='right'else 0.),dtype=torch.float32),
                target=torch.from_numpy(target).unsqueeze(0),mask=torch.from_numpy(mask).unsqueeze(0),
                index=torch.tensor([i]),unique=torch.tensor([i]),repeat=torch.tensor([0]),center=torch.tensor([center]),episode=torch.tensor([w]),
                hand=[hand],split=[split],user=[user],sample_input_sha256=[record_token],input_sha256=digest([record])))
            identities.append(meta);records.append(record)
    return dict(parts=parts,source_indices=list(range(12)),request_identities=identities),records


def test_independent_user_mean_population_sd_and_invalid_denominators():
    payload,records=example();v=api().independent_point_statistics(payload,records)
    assert v['consumed_samples']==12 and v['valid_samples']==10 and v['invalid_samples']==2
    assert v['overall']['valid_angle_count']==220 and v['invalid_angle_count']==44
    assert v['overall']['mae_deg']==pytest.approx(4.1*180/math.pi)
    g=v['by_split_user_summary']['gesture']
    assert g['user_count']==2 and g['mean_mae_deg']==pytest.approx(3.5*180/math.pi)
    assert g['std_mae_deg']==pytest.approx(2.*180/math.pi)
    assert v['by_split']['gesture']['mae_deg']==pytest.approx(17/6*180/math.pi)
    assert v['by_split_user_summary']['user']['user_count']==0
    assert v['by_split_user_summary']['user']['mean_mae_deg']is None
    assert set(v['by_user'])=={'user_000','user_001'}and'private_'not in json.dumps(v)


@pytest.mark.parametrize('damage',['missing_invalid','order','target','mask','private_identity','BF16','nan','duplicate','hand_pair'])
def test_independent_reader_rejects_changed_coverage_targets_and_typed_sources(damage):
    p,r=example()
    if damage=='missing_invalid':p['parts'].pop(3)
    elif damage=='order':p['source_indices'].reverse()
    elif damage=='target':p['parts'][0]['target'][0,0,0]=1.
    elif damage=='mask':p['parts'][3]['mask'][0,0]=True
    elif damage=='private_identity':p['request_identities'][0]['subject']='replaced_subject'
    elif damage=='BF16':p['parts'][0]['pred']=p['parts'][0]['pred'].to(torch.bfloat16)
    elif damage=='nan':p['parts'][0]['pred'][0,0,0]=float('nan')
    elif damage=='duplicate':p['parts'][1]=p['parts'][0]
    else:p['parts'][6]['center'][0]+=1
    with pytest.raises(ValueError):api().independent_point_statistics(p,r)


@pytest.mark.parametrize('damage',['wrong_average','wrong_sd','private_alias','wrong_count'])
def test_independent_public_arithmetic_comparison_rejects_changed_report(damage):
    p,r=example();expected=api().independent_point_statistics(p,r);actual=copy.deepcopy(expected)
    if damage=='wrong_average':actual['overall']['mae_deg']+=0.01
    elif damage=='wrong_sd':actual['by_split_user_summary']['gesture']['std_mae_deg']*=math.sqrt(2)
    elif damage=='private_alias':actual['by_user']['private_A']=actual['by_user'].pop('user_000')
    else:actual['overall']['valid_angle_count']-=22
    with pytest.raises(ValueError):api().compare_public_statistics(expected,actual)


def test_short_qualification_cannot_be_promoted_to_full_reference():
    p,r=example();v=api().independent_point_statistics(p,r)
    with pytest.raises(ValueError):api().require_full_independent_population(v)


def test_csv_numeric_null_and_boolean_fields_match_expected_rows():
    rows=[dict(method='S',mean=1.25,n=22,sd=None,population_equivalence=False)]
    api().check_exact_csv('method,mean,n,sd,population_equivalence\nS,1.25,22,,False\n',rows)


@pytest.mark.parametrize('text',[
    'method,mean,n\nS,1.26,22\n',
    'method,mean,n\nS,1.25,0\n',
    'method,mean,n\n',
    'method,mean,n,private_user\nS,1.25,22,private_A\n',
])
def test_csv_modified_mean_denominator_missing_row_or_private_column_is_rejected(text):
    with pytest.raises(ValueError):api().check_exact_csv(text,[dict(method='S',mean=1.25,n=22)])
