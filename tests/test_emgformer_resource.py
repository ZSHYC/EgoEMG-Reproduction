"""Typed idle-device admission; these tests make no hardware claims."""
import copy
import importlib

import pytest

U1='GPU-11111111-1111-1111-1111-111111111111'
U3='GPU-33333333-3333-3333-3333-333333333333'
U4='GPU-44444444-4444-4444-4444-444444444444'


def api():
    try:return importlib.import_module('egoemg_repro.emgformer_resource').select_idle_emgformer_gpu
    except ModuleNotFoundError as exc:
        if exc.name=='egoemg_repro.emgformer_resource':pytest.fail('native EMGFormer idle-device admission missing')
        raise


def inventory():
    return [dict(index=i,uuid=u,free_mib=80000,total_mib=81920)for i,u in ((1,U1),(3,U3),(4,U4))]


def test_ordered_explicit_idle_candidate_and_returned_copy():
    g=inventory();before=copy.deepcopy(g);value=api()(g,[],candidates=[U3,U1])
    assert value==g[1]and value is not g[1]and g==before
    value['free_mib']=0;assert g==before


@pytest.mark.parametrize('pid',[123,456])
def test_any_application_blocks_even_above_reserve(pid):
    g=inventory();assert api()(g,[dict(pid=pid,uuid=U1)],candidates=[U1,U3])==g[1]
    assert api()(g,[dict(pid=pid,uuid=U1)],candidates=[U1])is None


def test_reserved_idle_device_is_not_implicitly_added():
    g=inventory();apps=[dict(pid=10,uuid=U1),dict(pid=11,uuid=U3)]
    assert api()(g,apps,candidates=[U1,U3])is None
    assert g[2]['uuid']==U4 and g[2]['free_mib']>65536


@pytest.mark.parametrize('free,eligible',[(65535,False),(65536,True),(65537,True)])
def test_fixed_64gib_admission_boundary(free,eligible):
    g=inventory();g[0]['free_mib']=free
    assert (api()(g,[],candidates=[U1])is not None)is eligible


@pytest.mark.parametrize('damage',['gpu_bool_index','gpu_bool_free','gpu_float_free','gpu_negative_free',
    'gpu_over_total','gpu_missing_free','duplicate_gpu_uuid','duplicate_gpu_index','bad_gpu_uuid',
    'partial_gpu_inventory','empty_candidates','duplicate_candidates','string_candidates',
    'unknown_application_gpu','duplicate_application','bad_pid_bool','bad_pid_zero','application_missing_uuid'])
def test_malformed_or_incomplete_inventory_fails_closed(damage):
    g=inventory();apps=[];c=[U1,U3]
    if damage=='gpu_bool_index':g[0]['index']=True
    elif damage=='gpu_bool_free':g[0]['free_mib']=True
    elif damage=='gpu_float_free':g[0]['free_mib']=80000.
    elif damage=='gpu_negative_free':g[0]['free_mib']=-1
    elif damage=='gpu_over_total':g[0]['free_mib']=90000
    elif damage=='gpu_missing_free':del g[0]['free_mib']
    elif damage=='duplicate_gpu_uuid':g[1]['uuid']=U1
    elif damage=='duplicate_gpu_index':g[1]['index']=1
    elif damage=='bad_gpu_uuid':g[0]['uuid']='cuda:0'
    elif damage=='partial_gpu_inventory':g.pop(1)
    elif damage=='empty_candidates':c=[]
    elif damage=='duplicate_candidates':c=[U1,U1]
    elif damage=='string_candidates':c=U1
    elif damage=='unknown_application_gpu':apps=[dict(pid=10,uuid='GPU-99999999-9999-9999-9999-999999999999')]
    elif damage=='duplicate_application':apps=[dict(pid=10,uuid=U1)]*2
    elif damage=='bad_pid_bool':apps=[dict(pid=True,uuid=U1)]
    elif damage=='bad_pid_zero':apps=[dict(pid=0,uuid=U1)]
    elif damage=='application_missing_uuid':apps=[dict(pid=10)]
    with pytest.raises(ValueError):api()(g,apps,candidates=c)
