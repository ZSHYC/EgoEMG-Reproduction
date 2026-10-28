"""Hand-counted frame masks catch hand duplication and invalid-label cleanup."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from test_semantic_tables import metadata_fixture
from egoemg_repro.data import file_sha256


def module():
    return importlib.import_module('egoemg_repro.semantic_frames')


@pytest.fixture
def frames(tmp_path):
    root=metadata_fixture(tmp_path/'PRIVATE-FRAMES')
    arrays={'label_gesture_class':np.array([0,0,1,50,-1,33,33,59],dtype='<i4'),
            'label_gesture_active':np.array([1,0,1,1,0,1,0,1],dtype='?'),
            'frame_split_id':np.array([0,0,1,2,3,2,2,3],dtype='i1'),
            'generated_label_valid':np.ones((8,2),dtype='?')}
    for hand in ('left','right'):
        arrays[f'generated_joint_angles_{hand}']=np.ones((8,20),dtype='<f4')
        for axis in ('pitch','yaw'):arrays[f'mocap_{hand}_wrist_{axis}']=np.full(8,90,dtype='<f4')
        arrays[f'mocap_{hand}_wrist_angles_valid']=np.ones(8,dtype='?')
    arrays['generated_label_valid'][2,0]=False
    arrays['generated_label_valid'][3,1]=False
    arrays['mocap_left_wrist_angles_valid'][4]=False
    arrays['generated_joint_angles_left'][1,3]=np.nan
    arrays['generated_joint_angles_right'][3,0]=np.nan
    arrays['mocap_left_wrist_yaw'][4]=np.nan
    arrays['mocap_right_wrist_yaw'][5]=np.inf
    ledger={'ledger_version':1,'files':{}}
    for name,a in arrays.items():
        p=root/(name+'.dat');a.tofile(p)
        ledger['files'][p.name]={'bytes':p.stat().st_size,'sha256':file_sha256(p),'checksum_basis':'local_transfer'}
    (root/'acquisition_ledger.json').write_text(json.dumps(ledger))
    return root


def collect(root,chunk=3):
    return module().collect_frame_validity(root,upstream=Path(__file__).parents[1]/'upstream',
        transfer_receipt_sha256=file_sha256(root/'acquisition_ledger.json'),chunk_rows=chunk)


def test_full_rows_use_separate_hands_and_finite_before_cleanup(frames):
    p=collect(frames)['public_projection'];o=p['overall']
    assert p['timeline_frames']==8 and o['hand_frames']==16
    assert o['possible_angle_elements']==352 and o['valid_angle_elements']==308
    assert sum(o['finite_by_angle'])==348
    assert o['generated_valid_hand_frames']==14 and o['wrist_valid_hand_frames']==15
    assert o['fingers_all20_valid_hand_frames']==13
    assert o['wrist_all2_valid_hand_frames']==14
    assert o['all22_valid_hand_frames']==11 and o['any_angle_valid_hand_frames']==16
    assert o['active_hand_frames']==10
    assert p['by_hand']['left']['valid_angle_elements']==153
    assert p['by_hand']['right']['valid_angle_elements']==155
    assert {s:x['timeline_frames'] for s,x in p['by_split'].items()}==dict(train=2,user=1,gesture=3,both=2)
    assert p['unassigned']['timeline_frames']==1
    assert len(p['by_gesture'])==60 and p['by_gesture'][50]['timeline_frames']==1
    assert p['by_gesture'][33]['timeline_frames']==2
    assert p['model_errors_computed'] is False


def test_chunking_and_public_privacy(frames):
    a=collect(frames,1);b=collect(frames,128)
    assert a['public_projection']==b['public_projection']
    assert 'PRIVATE' not in json.dumps(a['public_projection'])
    assert a['private_receipt']['all_input_nine_attributes_unchanged']
    assert a['public_projection']['source_identity']['complete_payload_rehash_performed']


def test_same_size_payload_tampering_is_not_accepted(frames):
    p=frames/'generated_joint_angles_left.dat'
    with p.open('r+b') as f:f.write(np.array([2.],dtype='<f4').tobytes())
    with pytest.raises(ValueError,match='SHA256'):collect(frames)


@pytest.mark.parametrize('field,first_byte',[
    ('generated_label_valid',2),('label_gesture_active',2),
    ('label_gesture_class',60),('frame_split_id',4)])
def test_invalid_raw_categories_rejected_even_with_matching_ledger(frames,field,first_byte):
    p=frames/(field+'.dat')
    with p.open('r+b') as f:f.write(bytes([first_byte]))
    ledger=json.loads((frames/'acquisition_ledger.json').read_text())
    ledger['files'][p.name]['sha256']=file_sha256(p)
    (frames/'acquisition_ledger.json').write_text(json.dumps(ledger))
    with pytest.raises(ValueError):collect(frames)


def test_zero_chunk_is_rejected(frames):
    with pytest.raises(ValueError):collect(frames,0)


@pytest.mark.parametrize('mode',['flag','environment'])
def test_replay_refuses_optimized_python_before_opening_inputs(tmp_path,mode):
    import os
    import subprocess
    import sys
    script=Path(__file__).parents[1]/'scripts/replay_frame_validity.py'
    environment=dict(os.environ)
    flags=['-O'] if mode=='flag' else []
    if mode=='environment':environment['PYTHONOPTIMIZE']='1'
    result=subprocess.run([sys.executable,*flags,str(script)],cwd=tmp_path,
                          env=environment,text=True,capture_output=True)
    assert result.returncode!=0
    assert 'optimized Python cannot verify frame validity' in result.stderr
    assert 'FileNotFoundError' not in result.stderr
