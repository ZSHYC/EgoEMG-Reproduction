"""Bind actual native point-reader inputs, without requiring unrelated raw EMG."""
import json

import pytest

from test_vision_literal_data import sample_store,crop_store,config,UPSTREAM


def test_native_point_identity_uses_actual_opened_files_and_detects_changes(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('pytorch_lightning')
    from egoemg_repro import vision_literal_check as check
    data=tmp_path/'data';sample_store(data);crops=tmp_path/'crops';crop_store(crops,flat=True)
    manifest=json.loads((data/'manifest.json').read_text())
    manifest['fields']['emg_left_raw']=dict(filename='unavailable-raw.dat',shape=[3*7790,8],dtype='float32')
    (data/'manifest.json').write_text(json.dumps(manifest))
    # Real released cohorts can retain declarations of unacquired raw fields.
    # The pure vision point reader never opens raw or filtered EMG arrays.
    for hand in ('left','right'):(data/f'emg_{hand}_filtered.dat').unlink()
    datasets=check._datasets(data,crops,UPSTREAM,config())
    initial=check._data_identity(datasets)
    assert initial['consumed']['val']==4
    path=data/'generated_joint_angles_left.dat'
    with path.open('r+b') as stream:
        stream.seek(0);stream.write(b'\x00\x00\x00\x3f')
    assert check._data_identity(datasets)['file_stat_sha256']!=initial['file_stat_sha256']
