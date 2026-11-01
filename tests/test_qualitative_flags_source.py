"""Real current boolean files and the pinned legacy geometry byte reference."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_occlusion_dense_data import UPSTREAM

ROOT=Path(__file__).resolve().parents[1]
PAIRED=Path(os.environ.get('EGOEMG_QUALITATIVE_PAIRED_ROOT','/data/shihao/egoemg-qualitative-ordinary-paired-io-3ddc1a6b03eb-20261101'))
DATA=Path('/home/shihao/projects/EgoEMG-Reproduction/data/legacy')
GEOMETRY=Path('/data/shihao/egoemg-legacy-geometry-20261028')
pytestmark=pytest.mark.skipif(not(PAIRED.exists()and DATA.exists()and GEOMETRY.exists()),reason='actual legacy boolean files required; no fake full positive')


@pytest.fixture(scope='module')
def evidence():
    from egoemg_repro.qualitative_run import load_qualitative_evidence
    return load_qualitative_evidence(evidence=ROOT/'results/native-qualitative',paired_root=PAIRED,upstream=UPSTREAM)


@pytest.fixture(scope='module')
def flags(evidence):
    from egoemg_repro.qualitative_flags import admit_qualitative_flags
    return admit_qualitative_flags(evidence,data=DATA,geometry=GEOMETRY)


def test_actual_all2322_flags_equal_independently_read_current_raw_boolean_files(evidence,flags):
    fields={}
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name in('mocap_webcam_tracked','image_webcam_stale','mocap_left_valid','mocap_right_valid'):
        spec=manifest['fields'][name];path=(DATA if name.startswith('image_')else GEOMETRY)/spec['filename']
        fields[name]=np.memmap(path,mode='r',dtype=spec['dtype'],shape=tuple(spec['shape']))
    rows=flags.records();assert len(rows)==2322
    for k,r in zip(evidence.keys(),rows):
        c=k['center_idx']
        assert r['mocap_tracked']==bool(fields['mocap_webcam_tracked'][c])
        assert r['image_stale']==bool(fields['image_webcam_stale'][c])
        assert r['markers_valid']==bool(fields['mocap_'+k['hand']+'_valid'][c].any())
        for n in('sequence','split','hand','ep_idx','center_idx','window_start','window_end'):assert r[n]==k[n]
    v=flags.identity
    assert v['selected_hand_points']==2322 and v['full_boolean_source_bytes']==2911115900
    assert v['source_aliases']['mocap_head_valid']=='mocap_webcam_tracked'
    assert v['source_aliases']['image_head_stale']=='image_webcam_stale'
    assert not v['model_or_geometry_permit']and not v['figure7_reproduced']


def test_exported_copies_and_unregistered_flag_objects_cannot_change_live_admission(flags):
    original=flags.records()[0];flags.records()[0].clear();flags.identity.clear()
    assert flags.records()[0]==original
    with pytest.raises(ValueError,match='registered'):copy.copy(flags).check()


def test_resealed_raw_source_reference_cannot_bypass_the_accepted_boolean_byte_hashes(evidence,tmp_path,monkeypatch):
    from egoemg_repro import qualitative_flags
    v=json.loads(qualitative_flags.REFERENCE.read_text());v['identity']['raw_input_files_sha256']['field:mocap_left_valid']='a'*64
    p=tmp_path/'new-reference.json';p.write_text(json.dumps(v));monkeypatch.setattr(qualitative_flags,'REFERENCE',p)
    with pytest.raises(ValueError):qualitative_flags.admit_qualitative_flags(evidence,data=DATA,geometry=GEOMETRY)
