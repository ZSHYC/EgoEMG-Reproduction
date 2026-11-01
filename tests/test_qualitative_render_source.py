"""Actual completed dual-pass predictions and original input consumer."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_occlusion_dense_data import UPSTREAM

ROOT=Path(__file__).resolve().parents[1]
PAIRED=Path(os.environ.get('EGOEMG_QUALITATIVE_PAIRED_ROOT','/data/shihao/egoemg-qualitative-ordinary-paired-io-3ddc1a6b03eb-20261101'))
PRED=Path('/data/shihao/egoemg-vits-ordinary-frozen-full-cpu-7fe8ea87a630-20261101')
DATA=Path('/home/shihao/projects/EgoEMG-Reproduction/data/legacy');CROPS=DATA.parent/'crops'
GEOM=Path('/data/shihao/egoemg-legacy-geometry-20261028')
pytestmark=pytest.mark.skipif(not(PAIRED.exists()and PRED.exists()and DATA.exists()),reason='actual completed neural source and corpus required; no fake full positive')


@pytest.fixture(scope='module')
def consumers():
    from egoemg_repro.qualitative_run import load_qualitative_evidence,admit_qualitative_run_inputs
    from egoemg_repro.qualitative_flags import admit_qualitative_flags
    state=json.loads((PRED/'private/execution-state.json').read_text());assert state['status']=='complete'and state['actual_exit_code']==0
    e=load_qualitative_evidence(evidence=ROOT/'results/native-qualitative',paired_root=PAIRED,upstream=UPSTREAM,producer_root=PRED)
    inputs=admit_qualitative_run_inputs(e,data=DATA,crops=CROPS)
    flags=admit_qualitative_flags(e,data=DATA,geometry=GEOM)
    return inputs,flags


def test_actual_top_displayable_sample_matches_full_original_rank_raw_arrays_and_image(consumers,tmp_path):
    from PIL import Image
    from egoemg_repro.qualitative_render import export_qualitative_candidates
    from egoemg_repro.qualitative_run import read_qualitative_pass
    from egoemg_repro.occlusion_dense_data import _typed_sha
    inputs,flags=consumers
    result=export_qualitative_candidates(inputs,PRED/'private/pass-a',PRED/'private/pass-b',flags=flags,output=tmp_path/'report',limit=1)
    assert result['counts']['selected']==2322 and result['counts']['tracked_filter_pass']==2242
    assert result['rendered_candidates']==1 and not result['condition_examples_selected']
    index=json.loads((tmp_path/'report/index.json').read_text());item=index['samples'][0]
    raw=read_qualitative_pass(PRED/'private/pass-a',evidence=inputs.evidence)
    rows=flags.records();a=raw['arrays'];order=sorted(range(2322),key=lambda i:float(a['mae_vision_rad'][i]-a['mae_fusion_rad'][i]),reverse=True)
    expected=next(i for i in order if rows[i]['mocap_tracked']and not rows[i]['image_stale']and rows[i]['markers_valid']and bool(a['crop_valid_mask'][i,0]))
    assert item['key']==inputs.evidence.keys()[expected]
    with np.load(tmp_path/'report'/item['arrays_file'],allow_pickle=False)as z:
        np.testing.assert_array_equal(z['pose:gt'],a['targets'][expected])
        np.testing.assert_array_equal(z['pose:vision'],a['vision_prediction'][expected])
        assert z['rgb'].dtype==np.uint8 and z['rgb'].shape==(256,256,3)
    meta=json.loads((tmp_path/'report'/item['metadata_file']).read_text())
    assert meta['panel_metadata']['errors_rad']['fusion']==float(a['mae_fusion_rad'][expected])
    assert Image.open(tmp_path/'report'/item['png_file']).size==(2160,512)
    assert result['dual_full_frozen_passes_verified']
    assert result['source_model_identity_sha256']=='7543495ccd161d11bc2db25f277a7dfb1fddd4fd0051b9f1d0531e5faeea41fd'
    assert not result['figure7_reproduced']and not result['full_paper_complete']


def test_forged_consumer_or_resealed_neural_completion_proof_cannot_export(consumers,tmp_path,monkeypatch):
    from egoemg_repro import qualitative_render
    inputs,flags=consumers
    with pytest.raises(ValueError,match='registered'):
        qualitative_render.export_qualitative_candidates(copy.copy(inputs),PRED/'private/pass-a',PRED/'private/pass-b',flags=flags,output=tmp_path/'forged',limit=1)
    v=json.loads(qualitative_render.PREDICTION_REFERENCE.read_text());v['points_per_pass']=1
    p=tmp_path/'resealed-neural-proof.json';p.write_text(json.dumps(v));monkeypatch.setattr(qualitative_render,'PREDICTION_REFERENCE',p)
    with pytest.raises(ValueError):
        qualitative_render.export_qualitative_candidates(inputs,PRED/'private/pass-a',PRED/'private/pass-b',flags=flags,output=tmp_path/'changed',limit=1)


def test_existing_output_and_same_pass_cannot_be_promoted_as_a_new_pair(consumers,tmp_path):
    from egoemg_repro.qualitative_render import export_qualitative_candidates
    inputs,flags=consumers
    with pytest.raises(ValueError):
        export_qualitative_candidates(inputs,PRED/'private/pass-a',PRED/'private/pass-a',flags=flags,output=tmp_path/'same',limit=1)


def test_historical_reader_inputs_cannot_start_a_new_forward(consumers,tmp_path):
    from egoemg_repro.qualitative_run import run_qualitative_pass
    inputs,_=consumers
    with pytest.raises(ValueError,match='read-only'):
        run_qualitative_pass(inputs,tmp_path/'must-not-read',tmp_path/'must-not-create')
    assert not(tmp_path/'must-not-create').exists()
