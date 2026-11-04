"""Fail before sources, geometry, or CUDA for invalid full-IK requests."""
import copy
from pathlib import Path

import pytest

from scripts.paper import run_native_mano_ik as workflow


def request(tmp_path):
    return dict(data=str(tmp_path/'data'), geometry=str(tmp_path/'geometry'),
        upstream=str(tmp_path/'upstream'), manotorch=str(tmp_path/'manotorch'),
        chumpy=str(tmp_path/'chumpy'), mano_assets=str(tmp_path/'assets'),
        engineering=True, batch_size=2, device='cpu', gpu_uuid=None,
        qualification=None, minimum_free_bytes=15*1024**3)


def test_unknown_action_rejected_before_io(tmp_path):
    with pytest.raises(ValueError, match='action'):
        workflow.run_workflow('all-maybe', request(tmp_path), tmp_path/'output')


@pytest.mark.parametrize('key,value', [('engineering', 1), ('batch_size', True),
    ('batch_size', 0), ('batch_size', 10097), ('device', 'cuda:7'),
    ('gpu_uuid', 'GPU-unadmitted'), ('minimum_free_bytes', -1)])
def test_invalid_configuration_never_starts_sources(tmp_path, key, value):
    cfg = request(tmp_path); cfg[key] = value
    with pytest.raises(ValueError):
        workflow.validate_request('workflow', cfg)


def test_full_ik_cannot_run_on_cpu_or_smaller_batches(tmp_path):
    cfg = request(tmp_path); cfg['engineering'] = False
    with pytest.raises(ValueError, match='10096|CUDA|qualification'):
        workflow.validate_request('workflow', cfg)
    cfg.update(batch_size=10096, device='cuda:0', gpu_uuid='GPU-'+'1'*36)
    with pytest.raises(ValueError, match='qualification'):
        workflow.validate_request('workflow', cfg)


def test_only_original_full_batches_may_qualify(tmp_path):
    cfg = request(tmp_path)
    with pytest.raises(ValueError, match='qualification|10096|CUDA'):
        workflow.validate_request('qualify', cfg)


def test_full_workflow_reserves_raw_states_and_separate_labels(tmp_path):
    cfg = request(tmp_path)
    cfg.update(engineering=False, batch_size=10096, device='cuda:0',
        gpu_uuid='GPU-'+'1'*36, qualification=str(tmp_path/'permit'), minimum_free_bytes=15*1024**3)
    with pytest.raises(ValueError, match='storage|200|reserve'):
        workflow.validate_request('workflow', cfg)


def test_existing_workflow_output_is_not_reused(tmp_path):
    output = tmp_path/'output'; output.mkdir()
    with pytest.raises(ValueError, match='existing|new|output'):
        workflow.run_workflow('workflow', request(tmp_path), output)


def test_valid_cpu_engineering_request_retains_all_six_shards(tmp_path):
    cfg = request(tmp_path)
    assert workflow.validate_request('workflow', cfg) == cfg
    assert cfg == request(tmp_path)


def test_workflow_sources_are_a_torch_free_real_file_snapshot(request):
    from torch_free_case import run_torch_free_case
    if run_torch_free_case(request): return
    import sys
    assert 'torch' not in sys.modules
    sources = workflow.workflow_sources()
    assert 'scripts/paper/run_native_mano_ik.py' in sources
    assert 'egoemg_repro/native_ik_results.py' in sources
    assert len(sources['egoemg_repro/native_ik_results.py']) == 64
    assert 'torch' not in sys.modules
