"""Shape statistics must count referenced beta rows once and publish aggregates."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from egoemg_repro import dataset_statistics as statistics


UPSTREAM = Path(os.environ.get('EGOEMG_UPSTREAM', Path(__file__).parents[1] / 'upstream'))


def write_fixture(root):
    metadata, beta = root / 'metadata-private-Alice', root / 'beta-private-Bob'
    metadata.mkdir()
    beta.mkdir()
    # Selected hand values per dimension are 0, 2, 4, 6, 8, 10, plus d.
    # Row 2 is deliberately unreferenced, and row 3 is referenced twice.
    offset = np.arange(10, dtype=np.float32)
    left = np.array([0, 2, 10000, 4], dtype=np.float32)[:, None] + offset
    right = np.array([6, 8, -10000, 10], dtype=np.float32)[:, None] + offset
    fields = {}
    for hand, values in [('left', left), ('right', right)]:
        name = f'generated_mano_{hand}_beta'
        values.tofile(beta / (name + '.dat'))
        fields[name] = dict(filename=name + '.dat', dtype='float32', shape=[4, 10])
    (metadata / 'manifest.json').write_text(json.dumps(dict(
        format_version='egoemg_v2_memmap', episode_fields=fields,
        private_note='PRIVATE-MANIFEST-ALICE')))
    np.savez(metadata / 'metadata.npz', episode_beta_idx=np.array([3, 0, 3, 1], dtype=np.int32),
             episode_id=np.array(['PRIVATE-EPISODE-ALICE'] * 4),
             episode_subject=np.array(['PRIVATE-SUBJECT-BOB'] * 4))
    refresh_ledger(beta)
    return metadata, beta


def refresh_ledger(beta):
    files = {}
    for path in beta.glob('*.dat'):
        files[path.name] = dict(bytes=path.stat().st_size,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                checksum_basis='local_transfer',
                                url='https://private.example/PRIVATE-LEDGER-ALICE')
    (beta / 'acquisition_ledger.json').write_text(json.dumps(dict(ledger_version=1, files=files)))


def rewrite_indices(metadata, indices):
    np.savez(metadata / 'metadata.npz', episode_beta_idx=np.asarray(indices),
             episode_id=np.array(['PRIVATE-EPISODE-ALICE'] * len(indices)))


def mutate_manifest(metadata, mutate):
    path = metadata / 'manifest.json'
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


def test_hand_calculated_statistics_deduplicate_references_and_exclude_unused_rows(tmp_path):
    metadata, beta = write_fixture(tmp_path)
    result = statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)
    assert result['counts'] == dict(episodes=4, available_beta_rows=4,
                                    referenced_beta_rows=3, unreferenced_beta_rows=1,
                                    left_hands=3, right_hands=3, combined_hands=6,
                                    dimensions=10)
    assert result['method']['ddof'] == 0
    assert result['method']['percentile_method'] == 'linear'
    for dim, actual in enumerate(result['dimensions']):
        assert actual['dimension'] == f'beta{dim + 1}'
        # Population variance of 0,2,4,6,8,10 is 35/3. Linear P5/P95
        # interpolate a quarter of the way between the end two values.
        expected = dict(mean=5 + dim, std=3.415650255319866,
                        min=dim, max=10 + dim, range=10,
                        p05=0.5 + dim, p95=9.5 + dim, p95_minus_p05=9)
        assert {key: actual[key] for key in expected} == pytest.approx(expected, abs=1e-6)
    rewrite_indices(metadata, np.array([1, 3, 0, 3], dtype=np.int32))
    reordered = statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)
    assert reordered['dimensions'] == result['dimensions']
    assert reordered['counts'] == result['counts']


@pytest.mark.parametrize('extra', [b'', b'x'])
def test_rejects_truncated_or_extra_beta_bytes(tmp_path, extra):
    metadata, beta = write_fixture(tmp_path)
    path = beta / 'generated_mano_left_beta.dat'
    path.write_bytes(path.read_bytes()[:-1] if not extra else path.read_bytes() + extra)
    refresh_ledger(beta)
    with pytest.raises(ValueError, match='byte size'):
        statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_beta_even_in_an_unreferenced_row(tmp_path, bad):
    metadata, beta = write_fixture(tmp_path)
    path = beta / 'generated_mano_right_beta.dat'
    values = np.fromfile(path, dtype=np.float32).reshape(4, 10)
    values[2, 0] = bad
    values.tofile(path)
    refresh_ledger(beta)
    with pytest.raises(ValueError, match='finite'):
        statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)


@pytest.mark.parametrize('indices', [np.array([-1, 0]), np.array([4, 0]),
                                     np.array([0.5, 1]), np.array([[0, 1]]),
                                     np.array([], dtype=np.int32), np.array([True, False]),
                                     np.array([0, 1], dtype=object)])
def test_rejects_invalid_beta_references(tmp_path, indices):
    metadata, beta = write_fixture(tmp_path)
    rewrite_indices(metadata, indices)
    with pytest.raises(ValueError, match='episode_beta_idx'):
        statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)


@pytest.mark.parametrize(('key', 'value'), [('dtype', 'float64'), ('dtype', 'object'),
                                           ('dtype', '>f4'),
                                           ('shape', [4, 9]), ('shape', [4, 10, 1]),
                                           ('shape', [4.0, 10]),
                                           ('filename', '../private.dat')])
def test_rejects_nonofficial_field_specification(tmp_path, key, value):
    metadata, beta = write_fixture(tmp_path)
    mutate_manifest(metadata, lambda m: m['episode_fields']['generated_mano_left_beta'].__setitem__(key, value))
    with pytest.raises(ValueError, match='beta field'):
        statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)


@pytest.mark.parametrize('fields', [None, [], {}, {'generated_mano_left_beta': []}])
def test_rejects_malformed_or_missing_episode_field_mapping(tmp_path, fields):
    metadata, beta = write_fixture(tmp_path)
    mutate_manifest(metadata, lambda m: m.__setitem__('episode_fields', fields))
    with pytest.raises(ValueError, match='beta field'):
        statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)


def test_rejects_different_left_right_row_counts(tmp_path):
    metadata, beta = write_fixture(tmp_path)
    path = beta / 'generated_mano_right_beta.dat'
    path.write_bytes(path.read_bytes()[:-40])
    mutate_manifest(metadata, lambda m: m['episode_fields']['generated_mano_right_beta'].__setitem__('shape', [3, 10]))
    refresh_ledger(beta)
    with pytest.raises(ValueError, match='shapes must match'):
        statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)


@pytest.mark.parametrize('which', ['manifest.json', 'metadata.npz', 'acquisition_ledger.json'])
def test_rejects_malformed_source_files(tmp_path, which):
    metadata, beta = write_fixture(tmp_path)
    path = (beta if which == 'acquisition_ledger.json' else metadata) / which
    path.write_bytes(b'invalid source bytes')
    with pytest.raises(ValueError, match='invalid|metadata'):
        statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)


@pytest.mark.parametrize('missing', ['metadata.npz', 'manifest.json', 'generated_mano_right_beta.dat',
                                    'acquisition_ledger.json'])
def test_missing_required_input_fails_without_outputs(tmp_path, missing):
    metadata, beta = write_fixture(tmp_path)
    root = metadata if missing in {'metadata.npz', 'manifest.json'} else beta
    (root / missing).unlink()
    with pytest.raises(ValueError, match='required'):
        statistics.export_shape_statistics(metadata, beta, tmp_path / 'out', upstream=UPSTREAM)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('fault', ['digest', 'bytes', 'missing_entry', 'version', 'basis'])
def test_requires_matching_acquisition_receipt(tmp_path, fault):
    metadata, beta = write_fixture(tmp_path)
    path = beta / 'acquisition_ledger.json'
    ledger = json.loads(path.read_text())
    record = ledger['files']['generated_mano_left_beta.dat']
    if fault == 'digest':
        record['sha256'] = '0' * 64
    elif fault == 'bytes':
        record['bytes'] += 1
    elif fault == 'missing_entry':
        del ledger['files']['generated_mano_left_beta.dat']
    elif fault == 'basis':
        record['checksum_basis'] = 'unverified'
    else:
        ledger['ledger_version'] = 99
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='acquisition'):
        statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)


@pytest.mark.parametrize('missing', ['episode_beta_idx', 'episode_id'])
def test_rejects_missing_metadata_fields(tmp_path, missing):
    metadata, beta = write_fixture(tmp_path)
    fields = dict(episode_beta_idx=np.array([0]), episode_id=np.array(['PRIVATE-ALICE']))
    del fields[missing]
    np.savez(metadata / 'metadata.npz', **fields)
    with pytest.raises(ValueError, match='metadata'):
        statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)


def test_rejects_episode_count_mismatch(tmp_path):
    metadata, beta = write_fixture(tmp_path)
    np.savez(metadata / 'metadata.npz', episode_beta_idx=np.array([0, 1]), episode_id=np.array(['PRIVATE']))
    with pytest.raises(ValueError, match='metadata'):
        statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)


def test_pdf_comparison_reports_differences_without_changing_measurements(tmp_path):
    metadata, beta = write_fixture(tmp_path)
    for path in beta.glob('*.dat'):
        values = np.fromfile(path, dtype=np.float32).reshape(4, 10)
        values[:, 9] = -0.738506
        values.tofile(path)
    refresh_ledger(beta)
    result = statistics.compute_shape_statistics(metadata, beta, upstream=UPSTREAM)
    comparison = result['paper_comparison']
    assert comparison['table'] == 5
    assert comparison['decimal_places'] == 3
    assert comparison['compared_entries'] == 50
    assert comparison['matching_entries'] == 0
    difference = next(item for item in comparison['differences']
                      if item['dimension'] == 'beta10' and item['metric'] == 'mean')
    assert difference['pdf_printed'] == '-0.738'
    assert difference['computed_rounded'] == '-0.739'
    assert difference['computed'] == result['dimensions'][9]['mean']
    assert difference['computed'] == pytest.approx(-0.738506, abs=1e-6)


@pytest.mark.parametrize('fault', ['revision', 'tracked', 'untracked'])
def test_rejects_unknown_or_dirty_official_source(tmp_path, fault):
    metadata, beta = write_fixture(tmp_path)
    upstream = tmp_path / 'upstream'
    subprocess.run(['git', 'clone', '--quiet', '--shared', str(UPSTREAM), str(upstream)], check=True)
    if fault == 'revision':
        previous = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD~1'], text=True).strip()
        subprocess.run(['git', '-C', str(upstream), 'checkout', '--quiet', previous], check=True)
    elif fault == 'tracked':
        with (upstream / 'scripts/paper/plot_beta_diversity.py').open('a') as stream:
            stream.write('\n# changed source\n')
    else:
        (upstream / 'untracked-statistics.py').write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError, match='upstream'):
        statistics.compute_shape_statistics(metadata, beta, upstream=upstream)


def test_cli_outputs_aggregate_artifacts_with_content_hashes_and_no_private_data(tmp_path):
    metadata, beta = write_fixture(tmp_path)
    output = tmp_path / 'out'
    completed = subprocess.run([sys.executable, '-m', 'egoemg_repro.dataset_statistics',
                                '--metadata-root', str(metadata), '--beta-root', str(beta),
                                '--upstream', str(UPSTREAM), '--output', str(output)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert {p.name for p in output.iterdir()} == {
        'shape-statistics.json', 'shape-statistics.md', 'shape-distribution.png', 'shape-distribution.svg'}
    result = json.loads((output / 'shape-statistics.json').read_text())
    source = result['sources']
    for key, path in [('manifest', metadata / 'manifest.json'), ('metadata', metadata / 'metadata.npz'),
                      ('acquisition_ledger', beta / 'acquisition_ledger.json')]:
        assert source[key]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    for hand in ('left', 'right'):
        field = source['beta'][hand]
        assert field['shape'] == [4, 10]
        assert field['dtype'] == 'float32'
        assert field['sha256'] == hashlib.sha256((beta / f'generated_mano_{hand}_beta.dat').read_bytes()).hexdigest()
    assert source['official_script']['sha256'] == hashlib.sha256((UPSTREAM / 'scripts/paper/plot_beta_diversity.py').read_bytes()).hexdigest()
    assert source['official_script']['revision'] == 'f7cbd048b7092139852270491d3859cd472e0311'
    for path in output.iterdir():
        data = path.read_bytes()
        for private in [b'PRIVATE', b'Alice', b'Bob', str(tmp_path).encode(), b'episode_id',
                        b'episode_subject', b'episode_beta_idx', b'private.example']:
            assert private not in data, (path.name, private)
    markdown = (output / 'shape-statistics.md').read_text()
    assert 'P95' in markdown and 'range' in markdown and 'max - min' in markdown
    assert 'independent people' in markdown
    svg = (output / 'shape-distribution.svg').read_text()
    assert all(f'beta{i}' in svg for i in range(1, 11))
    assert 'individual' in svg and 'bar chart' in svg
