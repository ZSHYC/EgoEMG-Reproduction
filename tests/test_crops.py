"""JPEG-in-LMDB fixtures catch row/video confusion and crop population loss."""
import io
import json
from pathlib import Path

import numpy as np
import pytest

from egoemg_repro.crops import CropStore, extract_centers, video_frame_at_center
from egoemg_repro.data import LegacyStore
from test_data import make_store, window


def add_video_indices(root, indices):
    manifest = json.loads((root / 'manifest.json').read_text())
    values = np.asarray(indices, dtype=np.int32)
    values.tofile(root / 'video.dat')
    manifest['fields']['image_webcam_frame_index'] = dict(filename='video.dat', dtype='int32', shape=list(values.shape))
    (root / 'manifest.json').write_text(json.dumps(manifest))


def make_crops(root, episode='episode_0', frame=17):
    lmdb = pytest.importorskip('lmdb')
    Image = pytest.importorskip('PIL.Image')
    root.mkdir(exist_ok=True)
    # An asymmetric image detects a second, incorrect left flip.
    rgb = np.zeros((256, 256, 3), np.uint8)
    rgb[:, :128, 0] = 255
    rgb[:, 128:, 2] = 255
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format='JPEG', quality=100, subsampling=0)
    env = lmdb.open(str(root / f'{episode}.lmdb'), subdir=False, map_size=2**20)
    with env.begin(write=True) as txn:
        txn.put(f'{frame:08d}_L'.encode(), stream.getvalue())
        txn.put(b'99999999_L', b'unused')
    env.close()
    return stream.getvalue()


def test_uses_video_mapping_at_emg_center_and_rejects_absent_mapping(tmp_path):
    make_store(tmp_path)
    add_video_indices(tmp_path, np.arange(47) + 15)
    assert video_frame_at_center(LegacyStore(tmp_path), window()) == 17
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    del manifest['fields']['image_webcam_frame_index']
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(KeyError, match='image_'):
        video_frame_at_center(LegacyStore(tmp_path), window())


def test_missing_key_is_explicit_but_absent_episode_file_fails(tmp_path):
    payload = make_crops(tmp_path)
    reader = CropStore(tmp_path)
    assert reader.read('episode_0', 17, 'left') == payload
    assert reader.read('episode_0', 17, 'right') is None
    with pytest.raises(FileNotFoundError):
        reader.read('episode_absent', 17, 'left')


def test_cache_retains_only_selected_keys_and_records_missing_hand(tmp_path):
    data, crops, cache = [tmp_path / name for name in ('data', 'crops', 'cache')]
    data.mkdir()
    make_store(data)
    add_video_indices(data, np.arange(47) + 15)
    payload = make_crops(crops)
    report = extract_centers(data, [window()], crops, cache)
    assert report['requested_samples'] == 2
    assert report['missing_samples'] == 1
    reader = CropStore(cache)
    assert reader.read('episode_0', 17, 'left') == payload
    assert reader.read('episode_0', 99999999, 'left') is None
    assert reader.read('episode_0', 17, 'right') is None
    assert report['samples'][0]['center'] == report['samples'][1]['center'] == 2
    import hashlib
    written = (cache / 'episode_0.lmdb').read_bytes()
    assert report['payload_files']['episode_0.lmdb']['bytes'] == len(written)
    assert report['payload_files']['episode_0.lmdb']['sha256'] == hashlib.sha256(written).hexdigest()


def test_flat_download_filename_is_read_as_the_same_lmdb_schema(tmp_path):
    payload = make_crops(tmp_path)
    (tmp_path / 'episode_0.lmdb').rename(tmp_path / 'episode_0.data.mdb')
    assert CropStore(tmp_path).read('episode_0', 17, 'left') == payload


def test_selective_download_inventory_selects_only_requested_episode_and_fields():
    from egoemg_repro.crops import select_crop_assets
    inventory = dict(files=[dict(episode_id='episode_a', filename='a.data.mdb', drive_id='a', expected_bytes=1024),
                            dict(episode_id='episode_b', filename='b.data.mdb', drive_id='b', expected_bytes=2048)],
                     required_frame_fields=[dict(field='image_webcam_frame_index', filename='index.dat', drive_id='i', expected_bytes=100)])
    crops, fields = select_crop_assets(inventory, episodes=['episode_b'])
    assert [a.filename for a in crops] == ['b.data.mdb']
    assert [a.filename for a in fields] == ['index.dat']
    with pytest.raises(ValueError, match='Unknown'):
        select_crop_assets(inventory, episodes=['episode_missing'])


def _fork_read(path, queue):
    try:
        queue.put(CropStore(path).read('episode_0', 17, 'left') is not None)
    except Exception as error:
        queue.put(str(error))


def test_worker_fork_reopens_after_parent_read(tmp_path):
    import multiprocessing
    if 'fork' not in multiprocessing.get_all_start_methods():
        pytest.skip('fork worker behavior is platform-specific')
    make_crops(tmp_path)
    assert CropStore(tmp_path).read('episode_0', 17, 'left') is not None
    context = multiprocessing.get_context('fork')
    queue = context.Queue()
    child = context.Process(target=_fork_read, args=(tmp_path, queue))
    child.start()
    child.join(timeout=10)
    try:
        assert child.exitcode == 0
        assert queue.get(timeout=2) is True
    finally:
        if child.is_alive():
            child.terminate()
            child.join()


def test_download_help_runs_without_numpy_or_any_site_packages():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-S', '-m', 'egoemg_repro.crops', 'download', '--help'],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--episodes' in result.stdout
