"""Read official per-episode JPEG LMDBs and cache only fixed manifest centers."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil

from .download import Asset, MIN_FREE_BYTES, download_many

# LMDB disallows opening the same file twice in a process. Share bounded read-only
# handles across dataset instances; the PID key gives each worker its own opens.
_READERS = OrderedDict()


def video_frame_at_center(store, window):
    """The EMG row is not a video frame number; use the released alignment field."""
    from .data import _validate_window
    _validate_window(store, window)
    name = next((name for name in ('image_webcam_frame_index', 'image_head_frame_index')
                 if name in store.fields), None)
    if name is None:
        raise KeyError('image_webcam_frame_index (or image_head_frame_index) is required')
    value = int(store.field(name)[int(window['center'])])
    if value < 0:
        raise ValueError('Negative source video frame index')
    return value


class CropStore:
    """Lazy, worker-local readonly reader. Missing keys differ from absent files."""
    def __init__(self, root):
        self.root = Path(root).resolve()

    def read(self, episode_id, video_frame_idx, hand):
        import lmdb
        if Path(episode_id).name != episode_id or episode_id in ('.', '..'):
            raise ValueError('Episode ID must be a single path component')
        if hand not in ('left', 'right') or int(video_frame_idx) < 0:
            raise ValueError('Invalid crop hand or video frame index')
        path = self.root / f'{episode_id}.lmdb'
        if not path.exists():
            path = self.root / f'{episode_id}.data.mdb'
        if not path.exists():
            raise FileNotFoundError(f'Crop LMDB unavailable: {path}')
        identity = (os.getpid(), str(path))
        if identity not in _READERS:
            while len(_READERS) >= 8:
                _, env = _READERS.popitem(last=False)
                env.close()
            _READERS[identity] = lmdb.open(str(path), subdir=path.is_dir(), readonly=True,
                                         lock=False, readahead=False, create=False)
        env = _READERS[identity]
        _READERS.move_to_end(identity)
        key = f'{int(video_frame_idx):08d}_{"L" if hand == "left" else "R"}'.encode('ascii')
        with env.begin() as transaction:
            return transaction.get(key)


def extract_centers(root, windows, crops_root, output):
    """Copy JPEG bytes at paired fixed centers, keeping absent keys in the report.

    Sources must already have been selectively downloaded. No row/video
    resampling, image recoding or dataset split decisions happen here.
    """
    import lmdb
    from .data import LegacyStore, file_sha256
    store, reader, output = LegacyStore(root), CropStore(crops_root), Path(output)
    windows = list(windows)
    if output.resolve() == Path(crops_root).resolve():
        raise ValueError('Cache output must differ from source crops')
    output.mkdir(parents=True, exist_ok=True)
    grouped, samples = {}, []
    for window in windows:
        frame = video_frame_at_center(store, window)
        episode = store.episodes[int(window['episode'])]['id']
        for hand in ('left', 'right'):
            key = f'{frame:08d}_{"L" if hand == "left" else "R"}'.encode('ascii')
            sample = dict(window, episode_id=episode, hand=hand, video_frame_index=frame)
            samples.append(sample)
            grouped.setdefault(episode, {}).setdefault((key, frame, hand), []).append(sample)
    unique_crops = 0
    payload_files = {}
    for episode, requests in grouped.items():
        destination = output / f'{episode}.lmdb'
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite an existing center cache: {destination}')
        # Bound payload memory by one episode, not the full selected dataset.
        values = {}
        for (key, frame, hand), references in requests.items():
            payload = reader.read(episode, frame, hand)
            if payload is not None:
                values[key] = payload
            for sample in references:
                sample['vision_valid'] = payload is not None
        unique_crops += len(values)
        size = sum(len(value) + 128 for value in values.values()) * 2 + 2**20
        if shutil.disk_usage(output).free - size < MIN_FREE_BYTES:
            raise OSError('Crop cache would violate the 15 GiB free-space reserve')
        temporary = output / f'{episode}.lmdb.partial'
        try:
            env = lmdb.open(str(temporary), subdir=False, map_size=size, lock=False)
            try:
                with env.begin(write=True) as transaction:
                    for key, payload in sorted(values.items()):
                        transaction.put(key, payload)
                env.sync()
            finally:
                env.close()
            temporary.replace(destination)
            payload_files[destination.name] = dict(bytes=destination.stat().st_size, sha256=file_sha256(destination),
                                                    checksum_basis='generated_cache_bytes')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    window_digest = hashlib.sha256(json.dumps(windows, sort_keys=True).encode()).hexdigest()
    report = dict(format='egoemg_fixed_center_jpeg_lmdb_v1', requested_samples=len(samples),
                  missing_samples=sum(not sample['vision_valid'] for sample in samples),
                  unique_crops=unique_crops, samples=samples,
                  payload_files=payload_files, windows_sha256=window_digest, manifest_sha256=file_sha256(store.manifest_path),
                  metadata_sha256=file_sha256(store.metadata_path),
                  mapping_field='image_webcam_frame_index' if 'image_webcam_frame_index' in store.fields else 'image_head_frame_index')
    (output / 'center_cache_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def select_crop_assets(inventory, episodes=None):
    selected = set(episodes) if episodes else {entry['episode_id'] for entry in inventory['files']}
    unknown = selected - {entry['episode_id'] for entry in inventory['files']}
    if unknown:
        raise ValueError(f'Unknown crop episodes: {sorted(unknown)}')
    def asset(entry, group):
        return Asset(entry['filename'], entry['drive_id'], entry['expected_bytes'],
                     entry.get('sha256'), group)
    crops = [asset(entry, 'visual_crops') for entry in inventory['files'] if entry['episode_id'] in selected]
    fields = [asset(entry, 'visual_alignment') for entry in inventory.get('required_frame_fields', [])]
    return crops, fields


def download_crops(inventory_path, root, *, data=None, index_root=None, workers=2,
                   episodes=None, fields_only=False):
    """Reuse atomic, resumable, byte-validated downloads and the 15 GiB reserve.

    Pass index_root during concurrent signal acquisition to isolate its receipt.
    Linking verified alignment fields into the data root is an explicit later
    integration step once both independent acquisition processes finish.
    """
    inventory = json.loads(Path(inventory_path).read_text())
    crops, fields = select_crop_assets(inventory, episodes)
    field_report = None
    if fields_only or index_root is not None:
        destination = index_root if index_root is not None else data
        if destination is None:
            raise ValueError('--fields-only requires --index-root or --data')
        download_many(fields, Path(destination), workers=workers)
        field_report = dict(root=str(destination), files=[a.filename for a in fields])
    crop_report = None
    if not fields_only:
        download_many(crops, Path(root), workers=workers)
        crop_report = dict(root=str(root), files=[a.filename for a in crops])
    return dict(crops=crop_report, fields=field_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cache = commands.add_parser('cache', help='Extract only manifest-selected center JPEGs')
    cache.add_argument('--data', required=True)
    cache.add_argument('--partitions', required=True)
    cache.add_argument('--crops', required=True)
    cache.add_argument('--output', required=True)
    download = commands.add_parser('download', help='Selectively download published crops or alignment fields')
    download.add_argument('--inventory', default='assets_crops_manifest.json')
    download.add_argument('--root', required=True)
    download.add_argument('--data')
    download.add_argument('--index-root')
    download.add_argument('--episodes', nargs='+')
    download.add_argument('--workers', type=int, default=2)
    download.add_argument('--fields-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'cache':
        manifest = json.loads(Path(args.partitions).read_text())
        report = extract_centers(args.data, manifest['windows'], args.crops, args.output)
        print(json.dumps({key: value for key, value in report.items() if key != 'samples'}, indent=2))
    else:
        report = download_crops(args.inventory, args.root, data=args.data, index_root=args.index_root,
                                workers=args.workers, episodes=args.episodes, fields_only=args.fields_only)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
