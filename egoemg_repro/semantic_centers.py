"""Audit only explicitly frozen legacy center rows; never rebuild a partition.

The returned public_projection is constructed from aggregate counts. Private
paths and stat receipts remain in private_receipt. Recorded transfer hashes are
not a claim that this invocation rehashed complete source payloads.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile

import numpy as np

from . import semantic_tables as definitions

FORMAT = 'egoemg_legacy_center_audit_v1'
ROLES = ('train', 'val', 'test')
HANDS = ('left', 'right')
PARTITION_BUILDER_SHA256 = 'ec816483e5d567cec45df77bd6823e97c201d3fd20f45b0d8004827565fdf678'
MAX_SMALL_SOURCE_BYTES = 128 * 1024**2
COUNT_KEYS = ('window_centers', 'active_window_centers', 'hand_centers', 'active_hand_centers',
              'generated_valid_hand_centers', 'wrist_valid_hand_centers',
              'possible_angle_elements', 'valid_angle_elements')


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _safe(path):
    path = Path(path).expanduser()
    if '..' in path.parts:
        raise ValueError('parent traversal is not allowed')
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink source is not allowed')
    return path


def _stamp(value):
    return tuple(getattr(value, name) for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def _open_source(path, expected=None):
    path = _safe(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        actual = os.fstat(descriptor)
        if (not stat.S_ISREG(actual.st_mode) or _stamp(actual) != _stamp(path.stat()) or
                (expected is not None and actual.st_size != expected)):
            raise ValueError('source file type, identity or size differs')
    except (OSError, ValueError) as exc:
        if 'descriptor' in locals():
            os.close(descriptor)
        raise ValueError('required center source file is missing or invalid') from exc
    return descriptor, _stamp(actual)


def _read_small(path):
    descriptor, before = _open_source(path)
    try:
        if before[2] > MAX_SMALL_SOURCE_BYTES:
            raise ValueError('metadata/partition receipt exceeds small-source bound')
        blocks = []
        while block := os.read(descriptor, 1024**2):
            blocks.append(block)
        raw = b''.join(blocks)
        if len(raw) != before[2] or _stamp(os.fstat(descriptor)) != before or _stamp(_safe(path).stat()) != before:
            raise ValueError('small center source changed during inspection')
        return raw, before
    finally:
        os.close(descriptor)


def _json(raw, name):
    return definitions._json(raw, name)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('invalid integer ' + name)
    return value


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _field_specs(manifest):
    n = manifest['total_rows']
    expected = {'label_gesture_class': ('<i4', [n]), 'label_gesture_active': ('|b1', [n]),
                'frame_split_id': ('|i1', [n]), 'generated_label_valid': ('|b1', [n, 2])}
    for hand in HANDS:
        expected[f'generated_joint_angles_{hand}'] = ('<f4', [n, 20])
        for axis in ('pitch', 'yaw'):
            expected[f'mocap_{hand}_wrist_{axis}'] = ('<f4', [n])
        expected[f'mocap_{hand}_wrist_angles_valid'] = ('|b1', [n])
    result = {}
    for name, (dtype, shape) in expected.items():
        spec = manifest['fields'][name]
        if (spec.get('filename') != name + '.dat' or spec.get('shape') != shape or
                np.dtype(spec.get('dtype', '')) != np.dtype(dtype)):
            raise ValueError('center field schema differs')
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * np.dtype(dtype).itemsize
        result[name] = dict(filename=name + '.dat', dtype=np.dtype(dtype).str,
                            shape=shape, bytes=n * row_bytes, row_bytes=row_bytes)
    return result


def _windows(partition, metadata_raw, manifest_sha, metadata_sha, frame_sha):
    if partition.get('schema_version') != 1:
        raise ValueError('unknown frozen partition schema')
    provenance, settings = partition.get('provenance', {}), partition.get('settings', {})
    declared = provenance.get('partition_sha256')
    canonical = dict(partition, provenance={k: v for k, v in provenance.items() if k != 'partition_sha256'})
    computed = _hash(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode())
    if declared != computed:
        raise ValueError('frozen partition canonical hash differs')
    if (provenance.get('manifest_sha256') != manifest_sha or provenance.get('metadata_sha256') != metadata_sha
            or provenance.get('frame_split_sha256') != frame_sha
            or provenance.get('protocol') != 'legacy-prefiltered-temporal-validation'
            or provenance.get('interval_convention') != 'half-open [start,end)'):
        raise ValueError('frozen partition provenance differs from metadata/transfer identities')
    width = _integer(settings.get('window_length'), 'window length', 1)
    for name in ('train_stride', 'eval_stride'):
        _integer(settings.get(name), name, 1)
    fraction = settings.get('val_fraction')
    if (settings.get('jitter') is not False or type(fraction) not in (int, float)
            or not math.isfinite(fraction) or not 0 <= fraction < 1):
        raise ValueError('frozen partition settings are invalid')
    with np.load(io.BytesIO(metadata_raw), allow_pickle=False) as md:
        starts, ends = md['episode_start_idx'].copy(), md['episode_end_idx'].copy()
    episodes = partition.get('episodes')
    if not isinstance(episodes, list) or len(episodes) != len(starts):
        raise ValueError('partition numeric bounds differ from metadata')
    for i, (episode, lo, hi) in enumerate(zip(episodes, starts, ends)):
        if (not isinstance(episode, dict) or type(episode.get('index')) is not int or episode['index'] != i
                or type(episode.get('start')) is not int or episode['start'] != int(lo)
                or type(episode.get('end')) is not int or episode['end'] != int(hi)):
            raise ValueError('partition numeric bounds differ from metadata')
    supplied = partition.get('windows')
    if not isinstance(supplied, list) or not supplied:
        raise ValueError('frozen partition windows are missing')
    windows, seen, counts = [], set(), Counter()
    intervals = {role: [] for role in ROLES}
    for item in supplied:
        if not isinstance(item, dict):
            raise ValueError('invalid frozen window')
        ep, start, end, center = [_integer(item.get(k), 'window ' + k) for k in ('episode', 'start', 'end', 'center')]
        role, split = item.get('role'), item.get('split')
        if (role not in ROLES or split not in definitions.SPLITS or
                (role in ('train', 'val') and split != 'train') or (role == 'test' and split == 'train')):
            raise ValueError('unknown or inconsistent frozen split/role')
        if (ep >= len(starts) or start < int(starts[ep]) or end > int(ends[ep]) or end - start != width
                or center != start + width // 2):
            raise ValueError('frozen window crosses its bounds or has wrong center/length')
        if center in seen:
            raise ValueError('duplicate frozen window/center')
        seen.add(center)
        windows.append(dict(episode=ep, start=start, end=end, center=center, role=role, split=split))
        counts[role] += 1
        intervals[role].append((start, end))
    coverage = partition.get('coverage', {})
    for role in ROLES:
        item = coverage.get(role, {})
        if (type(item.get('windows')) is not int or item['windows'] != counts[role]
                or type(item.get('hand_samples')) is not int or item['hand_samples'] != 2 * counts[role]):
            raise ValueError('partition saved coverage differs from records')
    for i, left in enumerate(ROLES):
        for right in ROLES[i + 1:]:
            a, b = sorted(intervals[left]), sorted(intervals[right])
            j = 0
            for lo, hi in a:
                while j < len(b) and b[j][1] <= lo:
                    j += 1
                if j < len(b) and b[j][0] < hi:
                    raise ValueError('frozen role footprints overlap')
    return sorted(windows, key=lambda w: w['center']), computed


def _point_row(descriptor, spec, center):
    raw = os.pread(descriptor, spec['row_bytes'], center * spec['row_bytes'])
    if len(raw) != spec['row_bytes']:
        raise ValueError('source changed or truncated during center point read')
    if np.dtype(spec['dtype']).kind == 'b' and any(value not in (0, 1) for value in raw):
        raise ValueError('noncanonical boolean center payload')
    return raw


def _angle_mask(rows, hand_index):
    hand = HANDS[hand_index]
    fingers = np.frombuffer(rows[f'generated_joint_angles_{hand}'], dtype='<f4')
    wrist = np.array([np.frombuffer(rows[f'mocap_{hand}_wrist_{axis}'], dtype='<f4')[0]
                      for axis in ('pitch', 'yaw')], dtype=np.float32)
    angles = np.concatenate([fingers, np.deg2rad(wrist)]).astype(np.float32)
    generated = bool(rows['generated_label_valid'][hand_index])
    wrist_valid = bool(rows[f'mocap_{hand}_wrist_angles_valid'][0])
    finite = np.isfinite(angles)
    mask = np.array([generated] * 20 + [wrist_valid] * 2) & finite
    clean = np.nan_to_num(angles, nan=0, posinf=0, neginf=0)
    return clean, mask, finite, generated, wrist_valid


def _bucket():
    return {**dict.fromkeys(COUNT_KEYS, 0), 'finite_by_angle': [0] * 22, 'valid_by_angle': [0] * 22}


def _window(bucket, active):
    bucket['window_centers'] += 1
    bucket['active_window_centers'] += int(active)


def _hand(bucket, active, mask, finite, generated, wrist):
    bucket['hand_centers'] += 1
    bucket['active_hand_centers'] += int(active)
    bucket['generated_valid_hand_centers'] += int(generated)
    bucket['wrist_valid_hand_centers'] += int(wrist)
    bucket['possible_angle_elements'] += 22
    bucket['valid_angle_elements'] += int(mask.sum())
    bucket['finite_by_angle'] = [a + int(b) for a, b in zip(bucket['finite_by_angle'], finite)]
    bucket['valid_by_angle'] = [a + int(b) for a, b in zip(bucket['valid_by_angle'], mask)]


def _sum_buckets(buckets):
    result = _bucket()
    for item in buckets:
        for key in COUNT_KEYS:
            result[key] += item[key]
        for key in ('finite_by_angle', 'valid_by_angle'):
            result[key] = [a + b for a, b in zip(result[key], item[key])]
    return result


def _view(gestures, classes, excluded):
    selected = [row for row in gestures if row['id'] not in excluded]
    rows = [dict(class_id=g['id'], name=g['name'], family=g['family'], **classes[g['id']]) for g in selected]
    families = [dict(family=family, definition_count=sum(g['family'] == family for g in selected),
                     **_sum_buckets(classes[g['id']] for g in selected if g['family'] == family))
                for family in definitions.FAMILY_COUNTS]
    return dict(definition_count=len(selected), excluded_class_ids=list(excluded),
                observed_class_count=sum(r['window_centers'] > 0 for r in rows),
                active_observed_class_count=sum(r['active_window_centers'] > 0 for r in rows),
                totals=_sum_buckets(classes[g['id']] for g in selected), classes=rows, families=families)


def collect_center_coverage(data_root: Path, partition: Path, *, upstream: Path,
                            partition_sha256: str, transfer_receipt_sha256: str,
                            transfer_receipt: Path | None = None) -> dict:
    """Point-read all frozen centers once; return safe aggregates and a private receipt.

    partition_sha256 is the independently frozen *file* SHA256, distinct from
    the partition builder's canonical hash. No role/class/activity filtering is
    applied. Raw local-transfer hashes are preserved as recorded identities.
    """
    root, partition, upstream = _safe(data_root), _safe(partition), _safe(upstream)
    receipt_path = _safe(transfer_receipt) if transfer_receipt is not None else root / 'acquisition_ledger.json'
    small_paths = [root / n for n in ('manifest.json', 'metadata.npz', 'metadata_summary.json')]
    small_paths += [partition, receipt_path, Path(__file__), Path(__file__).with_name('partitions.py')]
    snapshots = {path: _read_small(path) for path in small_paths}
    raw = {path: value[0] for path, value in snapshots.items()}
    if not _sha(partition_sha256) or _hash(raw[partition]) != partition_sha256:
        raise ValueError('explicit frozen partition file hash differs')
    if not _sha(transfer_receipt_sha256) or _hash(raw[receipt_path]) != transfer_receipt_sha256:
        raise ValueError('explicit frozen transfer receipt hash differs')
    if _hash(raw[Path(__file__).with_name('partitions.py')]) != PARTITION_BUILDER_SHA256:
        raise ValueError('partition builder source differs from the inspected frozen protocol')
    tables = definitions.build_semantic_tables(upstream, root)
    manifest = _json(raw[root / 'manifest.json'], 'center manifest')
    frozen = _json(raw[partition], 'frozen partition')
    ledger = _json(raw[receipt_path], 'center transfer receipt')
    if ledger.get('ledger_version') != 1 or not isinstance(ledger.get('files'), dict):
        raise ValueError('invalid original transfer ledger')
    specs = _field_specs(manifest)
    fields = {}
    for name, spec in specs.items():
        record = ledger['files'].get(spec['filename'], {})
        if (type(record.get('bytes')) is not int or record['bytes'] != spec['bytes']
                or record.get('checksum_basis') != 'local_transfer' or not _sha(record.get('sha256'))):
            raise ValueError('required field lacks exact original local-transfer identity')
        fields[name] = dict(spec, transfer_sha256=record['sha256'], checksum_basis='local_transfer')
    windows, canonical_sha = _windows(frozen, raw[root / 'metadata.npz'], _hash(raw[root / 'manifest.json']),
                                      _hash(raw[root / 'metadata.npz']), fields['frame_split_id']['transfer_sha256'])
    overall, unassigned = _bucket(), _bucket()
    by_role = {key: _bucket() for key in ROLES}
    by_split = {key: _bucket() for key in definitions.SPLITS}
    by_hand = {key: _bucket() for key in HANDS}
    classes = {gid: _bucket() for gid in range(60)}
    contingency = {}
    opened, digests = {}, {name: hashlib.sha256() for name in fields}
    angle_digest, mask_digest = hashlib.sha256(), hashlib.sha256()
    read_bytes = 0
    try:
        for name, spec in fields.items():
            descriptor, before = _open_source(root / spec['filename'], spec['bytes'])
            opened[name] = descriptor, before
        for window in windows:
            center = window['center']
            rows = {}
            for name, spec in fields.items():
                row = _point_row(opened[name][0], spec, center)
                digests[name].update(struct.pack('<Q', center) + row)
                read_bytes += len(row)
                rows[name] = row
            gid = int(np.frombuffer(rows['label_gesture_class'], dtype='<i4')[0])
            split_id = int(np.frombuffer(rows['frame_split_id'], dtype='i1')[0])
            active = bool(rows['label_gesture_active'][0])
            if not -1 <= gid < 60 or not 0 <= split_id < len(definitions.SPLITS):
                raise ValueError('unknown gesture class or split at frozen center')
            if definitions.SPLITS[split_id] != window['split']:
                raise ValueError('frozen window split differs from actual center split')
            key = window['role'], window['split'], gid, active
            if key not in contingency:
                contingency[key] = _bucket()
            buckets = [overall, by_role[window['role']], by_split[window['split']],
                       classes[gid] if gid >= 0 else unassigned, contingency[key]]
            for bucket in buckets:
                _window(bucket, active)
            for index, hand in enumerate(HANDS):
                angles, mask, finite, generated, wrist = _angle_mask(rows, index)
                angle_digest.update(angles.tobytes())
                mask_digest.update(mask.tobytes())
                for bucket in buckets:
                    _hand(bucket, active, mask, finite, generated, wrist)
                _window(by_hand[hand], active)
                _hand(by_hand[hand], active, mask, finite, generated, wrist)
        identity_fields = [dict(name=name, dtype=item['dtype'], shape=item['shape'], bytes=item['bytes'],
                                recorded_transfer_sha256=item['transfer_sha256'], checksum_basis='local_transfer',
                                point_rows=len(windows), point_bytes=len(windows) * item['row_bytes'],
                                center_and_raw_bytes_sha256=digests[name].hexdigest()) for name, item in fields.items()]
        identity = dict(partition_file_sha256=partition_sha256, partition_canonical_sha256=canonical_sha,
                        manifest_sha256=_hash(raw[root / 'manifest.json']), metadata_sha256=_hash(raw[root / 'metadata.npz']),
                        metadata_summary_sha256=_hash(raw[root / 'metadata_summary.json']),
                        transfer_receipt_sha256=_hash(raw[receipt_path]), source_sha256=_hash(raw[Path(__file__)]),
                        official_revision=definitions.UPSTREAM_COMMIT, full_payload_rehash_performed=False,
                        transfer_hash_scope='recorded local-transfer identities; complete payloads not rehashed here',
                        point_digest_definition='SHA256 of sorted center uint64 little-endian followed by raw row bytes per field',
                        fields=identity_fields)
        gestures = tables['gesture_vocabulary']['rows']
        not_computed = lambda: dict(status='not_computed', count=None)
        public = dict(format=FORMAT, scope='all explicitly frozen windows; shared center labels, two independent hand masks',
                      source_identity=identity, denominators={key: overall[key] for key in
                          ('window_centers', 'hand_centers', 'possible_angle_elements', 'valid_angle_elements')},
                      overall=overall, by_role=by_role, by_split=by_split, by_hand=by_hand,
                      center_contingency=[dict(role=role, split=split, class_id=gid, active=active, **contingency[key])
                          for key in sorted(contingency, key=lambda k: (ROLES.index(k[0]), definitions.SPLITS.index(k[1]), k[2], k[3]))
                          for role, split, gid, active in [key]],
                      full60_view=_view(gestures, classes, []), exclude_raw59_view=_view(gestures, classes, [50]),
                      unassigned_centers=unassigned, timeline_frames=not_computed(),
                      raw_mask_hand_frames=not_computed(), model_error_statistics=not_computed(),
                      mask_protocol='legacy reproduction: independent first20 generated hand validity / last2 wrist validity AND per-angle finite before cleanup',
                      units=dict(finger_disk='rad', wrist_disk='deg', model='rad', cleanup='nonfinite to zero after mask'))
        private = dict(data_root=str(root), partition_path=str(partition), transfer_receipt_path=str(receipt_path),
                       observed_utc=datetime.now(timezone.utc).isoformat(), point_read_payload_bytes=read_bytes,
                       model_angle_radians_sha256=angle_digest.hexdigest(), angle_valid_mask_sha256=mask_digest.hexdigest(),
                       source_stats={name: list(before) for name, (_, before) in opened.items()},
                       small_source_stats={str(path): list(value[1]) for path, value in snapshots.items()},
                       complete_payload_rehash_performed=False, numpy_version=np.__version__)
        definitions._check_upstream(upstream)
        for path, value in snapshots.items():
            if _read_small(path) != value:
                raise ValueError('small source changed during center point inspection')
        for name, (descriptor, before) in opened.items():
            if (_stamp(os.fstat(descriptor)) != before or
                    _stamp(_safe(root / fields[name]['filename']).stat()) != before):
                raise ValueError('large source changed during center point inspection')
        return dict(format=FORMAT, public_projection=public, private_receipt=private)
    except (OSError, KeyError, IndexError, TypeError) as exc:
        raise ValueError('invalid or changed frozen center source') from exc
    finally:
        for descriptor, _ in opened.values():
            os.close(descriptor)


def _write_private(path, audit):
    path = _safe(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix='.center-audit-')
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'partition', 'upstream', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--partition-sha256', required=True)
    parser.add_argument('--transfer-receipt-sha256', required=True)
    parser.add_argument('--transfer-receipt', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('private audit output already exists')
        audit = collect_center_coverage(args.data_root, args.partition, upstream=args.upstream,
                                        partition_sha256=args.partition_sha256,
                                        transfer_receipt_sha256=args.transfer_receipt_sha256,
                                        transfer_receipt=args.transfer_receipt)
        _write_private(args.output, audit)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Center audit failed: {exc}\n')
    print(json.dumps(audit['public_projection']['denominators'], sort_keys=True))


if __name__ == '__main__':
    main()
