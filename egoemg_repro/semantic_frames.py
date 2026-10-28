"""Stream every released row for aggregate angle validity, without models.

This measures the declared legacy reproduction mask, not annotation accuracy
or the official ordinary-path loss mask. Public output contains counts only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import semantic_centers as centers, semantic_tables as definitions

HANDS = ('left', 'right')
SCALARS = ('hand_frames', 'active_hand_frames', 'generated_valid_hand_frames',
           'wrist_valid_hand_frames', 'fingers_all20_valid_hand_frames',
           'wrist_all2_valid_hand_frames', 'all22_valid_hand_frames',
           'any_angle_valid_hand_frames', 'possible_angle_elements', 'valid_angle_elements')
ATTRIBUTES = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns',
              'st_nlink', 'st_mode', 'st_uid', 'st_gid')


def _stamp(value):
    return tuple(getattr(value, name) for name in ATTRIBUTES)


def _empty():
    return dict(**dict.fromkeys(SCALARS, 0), finite_by_angle=[0]*22, valid_by_angle=[0]*22)


def _sum(values):
    out = _empty()
    for item in values:
        for key in SCALARS:
            out[key] += item[key]
        for key in ('finite_by_angle', 'valid_by_angle'):
            out[key] = [a+b for a, b in zip(out[key], item[key])]
    return out


def _count(finite, mask, generated, wrist, active):
    n = len(mask)
    return dict(hand_frames=n, active_hand_frames=n*int(active),
                generated_valid_hand_frames=int(np.count_nonzero(generated)),
                wrist_valid_hand_frames=int(np.count_nonzero(wrist)),
                fingers_all20_valid_hand_frames=int(np.count_nonzero(mask[:, :20].all(axis=1))),
                wrist_all2_valid_hand_frames=int(np.count_nonzero(mask[:, 20:].all(axis=1))),
                all22_valid_hand_frames=int(np.count_nonzero(mask.all(axis=1))),
                any_angle_valid_hand_frames=int(np.count_nonzero(mask.any(axis=1))),
                possible_angle_elements=n*22, valid_angle_elements=int(np.count_nonzero(mask)),
                finite_by_angle=finite.sum(axis=0, dtype=np.int64).tolist(),
                valid_by_angle=mask.sum(axis=0, dtype=np.int64).tolist())


def collect_frame_validity(root, *, upstream, transfer_receipt_sha256, chunk_rows=65536):
    if type(chunk_rows) is not int or not 1 <= chunk_rows <= 262144:
        raise ValueError('bounded positive integer chunk size required')
    root = centers._safe(root)
    small = [root/n for n in ('manifest.json', 'metadata_summary.json', 'metadata.npz', 'acquisition_ledger.json')]
    small += [Path(__file__), Path(centers.__file__), Path(definitions.__file__)]
    snapshots = {p: centers._read_small(p) for p in small}
    raw = {p: value[0] for p, value in snapshots.items()}
    if centers._hash(raw[root/'acquisition_ledger.json']) != transfer_receipt_sha256:
        raise ValueError('explicit transfer receipt SHA256 differs')
    tables = definitions.build_semantic_tables(upstream, root)
    manifest = json.loads(raw[root/'manifest.json'])
    ledger = json.loads(raw[root/'acquisition_ledger.json'])
    if ledger.get('ledger_version') != 1 or not isinstance(ledger.get('files'), dict):
        raise ValueError('invalid original transfer ledger')
    specs = centers._field_specs(manifest)
    fields = {}
    for name, spec in specs.items():
        record = ledger['files'].get(spec['filename'], {})
        if (type(record.get('bytes')) is not int or record['bytes'] != spec['bytes']
                or record.get('checksum_basis') != 'local_transfer' or not centers._sha(record.get('sha256'))):
            raise ValueError('field lacks original local-transfer identity')
        fields[name] = dict(spec, transfer_sha256=record['sha256'])
    opened, digests, records = {}, {n: hashlib.sha256() for n in fields}, {}
    try:
        for name, spec in fields.items():
            fd, _ = centers._open_source(root/spec['filename'], spec['bytes'])
            opened[name] = fd, _stamp(os.fstat(fd))
        total = manifest['total_rows']
        for lo in range(0, total, chunk_rows):
            n = min(chunk_rows, total-lo)
            arrays = {}
            for name, spec in fields.items():
                payload = os.pread(opened[name][0], n*spec['row_bytes'], lo*spec['row_bytes'])
                if len(payload) != n*spec['row_bytes']:
                    raise ValueError('truncated frame source')
                if np.dtype(spec['dtype']).kind == 'b' and np.any(np.frombuffer(payload, dtype='u1') > 1):
                    raise ValueError('noncanonical boolean frame payload')
                digests[name].update(payload)
                arrays[name] = np.frombuffer(payload, dtype=spec['dtype']).reshape(n, *spec['shape'][1:])
            gids, splits, activity = [arrays[k] for k in ('label_gesture_class', 'frame_split_id', 'label_gesture_active')]
            if np.any((gids < -1) | (gids > 59)) or np.any((splits < 0) | (splits > 3)):
                raise ValueError('unknown frame class or split')
            codes = (splits.astype('i8')*61 + gids + 1)*2 + activity
            groups = [(int(code), codes == code) for code in np.unique(codes)]
            for code, selected in groups:
                if code not in records:
                    records[code] = dict(timeline_frames=0, **{h: _empty() for h in HANDS})
                records[code]['timeline_frames'] += int(np.count_nonzero(selected))
            for index, hand in enumerate(HANDS):
                fingers = arrays[f'generated_joint_angles_{hand}']
                wrist_angles = np.column_stack([arrays[f'mocap_{hand}_wrist_{a}'] for a in ('pitch', 'yaw')])
                finite = np.concatenate([np.isfinite(fingers), np.isfinite(np.deg2rad(wrist_angles))], axis=1)
                generated = arrays['generated_label_valid'][:, index]
                wrist = arrays[f'mocap_{hand}_wrist_angles_valid']
                mask = finite.copy()
                mask[:, :20] &= generated[:, None]
                mask[:, 20:] &= wrist[:, None]
                for code, selected in groups:
                    item = _count(finite[selected], mask[selected], generated[selected], wrist[selected], code % 2)
                    records[code][hand] = _sum([records[code][hand], item])
        identities = []
        for name, spec in fields.items():
            actual = digests[name].hexdigest()
            if actual != spec['transfer_sha256']:
                raise ValueError('complete frame payload SHA256 differs: '+name)
            identities.append(dict(name=name, dtype=spec['dtype'], shape=spec['shape'], bytes=spec['bytes'],
                                   sha256=actual, matches_original_transfer=True))
        rows = []
        for code, item in sorted(records.items()):
            category, active = divmod(code, 2)
            split, gid = divmod(category, 61)
            rows.append(dict(split=definitions.SPLITS[split], class_id=gid-1, active=bool(active), **item))
        def aggregate(items):
            return dict(timeline_frames=sum(r['timeline_frames'] for r in items),
                        **_sum([r[h] for r in items for h in HANDS]))
        overall = _sum([r[h] for r in rows for h in HANDS])
        if sum(r['timeline_frames'] for r in rows) != total or overall['hand_frames'] != 2*total:
            raise ValueError('frame/hand denominators do not balance')
        public = dict(format='egoemg_legacy_full_frame_validity_v1', timeline_frames=total,
            episode_count=manifest['num_episodes'], overall=overall,
            by_hand={h: _sum([r[h] for r in rows]) for h in HANDS},
            by_split={s: aggregate([r for r in rows if r['split']==s]) for s in definitions.SPLITS},
            by_gesture=[dict(id=g['id'], name=g['name'], family=g['family'],
                            **aggregate([r for r in rows if r['class_id']==g['id']]))
                        for g in tables['gesture_vocabulary']['rows']],
            unassigned=aggregate([r for r in rows if r['class_id']==-1]), records=rows,
            mask_protocol='legacy reproduction: generated first20 / independent wrist last2 AND finite before cleanup',
            source_identity=dict(official_revision=definitions.UPSTREAM_COMMIT,
                complete_payload_rehash_performed=True, fields=identities,
                small_files_sha256={p.name: centers._hash(b) for p,b in raw.items()},
                local_transfer_hashes_do_not_establish_publisher_authenticity=True),
            model_errors_computed=False, labels_regenerated=False, paper_population_equivalence_established=False,
            official_ordinary_loss_mask_equivalence_claimed=False)
        definitions._check_upstream(upstream)
        for p, value in snapshots.items():
            if centers._read_small(p) != value:
                raise ValueError('small frame audit source changed')
        for name, (fd, before) in opened.items():
            if _stamp(os.fstat(fd)) != before or _stamp((root/fields[name]['filename']).stat()) != before:
                raise ValueError('frame input nine attributes changed')
        private = dict(data_root=str(root), observed_utc=datetime.now(timezone.utc).isoformat(),
            chunk_rows=chunk_rows, read_payload_bytes=sum(s['bytes'] for s in fields.values()),
            all_input_nine_attributes_unchanged=True,
            input_nine_attributes={n:list(value[1]) for n,value in opened.items()},
            numpy_version=np.__version__, torch_imported='torch' in sys.modules)
        return dict(public_projection=public, private_receipt=private)
    finally:
        for fd, _ in opened.values():
            os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'upstream', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--transfer-receipt-sha256', required=True)
    parser.add_argument('--chunk-rows', type=int, default=65536)
    a = parser.parse_args(argv)
    if a.output.exists():
        raise FileExistsError('frame audit output already exists')
    audit = collect_frame_validity(a.data_root, upstream=a.upstream,
        transfer_receipt_sha256=a.transfer_receipt_sha256, chunk_rows=a.chunk_rows)
    centers._write_private(a.output, audit)
    print(json.dumps(dict(timeline_frames=audit['public_projection']['timeline_frames'],
                          valid_angle_elements=audit['public_projection']['overall']['valid_angle_elements'])))


if __name__ == '__main__':
    main()
