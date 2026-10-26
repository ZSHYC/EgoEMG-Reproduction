"""Project a verified shared-frame audit; do not rescan arrays or infer windows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np

from . import semantic_tables as definitions


AUDIT_FORMAT = 'private_legacy_gesture_frame_audit_v1'
AUDIT_SCOPES = {
    'engineering fixture only': 'engineering_fixture',
    'All real shared per-row class/active/split arrays; no model/window/center or independent-subject count claims':
        'recorded_legacy_frame_audit',
}
FIELD_DTYPES = {'label_gesture_class': np.dtype('<i4'), 'label_gesture_active': np.dtype('bool'),
                'frame_split_id': np.dtype('i1')}


def _id_list(value, expected, description):
    if (not isinstance(value, list) or any(type(v) is not int for v in value)
            or value != sorted(set(value)) or value != expected):
        raise ValueError(f'audit {description} differs from its counted records')


def _validate_audit(audit, total_rows):
    if (audit.get('format') != AUDIT_FORMAT or type(audit.get('total_synchronized_rows')) is not int
            or audit['total_synchronized_rows'] != total_rows or audit.get('no_hand_duplication') is not True
            or audit.get('split_labels') != definitions.SPLITS):
        raise ValueError('audit format, row denominator or split identity is invalid')
    scope = audit.get('scope')
    if not isinstance(scope, str) or scope not in AUDIT_SCOPES:
        raise ValueError('audit scope is unrecognized')
    timestamp = audit.get('observed_utc')
    try:
        stamp = datetime.fromisoformat(timestamp)
        if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError('UTC timestamp required')
        timestamp = stamp.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError('audit requires a valid UTC observation timestamp') from exc
    supplied = audit.get('fields')
    if not isinstance(supplied, dict):
        raise ValueError('audit field identities are missing')
    fields = {}
    for name, dtype in FIELD_DTYPES.items():
        item = supplied.get(name)
        try:
            valid = (isinstance(item, dict) and isinstance(item.get('shape'), list)
                     and item['shape'] == [total_rows] and all(type(v) is int for v in item['shape'])
                     and np.dtype(item.get('dtype', '')) == dtype
                     and type(item.get('bytes')) is int and item['bytes'] == total_rows * dtype.itemsize
                     and item.get('matches_acquisition_receipt') is True
                     and item.get('checksum_basis') == 'local_transfer'
                     and isinstance(item.get('sha256'), str)
                     and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None)
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError('audit field schema or matching transfer identity is invalid')
        fields[name] = dict(filename=name + '.dat', dtype=dtype.str, shape=[total_rows],
                            bytes=total_rows * dtype.itemsize, sha256=item['sha256'],
                            checksum_basis='local_transfer', matches_acquisition_receipt=True)
    supplied_rows = audit.get('counts')
    if not isinstance(supplied_rows, list) or not supplied_rows:
        raise ValueError('audit count records are missing')
    rows, keys = [], set()
    for item in supplied_rows:
        if not isinstance(item, dict):
            raise ValueError('invalid audit count record')
        gid, split, active, count = [item.get(k) for k in ('class_id', 'split', 'active', 'synchronized_rows')]
        if (type(gid) is not int or not -1 <= gid < 60 or not isinstance(split, str)
                or split not in definitions.SPLITS or type(active) is not bool
                or type(count) is not int or count < 0):
            raise ValueError('audit class, split, activity or integer count is invalid')
        key = gid, split, active
        if key in keys:
            raise ValueError('duplicate audit count key')
        keys.add(key)
        rows.append(dict(class_id=gid, split=split, active=active, synchronized_rows=count))
    if sum(r['synchronized_rows'] for r in rows) != total_rows:
        raise ValueError('audit synchronized row counts do not balance')
    observed = sorted({r['class_id'] for r in rows if r['synchronized_rows'] > 0})
    active_observed = sorted({r['class_id'] for r in rows if r['active'] and r['synchronized_rows'] > 0})
    _id_list(audit.get('actual_class_ids'), observed, 'observed classes')
    _id_list(audit.get('actual_active_class_ids'), active_observed, 'active observed classes')
    value = audit.get('class_range')
    if (not isinstance(value, list) or any(type(v) is not int for v in value)
            or value != [min(observed), max(observed)]):
        raise ValueError('audit class range differs from its counted records')
    rows.sort(key=lambda r: (definitions.SPLITS.index(r['split']), r['class_id'], r['active']))
    return rows, fields, timestamp, AUDIT_SCOPES[scope]


def _totals(rows):
    total = sum(r['synchronized_rows'] for r in rows)
    active = sum(r['synchronized_rows'] for r in rows if r['active'])
    defined = sum(r['synchronized_rows'] for r in rows if r['class_id'] >= 0)
    return dict(total=total, active=active, inactive=total - active, defined=defined,
                unassigned=total - defined,
                active_defined=sum(r['synchronized_rows'] for r in rows if r['active'] and r['class_id'] >= 0),
                raw50=sum(r['synchronized_rows'] for r in rows if r['class_id'] == 50))


def _view(rows, gestures, excluded):
    selected = [g for g in gestures if g['id'] not in excluded]
    ids = {g['id'] for g in selected}
    counted = [r for r in rows if r['class_id'] in ids]
    observed = {r['class_id'] for r in counted if r['synchronized_rows'] > 0}
    active_observed = {r['class_id'] for r in counted if r['active'] and r['synchronized_rows'] > 0}
    membership = {g['id']: g['family'] for g in selected}
    family_definitions = {f: sum(g['family'] == f for g in selected) for f in definitions.FAMILY_COUNTS}
    family_rows = {f: sum(r['synchronized_rows'] for r in counted if membership[r['class_id']] == f)
                   for f in definitions.FAMILY_COUNTS}
    family_active = {f: sum(r['synchronized_rows'] for r in counted if r['active'] and membership[r['class_id']] == f)
                     for f in definitions.FAMILY_COUNTS}
    return dict(definition_count=len(ids), observed_class_count=len(observed),
                active_observed_class_count=len(active_observed),
                synchronized_rows=sum(r['synchronized_rows'] for r in counted),
                active_synchronized_rows=sum(r['synchronized_rows'] for r in counted if r['active']),
                family_definition_counts=family_definitions, family_rows=family_rows,
                family_active_rows=family_active,
                denominator='shared synchronized timeline rows in defined classes; unassigned rows are separate')


def _compile(audit_path, upstream, metadata_root):
    semantic, snapshots = definitions._compile(upstream, metadata_root)
    raw = definitions._snapshot(Path(audit_path), snapshots)
    audit = definitions._json(raw, 'frame audit')
    rows, fields, stamp, data_scope = _validate_audit(audit, semantic['total_rows'])
    gestures = semantic['gesture_vocabulary']['rows']
    all_totals = _totals(rows)
    full = _view(rows, gestures, set())
    excluded = _view(rows, gestures, {50})
    excluded.update(before_count=full['synchronized_rows'],
                    raw_excluded_count=full['synchronized_rows'] - excluded['synchronized_rows'],
                    after_count=excluded['synchronized_rows'], before_active=full['active_synchronized_rows'],
                    raw_excluded_active=full['active_synchronized_rows'] - excluded['active_synchronized_rows'],
                    after_active=excluded['active_synchronized_rows'], excluded_ids=[50],
                    scope='59-class frame view only; no model errors or error-analysis sample count')
    by_gesture = []
    for gesture in gestures:
        subset = [r for r in rows if r['class_id'] == gesture['id']]
        total = sum(r['synchronized_rows'] for r in subset)
        active = sum(r['synchronized_rows'] for r in subset if r['active'])
        by_gesture.append(dict(id=gesture['id'], name=gesture['name'], family=gesture['family'],
                               total=total, active=active, inactive=total - active, observed=total > 0))
    sources = dict(semantic['gesture_vocabulary']['sources'])
    sources['frame_audit'] = dict(filename='legacy-gesture-frame-audit.json', sha256=definitions._digest(raw))
    sources['fields'] = fields
    sources['coverage_reproduction'] = dict(filename='egoemg_repro/semantic_coverage.py',
                                           sha256=definitions._digest(definitions._snapshot(Path(__file__), snapshots)))
    result = dict(schema_version=1, status='complete_frame_audit_projection', data_scope=data_scope,
                  scope='shared synchronized timeline rows only; not hand, window, center or independent-person counts',
                  definition_count=60, observed_class_count=full['observed_class_count'],
                  active_observed_class_count=full['active_observed_class_count'],
                  timeline_frames=all_totals,
                  by_split={s: _totals([r for r in rows if r['split'] == s]) for s in definitions.SPLITS},
                  by_gesture=by_gesture, records=rows, full60_view=full,
                  exclude_raw59_frame_view=excluded, sources=sources,
                  provenance=dict(computation_input='verified_frame_audit',
                                  source_array_rehash_performed_by='recorded_audit',
                                  this_export_rescanned_raw_arrays=False,
                                  no_hand_duplication=True, observed_utc=stamp,
                                  verification_scope='schema and aggregate consistency checked here; raw-array hashes and full scan belong to the recorded audit',
                                  publisher_authenticity='not established by local-transfer hashes'),
                  balances=dict(aggregate_rows=sum(r['synchronized_rows'] for r in rows),
                                active_plus_inactive=all_totals['active'] + all_totals['inactive'],
                                defined_plus_unassigned=all_totals['defined'] + all_totals['unassigned'],
                                family_defined_rows=sum(full['family_rows'].values()),
                                after_raw_plus_excluded=excluded['after_count'] + excluded['raw_excluded_count']),
                  notes=['Activity is not a joint-label validity mask.',
                         'The complete vocabulary keeps raw ID 50; its exclusion is explicit only in the 59-class frame view.',
                         'Zero observed frames for a defined class do not imply zero model error.',
                         'No window, hand-center, valid-angle, participant or training result is inferred from frame counts.'])
    for name in ('window_centers', 'hand_centers', 'raw_mask_hand_frames', 'center_valid_angles', 'model_error_statistics'):
        result[name] = dict(status='not_computed', count=None)
    definitions._verify_snapshots(snapshots, Path(upstream))
    return result, snapshots


def build_frame_coverage(audit_path: Path, *, upstream: Path, metadata_root: Path) -> dict:
    """Validate and project a recorded frame audit without reading any arrays."""
    return _compile(audit_path, upstream, metadata_root)[0]


def _markdown(doc):
    total = doc['timeline_frames']
    title = 'Engineering fixture frame coverage' if doc['data_scope'] == 'engineering_fixture' else 'Legacy synchronized-frame coverage'
    lines = ['# ' + title, '',
             'Unit: shared synchronized timeline rows. This is not a hand, window, center or independent-person count.',
             'This export projects a verified audit; it is not a new raw-array scan.', '',
             f"Observed defined classes: {doc['observed_class_count']}/60; active observed classes: {doc['active_observed_class_count']}/60.",
             f"All rows: {total['total']:,}; active: {total['active']:,}; inactive: {total['inactive']:,}; unassigned: {total['unassigned']:,}.",
             f"raw (ID 50): {total['raw50']:,} rows; retained in the complete vocabulary.", '',
             '| Split | All rows | Active | Inactive | Unassigned |',
             '| --- | ---: | ---: | ---: | ---: |']
    for split, counts in doc['by_split'].items():
        lines.append(f"| {split} | {counts['total']:,} | {counts['active']:,} | {counts['inactive']:,} | {counts['unassigned']:,} |")
    lines.extend(['', '| Family | Definitions | Observed timeline rows | Active timeline rows |',
                  '| --- | ---: | ---: | ---: |'])
    full = doc['full60_view']
    for family, count in full['family_definition_counts'].items():
        lines.append(f"| {family} | {count} | {full['family_rows'][family]:,} | {full['family_active_rows'][family]:,} |")
    excluded = doc['exclude_raw59_frame_view']
    lines.extend(['', f"59-class frame view: {excluded['before_count']:,} defined-class rows - "
                  f"{excluded['raw_excluded_count']:,} raw rows = {excluded['after_count']:,} rows.",
                  'Its definition family sizes are 30/17/12. These are not Table 14/15 error-analysis samples.', '',
                  '| ID | Name | Family | All rows | Active | Inactive |',
                  '| ---: | --- | --- | ---: | ---: | ---: |'])
    for row in doc['by_gesture']:
        lines.append(f"| {row['id']} | {row['name']} | {row['family']} | {row['total']:,} | {row['active']:,} | {row['inactive']:,} |")
    lines.extend(['', 'The JSON includes the complete class/split/activity contingency records.', '',
                  'Window centers, hand centers, raw validity-mask hand frames, valid angles and model errors: not_computed (null).',
                  '', *doc['notes'], '', f"Recorded audit observation (UTC): {doc['provenance']['observed_utc']}.",
                  f"Recorded audit SHA256: `{doc['sources']['frame_audit']['sha256']}`.",
                  'Raw-array hashes and complete scanning belong to that recorded audit; this export checks schema and aggregate consistency.',
                  'Local-transfer hashes do not establish independent publisher authenticity.', '', 'Audited field identities:', ''])
    for item in doc['sources']['fields'].values():
        lines.append(f"- {item['filename']} ({item['bytes']:,} bytes): `{item['sha256']}`")
    lines.extend(definitions._source_markdown(dict(sources=doc['sources'], paper_reference=definitions._paper_reference(4, 15))))
    item = doc['sources']['coverage_reproduction']
    lines.append(f"- {item['filename']}: `{item['sha256']}`")
    return '\n'.join(lines) + '\n'


def export_frame_coverage(audit_path: Path, *, upstream: Path, metadata_root: Path, output: Path) -> dict:
    result, snapshots = _compile(audit_path, upstream, metadata_root)
    payloads = {'coverage.json': json.dumps(result, indent=2, allow_nan=False) + '\n',
                'coverage.md': _markdown(result)}
    definitions._verify_snapshots(snapshots, Path(upstream))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        (output / name).write_text(value, encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame-audit', type=Path, required=True)
    parser.add_argument('--metadata-root', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = export_frame_coverage(args.frame_audit, upstream=args.upstream,
                                       metadata_root=args.metadata_root, output=args.output)
    except ValueError as exc:
        parser.exit(2, f'Semantic coverage export failed: {exc}\n')
    print(json.dumps(dict(artifacts=2, synchronized_rows=result['timeline_frames']['total'],
                          observed_classes=result['observed_class_count']), sort_keys=True))


if __name__ == '__main__':
    main()
