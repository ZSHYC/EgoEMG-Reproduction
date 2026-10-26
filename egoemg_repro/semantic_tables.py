"""Export pinned gesture and angle definitions; observed coverage is separate."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np

from . import UPSTREAM_COMMIT


PAPER_SHA256 = '124318dce75d9af00ab4d8427acf01315cf83ae1992082b783d5a488c2474725'
FAMILY_COUNTS = {'Single-hand': 30, 'Symmetric bimanual': 18, 'Asymmetric bimanual': 12}
SPLITS = ['train', 'user', 'gesture', 'both']
GESTURE_SCRIPT = 'scripts/paper/generate_per_gesture_appendix.py'
CONSTANTS = 'egoemg/constants.py'
PLOT_SCRIPT = 'scripts/paper/plot_paper_figures.py'
HAND_MODEL = 'egoemg/UmeTrack/dataset/generic_hand_model.json'
OFFICIAL_HASHES = {
    CONSTANTS: '01f4ad9d6be5b587d46d28b01c2325d584a593a63bd14c79dd0f8e2a1e561099',
    GESTURE_SCRIPT: '590b85ee7f1675e82ca7cf02fa72464b6c3c335b068467ff8d35484fd91ecf1d',
    PLOT_SCRIPT: '0da286e14f434d8b66a3fa7cb297b4175a3b77f899ef1c7642409936cd755289',
    HAND_MODEL: '2eb8029a8e1cedc2cf5b40c14c0955e2004277024df3c6304ce0bb74106a7c31',
    'scripts/eval/analyze_per_gesture.py': '448e83f7a3698334d1813fda8ce0b7303672f9d95740442157ded4aa244d28a7',
    'egoemg/datasets/egoemg_memmap_dataset.py': '9f8999ccc6a2d964713ff85886781ecc4d41ddb18f2986e0fbe5c6cc11236060',
    'egoemg/metrics.py': '85aac28fc2cac11e4714bf6a18a52b2faeccc488d6135202aae8a8e974e95309',
    'egoemg/lightning.py': 'f32b97dab2bb94aff35d0513d964b15087a859caaf11f36030f79074edae1899',
    'scripts/ik/batch_ik_mesh.py': '794fdfcb82a692812de4ca7a15fd4e41c0c3f3717424dd917fb05e4f9508da2c',
}
LEGACY_NOTE = ("Derived from generated_mano_* by decomposing MANO relative joint rotations "
               "into scalar projections onto UmeTrack's predefined anatomical rotation axes "
               "(axis-angle projection, inverse of UmeTrack FK).")
LEGACY_READER_SHA256 = 'de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e'
PAPER_DESCRIPTIONS = {
    **{i: 'American Sign Language digits 1 through 9' for i in range(9)},
    12: '“I Love You” hand sign (thumb, index, pinky extended)',
    25: '“Rock on” hand sign (index and pinky extended)',
    29: 'Simulated grasping motion without contact',
    51: 'Fingertips of both hands touching (Thumb and Middle), then separating',
    52: 'Middle and Ring fingers of one hand oppose the other hand',
    53: 'Two hands crossing to make a “Bei” Chinese character',
    54: 'Two hands with thumbs forming a “Checky” sign',
    58: 'Two hands pinching and wringing',
}


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _snapshot(path, snapshots):
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError('required semantic source is missing or unreadable') from exc
    snapshots[path] = raw
    return raw


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key in semantic source')
        result[key] = value
    return result


def _json(raw, description):
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'invalid {description}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'invalid {description}: expected an object')
    return value


def _check_upstream(root):
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'],
                            capture_output=True, text=True)
    if revision.returncode or revision.stdout.strip() != UPSTREAM_COMMIT:
        raise ValueError('upstream must be the pinned official revision')
    if status.returncode or status.stdout.strip():
        raise ValueError('upstream must be clean, including untracked files')


def _verify_snapshots(snapshots, upstream):
    for path, raw in snapshots.items():
        try:
            unchanged = path.read_bytes() == raw
        except OSError:
            unchanged = False
        if not unchanged:
            raise ValueError('semantic source changed during inspection')
    _check_upstream(upstream)


def _constant(node):
    """Read only literals and the one official list(range(0, 30)) expression."""
    if isinstance(node, ast.Constant) and type(node.value) in (str, int):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_constant(item) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            key = _constant(key)
            if type(key) not in (int, str) or key in result:
                raise ValueError('duplicate or invalid AST constant key')
            result[key] = _constant(value)
        return result
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'list'
            and len(node.args) == 1 and not node.keywords):
        inner = node.args[0]
        if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == 'range'
                and len(inner.args) == 2 and not inner.keywords
                and all(isinstance(a, ast.Constant) and type(a.value) is int for a in inner.args)
                and [a.value for a in inner.args] == [0, 30]):
            return list(range(30))
    raise ValueError('unsupported AST constant expression')


def _assignment(source, name):
    try:
        tree = ast.parse(source)
    except (SyntaxError, UnicodeError) as exc:
        raise ValueError('invalid official constant AST') from exc
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if len(node.targets) != 1:
                raise ValueError('ambiguous AST constant assignment')
            nodes.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            nodes.append(node.value)
    if len(nodes) != 1:
        raise ValueError('official constant assignment must occur exactly once')
    return nodes[0]


def _gesture_constants(source):
    names = _constant(_assignment(source, 'GESTURE_NAMES'))
    families = _constant(_assignment(source, 'FAMILIES'))
    if (not isinstance(names, dict) or set(names) != set(range(60))
            or any(type(k) is not int for k in names)
            or any(not isinstance(v, (tuple, list)) or len(v) != 2
                   or any(not isinstance(s, str) or not s for s in v) for v in names.values())):
        raise ValueError('gesture constants must define exactly 60 ID/name/description records')
    if not isinstance(families, dict) or set(families) != set(FAMILY_COUNTS):
        raise ValueError('gesture family constants are missing or unknown')
    members = []
    for family, expected in FAMILY_COUNTS.items():
        ids = families[family]
        if not isinstance(ids, list) or len(ids) != expected or any(type(i) is not int for i in ids):
            raise ValueError('invalid gesture family membership')
        members.extend(ids)
    if len(set(members)) != 60 or set(members) != set(names):
        raise ValueError('duplicate or missing gesture family member')
    return names, families


def _joint_constants(source):
    if _constant(_assignment(source, 'NUM_JOINTS')) != 20:
        raise ValueError('official finger joint count must be 20')
    node = _assignment(source, 'JOINTS')
    if not isinstance(node, ast.List):
        raise ValueError('invalid joint constant AST')
    rows = []
    for entry in node.elts:
        if (not isinstance(entry, ast.Call) or not isinstance(entry.func, ast.Name)
                or entry.func.id != 'Joint' or len(entry.args) != 3 or entry.keywords):
            raise ValueError('unsupported joint constant AST')
        name, index, groups = [_constant(a) for a in entry.args]
        if (not isinstance(name, str) or type(index) is not int or not isinstance(groups, list)
                or len(groups) != 2 or groups[0] not in {'thumb', 'index', 'middle', 'ring', 'pinky'}
                or groups[1] not in {'proximal', 'mid', 'distal'}):
            raise ValueError('invalid joint semantics')
        rows.append(dict(name=name, index=index, groups=groups))
    if [r['index'] for r in rows] != list(range(20)) or len({r['name'] for r in rows}) != 20:
        raise ValueError('joint semantics must give 20 ordered unique joints')
    return rows


def _metadata_context(root, names, snapshots):
    root = Path(root)
    manifest = _json(_snapshot(root / 'manifest.json', snapshots), 'manifest')
    summary = _json(_snapshot(root / 'metadata_summary.json', snapshots), 'metadata summary')
    raw = _snapshot(root / 'metadata.npz', snapshots)
    n = manifest.get('total_rows')
    semantics = [name.lower() for name in names]
    if (manifest.get('format_version') != 'egoemg_v2_memmap' or type(n) is not int or n <= 0
            or type(manifest.get('num_episodes')) is not int or manifest['num_episodes'] <= 0
            or manifest.get('generated_joint_angles_semantics') != semantics
            or manifest.get('left_hand_strategy') != 'flip_local_z'
            or manifest.get('frame_split_labels') != SPLITS):
        raise ValueError('manifest metadata semantics or schema differ from the inspected legacy release')
    if (summary.get('format_version') != manifest['format_version'] or summary.get('total_rows') != n
            or summary.get('generated_joint_angles_semantics') != semantics
            or summary.get('left_hand_strategy') != 'flip_local_z'
            or summary.get('generated_joint_angles_note') != LEGACY_NOTE):
        raise ValueError('metadata summary semantics differ from the inspected legacy release')
    fields = manifest.get('fields')
    expected = {'label_gesture_class': ('<i4', [n]), 'label_gesture_active': ('|b1', [n]),
                'frame_split_id': ('|i1', [n]), 'generated_label_valid': ('|b1', [n, 2])}
    for hand in ('left', 'right'):
        expected[f'generated_joint_angles_{hand}'] = ('<f4', [n, 20])
        for axis in ('pitch', 'yaw'):
            expected[f'mocap_{hand}_wrist_{axis}'] = ('<f4', [n])
        expected[f'mocap_{hand}_wrist_angles_valid'] = ('|b1', [n])
    if not isinstance(fields, dict):
        raise ValueError('invalid manifest field schema')
    for name, (dtype, shape) in expected.items():
        spec = fields.get(name)
        try:
            valid = (isinstance(spec, dict) and spec.get('filename') == name + '.dat'
                     and spec.get('shape') == shape and isinstance(spec.get('shape'), list)
                     and all(type(v) is int for v in spec['shape'])
                     and np.dtype(spec.get('dtype', '')) == np.dtype(dtype))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError('invalid manifest semantic field schema')
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as md:
            start, end, length = [md[k] for k in ('episode_start_idx', 'episode_end_idx', 'episode_length')]
            splits, ids = md['splits_split'], md['splits_split_id']
            valid = (all(v.ndim == 1 and v.dtype.kind in 'iu' for v in (start, end, length, ids))
                     and len(start) == len(end) == len(length) == manifest['num_episodes']
                     and int(start[0]) == 0 and int(end[-1]) == n and np.all(end > start)
                     and np.array_equal(end - start, length) and np.array_equal(start[1:], end[:-1])
                     and splits.ndim == 1 and splits.dtype.kind in 'SU'
                     and [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in splits] == SPLITS
                     and np.array_equal(ids, np.arange(4)))
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, IndexError, TypeError) as exc:
        raise ValueError('invalid numeric metadata boundaries or split schema') from exc
    if not valid:
        raise ValueError('invalid numeric metadata boundaries or split schema')
    return n


def _paper_reference(table, page):
    return dict(filename='EgoEMG.pdf', sha256=PAPER_SHA256, table=table, page=page,
                kind='fixed manually checked transcription; not a runtime PDF read')


def _compile(upstream, metadata_root):
    upstream = Path(upstream).resolve()
    _check_upstream(upstream)
    snapshots, source = {}, {}
    for filename, expected in OFFICIAL_HASHES.items():
        raw = _snapshot(upstream / filename, snapshots)
        if _digest(raw) != expected:
            raise ValueError('official source content differs from the approved semantic sources')
        source[filename] = raw
    joints = _joint_constants(source[CONSTANTS])
    labels = _constant(_assignment(source[PLOT_SCRIPT], 'JOINT_LABELS_SIMPLE'))
    if labels != [r['name'] for r in joints] + ['WRIST_FE', 'WRIST_RU']:
        raise ValueError('official 22-angle semantics disagree')
    n = _metadata_context(metadata_root, labels[:20], snapshots)
    reader = Path(__file__).with_name('data.py')
    if _digest(_snapshot(reader, snapshots)) != LEGACY_READER_SHA256:
        raise ValueError('legacy angle reader differs from the inspected mask protocol')
    _snapshot(Path(__file__), snapshots)
    sources = dict(
        official=dict(revision=UPSTREAM_COMMIT, files=[dict(filename=k, sha256=_digest(v)) for k, v in source.items()],
                      policy='approved revision, clean source tree and pinned source contents'),
        reproduction=dict(filename='egoemg_repro/semantic_tables.py', sha256=_digest(snapshots[Path(__file__)])),
        legacy_reader=dict(filename='egoemg_repro/data.py', sha256=LEGACY_READER_SHA256))
    for key, filename in [('manifest', 'manifest.json'), ('metadata', 'metadata.npz'),
                          ('metadata_summary', 'metadata_summary.json')]:
        sources[key] = dict(filename=filename, sha256=_digest(snapshots[Path(metadata_root) / filename]))
    names, families = _gesture_constants(source[GESTURE_SCRIPT])
    membership = {gid: family for family, ids in families.items() for gid in ids}
    gestures = []
    for gid, (name, description) in sorted(names.items()):
        paper_description = PAPER_DESCRIPTIONS.get(gid, description)
        gestures.append(dict(id=gid, name=name, description_official=description, family=membership[gid],
                             paper_reference=dict(row_name='ASL1-ASL9' if gid < 9 else name,
                                                  description=paper_description),
                             description_matches_paper=description == paper_description,
                             analysis59_excluded=gid == 50))
    gesture_doc = dict(schema_version=1, scope='definitions only; actual frame coverage is separate',
                       definition_count=60, family_definition_counts=dict(FAMILY_COUNTS),
                       observed_class_count=None, coverage_status='not_computed',
                       coverage_artifact='coverage.json', rows=gestures, sources=sources,
                       paper_reference=_paper_reference(4, 15),
                       notes=['IDs come from pinned code; Table 4 prints names without numeric IDs.',
                              'raw (ID 50) remains in the complete vocabulary and symmetric family.',
                              'The later 59-class error analysis excludes raw explicitly; these definitions are not errors.',
                              'Official descriptions are retained; fixed PDF wording differences are recorded separately.'])
    hand = _json(source[HAND_MODEL], 'official hand model')
    try:
        limits = np.asarray(hand['joint_limits'], dtype=np.float64)
        valid_limits = limits.shape == (22, 2) and np.isfinite(limits).all() and np.all(limits[:, 0] <= limits[:, 1])
    except (KeyError, TypeError, ValueError):
        valid_limits = False
    if not valid_limits:
        raise ValueError('invalid official nominal joint limits')
    rows = []
    for index, name in enumerate(labels):
        finger = index < 20
        parts = name.split('_')
        rows.append(dict(index=index, name=name, finger=parts[0].lower() if finger else None,
                         joint=parts[1] if finger else 'wrist', motion=parts[-1],
                         groups=joints[index]['groups'] if finger else ['wrist'],
                         disk_field='generated_joint_angles_{hand}' if finger else
                         'mocap_{hand}_wrist_' + ('pitch' if index == 20 else 'yaw'),
                         disk_component=index if finger else None, disk_unit='rad' if finger else 'deg',
                         model_unit='rad', report_unit='deg', hand_index={'left': 0, 'right': 1},
                         mask_field='generated_label_valid' if finger else 'mocap_{hand}_wrist_angles_valid',
                         mask_policy='legacy_reproduction_angle_mask: raw hand/group validity AND finite angle before cleanup',
                         nominal_limit_rad=limits[index].tolist() if finger else None,
                         limit_scope=('model constraint, not a measured legacy range' if finger else
                                      'unknown for independent mocap wrist, not a measured range; generic model limits are not assigned'),
                         source_reference=CONSTANTS if finger else PLOT_SCRIPT))
    angle_doc = dict(schema_version=1, definition_count=22, finger_dimensions=20, wrist_dimensions=2,
                     rows=rows, sources=sources, paper_reference=_paper_reference(6, 20),
                     left_hand_strategy='flip_local_z',
                     sign_convention=dict(wrist_positive='extension and radial deviation',
                                          left_wrist_normal='L cross F', source='PDF B.2 page 19',
                                          verification='paper definition; legacy marker computation equivalence is unverified',
                                          finger='canonical anatomical rotation axes; no blanket left-hand sign inversion'),
                     mask_protocols=dict(
                         legacy_reproduction_angle_mask=dict(
                             fingers='generated_label_valid[row, hand_column] AND isfinite(finger_angle)',
                             wrist='mocap_{hand}_wrist_angles_valid[row] AND isfinite(wrist_angle)',
                             hand_columns={'left': 0, 'right': 1}, finite_before_cleanup=True,
                             source='egoemg_repro/data.py'),
                         official_literal_loader=dict(
                             ordinary_path='generated hand validity is surfaced; wrist validity is not surfaced to loss',
                             extra_wrist_mask='Lightning masks wrist for named datasets without wrist labels',
                             wrist_metric='a per-joint mask uses channel 20 for both wrist metrics',
                             equivalent_to_legacy_angle_mask=False,
                             source='egoemg/datasets/egoemg_memmap_dataset.py and egoemg/lightning.py')),
                     generation=dict(legacy_method='axis_angle_projection', paper_method='landmark_level_ik',
                                     equivalence_verified=False),
                     notes=['Finger angles are radians on disk; wrist pitch/yaw are degrees on disk.',
                            'The model uses radians and reports use degrees; reporting units do not change disk units.',
                            'Nominal limits are model constraints, not measured legacy ranges or proof they were enforced.',
                            'Independent mocap wrist limits are unknown; generic model final limits are not assigned.',
                            'Matching dimension names does not establish label-generation or fitting-error equivalence.'])
    result = dict(schema_version=1, total_rows=n, gesture_vocabulary=gesture_doc, joint_angle_semantics=angle_doc)
    _verify_snapshots(snapshots, upstream)
    return result, snapshots


def build_semantic_tables(upstream: Path, metadata_root: Path) -> dict:
    """Compile only approved definitions and explicitly selected public fields."""
    return _compile(upstream, metadata_root)[0]


def _source_markdown(doc):
    sources = doc['sources']
    lines = ['', f"Official revision: `{sources['official']['revision']}`.",
             f"PDF reference: fixed transcription, SHA256 `{doc['paper_reference']['sha256']}`.", '', 'Content identities:', '']
    for key in ('manifest', 'metadata', 'metadata_summary', 'legacy_reader', 'reproduction'):
        item = sources[key]
        lines.append(f"- {item['filename']}: `{item['sha256']}`")
    for item in sources['official']['files']:
        lines.append(f"- {item['filename']}: `{item['sha256']}`")
    return lines


def _gesture_markdown(doc):
    lines = ['# Gesture vocabulary', '',
             '60 definitions: 30 single-hand, 18 symmetric bimanual, 12 asymmetric bimanual.',
             'Observed coverage is not computed by this definition export; see [frame coverage](coverage.md).', '',
             '| ID | Name | Family | Official description |', '| ---: | --- | --- | --- |']
    for row in doc['rows']:
        lines.append(f"| {row['id']} | {row['name']} | {row['family']} | {row['description_official']} |")
    lines.extend(['', *doc['notes'], '', 'Fixed PDF wording differences:', ''])
    for row in doc['rows']:
        if not row['description_matches_paper']:
            lines.append(f"- ID {row['id']} ({row['paper_reference']['row_name']}): {row['paper_reference']['description']}")
    return '\n'.join(lines + _source_markdown(doc)) + '\n'


def _angle_markdown(doc):
    lines = ['# Joint angle semantics', '',
             '22 dimensions: 20 finger angles and 2 independent mocap wrist angles.',
             'Finger angles are radians on disk; wrist pitch/yaw are degrees on disk. The model uses radians; reports use degrees.',
             'Nominal model limits are not measured data ranges. Unknown mocap wrist limits remain unspecified.', '',
             '| Index | Name | Disk field | Component | Disk unit | Raw validity | Nominal limit (rad) |',
             '| ---: | --- | --- | ---: | --- | --- | --- |']
    for row in doc['rows']:
        limits = row['nominal_limit_rad']
        bound = f'[{limits[0]:.6f}, {limits[1]:.6f}]' if limits is not None else 'unknown'
        component = row['disk_component'] if row['disk_component'] is not None else 'scalar'
        lines.append(f"| {row['index']} | {row['name']} | {row['disk_field']} | {component} | {row['disk_unit']} | {row['mask_field']} | {bound} |")
    lines.extend(['', 'Hand columns: left=0, right=1. Both hands use the same semantic index order.',
                  'Left-hand strategy in the release metadata: flip_local_z; no blanket angle sign inversion is applied.',
                  'PDF wrist signs: extension/radial deviation positive, left normal L cross F; legacy implementation equivalence is unverified.', '',
                  '- legacy_reproduction_angle_mask: first 20 use generated hand validity, last 2 use independent wrist validity; each angle must be finite before cleanup.',
                  '- official_literal_loader: generated hand validity is surfaced, but ordinary-path wrist validity is not passed to loss; dataset-name wrist masking is separate.',
                  '- The official wrist metric uses channel 20 of a per-joint mask for both wrist components.', '',
                  'Generation discrepancy: legacy metadata describes axis-angle projection; PDF B.3 describes landmark-level IK. Matching names do not prove equivalence.',
                  '', *doc['notes']])
    return '\n'.join(lines + _source_markdown(doc)) + '\n'


def export_semantic_tables(upstream: Path, metadata_root: Path, output: Path) -> dict:
    result, snapshots = _compile(upstream, metadata_root)
    documents = {'gesture-vocabulary': result['gesture_vocabulary'], 'joint-angle-semantics': result['joint_angle_semantics']}
    payloads = {}
    for name, doc in documents.items():
        payloads[name + '.json'] = json.dumps(doc, indent=2, allow_nan=False) + '\n'
    payloads['gesture-vocabulary.md'] = _gesture_markdown(documents['gesture-vocabulary'])
    payloads['joint-angle-semantics.md'] = _angle_markdown(documents['joint-angle-semantics'])
    _verify_snapshots(snapshots, Path(upstream))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, text in payloads.items():
        (output / name).write_text(text, encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--metadata-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = export_semantic_tables(args.upstream, args.metadata_root, args.output)
    except ValueError as exc:
        parser.exit(2, f'Semantic definition export failed: {exc}\n')
    print(json.dumps(dict(artifacts=4, gesture_definitions=result['gesture_vocabulary']['definition_count'],
                          angle_definitions=result['joint_angle_semantics']['definition_count']), sort_keys=True))


if __name__ == '__main__':
    main()
