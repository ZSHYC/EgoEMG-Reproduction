"""Aggregate MANO shape statistics from the pinned legacy release.

Only distinct metadata-referenced beta rows contribute, once per hand. Public
artifacts contain aggregates and content identities, never episode labels or
the underlying hand parameter table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

import numpy as np

from . import UPSTREAM_COMMIT


OFFICIAL_SCRIPT = 'scripts/paper/plot_beta_diversity.py'
OFFICIAL_SCRIPT_SHA256 = 'c7c6179f06185954096ec32e97ce93f72dcaffa8d8276dfa78bca9b8c4eddc72'
# Fixed transcription of Table 5, page 16 of the supplied paper. Its last
# column is printed as P95-P5, but matches the official max-minus-min values.
PAPER_SHA256 = '124318dce75d9af00ab4d8427acf01315cf83ae1992082b783d5a488c2474725'
TABLE5 = [
    (1.087, 1.051, -0.966, 3.034, 4.000),
    (1.505, 0.793, 0.183, 3.506, 3.323),
    (-0.981, 1.076, -3.804, 2.547, 6.351),
    (-0.537, 1.069, -3.077, 2.050, 5.128),
    (1.054, 0.996, -0.461, 4.638, 5.099),
    (-2.269, 1.711, -10.253, 0.450, 10.703),
    (0.727, 1.437, -1.735, 8.254, 9.989),
    (0.371, 1.000, -1.802, 4.683, 6.485),
    (0.365, 1.464, -2.978, 5.344, 8.322),
    (-0.738, 0.553, -3.308, 0.366, 3.674),
]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required(path, description):
    if not path.is_file():
        raise ValueError(f'required {description} is missing')
    return path


def _json(path, description):
    _required(path, description)
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f'invalid {description}') from exc
    if not isinstance(result, dict):
        raise ValueError(f'invalid {description}: expected an object')
    return result


def _official_source(root):
    root = Path(root).resolve()
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
    if revision.returncode or revision.stdout.strip() != UPSTREAM_COMMIT:
        raise ValueError(f'upstream revision must be the approved {UPSTREAM_COMMIT}')
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain',
                             '--untracked-files=all'], capture_output=True, text=True)
    if status.returncode or status.stdout.strip():
        raise ValueError('upstream source must be clean, including untracked files')
    script = _required(root / OFFICIAL_SCRIPT, 'upstream statistical script')
    digest = _sha256(script)
    if digest != OFFICIAL_SCRIPT_SHA256:
        raise ValueError('upstream statistical script differs from the approved source')
    return dict(filename=OFFICIAL_SCRIPT, revision=UPSTREAM_COMMIT, sha256=digest,
                source_policy='approved revision, clean source tree and pinned script content')


def _beta_field(manifest, ledger, root, hand):
    name = f'generated_mano_{hand}_beta'
    fields = manifest.get('episode_fields')
    if not isinstance(fields, dict):
        raise ValueError('invalid beta field mapping in manifest')
    info = fields.get(name)
    if not isinstance(info, dict) or info.get('filename') != name + '.dat':
        raise ValueError(f'invalid {hand} beta field filename')
    shape = info.get('shape')
    if (not isinstance(shape, list) or len(shape) != 2 or
            any(type(n) is not int or n <= 0 for n in shape) or shape[1] != 10):
        raise ValueError(f'invalid {hand} beta field shape: expected positive rows by 10')
    try:
        dtype = np.dtype(info.get('dtype', ''))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid {hand} beta field dtype') from exc
    if dtype != np.dtype('<f4'):
        raise ValueError(f'invalid {hand} beta field dtype: expected float32')
    path = _required(root / (name + '.dat'), f'{hand} beta field')
    size = path.stat().st_size
    if size != shape[0] * shape[1] * dtype.itemsize:
        raise ValueError(f'{hand} beta byte size does not match the manifest')
    digest = _sha256(path)
    receipt = ledger.get('files', {}).get(path.name)
    if (not isinstance(receipt, dict) or type(receipt.get('bytes')) is not int or
            receipt['bytes'] != size or receipt.get('sha256') != digest or
            receipt.get('checksum_basis') not in {'local_transfer', 'pinned_metadata'}):
        raise ValueError(f'{hand} beta acquisition receipt does not match the payload')
    values = np.fromfile(path, dtype=dtype).reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError(f'{hand} beta values must all be finite, including unreferenced rows')
    identity = dict(filename=path.name, dtype='float32', shape=shape, bytes=size,
                    sha256=digest, checksum_basis=receipt['checksum_basis'])
    return values, identity


def _referenced_rows(path, row_count):
    _required(path, 'metadata')
    try:
        with np.load(path, allow_pickle=False) as metadata:
            if not {'episode_beta_idx', 'episode_id'}.issubset(metadata.files):
                raise ValueError('metadata requires episode_beta_idx and episode_id')
            indices = metadata['episode_beta_idx']
            if (indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in 'iu' or
                    np.any(indices < 0) or np.any(indices >= row_count)):
                raise ValueError('metadata episode_beta_idx must be nonempty 1D integer indices within beta rows')
            ids = metadata['episode_id']
            if ids.ndim != 1 or ids.size != indices.size:
                raise ValueError('metadata episode count differs from beta references')
    except (OSError, UnicodeError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f'invalid metadata episode_beta_idx or episode_id: {exc}') from exc
    return np.unique(indices), int(indices.size)


def _paper_comparison(dimensions):
    differences = []
    metrics = ('mean', 'std', 'min', 'max', 'range')
    for actual, reference in zip(dimensions, TABLE5):
        for metric, expected in zip(metrics, reference):
            printed, rounded = f'{expected:.3f}', f'{actual[metric]:.3f}'
            if rounded != printed:
                differences.append(dict(dimension=actual['dimension'], metric=metric,
                                        pdf_printed=printed, computed=actual[metric],
                                        computed_rounded=rounded))
    total = len(TABLE5) * len(metrics)
    return dict(filename='EgoEMG.pdf', sha256=PAPER_SHA256, table=5, page=16,
                reference_kind='fixed transcription of supplied PDF, not a runtime PDF read',
                decimal_places=3, compared_entries=total, matching_entries=total - len(differences),
                last_column_interpretation='official range (max - min), despite the PDF P95-P5 label',
                differences=differences)


def _analyze(metadata_root, beta_root, upstream):
    official = _official_source(upstream)
    metadata_root, beta_root = Path(metadata_root), Path(beta_root)
    manifest_path, metadata_path = metadata_root / 'manifest.json', metadata_root / 'metadata.npz'
    manifest = _json(manifest_path, 'manifest')
    if manifest.get('format_version') != 'egoemg_v2_memmap':
        raise ValueError('manifest must describe the inspected egoemg_v2_memmap release')
    ledger_path = beta_root / 'acquisition_ledger.json'
    ledger = _json(ledger_path, 'acquisition ledger')
    if ledger.get('ledger_version') != 1 or not isinstance(ledger.get('files'), dict):
        raise ValueError('unsupported acquisition ledger format')
    left, left_identity = _beta_field(manifest, ledger, beta_root, 'left')
    right, right_identity = _beta_field(manifest, ledger, beta_root, 'right')
    if left.shape != right.shape:
        raise ValueError('left and right beta field shapes must match')
    rows, episodes = _referenced_rows(metadata_path, left.shape[0])
    combined = np.concatenate([left[rows], right[rows]], axis=0)
    dimensions = []
    for dim in range(10):
        values = combined[:, dim]
        p05, p95 = np.percentile(values, [5, 95], method='linear')
        # Preserve the official float32 mean/std/min/max/range computation.
        with np.errstate(over='ignore', invalid='ignore'):
            metrics = dict(mean=float(values.mean()), std=float(values.std(ddof=0)),
                           min=float(values.min()), max=float(values.max()),
                           range=float(np.ptp(values)), p05=float(p05), p95=float(p95),
                           p95_minus_p05=float(p95 - p05))
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError('beta aggregate statistics must be finite')
        dimensions.append(dict(dimension=f'beta{dim + 1}', **metrics))
    count = int(rows.size)
    result = dict(
        schema_version=1,
        counts=dict(episodes=episodes, available_beta_rows=int(left.shape[0]),
                    referenced_beta_rows=count, unreferenced_beta_rows=int(left.shape[0]) - count,
                    left_hands=count, right_hands=count, combined_hands=2 * count, dimensions=10),
        method=dict(ddof=0, percentile_method='linear', computation_dtype='float32',
                    percentile_dtype='float64', numpy_version=np.__version__,
                    sampling='distinct metadata-referenced beta rows, once per left and right hand',
                    range='max - min', p95_minus_p05='linear P95 - linear P5'),
        sources=dict(
            manifest=dict(filename='manifest.json', sha256=_sha256(manifest_path)),
            metadata=dict(filename='metadata.npz', sha256=_sha256(metadata_path)),
            beta=dict(left=left_identity, right=right_identity),
            acquisition_ledger=dict(filename='acquisition_ledger.json', sha256=_sha256(ledger_path),
                                    scope='payload bytes rehashed and matched to local receipts; publisher authenticity is unverified'),
            official_script=official,
            reproduction=dict(filename='egoemg_repro/dataset_statistics.py', sha256=_sha256(Path(__file__)))),
        dimensions=dimensions,
        paper_comparison=_paper_comparison(dimensions),
        notes=[
            'Counts describe episodes and distinct referenced beta rows, not verified independent people.',
            'Official code computes range (max - min); the paper PDF labels the corresponding column P95-P5. These are different statistics.',
            'Actual P5, P95 and P95-P5 are separately reported using linear percentiles; values are not adjusted to match the PDF.',
            'The requested aggregate boxplot replaces the official per-episode bar chart. Individual points and outlier markers are hidden.',
            'beta1 through beta10 are neutral MANO coefficient labels; no fixed anatomical meaning is assigned.'
        ])
    return result, combined


def compute_shape_statistics(metadata_root, beta_root, *, upstream):
    """Validate source identities and return only aggregate, public-safe values."""
    return _analyze(metadata_root, beta_root, upstream)[0]


def _markdown(result):
    count = result['counts']
    lines = ['# Legacy MANO shape statistics', '',
             f"{count['episodes']} metadata episodes reference {count['referenced_beta_rows']} distinct beta rows; "
             f"{count['left_hands']} left + {count['right_hands']} right = {count['combined_hands']} hands.", '',
             'Population standard deviation uses ddof=0. Percentiles use the linear method.', '',
             '| Dimension | Mean | Std | Min | Max | range (max - min) | P5 | P95 | P95-P5 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    keys = ('mean', 'std', 'min', 'max', 'range', 'p05', 'p95', 'p95_minus_p05')
    for entry in result['dimensions']:
        lines.append('| ' + entry['dimension'] + ' | ' + ' | '.join(f'{entry[key]:.6f}' for key in keys) + ' |')
    comparison = result['paper_comparison']
    lines.extend(['', *result['notes'], '',
                  f"PDF Table 5 (page 16): {comparison['matching_entries']}/{comparison['compared_entries']} "
                  'entries match at three decimal places when the final column is interpreted as official range.', ''])
    for item in comparison['differences']:
        lines.append(f"- {item['dimension']} {item['metric']}: computed {item['computed']:.9f} "
                     f"(three decimals: {item['computed_rounded']}); PDF: {item['pdf_printed']}.")
    if comparison['differences']:
        lines.extend(['', 'Differences are retained; their cause is not established.'])
    lines.extend(['', f"Comparison reference: supplied PDF SHA256 `{comparison['sha256']}`; fixed Table 5 transcription.",
                  '', 'Content identities (SHA256):', ''])
    for source in ('manifest', 'metadata', 'acquisition_ledger', 'official_script', 'reproduction'):
        item = result['sources'][source]
        lines.append(f"- {item['filename']}: `{item['sha256']}`")
    for item in result['sources']['beta'].values():
        lines.append(f"- {item['filename']} ({item['shape'][0]} x {item['shape'][1]}, {item['dtype']}): `{item['sha256']}`")
    lines.extend(['', 'Acquisition hashes verify local transfer integrity; publisher authenticity is unverified.',
                  f'Official source revision: `{UPSTREAM_COMMIT}`.', '',
                  '![Aggregate shape distributions](shape-distribution.png)', ''])
    return '\n'.join(lines)


def _plot(combined, result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context({'svg.fonttype': 'none', 'font.size': 10}):
        fig, ax = plt.subplots(figsize=(12, 5.6))
        ax.boxplot(combined, tick_labels=[f'beta{i}' for i in range(1, 11)], showfliers=False,
                   whis=1.5, patch_artist=True,
                   boxprops=dict(facecolor='#9cc5dd', edgecolor='#25475e'),
                   medianprops=dict(color='#a53c27', linewidth=1.7),
                   whiskerprops=dict(color='#25475e'), capprops=dict(color='#25475e'))
        ax.axhline(0, color='#81909a', linewidth=0.7, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e0e6ea', linewidth=0.7)
        ax.set_ylabel('MANO shape coefficient')
        ax.set_xlabel('Neutral coefficient labels; no fixed anatomical meaning assigned')
        count = result['counts']
        ax.set_title(f"Legacy MANO shape distribution ({count['combined_hands']} hands)\n"
                     f"{count['referenced_beta_rows']} distinct referenced beta rows; left and right each counted once",
                     fontsize=12, pad=14)
        fig.text(0.08, 0.055, 'Boxes: Q1-Q3; line: median; whiskers: up to 1.5 IQR; individual points and outliers are hidden.\n'
                 'Aggregate boxplot replaces the official per-episode bar chart; counts do not verify independent people.',
                 fontsize=9, color='#435363')
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.83)
        try:
            fig.savefig(output / 'shape-distribution.png', dpi=180,
                        metadata={'Software': 'EgoEMG aggregate shape statistics'})
            fig.savefig(output / 'shape-distribution.svg', metadata={'Date': None,
                        'Creator': 'EgoEMG aggregate shape statistics'})
        finally:
            plt.close(fig)


def export_shape_statistics(metadata_root, beta_root, output, *, upstream):
    """Write JSON, Markdown, PNG and SVG only after every input passes validation."""
    result, combined = _analyze(metadata_root, beta_root, upstream)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _plot(combined, result, output)
    (output / 'shape-statistics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (output / 'shape-statistics.md').write_text(_markdown(result), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-root', type=Path, required=True)
    parser.add_argument('--beta-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=Path(os.environ.get('EGOEMG_UPSTREAM', 'upstream')))
    args = parser.parse_args(argv)
    try:
        result = export_shape_statistics(args.metadata_root, args.beta_root, args.output, upstream=args.upstream)
    except ValueError as exc:
        parser.exit(2, f'Shape statistics failed: {exc}\n')
    print(json.dumps(dict(artifacts=4, counts=result['counts']), sort_keys=True))


if __name__ == '__main__':
    main()
