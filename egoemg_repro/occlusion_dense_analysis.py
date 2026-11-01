"""Pinned author CSV/statistics, without importing or constructing pose models.

This pure analysis unit grants no prediction, dataset or Figure 13 admission.
The author calls the trend LOESS; its actual implementation is a weighted cubic
UnivariateSpline. Execute that implementation, including its filtering/edge rules.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import math
from numbers import Integral, Real
from pathlib import Path
import subprocess
from types import SimpleNamespace

from . import UPSTREAM_COMMIT
from .classic_emg import _approved_upstream, _hash
from .classic_emg_cohort import _safe, _StatGuard

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('scripts/eval/evaluate_occlusion_fusion.py',
                'scripts/paper/plot_occlusion_fusion.py')


def _sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


IMPORTED_SHA = _sha(__file__)


def _sources(upstream):
    root = _approved_upstream(upstream)
    paths = {n: _safe(root / n) for n in SOURCE_FILES}
    guard = _StatGuard([*paths.values(), _safe(__file__)])
    if _sha(__file__) != IMPORTED_SHA:
        raise ValueError('imported dense analysis source changed')
    raw = {n: p.read_bytes() for n, p in paths.items()}
    for name, value in raw.items():
        original = subprocess.run(['git', '-C', str(root), 'show',
            UPSTREAM_COMMIT + ':' + name], capture_output=True, timeout=30)
        if original.returncode or original.stdout != value:
            raise ValueError('pinned author analysis source differs: ' + name)
    guard.check()
    return paths, raw, guard


def source_analysis_contract(upstream):
    """Source-only metadata; no Torch, SciPy, Pandas or plotting import."""
    _, raw, guard = _sources(upstream)
    result = dict(format='pinned_author_dense_analysis_contract_v1',
        upstream_revision=UPSTREAM_COMMIT,
        source_files_sha256={n: hashlib.sha256(v).hexdigest() for n, v in raw.items()},
        adapter_source_sha256=IMPORTED_SHA,
        degree_factor=57.3, error_round_decimals=4, geometry_round_decimals=6,
        source_error_uses_mask=False, source_error_reduction='per-sample mean of all 22 joint absolute errors',
        filter='occlusion_score.notna() & (mae_vision_deg > 0.01)',
        bin_width=.05, bin_interval='[lo,hi)', min_samples_per_bin=100, minimum_final_bin=10,
        sem_ddof=1, spline_degree=3, spline_weights='bin counts',
        spline_smoothing='number of bins * 0.4', spline_points=200,
        trend_implementation='scipy.interpolate.UnivariateSpline', source_trend_label='LOESS trend',
        spearman_population='all filtered per-sample occlusion and rounded delta, including unbinned rows',
        scatter_seed=42, maximum_scatter_points=4000, scatter_sampling_does_not_limit_inference=True,
        scientific_eligible=False, figure13_reproduced=False, full_paper_complete=False)
    result['sha256'] = _hash(result)
    guard.check()
    return result


def _main(raw):
    tree = ast.parse(raw)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def _csv_expression(raw):
    nodes = [n.value.args[0] for n in ast.walk(_main(raw))
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'append'
        and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'rows'
        and len(n.value.args) == 1 and isinstance(n.value.args[0], ast.Dict)]
    if len(nodes) != 1:
        raise ValueError('unique pinned native CSV dictionary required')
    return compile(ast.Expression(body=nodes[0]), SOURCE_FILES[0], 'eval')


def _real(value, name, *, nan=False, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('numeric native value required: ' + name)
    value = float(value)
    if nan and math.isnan(value):
        return value
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError('finite nonnegative native value required: ' + name)
    return value


def source_csv_rows(records, *, upstream=None):
    """Keep every genuine raw record; never invent an absent branch prediction."""
    import numpy as np
    _, raw, guard = _sources(ROOT / 'upstream' if upstream is None else upstream)
    expression = _csv_expression(raw[SOURCE_FILES[0]])
    seen, rows = set(), []
    required = {'split', 'hand', 'ep_idx', 'center_idx', 'mocap_tracked', 'mocap_stale',
        'markers_valid', 'occlusion_score', 'visible_ratio', 'mae_vision_rad', 'mae_emg_rad', 'mae_fusion_rad'}
    for original in records:
        if not isinstance(original, dict) or not required.issubset(original):
            raise ValueError('complete actual native record and all three predictions required')
        r = dict(original)
        if r['split'] not in ('user', 'gesture', 'both') or r['hand'] not in ('left', 'right'):
            raise ValueError('explicit native split and hand required')
        for name in ('ep_idx', 'center_idx'):
            if isinstance(r[name], bool) or not isinstance(r[name], Integral) or r[name] < 0:
                raise ValueError('nonnegative integer native key required')
            r[name] = int(r[name])
        key = tuple(r[n] for n in ('split', 'hand', 'ep_idx', 'center_idx'))
        if key in seen:
            raise ValueError('duplicate native prediction key')
        seen.add(key)
        for name in ('mocap_tracked', 'mocap_stale', 'markers_valid'):
            if not isinstance(r[name], (bool, Integral, np.bool_)) or r[name] not in (0, 1):
                raise ValueError('actual native boolean flag required')
        for name in ('mae_vision_rad', 'mae_emg_rad', 'mae_fusion_rad'):
            r[name] = _real(r[name], name)
        for name in ('occlusion_score', 'visible_ratio'):
            # This rounding occurs in the native geometry loop before its CSV
            # dictionary; retain NaN rows exactly as the source does.
            r[name] = round(_real(r[name], name, nan=True), 6)
        rows.append(eval(expression, {'np': np}, {'r': r}))
    guard.check()
    return rows


def _assignment(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


class _ObserveMembership(ast.NodeTransformer):
    """Observe existing native bin memberships; do not change its computation."""
    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == 'append' and isinstance(call.func.value, ast.Name)
            and call.func.value.id == 'bin_counts'):
            return [node, ast.parse('_bin_members.append(merged.copy())').body[0]]
        return node


def _statistics_code(raw):
    body = _main(raw).body
    begin = next(i for i, n in enumerate(body) if _assignment(n, 'valid'))
    end = next(i for i, n in enumerate(body) if _assignment(n, 'y_smooth'))
    prefix = _ObserveMembership().visit(ast.Module(body=copy.deepcopy(body[begin:end + 1]), type_ignores=[])).body
    extra = [copy.deepcopy(n) for n in body if any(_assignment(n, k) for k in ('n_scatter', 'rng', 's_idx'))
        or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple)
            and [v.id for v in t.elts if isinstance(v, ast.Name)] == ['rho', 'pval'] for t in n.targets))]
    if len(extra) != 4:
        raise ValueError('pinned native scatter and Spearman statements required')
    module = ast.Module(body=ast.parse('_bin_members=[]').body + prefix + extra, type_ignores=[])
    return compile(ast.fix_missing_locations(module), SOURCE_FILES[1], 'exec')


def analyze_source_csv(csv_path, *, upstream):
    """Execute original statistics and expose exact filtered/bin/scatter keys.

    This accepts a CSV for analysis only. It does not certify its predictions,
    geometry, full inference population or experimental producer provenance.
    """
    paths, raw, guard = _sources(upstream)
    path = _safe(csv_path)
    guard.add(path)
    digest = _sha(path)
    contract = source_analysis_contract(upstream)
    import numpy as np
    import pandas as pd
    from scipy.interpolate import UnivariateSpline
    from scipy.stats import sem, spearmanr
    df = pd.read_csv(path)
    required = {'occlusion_score', 'mae_vision_deg', 'mae_fusion_deg', 'delta_fusion_deg'}
    if not required.issubset(df.columns):
        raise ValueError('native rounded CSV columns required')
    # Preserve native NaN geometry handling, but never silently produce a
    # numeric-looking result from corrupt/infinite branch errors.
    try:
        for name in required:
            values = df[name].to_numpy(dtype=float)
            if np.any(np.isinf(values)) or (name != 'occlusion_score' and np.any(~np.isfinite(values))):
                raise ValueError('nonfinite native CSV error values')
        if np.any(df['mae_vision_deg'].values < 0) or np.any(df['mae_fusion_deg'].values < 0):
            raise ValueError('native MAE cannot be negative')
    except (TypeError, ValueError) as exc:
        raise ValueError('malformed native analysis CSV') from exc
    ns = dict(np=np, df=df, args=SimpleNamespace(min_samples_per_bin=100),
        sem=sem, spearmanr=spearmanr, UnivariateSpline=UnivariateSpline)
    try:
        exec(_statistics_code(raw[SOURCE_FILES[1]]), ns)
    except Exception as exc:
        # SciPy 1.13.1's F2PY extension names its actual class dfitpack.error,
        # but does not export an `error` attribute. Normalize that observed
        # native insufficient-points failure, without swallowing other bugs.
        fitpack_error = type(exc).__module__ == 'dfitpack' and type(exc).__name__ == 'error'
        if not isinstance(exc, (ValueError, IndexError, ZeroDivisionError, RuntimeError)) and not fitpack_error:
            raise
        raise ValueError('native analysis cannot form its original cubic trend: ' + str(exc)) from exc
    filtered = ns['valid'].index.to_numpy(dtype=np.int64)
    bins = []
    for i, members in enumerate(ns['_bin_members']):
        bins.append(dict(count=int(ns['cnt'][i]), occlusion_mean=float(ns['bc'][i]),
            vision_mean_deg=float(ns['bv'][i]), vision_sem_deg=float(ns['bvs'][i]),
            fusion_mean_deg=float(ns['bf'][i]), fusion_sem_deg=float(ns['bfs'][i]),
            delta_mean_deg=float(ns['bin_delta_m'][i]), delta_sem_deg=float(ns['bin_delta_s'][i]),
            difference_of_rounded_column_means_deg=float(ns['bv'][i] - ns['bf'][i]),
            source_rows=filtered[members].tolist()))
    binned = {j for b in bins for j in b['source_rows']}
    rho, pvalue = float(ns['rho']), float(ns['pval'])
    result = dict(format='pinned_author_dense_csv_statistics_v1', source_contract=contract,
        input_csv_sha256=digest, input_rows=len(df), filtered_rows=len(filtered), binned_rows=len(binned),
        filtered_source_rows=filtered.tolist(),
        filtered_not_binned_source_rows=[int(j) for j in filtered if j not in binned], bins=bins,
        trend=dict(method=contract['trend_implementation'], source_label=contract['source_trend_label'],
            x=ns['x_smooth'].tolist(), y=ns['y_smooth'].tolist()),
        spearman=dict(rho=rho if math.isfinite(rho) else None,
            pvalue=pvalue if math.isfinite(pvalue) else None, sample_count=len(filtered),
            native_nonfinite=not(math.isfinite(rho) and math.isfinite(pvalue))),
        scatter_source_rows=filtered[ns['s_idx']].tolist(),
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas')},
        scientific_eligible=False, figure13_reproduced=False, full_paper_complete=False)
    guard.check()
    if _sha(path) != digest or any(_sha(paths[n]) != contract['source_files_sha256'][n] for n in paths):
        raise ValueError('native analysis inputs or source changed during execution')
    result['sha256'] = _hash(result)
    return result
