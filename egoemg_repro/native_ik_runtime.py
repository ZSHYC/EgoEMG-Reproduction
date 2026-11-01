"""Pinned original IK calls and an independent direct geometric readback.

Source admission is Torch-free. Geometry and original L-BFGS are available only
through registered objects; no learned pose model or original label is written.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import weakref

import numpy as np

from .classic_emg_cohort import _safe
from .data import file_sha256
from .native_ik_inputs import _Guard, _hash, SOURCE_SHA, UPSTREAM

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES_SHA = 'da59ca6c488e08c409ebd3bc202fd244aa365b0e7c6fb3c75a7984710d65d75d'
IMPORTED_SHA = file_sha256(__file__)
_SOURCES = weakref.WeakKeyDictionary()
_RUNTIMES = weakref.WeakKeyDictionary()


def _registered(obj, registry, cls):
    if type(obj) is not cls or obj not in registry:
        raise ValueError('registered original IK source/runtime object required')
    return registry[obj]


def _check_git(root):
    try:
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError('original pinned upstream source checkout required') from exc
    if head != UPSTREAM or dirty:
        raise ValueError('original pinned upstream source checkout differs')


def _provider(name, root):
    module = sys.modules.get(name)
    if module is not None:
        path = getattr(module, '__file__', None)
        if not path or not _safe(path).is_relative_to(root):
            raise ValueError('original IK dependency provider changed: ' + name)


@dataclass(frozen=True, eq=False)
class IKSource:
    @property
    def identity(self):
        self.check()
        return copy.deepcopy(_SOURCES[self]['identity'])

    def check(self):
        state = _registered(self, _SOURCES, IKSource)
        state['guard'].check()
        if _hash(state['identity']) != state['identity_hash']:
            raise ValueError('original IK source identity changed')

    def build(self, device='cpu'):
        self.check()
        if device not in ('cpu', 'cuda:0'):
            raise ValueError('explicit original IK device cpu or cuda:0 required')
        s = _SOURCES[self]
        _provider('manotorch', s['manotorch'])
        _provider('mano', s['manotorch'])
        _provider('chumpy', s['chumpy'])
        sys.path[:0] = [str(s[k]) for k in ('upstream', 'manotorch', 'chumpy')]
        import torch
        if np.__version__ != '1.26.4' or torch.__version__.split('+')[0] != '2.2.2':
            raise ValueError('admitted IK requires NumPy1.26.4 and Torch2.2.2')
        if torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
            raise ValueError('original IK FP32 default without autocast required')
        if device == 'cuda:0' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise ValueError('one explicitly admitted visible CUDA device required')
        if device == 'cpu' and torch.cuda.is_initialized():
            raise ValueError('CPU IK engineering must not initialize CUDA')
        restored = []
        for name, value in dict(bool=bool, int=int, float=float, complex=complex,
                                object=object, unicode=str, str=str).items():
            if name not in np.__dict__:
                setattr(np, name, value)
                restored.append('numpy.' + name)
        if not hasattr(inspect, 'getargspec'):
            inspect.getargspec = inspect.getfullargspec
            restored.append('inspect.getargspec')
        from manotorch.manolayer import ManoLayer
        from egoemg.kinematics import load_default_hand_model, apply_to_hand_model
        _provider('manotorch.manolayer', s['manotorch'])
        _provider('mano', s['manotorch'])
        _provider('chumpy', s['chumpy'])
        self.check()
        spec = importlib.util.spec_from_file_location('_pinned_egoemg_native_ik', s['upstream'] / 'scripts/ik/batch_ik_mesh.py')
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        self.check()
        mano = ManoLayer(rot_mode='axisang', side='right', mano_assets_root=str(s['assets']),
                         use_pca=False, flat_hand_mean=True).to(device)
        hand = apply_to_hand_model(load_default_hand_model(), lambda t: t.float().to(device))
        limits = np.asarray(json.loads((s['upstream'] / 'egoemg/UmeTrack/dataset/generic_hand_model.json').read_text())['joint_limits'][:20], np.float32)
        self.check()
        result = IKRuntime()
        names = ('optimize_batch', 'mano_fk_batch', 'umetrack_fk_batch', 'axisangle_to_rotmat_batch')
        _RUNTIMES[result] = dict(source=self, native=native, functions={n: getattr(native, n) for n in names},
            constants=dict(scale=native.ALIGN_SCALE, flip=native.FLIP_MATRIX.copy(), trans=native.ALIGN_TRANS.copy()),
            mano=mano, hand=hand, device=device, lo=torch.from_numpy(limits[:, 0].copy()).to(device),
            hi=torch.from_numpy(limits[:, 1].copy()).to(device),
            identity=dict(format='original_native_ik_geometry_runtime_v1', source_identity_sha256=_hash(s['identity']),
                device=device, precision='float32', numpy=np.__version__, torch=torch.__version__,
                torch_threads=torch.get_num_threads(), actual_mano_geometry_layers=1,
                compatibility_names_restored=restored, learned_pose_models_constructed=0,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                scientific_eligible=False, full_paper_complete=False))
        result.check()
        return result


def admit_ik_sources(upstream, manotorch, chumpy, mano_assets):
    upstream, manotorch, chumpy, assets = [_safe(p) for p in (upstream, manotorch, chumpy, mano_assets)]
    document = _safe(ROOT / 'assets/native_ik_dependency_sources.json')
    if file_sha256(document) != DEPENDENCIES_SHA or file_sha256(__file__) != IMPORTED_SHA:
        raise ValueError('original IK dependency/source manifest differs')
    manifest = json.loads(document.read_text())
    _check_git(upstream)
    providers = dict(manotorch=manotorch, chumpy=chumpy)
    paths = {'dependency-manifest': document, 'egoemg_repro/native_ik_runtime.py': _safe(__file__),
             'egoemg_repro/native_ik_inputs.py': _safe(ROOT / 'egoemg_repro/native_ik_inputs.py')}
    expected = {}
    for kind, root in providers.items():
        files = manifest[kind + '_files_sha256']
        scope = root if kind == 'manotorch' else root / 'chumpy'
        if {p.relative_to(root).as_posix() for p in scope.rglob('*.py')} != {n for n in files if n.endswith('.py')}:
            raise ValueError('original IK dependency Python source/provider set differs')
        for name, sha in files.items():
            key = kind + '/' + name
            paths[key] = _safe(root / name)
            expected[key] = sha
    for name, sha in SOURCE_SHA.items():
        key = 'upstream/' + name
        paths[key] = _safe(upstream / name)
        expected[key] = sha
    for p in upstream.glob('egoemg/**/*.py'):
        paths['upstream/' + p.relative_to(upstream).as_posix()] = _safe(p)
    paths['private/MANO_RIGHT.pkl'] = _safe(assets / 'models/MANO_RIGHT.pkl')
    expected['private/MANO_RIGHT.pkl'] = manifest['mano_right_sha256']
    guard = _Guard(paths.values(), [])
    hashes = {}
    for name, p in paths.items():
        guard.check()
        hashes[name] = file_sha256(p)
        guard.check()
        if name in expected and hashes[name] != expected[name]:
            raise ValueError('original IK source or MANO asset content differs: ' + name)
    _check_git(upstream)
    guard.check()
    identity = dict(format='original_native_ik_runtime_source_admission_v1', upstream_revision=UPSTREAM,
        manotorch_revision=manifest['manotorch_revision'], actual_dependency_source_files=dict(manotorch=25, chumpy=20),
        chumpy_version=manifest['chumpy_version'], files_sha256=hashes,
        author_external_checkout_identity_proven=False, model_constructed=False,
        native_policy=dict(right_mano_for_both_hands=True, flat_hand_mean=True, beta='internal zero',
            zero_input_global_orientation=True, extra_left_reflection=False, loss='landmark',
            max_outer_iterations=200, lbfgs_max_iter=50, lbfgs_lr=.1, lbfgs_history=30,
            line_search='strong_wolfe', opt_global_orient=True, opt_trans=True,
            regenerated_finger_angles=20, regenerated_wrist_angles=False),
        scientific_eligible=False, full_paper_complete=False)
    result = IKSource()
    _SOURCES[result] = dict(upstream=upstream, manotorch=manotorch, chumpy=chumpy, assets=assets,
                            guard=guard, identity=identity, identity_hash=_hash(identity))
    result.check()
    return result


def _poses(values):
    if (type(values) is not np.ndarray or values.dtype != np.dtype('float32') or values.ndim != 2
        or values.shape[1] != 48 or not 0 < len(values) <= 10096 or not values.flags.c_contiguous
        or not np.isfinite(values).all() or np.any(values[:, :3] != 0)):
        raise ValueError('original canonical finite contiguous float32[B,48] with zero global orientation required')
    return values


def _model_digest(s):
    h = hashlib.sha256()
    tensors = dict(s['mano'].state_dict())
    tensors.update({'umetrack.' + n: getattr(s['hand'], n) for n in s['hand']._fields})
    for name, value in sorted(tensors.items()):
        if value is None:
            h.update((name + ':None').encode())
        else:
            a = value.detach().cpu().contiguous().numpy()
            h.update((name + str(a.shape) + a.dtype.str).encode()); h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class IKRuntime:
    def check(self):
        s = _registered(self, _RUNTIMES, IKRuntime)
        s['source'].check()
        native = s['native']
        if (any(getattr(native, n) is not fn for n, fn in s['functions'].items())
            or native.ALIGN_SCALE != s['constants']['scale']
            or not np.array_equal(native.FLIP_MATRIX, s['constants']['flip'])
            or not np.array_equal(native.ALIGN_TRANS, s['constants']['trans'])):
            raise ValueError('original IK callable/constants binding changed')

    @property
    def identity(self):
        self.check()
        return copy.deepcopy(_RUNTIMES[self]['identity'])

    def rebuild(self, poses, angles, raw_angles, global_orient, trans_raw):
        self.check(); _poses(poses)
        s = _RUNTIMES[self]
        import torch
        from egoemg.kinematics import broadcast_hand_model_to, apply_to_hand_model
        from egoemg.UmeTrack.lib.common.hand_skinning import skin_landmarks
        b = len(poses)
        values = dict(angles=angles, raw_angles=raw_angles, global_orient=global_orient, trans_raw=trans_raw)
        for name, a in values.items():
            shape = (b, 20) if name in ('angles', 'raw_angles') else (b, 3)
            if type(a) is not np.ndarray or a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all():
                raise ValueError('finite original IK final-state shape/dtype required: ' + name)
        device = s['device']; native = s['native']; before = _model_digest(s)
        def tensor(a): return torch.from_numpy(a.copy()).to(device)
        pose, a, raw, orient, trans = [tensor(v) for v in (poses, angles, raw_angles, global_orient, trans_raw)]
        with torch.inference_mode():
            derived = s['lo'] + (s['hi'] - s['lo']) * torch.sigmoid(raw)
            if not torch.equal(derived, a) or not ((a >= s['lo']).all() and (a <= s['hi']).all()):
                raise ValueError('original IK returned angles differ from final raw parameters/limits')
            target = s['mano'](pose, torch.zeros(b, 10, device=device)).joints * 1000.0
            hm = apply_to_hand_model(broadcast_hand_model_to(s['hand'], (b,)), lambda t: t.float())
            predicted = skin_landmarks(hm, a, torch.eye(4, device=device).unsqueeze(0).expand(b, -1, -1))
            axis = torch.zeros(b, 3, 3, device=device)
            axis[:, 0, 1] = -orient[:, 2]; axis[:, 0, 2] = orient[:, 1]
            axis[:, 1, 0] = orient[:, 2]; axis[:, 1, 2] = -orient[:, 0]
            axis[:, 2, 0] = -orient[:, 1]; axis[:, 2, 1] = orient[:, 0]
            rotation = torch.linalg.matrix_exp(axis)
            target = native.ALIGN_SCALE * (target @ tensor(native.FLIP_MATRIX).T) @ rotation.transpose(-1, -2)
            target = target + (tensor(native.ALIGN_TRANS) + 10 * trans).unsqueeze(1)
            residual = predicted[:, native._UMETRACK_IDX.to(device)] - target[:, native._MANO_IDX.to(device)]
            loss = residual.square().sum(-1).mean().item()
            if not torch.isfinite(residual).all():
                raise ValueError('nonfinite original IK geometry')
            arrays = {name: value.detach().cpu().numpy().copy() for name, value in
                      dict(residual=residual, predicted_landmarks=predicted, aligned_mano_joints=target).items()}
        self.check()
        if _model_digest(s) != before:
            raise ValueError('original fixed IK geometry state changed')
        return dict(arrays=arrays, final_objective_mm2=loss, model_state_sha256=before,
                    actual_rebuild_optimizer_steps=0, source_identity_sha256=_hash(_SOURCES[s['source']]['identity']))

    def optimize(self, canonical_poses, *, guard=lambda: None):
        self.check(); _poses(canonical_poses); guard()
        import torch
        if (not torch.is_grad_enabled() or torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
            or torch.get_default_dtype() != torch.float32):
            raise ValueError('original IK requires FP32 autograd without autocast')
        s = _RUNTIMES[self]; native = s['native']; device = s['device']; b = len(canonical_poses)
        original_input = canonical_poses.tobytes(); poses = canonical_poses.copy()
        state_before = _model_digest(s); start = time.monotonic(); trace = []; retained = []; closures = 0
        original = torch.optim.LBFGS.step
        def observed(optimizer, closure):
            nonlocal closures, retained
            self.check(); guard()
            def counted():
                nonlocal closures
                closures += 1
                return closure()
            try:
                returned = original(optimizer, counted)
            finally:
                self.check(); guard()
            params = optimizer.param_groups[0]['params']
            if [tuple(v.shape) for v in params] != [(b, 20), (b, 3), (b, 3)]:
                raise ValueError('original IK optimization parameter binding changed')
            retained = [v.detach().cpu().numpy().copy() for v in params]
            trace.append(dict(outer_step=len(trace) + 1, closure_calls_so_far=closures,
                              returned_pre_step_loss=float(returned)))
            return returned
        torch.optim.LBFGS.step = observed
        try:
            angles, loss = native.optimize_batch(hand_model=s['hand'], mano_layer=s['mano'],
                pose_batch=torch.from_numpy(poses).to(device),
                flip_t=torch.from_numpy(native.FLIP_MATRIX.copy()).float().to(device),
                angle_min=s['lo'], angle_max=s['hi'], max_iter=200,
                opt_global_orient=True, opt_trans=True, loss_type='landmark',
                chamfer_weight=1., landmark_weight=1.,
                mano_vidx=native._uniform_idx(778, 100).to(device),
                ut_vidx=native._uniform_idx(788, 100).to(device), device=device)
        finally:
            torch.optim.LBFGS.step = original
        if not retained or not 1 <= len(trace) <= 200 or canonical_poses.tobytes() != original_input:
            raise ValueError('original IK optimizer trace or immutable input differs')
        raw_angles, orient, trans = retained
        checked = self.rebuild(poses, angles, raw_angles, orient, trans)
        if checked['final_objective_mm2'] != loss or checked['model_state_sha256'] != state_before:
            raise ValueError('original IK returned final objective/state failed direct reconstruction')
        self.check(); guard()
        return dict(angles=angles, raw_angles=raw_angles, global_orient=orient, trans_raw=trans,
            **checked, native_final_loss=loss, actual_ik_outer_steps=len(trace),
            actual_ik_optimizer_steps=len(trace), actual_ik_closure_evaluations=closures, ik_step_trace=trace,
            actual_original_step_return_preserved=True, seconds=time.monotonic() - start)
