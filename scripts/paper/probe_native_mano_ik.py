"""CPU engineering call to the pinned original MANO-to-UmeTrack IK function.

Uses two synthetic poses and licensed local MANO data. Never writes dataset
labels or claims a whole-data labeling result. Dependencies stay unmodified.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

EGOEMG = 'f7cbd048b7092139852270491d3859cd472e0311'
MANOTORCH = 'a2a70c591f91551078b7bb2af9b5d9f275b626e0'
MANO_RIGHT = '45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
            s.st_nlink, s.st_mode, s.st_uid, s.st_gid]


def clean(root, expected):
    assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == expected
    assert not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
                                        '--untracked-files=all'], text=True).strip()


def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    assert not args.output.exists()
    args.output.mkdir(parents=True, mode=0o700)
    pid = os.getpid()
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    (args.output / 'owner.json').write_text(json.dumps(dict(pid=pid, start_ticks=fields[19])) + '\n')
    upstream, manotorch, chumpy, assets = [getattr(args, n).resolve() for n in
                                         ('upstream', 'manotorch', 'chumpy', 'mano_assets')]
    clean(upstream, EGOEMG); clean(manotorch, MANOTORCH)
    sources = {'egoemg/' + p.relative_to(upstream).as_posix(): p
               for p in (upstream / 'egoemg').rglob('*.py')}
    ik_file = upstream / 'scripts/ik/batch_ik_mesh.py'
    limits_file = upstream / 'egoemg/UmeTrack/dataset/generic_hand_model.json'
    sources.update({'egoemg/scripts/ik/batch_ik_mesh.py': ik_file,
                    'egoemg/egoemg/UmeTrack/dataset/generic_hand_model.json': limits_file})
    for base, prefix in ((manotorch, 'manotorch'), (chumpy, 'chumpy')):
        for p in base.rglob('*.py'):
            if '.git' not in p.relative_to(base).parts:
                sources[prefix + '/' + p.relative_to(base).as_posix()] = p
    sources['manotorch/LICENSE'] = manotorch / 'LICENSE'
    source_hashes = {n: sha(p) for n, p in sources.items()}
    model = assets / 'models/MANO_RIGHT.pkl'
    assert sha(model) == MANO_RIGHT
    before = {str(p): stamp(p) for p in [*sources.values(), model, Path(__file__)]}
    assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:'))) * 1024 > 2 * 1024**3
    sys.path[:0] = [str(upstream), str(manotorch), str(chumpy)]
    import numpy as np
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    # The original Chumpy0.70 is retained byte-for-byte. Its removed names are
    # restored only in this disposable process, as in the prior MANO admission.
    aliases = dict(bool=bool, int=int, float=float, complex=complex,
                   object=object, unicode=str, str=str)
    restored = []
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value); restored.append('numpy.' + name)
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
        restored.append('inspect.getargspec')
    spec = importlib.util.spec_from_file_location('original_egoemg_batch_ik', ik_file)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    from manotorch.manolayer import ManoLayer
    from egoemg.kinematics import load_default_hand_model, apply_to_hand_model
    import manotorch as imported_manotorch
    assert Path(imported_manotorch.__file__).resolve().is_relative_to(manotorch)
    started = time.monotonic()
    mano = ManoLayer(rot_mode='axisang', side='right', mano_assets_root=str(assets),
                     use_pca=False, flat_hand_mean=True).cpu()
    hand = apply_to_hand_model(load_default_hand_model(), lambda t: t.float().cpu())
    limits = np.asarray(json.loads(limits_file.read_text())['joint_limits'][:20], dtype=np.float32)
    lo, hi = torch.from_numpy(limits[:, 0].copy()), torch.from_numpy(limits[:, 1].copy())
    poses = torch.zeros(2, 48, dtype=torch.float32)
    poses[1, 3:] = torch.linspace(-.12, .18, 45)
    zero_beta = torch.zeros(2, 10)
    flip = torch.from_numpy(native.FLIP_MATRIX).float()
    translation = torch.from_numpy(native.ALIGN_TRANS).float()
    with torch.no_grad():
        target_joints, target_vertices = native.mano_fk_batch(mano, poses, zero_beta)
        initial_angles = (lo + hi).unsqueeze(0).repeat(2, 1) / 2
        a22 = torch.cat([initial_angles, torch.zeros(2, 2)], dim=1)
        initial_lm = native.umetrack_fk_batch(hand, a22, 'cpu', return_mesh=False, return_landmarks=True)
        initial_target = native.ALIGN_SCALE * (target_joints @ flip.T) + translation
        initial_loss = native.landmark_l2_batch(initial_lm[:, native._UMETRACK_IDX],
                                               initial_target[:, native._MANO_IDX]).item()
    trace = []; retained = {}; actual_steps = 0; closures = 0
    original_step = torch.optim.LBFGS.step

    def observed_step(optimizer, closure):
        nonlocal actual_steps, closures
        def counted():
            nonlocal closures
            closures += 1
            return closure()
        loss = original_step(optimizer, counted)
        actual_steps += 1
        params = optimizer.param_groups[0]['params']
        assert [tuple(p.shape) for p in params] == [(2, 20), (2, 3), (2, 3)]
        retained['params'] = [p.detach().clone() for p in params]
        trace.append(dict(outer_step=actual_steps, closure_calls_so_far=closures,
                          returned_pre_step_loss=float(loss)))
        return loss

    torch.optim.LBFGS.step = observed_step
    try:
        angles, final_loss = native.optimize_batch(
            hand_model=hand, mano_layer=mano, pose_batch=poses, flip_t=flip,
            angle_min=lo, angle_max=hi, max_iter=200, opt_global_orient=True,
            opt_trans=True, loss_type='landmark', chamfer_weight=1., landmark_weight=1.,
            mano_vidx=native._uniform_idx(778, 100), ut_vidx=native._uniform_idx(788, 100),
            device='cpu')
    finally:
        torch.optim.LBFGS.step = original_step
    raw_angles, global_orient, trans_raw = retained['params']
    with torch.no_grad():
        observed_angles = lo + (hi - lo) * torch.sigmoid(raw_angles)
        assert np.array_equal(observed_angles.numpy(), angles)
        assert torch.isfinite(observed_angles).all()
        assert (observed_angles >= lo).all() and (observed_angles <= hi).all()
        a22 = torch.cat([observed_angles, torch.zeros(2, 2)], dim=1)
        predicted = native.umetrack_fk_batch(hand, a22, 'cpu', return_mesh=False, return_landmarks=True)
        rotation = native.axisangle_to_rotmat_batch(global_orient)
        aligned = native.ALIGN_SCALE * (target_joints @ flip.T) @ rotation.transpose(-1, -2)
        aligned += (translation + 10 * trans_raw).unsqueeze(1)
        delta = predicted[:, native._UMETRACK_IDX] - aligned[:, native._MANO_IDX]
        recomputed = delta.square().sum(-1).mean().item()
        assert recomputed == final_loss and final_loss < initial_loss
        distance = delta.square().sum(-1).sqrt()
        assert torch.isfinite(predicted).all() and torch.isfinite(aligned).all()
    assert not torch.cuda.is_initialized()
    assert all(stamp(n) == s for n, s in before.items())
    assert all(sha(p) == source_hashes[n] for n, p in sources.items()) and sha(model) == MANO_RIGHT
    clean(upstream, EGOEMG); clean(manotorch, MANOTORCH)
    np.savez(args.output / 'private-ik-state.npz', synthetic_poses=poses.numpy(),
             angles=angles, global_orient=global_orient.numpy(), trans_raw=trans_raw.numpy())
    (args.output / 'private-input-stamps.json').write_text(json.dumps(before, sort_keys=True, indent=2) + '\n')
    result = dict(format='egoemg_original_native_mano_umetrack_ik_cpu_probe_v1',
                  upstream_revision=EGOEMG, manotorch_revision=MANOTORCH,
                  source_files_sha256=source_hashes, mano_right_sha256=MANO_RIGHT,
                  source_files_unmodified=True, nine_input_attributes_unchanged=True,
                  explicit_dependency_pin_not_author_checkout=True,
                  actual_native_mano_layer_constructions=1, synthetic_input_poses=2,
                  pose_global_orientation_zeroed=True, native_internal_beta_zero=True,
                  native_flat_hand_mean=True, native_loss='landmark', native_outer_budget=200,
                  native_lbfgs_max_iter_per_step=50, actual_ik_outer_steps=actual_steps,
                  actual_ik_closure_evaluations=closures, actual_original_step_return_preserved=True,
                  initial_landmark_squared_distance_mm2=initial_loss,
                  final_landmark_squared_distance_mm2=final_loss,
                  independently_recomputed_final_loss_exact=True,
                  per_pose_mean_landmark_distance_mm=distance.mean(-1).tolist(),
                  per_pose_rms_landmark_distance_mm=delta.square().sum(-1).mean(-1).sqrt().tolist(),
                  angles_finite_and_within_original_limits=True,
                  mano_vertex_shape=list(target_vertices.shape), mano_joint_shape=list(target_joints.shape),
                  actual_returned_finger_angle_shape=list(angles.shape),
                  ik_step_trace=trace, compatibility_names_restored_in_process=restored,
                  native_cli_limits_path_exists=(upstream / 'emg2pose/UmeTrack/dataset/generic_hand_model.json').exists(),
                  actual_limits_asset='egoemg/UmeTrack/dataset/generic_hand_model.json',
                  cpu_function_call_not_full_gpu_cli=True, dataset_labels_written=False,
                  full_frame_label_regeneration_complete=False, wrist_labels_regenerated=False,
                  participant_beta_geometry_validated=False, new_scientific_model_updates=0,
                  cuda_initialized=False, full_paper_complete=False,
                  python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
                  seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  script_sha256=sha(__file__),
                  private_state_sha256=sha(args.output / 'private-ik-state.npz'))
    (args.output / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('actual_ik_outer_steps', 'actual_ik_closure_evaluations',
                     'initial_landmark_squared_distance_mm2', 'final_landmark_squared_distance_mm2', 'seconds')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'manotorch', 'chumpy', 'mano-assets', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
