"""Real bilateral engineering only: two contiguous frames per hand, original IK.

The complete available pose source is freshly admitted, but the optimizer batch
is explicitly2 instead of10096. This command never qualifies or regenerates the
full dataset. The separate verify action constructs geometry, never optimizers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from egoemg_repro.classic_emg_cohort import _safe
from egoemg_repro.data import file_sha256
from egoemg_repro.emg2pose_memory_process import process_identity
from egoemg_repro.native_ik_inputs import admit_native_ik_inputs, _state, _hash, _Guard
from egoemg_repro.native_ik_runtime import admit_ik_sources


def seal(value):
    return dict(value, sha256=_hash(value))


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def sample(inputs, hand, center):
    inputs.check()
    indices = np.array([center - 1, center], dtype=np.int64)
    raw = np.array(_state(inputs)['arrays'][hand][indices], dtype=np.float32, copy=True)
    canonical = raw.copy(); canonical[:, :3] = 0.0
    if not np.any(raw != 0):
        raise ValueError('explicit real-frame engineering selection must include a nonzero pose')
    inputs.check()
    return dict(indices=indices, raw_poses=raw, canonical_poses=canonical)


def run(args):
    if args.action not in ('produce', 'verify'):
        raise ValueError('explicit produce or verify action required')
    if type(args.center_index) is not int or args.center_index <= 0:
        raise ValueError('positive integer engineering center index required')
    output = _safe(args.output, exists=False)
    if output.exists():
        raise ValueError('existing real-frame engineering output must not be overwritten')
    if args.action == 'verify' and args.original is None:
        raise ValueError('original exited producer output required')
    original = None
    guard = _Guard([Path(__file__).resolve()], [])
    if args.action == 'verify':
        directory = _safe(args.original)
        if not (directory / 'result.json').is_file():
            raise ValueError('original real-frame producer result required')
        guard = _Guard([Path(__file__).resolve(), directory / 'result.json'], [])
        original = json.loads((directory / 'result.json').read_text())
        if original.get('sha256') != _hash({k: v for k, v in original.items() if k != 'sha256'}):
            raise ValueError('original real-frame engineering result seal differs')
        actor = process_identity(original['actor']['pid'])
        if actor is not None and actor['start_ticks'] == original['actor']['start_ticks']:
            raise ValueError('original IK producer must actually exit before fresh geometry verification')
        if original['script_sha256'] != file_sha256(__file__) or original['center_index'] != args.center_index:
            raise ValueError('original real-frame producer source/selection differs')
        for hand, record in original['hands'].items():
            path = _safe(directory / record['state_file'])
            if path.parent != directory or file_sha256(path) != record['state_file_sha256']:
                raise ValueError('original real-frame raw state bytes differ')
        guard = _Guard([*guard.stamps, *(directory / v['state_file'] for v in original['hands'].values())], [])
    start = time.monotonic()
    inputs = admit_native_ik_inputs(args.data, args.geometry, args.upstream)
    if args.center_index >= inputs.identity['total_rows']:
        raise ValueError('engineering frame index exceeds the admitted actual source')
    sources = admit_ik_sources(args.upstream, args.manotorch, args.chumpy, args.mano_assets)
    input_identity, source_identity = inputs.identity, sources.identity
    if original is not None:
        if original['input_identity_sha256'] != _hash(input_identity) or original['source_identity_sha256'] != _hash(source_identity):
            raise ValueError('fresh real-frame source identity differs')
    import torch
    torch.set_num_threads(2)
    if args.action == 'verify':
        def deny(*a, **k):
            raise RuntimeError('independent real-frame geometry verification cannot optimize')
        torch.optim.Optimizer.__init__ = deny
        torch.optim.LBFGS.step = deny
    output.mkdir(parents=True, mode=0o700)
    hands = {}
    def checked():
        inputs.check(); sources.check(); guard.check()
    for hand in ('right', 'left'):
        current = sample(inputs, hand, args.center_index)
        runtime = sources.build(device='cpu')
        checked()
        if args.action == 'produce':
            optimized = runtime.optimize(current['canonical_poses'], guard=checked)
            arrays = {**current, **{k: optimized[k] for k in ('angles', 'raw_angles', 'global_orient', 'trans_raw')},
                      **optimized['arrays']}
            state_file = hand + '-private-state.npz'
            np.savez(output / state_file, **arrays)
            record = {k: v for k, v in optimized.items() if k not in ('angles', 'raw_angles', 'global_orient', 'trans_raw', 'arrays')}
            distance = np.sqrt(np.square(arrays['residual'].astype(np.float64)).sum(-1))
            record.update(state_file=state_file, state_file_sha256=file_sha256(output / state_file),
                runtime_identity=runtime.identity, selected_indices=current['indices'].tolist(),
                actual_geometry_batch_size=2, original_default_batch_size=10096,
                per_frame_mean_landmark_distance_mm=distance.mean(-1).tolist(),
                per_frame_rms_landmark_distance_mm=np.sqrt(np.square(distance).mean(-1)).tolist())
            hands[hand] = record
        else:
            record = original['hands'][hand]
            with np.load(directory / record['state_file'], allow_pickle=False) as saved:
                arrays = {n: saved[n].copy() for n in saved.files}
            for name in current:
                if arrays[name].dtype != current[name].dtype or not np.array_equal(arrays[name], current[name]):
                    raise ValueError('real-frame raw/canonical/source index witness differs')
            if record['runtime_identity'] != runtime.identity:
                raise ValueError('fresh real-frame geometry runtime identity differs')
            rebuilt = runtime.rebuild(current['canonical_poses'], *(arrays[n] for n in ('angles', 'raw_angles', 'global_orient', 'trans_raw')))
            if rebuilt['final_objective_mm2'] != record['native_final_loss'] or rebuilt['model_state_sha256'] != record['model_state_sha256']:
                raise ValueError('fresh real-frame final objective or fixed geometry state differs')
            for name, value in rebuilt['arrays'].items():
                if arrays[name].dtype != value.dtype or not np.array_equal(arrays[name], value):
                    raise ValueError('fresh real-frame direct geometric arrays differ: ' + name)
            hands[hand] = dict(actual_geometry_batch_size=2, selected_indices=current['indices'].tolist(),
                all_typed_geometric_arrays_exact=True, native_final_loss=rebuilt['final_objective_mm2'],
                model_state_sha256=rebuilt['model_state_sha256'], actual_ik_optimizer_steps=0,
                original_state_file_sha256=record['state_file_sha256'])
        checked()
    if torch.cuda.is_initialized():
        raise ValueError('CPU real-frame engineering unexpectedly initialized CUDA')
    result = seal(dict(format='actual_native_ik_bilateral_real_frame_engineering_' + args.action + '_v1',
        action=args.action, actor=process_identity(), input_identity_sha256=_hash(input_identity),
        source_identity_sha256=_hash(source_identity), hands=hands, center_index=args.center_index,
        selected_rows_per_hand=2, actual_native_mano_geometry_layers=2, actual_learned_pose_models_constructed=0,
        actual_native_optimizer_calls=2 if args.action == 'produce' else 0,
        actual_ik_optimizer_steps=sum(v['actual_ik_optimizer_steps'] for v in hands.values()),
        original_actor_exited=original is not None, original_result_file_sha256=file_sha256(directory / 'result.json') if original is not None else None,
        original_raw_source_files_unmodified=True, script_sha256=file_sha256(__file__),
        actual_complete_pose_payloads_freshly_admitted=True, original_default_batch10096_qualified=False,
        engineering_batch_override_explicit=True, cpu_function_engineering_only=True,
        regenerated_source_dataset_labels=False, actual_scientific_model_training_updates=0,
        cuda_initialized=False, scientific_eligible=False, full_paper_complete=False, seconds=time.monotonic() - start))
    checked(); write(output / 'result.json', result); checked()
    print(json.dumps({k: result[k] for k in ('action', 'input_identity_sha256', 'actual_native_optimizer_calls', 'actual_ik_optimizer_steps', 'seconds')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('produce', 'verify'))
    for name in ('data', 'geometry', 'upstream', 'manotorch', 'chumpy', 'mano-assets', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--center-index', type=int, default=331075)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
