from pathlib import Path
import ast
import argparse
import datetime
import gc
import hashlib
import inspect
import json
import os
import pickle
import resource
import re
import shutil
import sys
import time
import numpy as np

PUBLIC_BLOBS = {
    'model.py': '42f7b84eced4fc2dd14fcb0f732b7be3728b51c8',
    'view_samples.py': '25e90ea14d1bb1c37145cf208aee3717168beb90',
    'fh_utils.py': '103b3e01aaea148939f72df947a69c5e60eb0385',
    'freihand-README.md': '4248c5cb5cdbf5c414fa6d4300a5ab8ecf87ff4e',
    'mano_utils.py': 'b8d2213873f49f526e3c266108e93e075f574ba5',
    'setup_mano.py': '08d59cb34a2dd4c217f2d55074484a24a9b4fd49',
}
LICENSED_INPUTS = {
    'models/MANO_RIGHT.pkl': ('fd5a9d35f914987cf1cc04ffe338caa1', '45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767'),
    'webuser/verts.py': ('998c30fd83c473da6178aa2cb23b6a5d', 'b6d46e2688aef5a37124de190ca86ede19ed39a84818f3ac9ef022f3092fc7ee'),
    'webuser/posemapper.py': ('c5e9eacc535ec7d03060e0c8d6f80f45', '40c3bf9957b359db5eb52f6089ead9e44ad9ff5f22d4d0e20814ab149e94154d'),
    'webuser/lbs.py': ('d11c767d5db8d4a55b4ece1c46a4e4ac', 'bec1572ed1476a004973d9e55cff1afc27e8f8656b710509e428197191385049'),
    'webuser/smpl_handpca_wrapper_HAND_only.py': ('5afc7a3eb1a6ce0c2dac1483338a5f58', '080347cf7c87e7931aad33517f0284437c86b6c8de52e6a4cb41507cc6d9497e'),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def checked_bytes(path):
    need(path.is_file() and not path.is_symlink(), 'nonregular source input')
    before = path.stat()
    body = path.read_bytes()
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'source input generation changed')
    return body


def check_sources(request):
    public = Path(request['public_root'])
    licensed = Path(request['licensed_root'])
    result = {'public_files': {}, 'licensed_files': {}}
    for name, blob in PUBLIC_BLOBS.items():
        body = checked_bytes(public / name)
        need(hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest() == blob, 'pinned original FreiHAND source differs')
        result['public_files'][name] = {'bytes': len(body), 'git_blob_sha1': blob, 'sha256': hashlib.sha256(body).hexdigest()}
    for name, (expected_md5, expected_sha) in LICENSED_INPUTS.items():
        body = checked_bytes(licensed / name)
        need(hashlib.md5(body).hexdigest() == expected_md5 and hashlib.sha256(body).hexdigest() == expected_sha, 'registered model/core source differs from official setup identity')
        result['licensed_files'][name] = {'bytes': len(body), 'md5': expected_md5, 'sha256': expected_sha}
    return result


def assemble_joints(vertices, regressed):
    need(isinstance(vertices, np.ndarray) and vertices.dtype == np.float64 and vertices.shape == (778, 3) and np.isfinite(vertices).all(), 'complete finite float64 posed mesh required')
    need(isinstance(regressed, np.ndarray) and regressed.dtype == np.float64 and regressed.shape == (16, 3) and np.isfinite(regressed).all(), 'complete finite float64 posed-mesh joint regression required')
    mapping = {0: 0, 1: 5, 2: 6, 3: 7, 4: 9, 5: 10, 6: 11, 7: 17, 8: 18, 9: 19, 10: 13, 11: 14, 12: 15, 13: 1, 14: 2, 15: 3}
    tips = {4: 744, 8: 320, 12: 443, 16: 555, 20: 672}
    joints = np.empty((21, 3), dtype=np.float64)
    for original, label in mapping.items():
        joints[label] = regressed[original]
    for label, vertex in tips.items():
        joints[label] = vertices[vertex]
    return joints


def camera_translation(mano, camera, posed_joints):
    validate_row(mano, camera)
    need(isinstance(posed_joints, np.ndarray) and posed_joints.dtype == np.float64 and posed_joints.shape == (21, 3) and np.isfinite(posed_joints).all(), 'complete posed source keypoints required')
    focal = 0.5 * (camera[0, 0] + camera[1, 1])
    root = np.array([mano[0, 58] - camera[0, 2], mano[0, 59] - camera[1, 2], focal], dtype=np.float64) / mano[0, 60]
    return root - posed_joints[9]


def validate_row(mano, camera):
    need(isinstance(mano, np.ndarray) and mano.dtype == np.float64 and mano.shape == (1, 61) and np.isfinite(mano).all(), 'original finite float64[1,61] row required')
    need(isinstance(camera, np.ndarray) and camera.dtype == np.float64 and camera.shape == (3, 3) and np.isfinite(camera).all(), 'original finite float64[3,3] intrinsics required')
    need(mano[0, 60] > 0, 'positive original weak-perspective scale required; no repair')


def check_runtime(request):
    runtime = Path(request['runtime_root'])
    expected = request['runtime_files_sha256']
    names = {'data/MANO_RIGHT.pkl', 'utils/mano_core/verts.py', 'utils/mano_core/posemapper.py', 'utils/mano_core/lbs.py', 'utils/mano_core/mano_loader.py'}
    need(set(expected) == names, 'complete original setup outputs required')
    for name, digest in expected.items():
        need(hashlib.sha256(checked_bytes(runtime / name)).hexdigest() == digest, 'private official-setup output changed')
    need(expected['data/MANO_RIGHT.pkl'] == LICENSED_INPUTS['models/MANO_RIGHT.pkl'][1], 'private model identity differs')
    for group in ['chumpy_files', 'smplx_files']:
        for path, digest in request[group].items():
            need(hashlib.sha256(checked_bytes(Path(path))).hexdigest() == digest, 'existing package provider changed')


def python_context(request):
    for path in reversed([str(Path(request['runtime_root']) / 'utils' / 'mano_core'), *request['package_paths']]):
        if path not in sys.path:
            sys.path.insert(0, path)
    aliases = []
    for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex, 'object': object, 'unicode': str, 'str': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
            aliases.append(name)
    inspect_alias = not hasattr(inspect, 'getargspec')
    if inspect_alias:
        inspect.getargspec = inspect.getfullargspec
    if 'cPickle' in sys.modules:
        need(sys.modules['cPickle'] is pickle, 'unexpected cPickle provider')
    else:
        sys.modules['cPickle'] = pickle
    import chumpy as ch
    import cv2
    need(ch.__version__ == '0.70' and Path(ch.__file__).resolve().is_relative_to(Path(request['package_paths'][0]).resolve()), 'original private Chumpy0.70 provider required')
    need(np.__version__ == '1.26.4', 'existing pinned NumPy provider required')
    return ch, {'numpy_version': np.__version__, 'chumpy_version': ch.__version__, 'opencv_version': cv2.__version__, 'process_local_numpy_aliases_added': aliases, 'inspect_getargspec_alias_added': inspect_alias, 'cPickle_alias_to_standard_pickle': True, 'shared_package_source_modified': False}


def build_original(request):
    identity = check_sources(request)
    check_runtime(request)
    ch, compatibility = python_context(request)
    runtime = Path(request['runtime_root'])
    loader_path = runtime / 'utils' / 'mano_core' / 'mano_loader.py'
    loader_text = loader_path.read_text()
    parts = loader_text.split("\nif __name__ == '__main__':\n")
    need(len(parts) == 2, 'original standalone-demo boundary differs')
    # The only omitted text is the unused Python2 standalone demo. All loader
    # definitions retain the bytes produced by the unchanged official setup.
    loader_env = {'__name__': 'freihand_private_original_loader'}
    exec(compile(parts[0], str(loader_path), 'exec', dont_inherit=True), loader_env)
    need(set(n.name for n in ast.parse(parts[0]).body if isinstance(n, ast.FunctionDef)) == {'ready_arguments', 'load_model'}, 'complete original loader definitions required')
    constructions = []

    def decoded_original_model(filename, **kwargs):
        need(filename == './data/MANO_RIGHT.pkl' and kwargs == {'ncomps': 45, 'flat_hand_mean': False, 'use_pca': False}, 'exact original FreiHAND no-PCA/mean-pose flags required')
        path = runtime / 'data' / 'MANO_RIGHT.pkl'
        need(hashlib.sha256(checked_bytes(path)).hexdigest() == LICENSED_INPUTS['models/MANO_RIGHT.pkl'][1], 'registered model changed before trusted decoding')
        with path.open('rb') as stream:
            data = pickle.load(stream, encoding='latin1')
        need(type(data) is dict, 'registered model is not an original dictionary')
        # Passing the decoded dictionary avoids the original Python2 text-mode
        # file read; the original loader/ready_arguments arithmetic is retained.
        model = loader_env['load_model'](data, **kwargs)
        constructions.append({'ncomps': 45, 'flat_hand_mean': False, 'use_pca': False})
        return model

    helpers = {'__name__': 'freihand_original_mesh_helpers'}
    exec(compile((Path(request['public_root']) / 'mano_utils.py').read_bytes(), str(Path(request['public_root']) / 'mano_utils.py'), 'exec', dont_inherit=True), helpers)
    source_path = Path(request['public_root']) / 'model.py'
    tree = ast.parse(source_path.read_bytes())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    need({node.name for node in nodes} == {'pose_hand', 'split_theta', 'get_focal_pp', 'backproject_ortho', 'recover_root', 'HandModel'}, 'all original geometry definitions required')
    env = {'__name__': 'freihand_original_geometry', 'np': np, 'ch': ch, 'renderer': None, 'load_model': decoded_original_model, 'get_keypoints_from_mesh_ch': helpers['get_keypoints_from_mesh_ch']}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), 'exec', dont_inherit=True), env)
    env['renderer'] = env['HandModel'](use_mean_pca=False, use_mean_pose=True)
    need(len(constructions) == 1, 'exactly one original model construction required')
    core = runtime / 'utils' / 'mano_core'
    for name in ['posemapper', 'lbs', 'verts']:
        module = sys.modules.get(name)
        need(module is not None and Path(module.__file__).resolve().parent == core.resolve(), 'original isolated MANO operator provider required')

    def forward(mano, camera):
        validate_row(mano, camera)
        before = (mano.tobytes(), camera.tobytes())
        xyz, vertices = env['pose_hand'](mano, camera, use_mean_pose=True)
        need(xyz.dtype == vertices.dtype == np.float64 and xyz.shape == (21, 3) and vertices.shape == (778, 3), 'complete original float64 geometry required')
        need(np.isfinite(xyz).all() and np.isfinite(vertices).all(), 'original geometry is nonfinite')
        need(before == (mano.tobytes(), camera.tobytes()), 'original source row mutated')
        return xyz, vertices

    return forward, {'backend': 'pinned_original_FreiHAND_Chumpy', 'source_identity': identity, 'compatibility': compatibility, 'geometry_function_AST_sha256': hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()).hexdigest(), 'rendering_imports_unused_and_omitted': True, 'Python2_standalone_demo_omitted': True, 'trusted_model_dictionary_supplied_to_original_loader': True, 'original_numpy_loader_seed': 1, 'model_constructions': len(constructions), 'model_flags': constructions[0], 'geometry_arithmetic_modified': False}


def build_independent(request):
    identity = check_sources(request)
    check_runtime(request)
    _, compatibility = python_context(request)
    import torch
    import smplx
    need(not torch.cuda.is_initialized(), 'CPU-only geometry required')
    need(Path(smplx.__file__).resolve().is_relative_to(Path(request['package_paths'][1]).resolve()), 'existing admitted SMPL-X package provider required')
    torch.set_num_threads(2)
    model = smplx.MANO(model_path=str(Path(request['runtime_root']) / 'data'), is_rhand=True, use_pca=False, num_pca_comps=45, flat_hand_mean=False, dtype=torch.float64).eval()
    need(model.v_template.dtype == torch.float64 and model.J_regressor.dtype == torch.float64 and model.v_template.device.type == 'cpu', 'float64 CPU MANO buffers required')
    regressor = model.J_regressor.detach().cpu().numpy().copy()

    def forward(mano, camera):
        validate_row(mano, camera)
        before = (mano.tobytes(), camera.tobytes())
        pose = torch.from_numpy(mano[:, :48].copy())
        beta = torch.from_numpy(mano[:, 48:58].copy())
        with torch.no_grad():
            output = model(global_orient=pose[:, :3], hand_pose=pose[:, 3:], betas=beta, transl=torch.zeros((1, 3), dtype=torch.float64, device='cpu'))
        raw_vertices = output.vertices.detach().cpu().numpy()[0].copy()
        local_joints = assemble_joints(raw_vertices, regressor @ raw_vertices)
        translation = camera_translation(mano, camera, local_joints)
        vertices = raw_vertices + translation
        # The original keypoints are regressed again after global translation;
        # do not assume exact unit row sums or substitute the library joint API.
        xyz = assemble_joints(vertices, regressor @ vertices)
        need(vertices.dtype == xyz.dtype == np.float64 and np.isfinite(vertices).all(), 'complete independent float64 geometry required')
        need(not torch.cuda.is_initialized() and before == (mano.tobytes(), camera.tobytes()), 'CPU/input generation policy changed')
        return xyz, vertices

    return forward, {'backend': 'existing_SMPLX_float64_CPU', 'source_identity': identity, 'compatibility': compatibility, 'torch_version': torch.__version__, 'model_constructions': 1, 'model_flags': {'use_pca': False, 'num_pca_comps': 45, 'flat_hand_mean': False, 'is_rhand': True}, 'dtype': 'float64', 'library_joint_output_used': False, 'posed_mesh_regression_recomputed_after_translation': True, 'cuda_initialized': False}


ROLE_ROWS = {'training': 32560, 'evaluation': 3960}
BACKEND_TOLERANCE_METERS = 1e-7
ERROR_COLUMNS = ['original_vertex_mean_m', 'original_joint_mean_m', 'independent_vertex_mean_m', 'independent_joint_mean_m', 'backend_vertex_max_abs_m', 'backend_joint_max_abs_m']


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp(path):
    need(path.is_file() and not path.is_symlink(), 'nonregular immutable input')
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def current_actor():
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': os.getpid(), 'start_ticks': int(fields[19])}


def closed(owner):
    try:
        fields = Path('/proc', str(owner['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[19]) != owner['start_ticks'] or fields[0] in ['Z', 'X']
    except FileNotFoundError:
        return True


def write_json(path, value, *, atomic=False):
    target = path.with_name(path.name + '.pending') if atomic else path
    with target.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if atomic:
        os.replace(target, path)


def resources(*, admission=False):
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    need(available >= (256 + (16 if admission else 0)) * 1024**3, 'unchanged256GiB host reserve and admitted workspace required')
    need(shutil.disk_usage('/data').free >= 200 * 1024**3, 'unchanged200GiB disk reserve required')
    need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 16 * 1024**3, 'geometry worker16GiB workspace exceeded')


def bind_inputs(request, *, hash_data):
    paths = {}
    for group in ['public_root', 'licensed_root']:
        root = Path(request[group])
        names = PUBLIC_BLOBS if group == 'public_root' else LICENSED_INPUTS
        paths.update({str(root / name): None for name in names})
    runtime = Path(request['runtime_root'])
    paths.update({str(runtime / name): digest for name, digest in request['runtime_files_sha256'].items()})
    for group in ['chumpy_files', 'smplx_files']:
        paths.update(request[group])
    for role in ROLE_ROWS:
        need(set(request['raw_files'][role]) == {'K', 'mano', 'xyz', 'verts'}, 'complete original geometry fields required')
        for identity in request['raw_files'][role].values():
            path = Path(identity['path'])
            need(path.stat().st_size == identity['bytes'], 'original raw input byte length differs')
            if hash_data:
                need(file_sha(path) == identity['sha256'], 'original complete raw input SHA differs')
            paths[str(path)] = None
    paths[str(Path(__file__))] = request['worker_source_sha256']
    result = {}
    for raw_path, digest in paths.items():
        path = Path(raw_path)
        result[path] = stamp(path)
        if digest is not None:
            need(file_sha(path) == digest, 'frozen source/provider bytes differ')
    check_sources(request)
    check_runtime(request)
    return result


def check_stamps(inputs):
    need(all(stamp(path) == old for path, old in inputs.items()), 'input/source generation changed during use')
    resources()


def read_role(request, role):
    resources(admission=True)
    shapes = {'K': (3, 3), 'mano': (1, 61), 'xyz': (21, 3), 'verts': (778, 3)}
    arrays = {}
    for name, width in shapes.items():
        with Path(request['raw_files'][role][name]['path']).open() as stream:
            raw = json.load(stream)
        array = np.asarray(raw)
        del raw
        need(array.dtype == np.float64 and array.shape == (ROLE_ROWS[role], *width) and np.isfinite(array).all(), 'complete original role array differs')
        arrays[name] = array
        resources()
    return arrays


def error_row(original_xyz, original_v, independent_xyz, independent_v, gt_xyz, gt_v):
    return np.array([np.linalg.norm(original_v - gt_v, axis=1).mean(), np.linalg.norm(original_xyz - gt_xyz, axis=1).mean(), np.linalg.norm(independent_v - gt_v, axis=1).mean(), np.linalg.norm(independent_xyz - gt_xyz, axis=1).mean(), np.max(np.abs(original_v - independent_v)), np.max(np.abs(original_xyz - independent_xyz))], dtype=np.float64)


def geometry_output_shapes(target_rows):
    need(type(target_rows) is int and target_rows in (2, sum(ROLE_ROWS.values())), 'explicit complete geometry output scope required')
    return {'original_vertices.npy': (target_rows, 778, 3), 'original_joints.npy': (target_rows, 21, 3), 'independent_vertices.npy': (target_rows, 778, 3), 'independent_joints.npy': (target_rows, 21, 3), 'row_errors.npy': (target_rows, len(ERROR_COLUMNS))}


def read_geometry_arrays(output, identities, target_rows):
    shapes = geometry_output_shapes(target_rows)
    need(type(identities) is dict and set(identities) == set(shapes), 'exact complete safe geometry array membership required')
    result = {}
    for name, shape in shapes.items():
        identity = identities[name]
        need(type(identity) is dict and set(identity) == {'shape', 'dtype', 'bytes', 'sha256'}, 'exact typed geometry descriptor required')
        need(identity['shape'] == list(shape) and identity['dtype'] == 'float64', 'fixed complete778mesh/21joint geometry shape required')
        path = output / name
        before = stamp(path)
        need(type(identity['bytes']) is int and before[2] == identity['bytes'] and file_sha(path) == identity['sha256'], 'complete typed geometry bytes differ')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        need(array.dtype == np.float64 and array.shape == shape and stamp(path) == before, 'actual fixed geometry dtype/shape/generation differs')
        result[name] = array
    return result


def check_output_stamps(output, before):
    need(set(output.iterdir()) == set(before), 'closed geometry output membership changed while reopening')
    need(all(stamp(path) == old for path, old in before.items()), 'closed geometry output generation changed while reopening')


def produce(request, phase, output):
    resources(admission=True)
    inputs = bind_inputs(request, hash_data=True)
    output.mkdir(mode=0o700, exist_ok=False)
    owner = current_actor()
    target_rows = 2 if phase == 'qualification' else sum(ROLE_ROWS.values())
    need(phase in ['qualification', 'full'], 'explicit geometry phase required')
    write_json(output / 'started.json', {'observed_utc': utc(), 'actor': owner, 'phase': phase, 'target_rows': target_rows, 'source_sha256': request['worker_source_sha256'], 'backend_tolerance_m': BACKEND_TOLERANCE_METERS})
    completed = 0
    started = time.monotonic()
    try:
        # Real models are built only after full source/asset/input admission.
        original, original_identity = build_original(request)
        independent, independent_identity = build_independent(request)
        shapes = geometry_output_shapes(target_rows)
        outputs = {name: np.lib.format.open_memmap(output / name, mode='w+', dtype=np.float64, shape=shape) for name, shape in shapes.items()}
        membership = []
        for role in ROLE_ROWS:
            data = read_role(request, role)
            indices = range(1) if phase == 'qualification' else range(ROLE_ROWS[role])
            for index in indices:
                mano = data['mano'][index].copy()
                camera = data['K'][index].copy()
                original_xyz, original_v = original(mano, camera)
                independent_xyz, independent_v = independent(mano, camera)
                row = error_row(original_xyz, original_v, independent_xyz, independent_v, data['xyz'][index], data['verts'][index])
                need(np.isfinite(row).all(), 'nonfinite source geometry errors')
                need(max(row[-2:]) <= BACKEND_TOLERANCE_METERS, 'backend geometry disagreement exceeds fixed1e-7meter bound; diagnose without changing tolerance')
                outputs['original_vertices.npy'][completed] = original_v
                outputs['original_joints.npy'][completed] = original_xyz
                outputs['independent_vertices.npy'][completed] = independent_v
                outputs['independent_joints.npy'][completed] = independent_xyz
                outputs['row_errors.npy'][completed] = row
                membership.append([role, index])
                completed += 1
                if completed % 100 == 0 or phase == 'qualification':
                    check_stamps(inputs)
                    for array in outputs.values():
                        array.flush()
                    write_json(output / 'progress.json', {'observed_utc': utc(), 'actor': owner, 'phase': phase, 'actual_rows_written': completed, 'target_rows': target_rows, 'last_source_key': [role, index], 'preallocated_length_is_not_coverage': True, 'full_paper_complete': False}, atomic=True)
                    print(json.dumps({'phase': phase, 'actual_rows': completed, 'target_rows': target_rows, 'last_key': [role, index], 'seconds': time.monotonic() - started}), flush=True)
                if completed % 1000 == 0:
                    gc.collect()
            del data
            gc.collect()
        need(completed == target_rows, 'full source row coverage incomplete')
        for array in outputs.values():
            array.flush()
        del outputs
        gc.collect()
        for name in shapes:
            with (output / name).open('rb') as stream:
                os.fsync(stream.fileno())
        write_json(output / 'membership.json', membership)
        check_stamps(inputs)
        bind_inputs(request, hash_data=True)
        identities = {name: {'shape': list(shape), 'dtype': 'float64', 'bytes': (output / name).stat().st_size, 'sha256': file_sha(output / name)} for name, shape in shapes.items()}
        result = {'format': 'original_and_independent_freihand_geometry_producer_v1', 'observed_utc': utc(), 'actor': owner, 'phase': phase, 'actual_rows_written': completed, 'role_counts': {role: sum(key[0] == role for key in membership) for role in ROLE_ROWS}, 'source_sha256': request['worker_source_sha256'], 'original_backend': original_identity, 'independent_backend': independent_identity, 'arrays': identities, 'membership_sha256': file_sha(output / 'membership.json'), 'error_columns': ERROR_COLUMNS, 'fixed_backend_max_abs_tolerance_m': BACKEND_TOLERANCE_METERS, 'all_source_generations_unchanged': True, 'original_input_roles_and_values_preserved': True, 'source_parameter_conversion_performed': False, 'explicit_source_row_calls_per_backend': completed, 'new_optimizer_updates': 0, 'actual_gpu_actions': 0, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'seconds': time.monotonic() - started, 'full_geometry_rows_produced': phase == 'full', 'postexit_independent_reopen_complete': False, 'author_mixture_reconstructed': False, 'markers2mano_reproduced': False, 'full_paper_complete': False}
        write_json(output / 'report.json', result)
        write_json(output / 'completed.json', {'status': 'complete', 'actor': owner, 'phase': phase, 'actual_rows': completed, 'source_sha256': request['worker_source_sha256'], 'report_sha256': file_sha(output / 'report.json')})
    except BaseException as error:
        write_json(output / 'failed.json', {'observed_utc': utc(), 'actor': owner, 'phase': phase, 'actual_rows_written': completed, 'error_type': type(error).__name__, 'message': str(error)[:700], 'original_inputs_preserved': True, 'automatic_retry': False})
        raise


def reopen(request, phase, output, result_path, expected_completed_sha, expected_actor):
    resources(admission=True)
    completion = verify_producer_closure(output, phase, expected_completed_sha, expected_actor)
    inputs = bind_inputs(request, hash_data=True)
    producer = json.loads((output / 'report.json').read_text())
    need(completion['status'] == 'complete' and closed(completion['actor']), 'actual source producer closure required before reopening')
    need(file_sha(output / 'report.json') == completion['report_sha256'], 'producer report identity differs')
    need(completion['actor'] == producer['actor'] and completion['source_sha256'] == producer['source_sha256'] == request['worker_source_sha256'], 'producer/source binding differs')
    target_rows = 2 if phase == 'qualification' else sum(ROLE_ROWS.values())
    need(phase == producer['phase'] == completion['phase'] and producer['actual_rows_written'] == completion['actual_rows'] == target_rows, 'source scope/coverage differs')
    membership = json.loads((output / 'membership.json').read_text())
    expected_keys = [[role, index] for role in ROLE_ROWS for index in (range(1) if phase == 'qualification' else range(ROLE_ROWS[role]))]
    need(membership == expected_keys and file_sha(output / 'membership.json') == producer['membership_sha256'], 'original ordered source membership differs')
    output_before = {path: stamp(path) for path in output.iterdir()}
    arrays = read_geometry_arrays(output, producer['arrays'], target_rows)
    recomputed = np.empty((target_rows, len(ERROR_COLUMNS)), dtype=np.float64)
    offset = 0
    for role in ROLE_ROWS:
        data = read_role(request, role)
        for index in (range(1) if phase == 'qualification' else range(ROLE_ROWS[role])):
            recomputed[offset] = error_row(arrays['original_joints.npy'][offset], arrays['original_vertices.npy'][offset], arrays['independent_joints.npy'][offset], arrays['independent_vertices.npy'][offset], data['xyz'][index], data['verts'][index])
            offset += 1
            if offset % 1000 == 0:
                check_stamps(inputs)
        del data
        gc.collect()
    need(offset == target_rows and np.array_equal(recomputed, arrays['row_errors.npy']), 'fresh model-free full error reconstruction differs')
    need(np.isfinite(recomputed).all() and recomputed[:, -2:].max() <= BACKEND_TOLERANCE_METERS, 'source geometry backend bound differs')
    summaries = {}
    for role in [*ROLE_ROWS, 'all']:
        values = recomputed if role == 'all' else recomputed[np.array([key[0] == role for key in membership])]
        summaries[role] = {'rows': len(values), **{name: {'mean': float(values[:, col].mean()), 'min': float(values[:, col].min()), 'max': float(values[:, col].max())} for col, name in enumerate(ERROR_COLUMNS)}}
    check_stamps(inputs)
    bind_inputs(request, hash_data=True)
    check_output_stamps(output, output_before)
    need('torch' not in sys.modules and 'chumpy' not in sys.modules, 'model-free reader imported a model provider')
    write_json(result_path, {'format': 'independent_model_free_freihand_complete_geometry_reopen_v1', 'observed_utc': utc(), 'actor': current_actor(), 'phase': phase, 'producer_actor': completion['actor'], 'producer_closed': True, 'source_sha256': request['worker_source_sha256'], 'producer_report_sha256': completion['report_sha256'], 'complete_rows': target_rows, 'all_ordered_source_keys_exact': True, 'all_complete_typed_array_SHA256_exact': True, 'all_raw_per_row_error_values_exact': True, 'source_and_output_generations_unchanged': True, 'summaries_meters': summaries, 'new_model_constructions': 0, 'new_optimizer_updates': 0, 'actual_gpu_actions': 0, 'full_source_geometry_reopened': phase == 'full', 'full_paper_complete': False})


def verify_producer_closure(output, phase, expected_sha256, expected_actor):
    need(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_sha256) is not None, 'externally captured actual producer completion SHA required')
    need(type(expected_actor) is dict and set(expected_actor) == {'pid', 'start_ticks'} and all(type(value) is int and value > 0 for value in expected_actor.values()), 'actual independently captured producer handle required')
    need(file_sha(output / 'completed.json') == expected_sha256, 'producer completion changed after actual owned exit')
    completion = json.loads((output / 'completed.json').read_text())
    need(completion['status'] == 'complete' and completion['phase'] == phase and completion['actor'] == expected_actor and closed(expected_actor), 'source producer scope/owner closure differs')
    return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--phase', required=True, choices=['qualification', 'full'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--reopen-result')
    parser.add_argument('--producer-completed-sha')
    parser.add_argument('--producer-pid', type=int)
    parser.add_argument('--producer-start', type=int)
    args = parser.parse_args()
    os.nice(10)
    request = json.loads(Path(args.request).read_text())
    need(file_sha(Path(__file__)) == request['worker_source_sha256'], 'frozen geometry worker identity differs')
    if args.reopen_result is not None:
        reopen(request, args.phase, Path(args.output), Path(args.reopen_result), args.producer_completed_sha, {'pid': args.producer_pid, 'start_ticks': args.producer_start})
    else:
        produce(request, args.phase, Path(args.output))


if __name__ == '__main__':
    main()
