from pathlib import Path
import importlib.util
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import numpy as np

BASE = Path(os.environ.get('EGOEMG_FREIHAND_GEOMETRY_CHECK_ROOT', 'tmp/freihand-geometry-qualification-20261102'))
WORKER = Path(os.environ.get('EGOEMG_FREIHAND_GEOMETRY_CHECK_WORKER', str(Path(__file__).with_name('freihand-official-geometry-worker-20261102.py'))))
spec = importlib.util.spec_from_file_location('freihand_geometry', WORKER)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


class OriginalSourceContract(unittest.TestCase):
    def request(self):
        return {'public_root': str(BASE / 'official'), 'licensed_root': str(BASE / 'licensed-mano-source')}

    def test_changed_public_geometry_source_rejected_without_ML_import(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for path in (BASE / 'official').glob('*.py'):
                shutil.copyfile(path, target / path.name)
            shutil.copyfile(BASE / 'official' / 'freihand-README.md', target / 'freihand-README.md')
            path = target / 'model.py'
            path.write_bytes(path.read_bytes() + b'\n# changed binding\n')
            request = self.request()
            request['public_root'] = str(target)
            with self.assertRaises(ValueError):
                geometry.check_sources(request)
            self.assertNotIn('torch', sys.modules)

    def test_same_length_corrupted_registered_model_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'licensed'
            shutil.copytree(BASE / 'licensed-mano-source', target)
            path = target / 'models' / 'MANO_RIGHT.pkl'
            body = bytearray(path.read_bytes())
            body[-1] ^= 1
            path.write_bytes(body)
            request = self.request()
            request['licensed_root'] = str(target)
            with self.assertRaises(ValueError):
                geometry.check_sources(request)
            self.assertEqual(path.read_bytes(), body)

    def test_both_model_builders_reject_bad_sources_before_ML_import(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request()
            request['public_root'] = directory
            for builder in [geometry.build_original, geometry.build_independent]:
                with self.subTest(builder=builder.__name__):
                    with self.assertRaises(ValueError):
                        builder(request)
                    self.assertNotIn('torch', sys.modules)

    def test_full21_joint_order_and_exact_fingertip_vertices(self):
        vertices = np.arange(778 * 3, dtype=np.float64).reshape(778, 3)
        regressed = vertices[100:116].copy()
        got = geometry.assemble_joints(vertices, regressed)
        self.assertIsInstance(got, np.ndarray)
        wanted = vertices[[100, 113, 114, 115, 744, 101, 102, 103, 320, 104, 105, 106, 443, 110, 111, 112, 555, 107, 108, 109, 672]]
        np.testing.assert_array_equal(got, wanted)

    def test_UV_scale_mean_focal_and_posed_joint9_translation(self):
        mano = np.zeros((1, 61), dtype=np.float64)
        mano[0, 58:] = [11, 12, 2]
        camera = np.array([[2., 0., 10.], [0., 4., 10.], [0., 0., 1.]])
        joints = np.zeros((21, 3), dtype=np.float64)
        joints[9] = [5., 6., 7.]
        got = geometry.camera_translation(mano, camera, joints)
        self.assertIsInstance(got, np.ndarray)
        np.testing.assert_array_equal(got, [-4.5, -5., -5.5])
        np.testing.assert_array_equal(joints[9] + got, [.5, 1., 1.5])

    def test_nonpositive_weak_perspective_scale_is_not_repaired(self):
        for scale in [0., -1.]:
            with self.subTest(scale=scale):
                mano = np.zeros((1, 61), dtype=np.float64)
                mano[0, 60] = scale
                with self.assertRaises(ValueError):
                    geometry.validate_row(mano, np.eye(3, dtype=np.float64))
                self.assertEqual(mano[0, 60], scale)

    def test_flattened_source_row_is_not_silently_reshaped(self):
        mano = np.zeros(61, dtype=np.float64)
        mano[60] = 2.
        with self.assertRaises(ValueError):
            geometry.validate_row(mano, np.eye(3, dtype=np.float64))

    def test_nonfinite_camera_rejected(self):
        mano = np.zeros((1, 61), dtype=np.float64)
        mano[0, 60] = 2.
        camera = np.eye(3, dtype=np.float64)
        camera[0, 0] = np.nan
        with self.assertRaises(ValueError):
            geometry.validate_row(mano, camera)

    def test_reader_requires_externally_captured_completion_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                geometry.verify_producer_closure(Path(directory), 'qualification', None, {'pid': 99999999, 'start_ticks': 1})

    def test_resealed_scope_cannot_replace_captured_producer_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = {'pid': 99999999, 'start_ticks': 1}
            completion = {'status': 'complete', 'phase': 'qualification', 'actor': owner, 'actual_rows': 2, 'source_sha256': 'a' * 64, 'report_sha256': 'b' * 64}
            path = root / 'completed.json'
            path.write_text(json.dumps(completion) + '\n')
            captured = hashlib.sha256(path.read_bytes()).hexdigest()
            completion.update(phase='full', actual_rows=36520)
            path.write_text(json.dumps(completion) + '\n')
            with self.assertRaises(ValueError):
                geometry.verify_producer_closure(root, 'full', captured, owner)

    def output_packet(self, directory, *, short_mesh=False):
        shapes = {'original_vertices.npy': (2, 1 if short_mesh else 778, 3),
                  'original_joints.npy': (2, 21, 3), 'independent_vertices.npy': (2, 778, 3),
                  'independent_joints.npy': (2, 21, 3), 'row_errors.npy': (2, 6)}
        manifest = {}
        for name, shape in shapes.items():
            path = directory / name
            np.save(path, np.zeros(shape, dtype=np.float64), allow_pickle=False)
            manifest[name] = {'shape': list(shape), 'dtype': 'float64', 'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        return manifest

    def test_resealed_short_mesh_cannot_broadcast_as_complete778_vertices(self):
        reader = getattr(geometry, 'read_geometry_arrays', None)
        self.assertTrue(callable(reader), 'geometry reader needs complete fixed output shapes')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.output_packet(root, short_mesh=True)
            with self.assertRaises(ValueError):
                reader(root, manifest, 2)

    def test_geometry_receipt_cannot_add_traversing_array_member(self):
        reader = getattr(geometry, 'read_geometry_arrays', None)
        self.assertTrue(callable(reader), 'geometry reader needs exact safe array membership')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.output_packet(root)
            manifest['../outside.npy'] = dict(manifest['original_joints.npy'])
            with self.assertRaises(ValueError):
                reader(root, manifest, 2)

    def test_output_membership_addition_is_rejected_after_reader_pass(self):
        check = getattr(geometry, 'check_output_stamps', None)
        self.assertTrue(callable(check), 'closed output membership needs independent revalidation')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'original_vertices.npy'
            path.write_bytes(b'original closed output')
            before = {path: geometry.stamp(path)}
            (root / 'extra.npy').write_bytes(b'late member')
            with self.assertRaises(ValueError):
                check(root, before)


if __name__ == '__main__':
    unittest.main()
