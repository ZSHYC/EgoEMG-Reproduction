from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('hot3d_intake', os.environ.get('EGOEMG_HOT3D_RECEIVER_FILE', str(Path(__file__).with_name('hot3d-full-train-receiver-20261102.py'))))
receiver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receiver)


class FullIntakeAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads(Path(os.environ.get('EGOEMG_HOT3D_MANIFEST_FILE', 'assets_hot3d_train_manifest.json')).read_text())

    def test_source_revision_change_is_rejected_before_any_download(self):
        changed = copy.deepcopy(self.inventory)
        changed['files'][5]['url'] = changed['files'][5]['url'].replace(changed['dataset_revision'], 'main')
        with self.assertRaises(ValueError):
            receiver.checked_assets(changed)

    def test_duplicate_path_cannot_replace_another_required_clip(self):
        changed = copy.deepcopy(self.inventory)
        changed['files'][-1] = copy.deepcopy(changed['files'][0])
        with self.assertRaises(ValueError):
            receiver.checked_assets(changed)

    def test_exceeding_real_available_capacity_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retained = root / 'existing.bin'
            retained.write_bytes(b'retained')
            free = shutil.disk_usage(root).free
            with self.assertRaises(RuntimeError):
                receiver.verify_capacity(root, remaining=1, reserve=free)
            self.assertEqual(retained.read_bytes(), b'retained')
            self.assertEqual({p.name for p in root.iterdir()}, {'existing.bin'})

    def test_corrupt_payload_is_rejected_and_preserved_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / 'clip-000001.tar'
            original = b'original incomplete transport bytes'
            payload.write_bytes(original)
            expected = {'expected_bytes': len(original), 'lfs_sha256': hashlib.sha256(b'x' * len(original)).hexdigest()}
            with self.assertRaises(ValueError):
                receiver.verify_payload(payload, expected)
            self.assertEqual(payload.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
