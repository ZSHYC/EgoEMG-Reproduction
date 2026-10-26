"""Training boundaries plus opt-in official-model finite-step integration."""
import importlib.util
import json
import os
import pickle
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

from egoemg_repro.train import (load_config, source_identity, build_model, align_batch, make_optimizer, validation_best, data_identity, augment_training_batch,
    validate_upstream, reproduction_identity, assert_compatible_provenance,
    checkpoint_with_best, selected_best_checkpoint, publish_checkpoints, save_checkpoint, prepare_output,
)


class TrainingConfigTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable')
    def test_dinov2_initialization_policy_and_resume_identity_are_not_imagenet(self):
        from egoemg_repro.train import initialization_identity
        config = {'model_kind': 'vision', 'seed': 42, 'vision_pretrained': True,
                  'upstream_experiment': 'fusion/vision_vit_small', 'vision_backbone_type': 'vit_base'}
        identity = initialization_identity(config, {})
        self.assertEqual(identity['policy'], 'DINOv2 backbone')
        self.assertEqual(identity['seed_before_model'], 42)
        identity['official_builder'] = {'pretrained_weights': {'family': 'DINOv2', 'source_type': 'hf-hub'}}
        restored = initialization_identity(config, {}, loaded={'provenance': {'initialization': identity}})
        self.assertEqual(restored, identity)
        self.assertEqual(initialization_identity(dict(config, vision_pretrained=False), {})['policy'], 'random')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable')
    def test_clean_vision_initializer_is_not_claimed_as_a_fresh_pretrained_load(self):
        import torch
        from egoemg_repro.train import initialization_identity
        state = checkpoint_fixture()
        state['provenance'].update(model_kind='vision', data={'sha256': 'same'}, visual_data={'sha256': 'crops'},
                                   partition_sha256='split', normalization_sha256='norm')
        state = checkpoint_with_best(state)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'selected-vision.pt'
            torch.save(state, path)
            config = {'model_kind': 'vision', 'seed': 42, 'vision_pretrained': True,
                      'upstream_experiment': 'fusion/vision_vit_small'}
            result = initialization_identity(config, state['provenance'], vision_checkpoint=path)
            self.assertEqual(result['policy'], 'clean vision checkpoint')
            self.assertEqual(result['vision']['model_kind'], 'vision')
            self.assertEqual(result['vision']['selection'], 'complete independent validation')

    def test_config_exposes_released_recipe_and_explicit_memory_adjustment(self):
        config = load_config(Path(__file__).parents[1] / 'configs/emgformer_s.json')
        self.assertGreater(config['batch_size'], 0)
        self.assertEqual(config['official_batch_size'], 500)
        self.assertEqual(config['epochs'], 200)
        self.assertEqual(config['lr'], 5e-5)
        self.assertEqual(config['eta_min'], 1e-5)

    def test_partial_validation_best_cannot_compete_with_full_validation(self):
        checkpoint = {'best_val_mae_deg': 1.0, 'validation_bounded': True}
        self.assertEqual(validation_best(checkpoint, bounded=True), 1.0)
        self.assertEqual(validation_best(checkpoint, bounded=False), float('inf'))

    def test_data_identity_covers_metadata_and_verified_payload_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'metadata.npz').write_bytes(b'labels-a')
            (root / 'emg.dat').write_bytes(b'data')
            record = {'bytes': 4, 'sha256': 'a' * 64, 'verified_at': 'first'}
            (root / 'acquisition_ledger.json').write_text(json.dumps({'files': {'emg.dat': record}}))
            first = data_identity(root)['sha256']
            record['verified_at'] = 'later'
            (root / 'acquisition_ledger.json').write_text(json.dumps({'files': {'emg.dat': record}}))
            self.assertEqual(first, data_identity(root)['sha256'])
            record['sha256'] = 'b' * 64
            (root / 'acquisition_ledger.json').write_text(json.dumps({'files': {'emg.dat': record}}))
            self.assertNotEqual(first, data_identity(root)['sha256'])
            second = data_identity(root)['sha256']
            (root / 'metadata.npz').write_bytes(b'labels-b')
            self.assertNotEqual(second, data_identity(root)['sha256'])

    def test_source_identity_changes_when_model_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'egoemg').mkdir()
            source = root / 'egoemg' / 'model.py'
            source.write_text('a')
            first = source_identity(root)['sha256']
            source.write_text('b')
            self.assertNotEqual(first, source_identity(root)['sha256'])


def serialize_fixture(state, path):
    with Path(path).open('wb') as stream:
        pickle.dump(state, stream)


def deserialize_fixture(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def checkpoint_fixture(value=1., score=3., step=1):
    return {'model': {'weight': [value]}, 'optimizer': {'momentum': [0.1]},
            'scheduler': {'epoch': step}, 'rng': {}, 'epoch': step, 'next_batch': 0,
            'global_step': step, 'best_val_mae_deg': score, 'validation_bounded': False,
            'provenance': {'source': {'sha256': 'official'},
                           'reproduction': {'sha256': 'adapter', 'revision': 'first', 'runtime_files_sha256': {'egoemg_repro/train.py': 'same-code'}}},
            'run_id': 'fixture-run'}


def ample_fixture_disk(_path):
    """Successful-write fixtures own capacity; host free space is irrelevant."""
    return SimpleNamespace(total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3)


class CheckpointSafetyTests(unittest.TestCase):
    @patch('egoemg_repro.train.shutil.disk_usage', ample_fixture_disk)
    def test_interrupted_best_publication_recovers_matching_weights_from_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            old = checkpoint_with_best(checkpoint_fixture(value=1., score=3., step=1))
            publish_checkpoints(output, old, improved=True, serializer=serialize_fixture)
            new = checkpoint_with_best(checkpoint_fixture(value=2., score=2., step=2))
            def interrupt_best(state, path):
                if Path(path).name == 'best.pt.tmp':
                    raise RuntimeError('simulated interruption')
                serialize_fixture(state, path)
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                publish_checkpoints(output, new, improved=True, serializer=interrupt_best)
            self.assertEqual(deserialize_fixture(output / 'best.pt')['model']['weight'], [1.])
            recovered = deserialize_fixture(output / 'last.pt')
            best = selected_best_checkpoint(recovered, recovered['provenance'], bounded=False)
            self.assertEqual(best['model']['weight'], [2.])
            self.assertEqual(best['best_val_mae_deg'], 2.)
            # Continuing without improvement must preserve the recovered best.
            latest = checkpoint_fixture(value=4., score=2., step=3)
            continued = checkpoint_with_best(latest, best)
            self.assertEqual(selected_best_checkpoint(continued)['model']['weight'], [2.])

    @patch('egoemg_repro.train.shutil.disk_usage', ample_fixture_disk)
    def test_resume_into_new_output_preserves_historical_best_without_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'relocated'
            first = checkpoint_with_best(checkpoint_fixture(value=1., score=2., step=1))
            resumed = checkpoint_fixture(value=5., score=2., step=2)
            state = checkpoint_with_best(resumed, selected_best_checkpoint(first))
            self.assertEqual(prepare_output(output, state, loader=deserialize_fixture), 'fixture-run')
            publish_checkpoints(output, state, improved=False, serializer=serialize_fixture)
            published = deserialize_fixture(output / 'best.pt')
            self.assertEqual(published['model']['weight'], [1.])
            self.assertEqual(published['optimizer'], first['optimizer'])
            self.assertEqual(selected_best_checkpoint(deserialize_fixture(output / 'last.pt'))['model']['weight'], [1.])

    def test_best_optimizer_snapshot_does_not_alias_resumed_training_state(self):
        first = checkpoint_with_best(checkpoint_fixture())
        historical = selected_best_checkpoint(first)
        resumed = checkpoint_with_best(checkpoint_fixture(value=5., step=2), historical)
        historical['optimizer']['momentum'][0] = 99.
        self.assertEqual(selected_best_checkpoint(resumed)['optimizer']['momentum'], [0.1])

    def test_best_score_or_provenance_disagreement_is_rejected(self):
        state = checkpoint_with_best(checkpoint_fixture())
        state['best_val_mae_deg'] = 0.5
        with self.assertRaisesRegex(ValueError, 'score'):
            selected_best_checkpoint(state)
        state = checkpoint_with_best(checkpoint_fixture())
        state['best']['provenance']['source']['sha256'] = 'unrelated'
        with self.assertRaisesRegex(ValueError, 'source'):
            selected_best_checkpoint(state)

    def test_low_disk_rejects_before_serialization_and_preserves_checkpoint(self):
        from egoemg_repro.download import MIN_FREE_BYTES
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            path.write_bytes(b'known-good')
            serialized = []
            def serializer(state, temporary):
                serialized.append(True)
                serialize_fixture(state, temporary)
            with patch('egoemg_repro.train.shutil.disk_usage', return_value=SimpleNamespace(free=MIN_FREE_BYTES)):
                with self.assertRaisesRegex(RuntimeError, 'reserve'):
                    save_checkpoint(path, checkpoint_with_best(checkpoint_fixture()), serializer=serializer)
            self.assertEqual(path.read_bytes(), b'known-good')
            self.assertEqual(serialized, [])
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_postserialization_capacity_check_preserves_previous_checkpoint(self):
        from egoemg_repro.download import MIN_FREE_BYTES
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            path.write_bytes(b'known-good')
            free_states = [SimpleNamespace(free=MIN_FREE_BYTES + 100 * 1024**2),
                           SimpleNamespace(free=MIN_FREE_BYTES - 1)]
            with patch('egoemg_repro.train.shutil.disk_usage', side_effect=free_states):
                with self.assertRaisesRegex(RuntimeError, 'reserve'):
                    save_checkpoint(path, checkpoint_with_best(checkpoint_fixture()), serializer=serialize_fixture)
            self.assertEqual(path.read_bytes(), b'known-good')
            self.assertFalse(Path(str(path) + '.tmp').exists())

    @patch('egoemg_repro.train.shutil.disk_usage', ample_fixture_disk)
    def test_failed_serialization_keeps_previous_checkpoint_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            path.write_bytes(b'known-good')
            def partial_write(state, temporary):
                Path(temporary).write_bytes(b'incomplete')
                raise RuntimeError('serialization interrupted')
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                save_checkpoint(path, checkpoint_with_best(checkpoint_fixture()), serializer=partial_write)
            self.assertEqual(path.read_bytes(), b'known-good')
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_fresh_run_and_unrelated_resume_cannot_replace_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'config.json').write_text('{"existing": true}')
            with self.assertRaisesRegex(ValueError, 'output'):
                prepare_output(output)
            (output / 'run.json').write_text(json.dumps({'run_id': 'another-run'}))
            with self.assertRaisesRegex(ValueError, 'run'):
                prepare_output(output, checkpoint_with_best(checkpoint_fixture()), loader=deserialize_fixture)
            self.assertEqual((output / 'config.json').read_text(), '{"existing": true}')

    @patch('egoemg_repro.train.shutil.disk_usage', ample_fixture_disk)
    def test_same_output_resume_checks_best_artifact_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            checkpoint = checkpoint_with_best(checkpoint_fixture())
            (output / 'run.json').write_text(json.dumps({'run_id': 'fixture-run'}))
            (output / 'provenance.json').write_text(json.dumps(checkpoint['provenance']))
            publish_checkpoints(output, checkpoint, improved=True, serializer=serialize_fixture)
            self.assertEqual(prepare_output(output, checkpoint, loader=deserialize_fixture), 'fixture-run')
            unrelated = checkpoint_with_best(checkpoint_fixture())
            unrelated['provenance']['source']['sha256'] = 'other-source'
            serialize_fixture(unrelated, output / 'best.pt')
            with self.assertRaisesRegex(ValueError, 'source'):
                prepare_output(output, checkpoint, loader=deserialize_fixture)


class RuntimeIdentityTests(unittest.TestCase):
    def test_wrong_upstream_revision_is_rejected_before_model_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / '.git' / 'HEAD').write_text('1' * 40 + '\n')
            with self.assertRaisesRegex(ValueError, 'revision'):
                build_model(root, {'seed': 42})

    def test_dirty_approved_upstream_is_rejected(self):
        from egoemg_repro import UPSTREAM_COMMIT
        responses = [subprocess.CompletedProcess([], 0, UPSTREAM_COMMIT + '\n', ''),
                     subprocess.CompletedProcess([], 0, ' M egoemg/models/modules/emgformer.py\n', '')]
        with patch('egoemg_repro.train.subprocess.run', side_effect=responses):
            with self.assertRaisesRegex(ValueError, 'clean'):
                validate_upstream('/unused-fixture')

    def test_changed_reproduction_runtime_code_prevents_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'egoemg_repro').mkdir()
            (root / 'configs').mkdir()
            for name in ('__init__', 'train', 'data', 'partitions', 'metrics', 'download'):
                (root / 'egoemg_repro' / (name + '.py')).write_text('# runtime')
            (root / 'configs' / 'emgformer_s.json').write_text('{}')
            first = reproduction_identity(root)
            provenance = {'reproduction': first}
            (root / 'egoemg_repro' / 'train.py').write_text('# changed loss or evaluation')
            second = reproduction_identity(root)
            self.assertNotEqual(first['sha256'], second['sha256'])
            with self.assertRaisesRegex(ValueError, 'reproduction'):
                assert_compatible_provenance(provenance, {'reproduction': second})
            # Commit metadata may advance without changing executed code.
            same_contents = dict(first, revision='a-new-docs-commit')
            assert_compatible_provenance(provenance, {'reproduction': same_contents})


@unittest.skipUnless(importlib.util.find_spec('torch') and os.getenv('EGOEMG_UPSTREAM'),
                     'set EGOEMG_UPSTREAM in a torch/hydra environment for official integration')
class OfficialTrainingTests(unittest.TestCase):
    def test_official_rotation_changes_channels_without_moving_time(self):
        import torch
        import numpy as np
        config = load_config(Path(__file__).parents[1] / 'configs/emgformer_s.json')
        # Import the official package via the same builder used in a real run.
        build_model(os.environ['EGOEMG_UPSTREAM'], config)
        np.random.seed(42)  # first official rotation is +1
        emg = torch.arange(8).view(1, 8, 1).expand(1, 8, 3).float()
        batch = augment_training_batch({'emg': emg}, config)
        self.assertEqual(batch['emg'][0, :, 0].tolist(), [7., 0., 1., 2., 3., 4., 5., 6.])
        self.assertTrue(torch.equal(batch['emg'][:, :, 0], batch['emg'][:, :, 2]))

    def test_official_one_step_is_finite_and_reload_is_identical(self):
        import torch
        config = load_config(Path(__file__).parents[1] / 'configs/emgformer_s.json')
        config['precision'] = 'float32'
        model = build_model(os.environ['EGOEMG_UPSTREAM'], config)
        torch.manual_seed(12)
        width = 12000
        batch = {'emg': torch.randn(1, 8, width), 'joint_angles': torch.zeros(1, 22, width),
                 'label_valid_mask': torch.ones(1, width, dtype=torch.bool),
                 'angle_valid_mask': torch.ones(1, 22, width, dtype=torch.bool)}
        batch['angle_valid_mask'][:, 20:] = False
        # A wrist label remains independent of finger validity.
        batch['label_valid_mask'][:, 600] = False
        batch['angle_valid_mask'][:, :20, 600] = False
        batch['angle_valid_mask'][:, 20:, 600] = True
        optimizer, scheduler = make_optimizer(model, config)
        from egoemg_repro.train import OfficialLoss
        prediction, target, valid = align_batch(model, batch)
        self.assertEqual(prediction.shape, (1, 22, 11490))
        self.assertEqual(int(valid[:, 20:].sum()), 2)
        self.assertTrue(valid[0, 20, 90])
        loss, terms = OfficialLoss()(prediction, target, valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            before = align_batch(model, batch)[0]
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'check.pt'
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, file)
            restored = build_model(os.environ['EGOEMG_UPSTREAM'], config).eval()
            state = torch.load(file, map_location='cpu', weights_only=False)
            restored.load_state_dict(state['model'])
            with torch.no_grad():
                after = align_batch(restored, batch)[0]
            self.assertTrue(torch.equal(before, after))
            self.assertGreater(len(state['optimizer']['state']), 0)


if __name__ == '__main__':
    unittest.main()


class VisualRoutingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional Torch runtime required')
    def test_visual_evaluation_is_center_only_with_matched_population(self):
        import torch
        from egoemg_repro.train import evaluate
        class ImageMean(torch.nn.Module):
            def forward(self, batch):
                return batch['vision_img'].mean((1, 2, 3))[:, None, None].expand(-1, 22, 1)
        batch = dict(emg=torch.empty(2, 8, 0), vision_img=torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)]),
                     joint_angles=torch.zeros(2,22,1), angle_valid_mask=torch.ones(2,22,1,dtype=torch.bool),
                     label_valid_mask=torch.ones(2,1,dtype=torch.bool), vision_valid_mask=torch.tensor([[True],[False]]),
                     hand=['left', 'right'], split=['user', 'both'], subject=['a', 'b'])
        report = evaluate(ImageMean(), [batch], 'cpu', {'precision': 'float32', 'model_kind': 'vision'})
        self.assertEqual(set(report['sampling']), {'center'})
        self.assertEqual(report['paired_population']['samples']['missing_crop'], 1)
        self.assertEqual(report['paired_population']['vision_valid']['overall']['mae_deg'], 0.)
        self.assertAlmostEqual(report['sampling']['center']['overall']['mae_deg'], 0.5 * 180 / 3.141592653589793)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional Torch runtime required')
    def test_optimizer_preserves_selected_adam_recipe(self):
        import torch
        model = torch.nn.Linear(2, 1)
        config = {'optimizer_class': 'Adam', 'lr': 0.0006, 'weight_decay': 0.0001,
                  'epochs': 150, 'eta_min': 1e-5}
        optimizer, scheduler = make_optimizer(model, config)
        self.assertIs(type(optimizer), torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.0006)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.0001)
        config['optimizer_class'] = 'AdamW'
        optimizer, _ = make_optimizer(model, config)
        self.assertIs(type(optimizer), torch.optim.AdamW)

    def test_visual_source_identity_tracks_selected_recipe_and_visual_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'egoemg_repro').mkdir()
            (root / 'configs').mkdir()
            for name in ('__init__', 'train', 'data', 'partitions', 'metrics', 'download', 'visual', 'crops'):
                (root / 'egoemg_repro' / (name + '.py')).write_text('# runtime')
            recipe = root / 'configs' / 'vision.json'
            recipe.write_text('{"model_kind":"vision"}')
            first = reproduction_identity(root, model_kind='vision', config_path=recipe)
            (root / 'egoemg_repro' / 'visual.py').write_text('# changed center alignment')
            second = reproduction_identity(root, model_kind='vision', config_path=recipe)
            self.assertNotEqual(first['sha256'], second['sha256'])
            recipe.write_text('{"model_kind":"vision","epochs":2}')
            self.assertNotEqual(second['sha256'], reproduction_identity(root, model_kind='vision', config_path=recipe)['sha256'])

    @unittest.skipUnless(os.getenv('EGOEMG_UPSTREAM'), 'set EGOEMG_UPSTREAM for official FK')
    def test_all_invalid_label_loss_is_finite_graph_zero(self):
        import torch
        from egoemg_repro.train import OfficialLoss, build_model
        build_model(os.environ['EGOEMG_UPSTREAM'],
                    load_config(Path(__file__).parents[1] / 'configs/emgformer_s.json'))
        prediction = torch.ones(1, 22, 1, requires_grad=True)
        loss, terms = OfficialLoss()(prediction, torch.zeros_like(prediction), torch.zeros_like(prediction,dtype=torch.bool))
        self.assertEqual(float(loss), 0.)
        self.assertEqual(terms['valid_angle_count'], 0)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(prediction.grad)), 0)


class VisualProvenanceTests(unittest.TestCase):
    def test_crop_receipt_and_alignment_payload_identify_visual_inputs(self):
        from egoemg_repro.train import visual_data_identity
        from test_data import make_store
        from test_crops import add_video_indices
        import hashlib
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root, crops = Path(tmp) / 'data', Path(tmp) / 'crops'
            root.mkdir(); crops.mkdir()
            make_store(root)
            add_video_indices(root, np.arange(47))
            payload = crops / 'episode_0.data.mdb'
            payload.write_bytes(b'fake-lmdb-payload')
            record = dict(bytes=payload.stat().st_size, sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
                          checksum_basis='local_transfer')
            (crops / 'acquisition_ledger.json').write_text(json.dumps({'files': {payload.name: record}}))
            first = visual_data_identity(root, crops)
            self.assertEqual(first['alignment']['field'], 'image_webcam_frame_index')
            self.assertEqual(first['payloads'][payload.name]['sha256'], record['sha256'])
            (root / 'video.dat').write_bytes((np.arange(47, dtype=np.int32) + 1).tobytes())
            self.assertNotEqual(first['sha256'], visual_data_identity(root, crops)['sha256'])
            (crops / 'acquisition_ledger.json').unlink()
            with self.assertRaisesRegex(ValueError, 'receipt'):
                visual_data_identity(root, crops)

    def test_initializer_requires_full_validation_selected_branch_and_matching_data(self):
        from egoemg_repro.train import validate_initializer
        state = checkpoint_fixture()
        state['provenance'].update(model_kind='emg', data={'sha256': 'same'},
                                   partition_sha256='split', normalization_sha256='norm')
        state = checkpoint_with_best(state)
        current = dict(state['provenance'])
        validate_initializer(state, 'emg', current)
        with self.assertRaisesRegex(ValueError, 'branch'):
            validate_initializer(state, 'vision', current)
        changed_code = dict(current, reproduction=dict(current['reproduction'],
            runtime_files_sha256={'egoemg_repro/train.py': 'different-loss'}))
        with self.assertRaisesRegex(ValueError, 'reproduction'):
            validate_initializer(state, 'emg', changed_code)
        changed = dict(current, normalization_sha256='different')
        with self.assertRaisesRegex(ValueError, 'normalization'):
            validate_initializer(state, 'emg', changed)
        state['validation_bounded'] = state['best']['validation_bounded'] = True
        with self.assertRaisesRegex(ValueError, 'validation'):
            validate_initializer(state, 'emg', current)

    def test_initializer_rejects_last_weights_that_are_not_the_selected_best(self):
        from egoemg_repro.train import validate_initializer
        best = checkpoint_fixture(step=1)
        best['provenance'].update(model_kind='emg', data={'sha256':'same'},
                                  partition_sha256='split', normalization_sha256='norm')
        latest = dict(best, model={'weight': [9.]}, global_step=2)
        state = checkpoint_with_best(latest, best)
        with self.assertRaisesRegex(ValueError, 'best'):
            validate_initializer(state, 'emg', best['provenance'])


@unittest.skipUnless(os.getenv('EGOEMG_UPSTREAM'), 'set EGOEMG_UPSTREAM for the real visual training driver')
class VisualDriverTests(unittest.TestCase):
    @patch('egoemg_repro.train.shutil.disk_usage', ample_fixture_disk)
    def test_real_vision_bounded_training_relocated_resume_and_repeat_evaluation(self):
        import hashlib
        import numpy as np
        from argparse import Namespace
        from egoemg_repro.train import run
        from test_data import make_store
        from test_crops import add_video_indices, make_crops
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data, crops, output = base / 'data', base / 'crops', base / 'run'
            data.mkdir(); crops.mkdir()
            splits = np.r_[np.zeros(36000, dtype=np.int8), np.ones(12000, dtype=np.int8)]
            make_store(data, splits=splits, lengths=(48000,))
            add_video_indices(data, np.full(48000, 17))
            make_crops(crops)
            payload = crops / 'episode_0.lmdb'
            record = dict(bytes=payload.stat().st_size, sha256=hashlib.sha256(payload.read_bytes()).hexdigest(), checksum_basis='fixture')
            (crops / 'acquisition_ledger.json').write_text(json.dumps({'files': {payload.name: record}}))
            config = load_config(Path(__file__).parents[1] / 'configs/resnet18.json')
            config.update(vision_pretrained=False, precision='float32', batch_size=2, num_workers=0,
                          epochs=2, train_stride=12000, bounded_eval_batches=1)
            recipe = base / 'vision.json'
            recipe.write_text(json.dumps(config))
            args = Namespace(data=str(data), crops=str(crops), upstream=os.environ['EGOEMG_UPSTREAM'],
                             output=str(output), config=str(recipe), max_steps=1, epochs=1,
                             batch_size=None, num_workers=None, resume=None, evaluate=None, device='cpu',
                             vision_checkpoint=None, emg_checkpoint=None)
            result = run(args)
            self.assertEqual(result['global_step'], 1)
            self.assertFalse(result['test_evaluated'])
            self.assertEqual(set(result['validation']['sampling']), {'center'})
            self.assertEqual(result['validation']['paired_population']['samples']['missing_crop'], 1)
            relocated = base / 'relocated'
            self.assertEqual(load_config(output / 'recipe_source.json')['epochs'], 2)
            self.assertEqual(load_config(output / 'config.json')['epochs'], 1)
            args.config, args.epochs = str(output / 'config.json'), None
            args.resume, args.output, args.max_steps = str(output / 'last.pt'), str(relocated), 2
            result = run(args)
            self.assertEqual(result['global_step'], 2)
            self.assertEqual(load_config(relocated / 'recipe_source.json')['epochs'], 2)
            self.assertEqual(load_config(relocated / 'config.json')['epochs'], 1)
            self.assertEqual(load_config(relocated / 'provenance.json')['initialization'],
                             load_config(output / 'provenance.json')['initialization'])
            changed = load_config(relocated / 'config.json')
            changed['lr'] *= 2
            bad_recipe = base / 'changed-config.json'
            bad_recipe.write_text(json.dumps(changed))
            args.config, args.output = str(bad_recipe), str(base / 'must-reject')
            with self.assertRaisesRegex(ValueError, 'config'):
                run(args)
            args.config, args.output = str(output / 'config.json'), str(relocated)
            args.resume, args.evaluate, args.max_steps = None, str(relocated / 'last.pt'), 1
            first = run(args)
            second = run(args)
            self.assertEqual(first, second)
            # Epoch exhaustion must never turn bounded integration into full test.
            args.evaluate, args.output, args.max_steps = None, str(base / 'exhausted'), 100
            exhausted = run(args)
            self.assertFalse(exhausted['test_evaluated'])
            self.assertEqual(exhausted['status'], 'bounded_integration_exhausted')
