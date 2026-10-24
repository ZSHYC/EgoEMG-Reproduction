"""Aggregate real angle observations, never average unequal group means."""
import math
import unittest
import numpy as np

from egoemg_repro.metrics import AngleAccumulator


class MetricTests(unittest.TestCase):
    def test_micro_average_uses_valid_angle_count_and_converts_once(self):
        meter = AngleAccumulator()
        meter.update(np.array([[1., 100.], [2., 100.]]), np.zeros((2, 2)),
                     np.array([[True, False], [True, False]]), hand='left', split='user', user='a')
        meter.update(np.array([[3.]]), np.zeros((1, 1)), np.ones((1, 1), bool),
                     hand='right', split='both', user='b')
        report = meter.report()
        self.assertEqual(report['overall']['valid_angle_count'], 3)
        self.assertAlmostEqual(report['overall']['absolute_error_sum_rad'], 6.)
        self.assertAlmostEqual(report['overall']['mae_deg'], 2 * 180 / math.pi)
        self.assertAlmostEqual(report['by_hand']['left']['mae_deg'], 1.5 * 180 / math.pi)
        self.assertEqual(report['by_split']['both']['valid_angle_count'], 1)
        self.assertEqual(report['by_user']['a']['valid_angle_count'], 2)

    def test_finger_wrist_hand_split_and_unweighted_user_dispersion_are_separate(self):
        meter = AngleAccumulator()
        prediction = np.zeros((22, 1))
        prediction[:20] = 1
        prediction[20:] = 3
        meter.update(prediction, np.zeros((22, 1)), np.ones((22, 1), bool),
                     hand='left', split='user', user='a')
        meter.update(np.array([[5.]]), np.zeros((1, 1)), np.ones((1, 1), bool),
                     hand='right', split='both', user='b')
        report = meter.report()
        self.assertEqual(report['by_component']['wrist']['valid_angle_count'], 2)
        self.assertAlmostEqual(report['by_component']['wrist']['mae_deg'], 3 * 180 / math.pi)
        self.assertEqual(report['by_hand_split']['left']['user']['valid_angle_count'], 22)
        # User A mean 26/22, user B mean 5: unweighted mean 34/11, std 21/11.
        self.assertAlmostEqual(report['user_summary']['mean_mae_deg'], (34 / 11) * 180 / math.pi)
        self.assertAlmostEqual(report['user_summary']['std_mae_deg'], (21 / 11) * 180 / math.pi)

    def test_center_and_trajectory_are_explicit_different_modes(self):
        meter = AngleAccumulator(mode='center')
        meter.update(np.array([[100., 1., 100.]]), np.zeros((1, 3)), np.ones((1, 3), bool),
                     hand='left', split='gesture', user='a', center_index=1)
        self.assertEqual(meter.report()['mode'], 'center')
        self.assertEqual(meter.report()['overall']['valid_angle_count'], 1)
        self.assertAlmostEqual(meter.report()['overall']['absolute_error_sum_rad'], 1.)

    def test_all_invalid_observations_report_null_instead_of_nan(self):
        meter = AngleAccumulator()
        meter.update(np.array([[float('nan')]]), np.zeros((1, 1)), np.zeros((1, 1), bool),
                     hand='left', split='user', user='a')
        self.assertIsNone(meter.report()['overall']['mae_deg'])

    def test_nonfinite_valid_label_is_rejected(self):
        with self.assertRaises(ValueError):
            AngleAccumulator().update(np.array([[float('nan')]]), np.zeros((1, 1)),
                                      np.ones((1, 1), bool), hand='left', split='user', user='a')


if __name__ == '__main__':
    unittest.main()


class PairedPopulationTests(unittest.TestCase):
    def test_missing_crops_remain_in_all_population_and_leave_matched_subset(self):
        from egoemg_repro.metrics import PairedCenterAccumulator
        meter = PairedCenterAccumulator()
        meter.update(np.array([[1.]]), np.zeros((1, 1)), np.ones((1, 1), bool),
                     hand='left', split='user', user='a', vision_valid=True)
        meter.update(np.array([[9.]]), np.zeros((1, 1)), np.ones((1, 1), bool),
                     hand='right', split='both', user='b', vision_valid=False)
        report = meter.report()
        self.assertEqual(report['samples'], {'all': 2, 'vision_valid': 1, 'missing_crop': 1})
        self.assertEqual(report['all']['overall']['valid_angle_count'], 2)
        self.assertAlmostEqual(report['all']['overall']['mae_deg'], 5 * 180 / math.pi)
        self.assertAlmostEqual(report['vision_valid']['overall']['mae_deg'], 180 / math.pi)
        self.assertEqual(report['missing_by_hand'], {'right': 1})
        self.assertEqual(report['missing_by_split'], {'both': 1})
        self.assertEqual(report['all']['mode'], 'center')
