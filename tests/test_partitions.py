import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from egoemg_repro.data import LegacyStore
from egoemg_repro.partitions import build_partitions
from test_data import make_store


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_fixed_official_centers_and_no_cross_role_footprint_overlap(self):
        split = np.zeros(107, dtype=np.int8)
        split[29:36] = 2  # Grid center 35 is gesture; footprint [30,40) includes train rows.
        split[80:100] = 1
        make_store(self.root, split, lengths=(100, 7))
        out = build_partitions(LegacyStore(self.root), window_length=10, train_stride=2, eval_stride=10, val_fraction=.2)
        self.assertEqual([(w['center'], w['split']) for w in out['windows'] if w['role'] == 'test'], [(35,'gesture'), (85,'user'), (95,'user')])
        byrole = {r: set() for r in ('train','val','test')}
        for w in out['windows']:
            byrole[w['role']].update(range(w['start'],w['end']))
            if w['role'] != 'test':
                self.assertTrue(np.all(split[w['start']:w['end']] == 0))
        for a,b in [('train','val'),('val','test'),('train','test')]:
            self.assertFalse(byrole[a] & byrole[b])
        self.assertTrue(all(byrole.values()))
        self.assertTrue(all(w['end'] <= 100 for w in out['windows']))
        self.assertEqual(out, build_partitions(LegacyStore(self.root),10,2,10,.2))
        self.assertEqual(len(out['provenance']['partition_sha256']),64)
        json.dumps(out)

    def test_split_names_resolved_by_table_not_numeric_assumptions(self):
        make_store(self.root, [1]*20+[0]*20, lengths=(40,), labels=('gesture','train','both','user'))
        out = build_partitions(LegacyStore(self.root),10,2,10,0)
        self.assertEqual([w['center'] for w in out['windows'] if w['role']=='test'], [25,35])
        self.assertTrue(all(w['split']=='gesture' for w in out['windows'] if w['role']=='test'))

    def test_short_train_runs_retained_without_overlapping_validation(self):
        make_store(self.root, lengths=(10,7))
        out = build_partitions(LegacyStore(self.root),10,2,10,.12)
        self.assertEqual([(w['start'],w['end'],w['role']) for w in out['windows']],[(0,10,'train')])
        self.assertEqual(out['coverage']['val']['windows'],0)

    def test_unknown_split_id_fails_closed(self):
        make_store(self.root, [4]*40, lengths=(40,))
        with self.assertRaisesRegex(ValueError,'split'):
            build_partitions(LegacyStore(self.root),10,2,10,.12)


if __name__ == '__main__':
    unittest.main()
