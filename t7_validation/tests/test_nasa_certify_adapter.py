"""NASA Certify adapter contract and regression tests."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT/'t7_validation/adapters/nasa/certify.py'
S = importlib.util.spec_from_file_location('test_nasa_certify_adapter_module', ADAPTER)
adapter = importlib.util.module_from_spec(S)
S.loader.exec_module(adapter)


class TestNASACertifyAdapter(unittest.TestCase):
    def setUp(self):
        self.rows = adapter._original_remember.load_projects()

    def test_complete_groups_and_pairs(self):
        for feature,goal in [('pcap','Effort-'),('cplx','Defects-')]:
            ans=adapter.group_answer(self.rows,feature,goal)
            keys=set(adapter._original_certify.group_values(self.rows,feature,goal))
            self.assertEqual(set(ans['groups']), keys)
            self.assertEqual(len(ans['comparisons']),len(keys)*(len(keys)-1)//2)
            self.assertEqual(sum(v['n'] for v in ans['groups'].values()),len(self.rows))

    def test_shuffle_preserves_group_measurements(self):
        for f,g in [('pcap','Effort-'),('cplx','Defects-')]:
            a=adapter.group_answer(self.rows,f,g)
            b=adapter.group_answer(list(reversed(self.rows)),f,g)
            self.assertEqual(a,b)

    def test_target_changes_recompute_group_stats_and_decisions(self):
        a=adapter.group_answer(self.rows,'pcap','Effort-')
        changed=[r.copy() for r in self.rows]
        for r in changed:
            if r['pcap']=='vh':
                r['Effort-']+=5000.
        b=adapter.group_answer(changed,'pcap','Effort-')
        self.assertNotEqual(a['groups']['vh']['mean'],b['groups']['vh']['mean'])
        self.assertNotEqual(a['groups']['vh']['median'],b['groups']['vh']['median'])

    def test_distributional_verdict_not_confused_with_original_p_value(self):
        groups=adapter._original_certify.group_values(self.rows,'pcap','Effort-')
        stats=adapter.comparison_details(groups['n'],groups['vh'])
        self.assertGreater(stats['original_permutation_p'],.05)
        self.assertEqual(stats['verdict'],'different')
        self.assertTrue(stats['ks_rejects'])


if __name__ == '__main__':
    unittest.main()
