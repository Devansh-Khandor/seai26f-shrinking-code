"""Original NASA T7 Certify tests: original module remains unchanged."""
import importlib.util
import math
import statistics
import sys
import unittest
from pathlib import Path

NASA = Path(__file__).resolve().parents[1]


def load(alias, path):
    s = importlib.util.spec_from_file_location(alias, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


remember = load('test_nasa_certify_original_remember', NASA/'remember.py')
old = sys.modules.get('remember')
try:
    sys.modules['remember'] = remember
    certify = load('test_nasa_certify_original_module', NASA/'certify.py')
finally:
    if old is None:
        sys.modules.pop('remember', None)
    else:
        sys.modules['remember'] = old


class TestOriginalNASACertify(unittest.TestCase):
    def test_grouping_preserves_all_values(self):
        rows = [dict(g='h', y=1.), dict(g='n', y=4.), dict(g='h', y=3.)]
        self.assertEqual(certify.group_values(rows, 'g', 'y'), {'h':[1.,3.], 'n':[4.]})

    def test_permutation_is_deterministic(self):
        a,b = [1.,2.,3.,4.,5.], [20.,21.,22.,23.,24.]
        self.assertEqual(certify.permutation_test(a,b,999,42),
                         certify.permutation_test(a,b,999,42))

    def test_permutation_identical_groups_have_p_one(self):
        diff,p = certify.permutation_test([1.,1.,1.], [1.,1.,1.], 99)
        self.assertEqual(diff, 0.)
        self.assertEqual(p,1.)

    def test_permutation_separated_groups(self):
        diff,p = certify.permutation_test(list(range(10)), list(range(100,110)),999)
        self.assertGreater(diff, 90)
        self.assertLess(p,.05)

    def test_full_dataset_expected_groups(self):
        rows = remember.load_projects()
        pcap=certify.group_values(rows,'pcap','Effort-')
        cplx=certify.group_values(rows,'cplx','Defects-')
        self.assertEqual(sum(map(len,pcap.values())),93)
        self.assertEqual(sum(map(len,cplx.values())),93)
        self.assertEqual(set(pcap), {'h','n','vh'})
        self.assertEqual(set(cplx), {'h','n','l','xh','vh'})

    def test_original_permutation_does_not_claim_distributional_separation(self):
        rows=remember.load_projects()
        groups=certify.group_values(rows,'pcap','Effort-')
        _,p=certify.permutation_test(groups['n'], groups['vh'],9999,42)
        self.assertGreater(p,.05)  # this real difference must be disclosed


if __name__ == '__main__':
    unittest.main()
