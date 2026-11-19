"""NASA Flow common-contract and future-data isolation regression tests."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'t7_validation/adapters/nasa/flow.py'
S=importlib.util.spec_from_file_location('nasa_flow_adapter_test_module',SRC)
adapter=importlib.util.module_from_spec(S)
S.loader.exec_module(adapter)

class TestNASAFlowAdapter(unittest.TestCase):
    def setUp(self):
        self.csv=ROOT/'data/raw/nasa93dem.csv'

    def test_arrival_protocol_and_checkpoints(self):
        alerts,summaries=adapter.answers(self.csv)
        self.assertEqual([e['row'] for e in alerts['assessments']], list(range(75,94)))
        self.assertEqual([e['after'] for e in summaries['checkpoints']],[0,9,19])
        for e,n in zip(summaries['checkpoints'],[74,83,93]):
            self.assertEqual(len(e['shortlist']),5)
            self.assertTrue(all(1<=r<=n for r in e['shortlist']))
            for g in ['Kloc+','Effort-','Defects-']:
                self.assertEqual(e['summary'][g]['n'],n)

    def test_future_target_modifications_cannot_change_early_decisions(self):
        alerts,_=adapter.answers(self.csv)
        with self.csv.open(newline='',encoding='utf-8') as fp:
            raw=list(csv.reader(fp))
        modified=[r[:] for r in raw]
        for rid in range(84,94):
            for goal in ['Effort-','Defects-','Kloc+']:
                modified[rid][raw[0].index(goal)]='1000000000'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'future.csv'
            with path.open('w',newline='',encoding='utf-8') as fp:
                csv.writer(fp).writerows(modified)
            changed,_=adapter.answers(path)
        self.assertEqual(alerts['assessments'][:9],changed['assessments'][:9])

    def test_shuffled_csv_produces_new_protocol_and_recomputed_summary(self):
        base_alerts,base_checkpoints=adapter.answers(self.csv)
        with self.csv.open(newline='',encoding='utf-8') as fp:
            raw=list(csv.reader(fp))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'shuffled.csv'
            with path.open('w',newline='',encoding='utf-8') as fp:
                csv.writer(fp).writerows([raw[0]]+list(reversed(raw[1:])))
            new_alerts,new_cps=adapter.answers(path)
        self.assertEqual([e['row'] for e in new_alerts['assessments']],list(range(75,94)))
        self.assertNotEqual(base_alerts['assessments'],new_alerts['assessments'])
        self.assertNotEqual(base_checkpoints['checkpoints'][0]['summary']['Effort-']['mean'],
                            new_cps['checkpoints'][0]['summary']['Effort-']['mean'])

    def test_import_isolation_maintains_apache_module_cache(self):
        # Adapter import must not overwrite `remember` and `guess` in sys.modules.
        import sys
        self.assertNotEqual(adapter.remember.__file__,str(ROOT/'human_t7/apache/remember.py'))
        self.assertEqual(adapter.flow.assess_proposal.__module__,'t7_nasa_original_flow')

if __name__ == '__main__':
    unittest.main()
