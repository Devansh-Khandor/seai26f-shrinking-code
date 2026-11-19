"""Regression tests for NASA Justify common-contract adapter."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'t7_validation/adapters/nasa/justify.py'
spec=importlib.util.spec_from_file_location('t7_nasa_justify_adapter_test',SRC)
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

class TestNASAJustifyAdapter(unittest.TestCase):
    def setUp(self):
        self.csv=ROOT/'data/raw/nasa93dem.csv'

    def test_real_historical_neighbor_values_and_heldout_project(self):
        a,_=adapter.answers(self.csv)
        projects=adapter.remember.load_projects(self.csv)
        by_id={p['row_number']:p for p in projects}
        self.assertEqual(a['project']%5,0)
        self.assertEqual(len(a['neighbors']),5)
        for entry in a['neighbors']:
            self.assertNotEqual(entry['row']%5,0)
            for g in adapter.remember.GOALS:
                self.assertEqual(entry['values'][g],by_id[entry['row']][g])
        training=[p for p in projects if p['row_number']%5]
        expected=adapter.justify.risk_estimates(by_id[a['project']],training,
                                                 adapter.guess.get_features(projects))['effort']
        self.assertEqual(a['estimate'],expected)

    def test_reported_pair_differences_and_observed_tradeoffs(self):
        _,b=adapter.answers(self.csv)
        projects=adapter.remember.load_projects(self.csv)
        rows={p['row_number']:p for p in projects}
        a1,a2=rows[b['a']],rows[b['b']]
        self.assertNotEqual(b['a'],b['b'])
        self.assertEqual(set(b['scores']),{str(b['a']),str(b['b'])})
        pref=rows[b['preferred']]
        other=rows[b['b'] if b['preferred']==b['a'] else b['a']]
        self.assertFalse(adapter._dominates(other,pref))
        self.assertEqual(set(b['tradeoffs']),{g for g,w in adapter.remember.GOALS.items()
                       if w*other[g]>w*pref[g]})
        for difference in b['differences']:
            self.assertEqual(difference['a'],a1[difference['feature']])
            self.assertEqual(difference['b'],a2[difference['feature']])
            self.assertNotEqual(difference['a'],difference['b'])

    def test_hidden_target_edits_cannot_change_explanation(self):
        a,b=adapter.answers(self.csv)
        with self.csv.open(newline='',encoding='utf-8') as f:
            rows=list(csv.reader(f))
        changed=[x[:] for x in rows]
        for rid in range(5,len(changed),5):
            for goal in ('Effort-','Defects-','Kloc+'):
                changed[rid][rows[0].index(goal)]='999999999'
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'masked.csv'
            with path.open('w',newline='',encoding='utf-8') as f:
                csv.writer(f).writerows(changed)
            aa,bb=adapter.answers(path)
        self.assertEqual(aa,a)
        # Proposal-selection logic does not use hidden target values.
        self.assertEqual({k:b[k] for k in ('a','b','preferred','scores','differences')},
                         {k:bb[k] for k in ('a','b','preferred','scores','differences')})

    def test_shuffled_records_recompute_neighbors(self):
        a,b=adapter.answers(self.csv)
        with self.csv.open(newline='',encoding='utf-8') as f:
            rows=list(csv.reader(f))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'shuffled.csv'
            with path.open('w',newline='',encoding='utf-8') as f:
                csv.writer(f).writerows([rows[0]]+list(reversed(rows[1:])))
            aa,bb=adapter.answers(path)
        self.assertEqual(aa['project']%5,0)
        self.assertNotEqual(a['neighbors'],aa['neighbors'])
        self.assertNotEqual(b['differences'],bb['differences'])

    def test_import_isolation(self):
        import sys
        self.assertEqual(Path(adapter.justify.__file__),ROOT/'human_t7/nasa/justify.py')
        self.assertTrue(adapter.guess.__file__.endswith('/nasa/guess.py'))
        self.assertNotEqual(adapter.remember.__file__,str(ROOT/'human_t7/apache/remember.py'))

if __name__=='__main__': unittest.main()
