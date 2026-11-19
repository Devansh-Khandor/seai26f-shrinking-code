"""Regression tests for unchanged NASA T7 Flow implementation."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / 'human_t7' / 'nasa'


def load(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


remember = load('original_nasa_flow_dep_remember', NASA/'remember.py')
prev_r, prev_g = sys.modules.get('remember'), sys.modules.get('guess')
try:
    sys.modules['remember'] = remember
    guess = load('original_nasa_flow_dep_guess', NASA/'guess.py')
    sys.modules['guess'] = guess
    flow = load('original_nasa_flow', NASA/'flow.py')
finally:
    for key, previous in [('remember',prev_r), ('guess',prev_g)]:
        if previous is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = previous


class TestOriginalNASAFlow(unittest.TestCase):
    def setUp(self):
        self.history = [
            {'row_number':1,'kind':'A','Kloc+':10.,'Effort-':10.,'Defects-':5.},
            {'row_number':2,'kind':'A','Kloc+':20.,'Effort-':20.,'Defects-':10.},
            {'row_number':3,'kind':'B','Kloc+':30.,'Effort-':30.,'Defects-':15.},
            {'row_number':4,'kind':'B','Kloc+':40.,'Effort-':40.,'Defects-':20.},
        ]
        self.features = ['kind']

    def test_percentile_nearest_rank(self):
        self.assertEqual(flow.percentile([40,10,30,20], .75),30)
        self.assertEqual(flow.percentile([10,20,30,40], 1),40)
        self.assertEqual(flow.percentile([10,20,30,40],0),10)

    def test_alert_limits_use_only_observed_history(self):
        self.assertEqual(flow.alert_limits(self.history), {'Effort-':30.,'Defects-':15.})
        changed = self.history + [{'row_number':5,'kind':'A','Kloc+':1.,'Effort-':1000.,'Defects-':999.}]
        self.assertEqual(flow.alert_limits(self.history)['Effort-'],30.)
        self.assertGreater(flow.alert_limits(changed)['Effort-'],30.)

    def test_assess_proposal_ignores_own_unknown_targets(self):
        proposal={'row_number':5,'kind':'B'}
        a=flow.assess_proposal(proposal,self.history,self.features)
        b=flow.assess_proposal({**proposal, 'Effort-':1e9,'Defects-':-1e9},self.history,self.features)
        self.assertEqual(a,b)
        self.assertEqual(a['row_number'],5)
        self.assertEqual(a['predicted_effort'],25.)
        self.assertEqual(a['predicted_defects'],12.5)

    def test_alert_flags_are_based_on_predicted_values(self):
        project={'row_number':5,'kind':'B'}
        result=flow.assess_proposal(project,self.history,self.features)
        self.assertEqual(result['effort_alert'],result['predicted_effort']>result['effort_limit'])
        self.assertEqual(result['defect_alert'],result['predicted_defects']>result['defect_limit'])

    def test_arrivals_update_history_after_assessing_each_proposal(self):
        arrivals=[
            {'row_number':5,'kind':'A','Kloc+':50.,'Effort-':1000.,'Defects-':2000.},
            {'row_number':6,'kind':'A','Kloc+':60.,'Effort-':100.,'Defects-':200.},
        ]
        before=flow.assess_proposal(arrivals[0],self.history,self.features)
        updated, assessments=flow.process_arrivals(self.history,arrivals,self.features)
        self.assertEqual(assessments[0],before)
        self.assertEqual([r['row_number'] for r in updated],[1,2,3,4,5,6])
        self.assertEqual(len(assessments),2)
        self.assertEqual(len(self.history),4)
        self.assertEqual(assessments[1],flow.assess_proposal(arrivals[1],self.history+arrivals[:1],self.features))

    def test_future_outcomes_cannot_change_earlier_assessments(self):
        arrivals=[
            {'row_number':5,'kind':'A','Kloc+':50.,'Effort-':100.,'Defects-':200.},
            {'row_number':6,'kind':'B','Kloc+':60.,'Effort-':300.,'Defects-':400.},
        ]
        _,base=flow.process_arrivals(self.history,arrivals,self.features)
        changed=[arrivals[0].copy(),{**arrivals[1],'Effort-':1e9,'Defects-':1e9,'Kloc+':1e9}]
        _,other=flow.process_arrivals(self.history,changed,self.features)
        self.assertEqual(base[0],other[0])
        self.assertEqual(base[1],other[1])  # assessed before own goals are revealed

    def test_all_nasa_arrivals_complete_without_losing_records(self):
        records=remember.load_projects()
        features=guess.get_features(records)
        updated,assessments=flow.process_arrivals(records[:74],records[74:],features)
        self.assertEqual(len(updated),93)
        self.assertEqual(len(assessments),19)
        self.assertEqual([a['row_number'] for a in assessments],list(range(75,94)))

if __name__ == '__main__':
    unittest.main()
