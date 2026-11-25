"""Unit tests for unchanged original NASA T7 Justify functions."""
import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
NASA=ROOT/'human_t7/nasa'

def load(alias,path):
    spec=importlib.util.spec_from_file_location(alias,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

remember=load('t7_nasa_original_justify_test_remember',NASA/'remember.py')
old={k:sys.modules.get(k) for k in ('remember','guess')}
try:
    sys.modules['remember']=remember
    guess=load('t7_nasa_original_justify_test_guess',NASA/'guess.py')
    sys.modules['guess']=guess
    justify=load('t7_nasa_original_justify_test',NASA/'justify.py')
finally:
    for k,v in old.items():
        if v is None: sys.modules.pop(k,None)
        else: sys.modules[k]=v

class TestOriginalNASAJustify(unittest.TestCase):
    def setUp(self):
        self.history=[
            {'row_number':1,'type':'a','Effort-':10.,'Defects-':100.,'Kloc+':4.},
            {'row_number':2,'type':'a','Effort-':20.,'Defects-':200.,'Kloc+':8.},
            {'row_number':3,'type':'b','Effort-':30.,'Defects-':300.,'Kloc+':12.},
            {'row_number':4,'type':'b','Effort-':40.,'Defects-':400.,'Kloc+':16.},
            {'row_number':5,'type':'a','Effort-':50.,'Defects-':500.,'Kloc+':20.}]
        self.features=['type']
        self.a={'row_number':6,'type':'a'}
        self.b={'row_number':7,'type':'b'}

    def test_risk_estimates_use_only_history(self):
        prediction=justify.risk_estimates(self.a,self.history,self.features)
        expected={'effort':30.,'defects':300.}
        self.assertEqual(prediction,expected)
        self.assertEqual(prediction,justify.risk_estimates(
            {**self.a,'Effort-':999999,'Defects-':-99999},self.history,self.features))

    def test_neighbor_explanation_returns_original_predictor_result(self):
        with redirect_stdout(io.StringIO()):
            got=justify.explain_effort(self.a,self.history,self.features)
        same=guess.predict(self.a,self.history,self.features,'Effort-',k=justify.K)
        self.assertEqual(got,same)
        self.assertEqual(len(got['neighbors']),5)

    def test_different_features_uses_input_columns(self):
        a={'row_number':8,'one':'a','two':'x','Effort-':5.}
        b={'row_number':9,'one':'b','two':'x','Effort-':1e10}
        self.assertEqual(justify.different_features(a,b,['one','two']),[('one','a','b')])

    def test_comparison_estimates_match_separate_predictions(self):
        with redirect_stdout(io.StringIO()):
            a,b,diffs=justify.compare_proposals(self.a,self.b,self.history,self.features)
        self.assertEqual(a,justify.risk_estimates(self.a,self.history,self.features))
        self.assertEqual(b,justify.risk_estimates(self.b,self.history,self.features))
        self.assertEqual(diffs,[('type','a','b')])

    def test_choose_proposals_ignores_unobserved_outcomes(self):
        proposals=[{'row_number':6,'type':'a'}, {'row_number':7,'type':'b'}]
        first=justify.choose_proposals(proposals,self.history,self.features)
        perturbed=[{**r,'Effort-':1e11,'Defects-':-1e11} for r in proposals]
        second=justify.choose_proposals(perturbed,self.history,self.features)
        self.assertEqual([r['row_number'] for r in first],
                         [r['row_number'] for r in second])

    def test_original_nasa_demo_real_data_no_exceptions(self):
        projects=remember.load_projects(ROOT/'data/raw/nasa93dem.csv')
        history=projects[:74]
        proposal=projects[74]
        features=guess.get_features(projects)
        with redirect_stdout(io.StringIO()):
            result=justify.explain_effort(proposal,history,features)
        self.assertEqual(len(result['neighbors']),5)
        self.assertTrue(all(row['row_number']<=74 for row in result['neighbors']))

if __name__=='__main__': unittest.main()
