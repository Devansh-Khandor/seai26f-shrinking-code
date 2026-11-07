import contextlib
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import unittest
from unittest.mock import patch

from shrinking_code.data import Row,Table
from shrinking_code.prediction import KNNRegressor
from shrinking_code.explanation import explain_prediction,Explainer
from shrinking_code.hypotheticals import search_changes
from shrinking_code.proposals import Proposal,Risk
from shrinking_code.selection import HistoricalOracle,select_candidates
from shrinking_code.anomalies import input_isolation,outcome_residuals
from shrinking_code.baselines import selection_race,prediction_race,summarize_budget
from shrinking_code import nasa
from shrinking_code.evaluation import train_test_split

ROOT=Path(__file__).resolve().parents[1]
T=Table.load(ROOT/'data/raw/nasa93dem.csv')


def synthetic(text):
    return Table.from_csv(io.StringIO(text))


class NasaBatchBTests(unittest.TestCase):
    def test_explain_neighbor_mean_and_mask(self):
        a,visible,_=nasa.proposal_context(T)
        model=KNNRegressor.fit(a); q=visible.rows[0]
        e=explain_prediction(model,q,'Effort-')
        self.assertEqual([r.number for r in e.prediction.neighbors],[1,2,3,6,7])
        self.assertAlmostEqual(e.prediction.value,57.12)
        self.assertEqual(e.prediction.value,statistics.mean([117.6,117.6,31.2,8.4,10.8]))
        self.assertEqual(e.neighbor_distances,(0,)*5)
        with self.assertRaisesRegex(ValueError,'masked'):
            explain_prediction(model,T.rows[q.number-1],'Effort-')

    def test_risk_pair_tradeoff_fallback_and_no_pair(self):
        _,_,assessments=nasa.proposal_context(T)
        a,b,tradeoff=nasa.risk_pair(assessments)
        self.assertEqual((a.number,b.number,tradeoff),(55,76,True))
        self.assertTrue(a.risks[0].prediction.value>b.risks[0].prediction.value)
        self.assertTrue(a.risks[1].prediction.value<b.risks[1].prediction.value)
        self.assertEqual([r.threshold for r in a.risks],[720,5092])
        self.assertIsNone(nasa.risk_pair((a,)))
        prediction=a.risks[0].prediction
        low=Proposal(1,74,(Risk('a',prediction,9999,False),))
        high=Proposal(2,74,(Risk('a',prediction,0,True),))
        self.assertEqual(nasa.risk_pair((low,high)),(low,high,False))

    def test_hidden_goals_do_not_change_justification_or_fix(self):
        a,b=train_test_split(T); ids={r.number for r in b.rows}
        altered=Table(T.columns,tuple(Row(r.number,tuple(999999 if c.role=='y' and r.number in ids else r.cells[c.index] for c in T.columns)) for r in T.rows))
        for func in [nasa.justify,nasa.fix]:
            outputs=[]
            for data in [T,altered]:
                out=io.StringIO()
                with contextlib.redirect_stdout(out):func(data)
                outputs.append(out.getvalue())
            self.assertEqual(*outputs)

    def test_historical_choice_and_inspections(self):
        self.assertEqual([r.number for r,_ in Explainer(T).ranked[:5]],[60,67,46,44,77])
        a,b=train_test_split(T); oracle=HistoricalOracle(b,10)
        result=select_candidates(a,oracle.visible,oracle.inspect,10)
        self.assertEqual(len(set(oracle.inspected)),10)
        self.assertEqual(result.prediction_count,435)
        self.assertEqual([result.checkpoint(i).best_row.number for i in (3,5,10)],[76]*3)
        self.assertTrue(all(r.cells[c.index] is None for r in oracle.visible.rows for c in T.columns if c.role!='x'))

    def test_allowlist_training_levels_and_real_no_solution(self):
        self.assertEqual(nasa.ADJUSTABLE,('docu','tool','site','sced'))
        self.assertEqual(nasa.ADJUSTMENT_VERSION,'NASA-adjustments-v1')
        a,visible,assessments=nasa.proposal_context(T)
        for target,number,limit in [('Effort-',80,1),('Defects-',76,2)]:
            q=next(r for r in visible.rows if r.number==number)
            result=search_changes(a,q,target,nasa.ADJUSTABLE,nasa.TARGETS,limit)
            self.assertEqual(result.candidates,())

    def test_single_changes_exact_predictions_tradeoff_and_observed(self):
        t=synthetic('fixed,tool,sced,Effort-,Defects-\na,l,l,100,100\na,h,l,150,50\na,l,h,90,80\n')
        q=Row(99,('a','l','l',None,None))
        result=search_changes(t,q,'Defects-',('tool','sced'),('Effort-',),1,.05,k=1)
        self.assertEqual(result.candidates[0].changes,(('tool','h'),))
        self.assertEqual(result.candidates[0].improvement,50)
        self.assertEqual(dict(result.candidates[0].predictions)['Effort-'].value,150)
        self.assertTrue(result.candidates[0].historically_observed)
        for v in result.candidates:
            self.assertEqual(v.query.cells[0],'a')
            self.assertEqual(v.query.cells[3:],(None,None))
            for name,value in v.changes:
                c=next(c for c in t.columns if c.name==name)
                self.assertIn(value,{r.cells[c.index] for r in t.rows})
        self.assertEqual(result,search_changes(t,q,'Defects-',('tool','sced'),('Effort-',),1,.05,k=1))

    def test_two_field_search_only_when_needed_and_minimum_threshold(self):
        t=synthetic('tool,sced,Defects-\nl,l,100\nh,l,99\nl,h,99\nh,h,50\n')
        q=Row(99,('l','l',None))
        first=search_changes(t,q,'Defects-',('tool','sced'),max_changes=1,k=1)
        self.assertEqual(first.candidates,())
        result=search_changes(t,q,'Defects-',('tool','sced'),max_changes=2,k=1)
        self.assertEqual(result.candidates[0].changes,(('sced','h'),('tool','h')))
        self.assertEqual(result.evaluations,3)
        self.assertEqual(result.candidates[0].improvement,50)
        smaller=search_changes(t,q,'Defects-',('tool','sced'),max_changes=2,min_fraction=.005,k=1)
        self.assertEqual(len(smaller.candidates[0].changes),1)
        self.assertEqual(smaller.evaluations,2)

    def test_search_rejects_invalid_queries_and_prohibited_roles(self):
        t=synthetic('tool,Size,SecretX,Defects-\nl,1,9,100\nh,2,8,50\n')
        q=Row(99,('l',1,None,None))
        for fields in [('Size',),('SecretX',),('Defects-',),('absent',),('tool','tool')]:
            with self.assertRaises(ValueError):search_changes(t,q,'Defects-',fields)
        with self.assertRaises(ValueError):search_changes(t,t.rows[0],'Defects-',('tool',))
        with self.assertRaises(ValueError):search_changes(t,q,'Defects-',('tool',),max_changes=3)
        with self.assertRaises(ValueError):search_changes(t,q,'Defects-',('tool',),min_fraction=0)

    def test_no_synthetic_x_combination_claimed_observed(self):
        t=synthetic('fixed,tool,Defects-\na,l,100\nb,h,50\n')
        q=Row(99,('z','l',None))
        result=search_changes(t,q,'Defects-',('tool',),k=1)
        self.assertEqual(result.candidates[0].improvement,50)
        self.assertFalse(result.candidates[0].historically_observed)
        self.assertEqual(result.candidates[0].query.cells,('z','h',None))

    def test_spot_and_residuals(self):
        results=input_isolation(T)
        self.assertEqual([r.row.number for r in results if r.flagged],[56,89,57,59,43])
        changed=Table(T.columns,tuple(Row(r.number,tuple(None if c.role=='y' else r.cells[c.index] for c in T.columns)) for r in T.rows))
        self.assertEqual([(r.row.number,r.score,r.neighbors) for r in results],[(r.row.number,r.score,r.neighbors) for r in input_isolation(changed)])
        a,b=train_test_split(T)
        r=outcome_residuals(a,b,'Effort-')[0]
        self.assertEqual(r.row.number,82); self.assertAlmostEqual(r.signed,-3066.42)
        self.assertAlmostEqual(r.score,3066.42)

    def test_race_paired_errors_and_random_summaries(self):
        a,b=train_test_split(T); p=prediction_race(a,b,'Effort-')
        self.assertEqual((p.wins,p.losses,p.ties),(15,4,0))
        self.assertEqual(len(p.pairs),19)
        self.assertAlmostEqual(sum(x.knn_error for x in p.pairs)/19,308.466315789)
        self.assertFalse(p.comparison.same)
        first=selection_race(T,budgets=(3,5,10))
        self.assertEqual(first,selection_race(T,budgets=(3,5,10)))
        for c in first.checkpoints:
            scores=[r.inspections[c.budget-1].best_score for r in first.random_trials]
            self.assertEqual((c.random_summary,c.match_or_beat_fraction),summarize_budget(c.model_score,scores))
        self.assertEqual([round(c.match_or_beat_fraction*30) for c in first.checkpoints],[4,6,16])

    def test_all_new_demos_and_previous_outputs(self):
        for skill in ['justify','choose','fix','spot','race']:
            out=subprocess.check_output([sys.executable,'-m','shrinking_code.demo','--domain','nasa','--skill',skill],cwd=ROOT,text=True)
            self.assertIn('NASA /',out)
        fixture=json.loads((ROOT/'tests/fixtures/pre_nasa_batch_b_hashes.json').read_text())
        for key,digest in fixture.items():
            domain,skill=key.split('/')
            out=subprocess.check_output([sys.executable,'-m','shrinking_code.demo','--domain',domain,'--skill',skill],cwd=ROOT)
            self.assertEqual(hashlib.sha256(out).hexdigest(),digest,key)
