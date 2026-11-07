import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from shrinking_code import apache
from shrinking_code.data import Row,Table
from shrinking_code.explanation import Explainer
from shrinking_code.evaluation import train_test_split
from shrinking_code.selection import HistoricalOracle,select_candidates
from shrinking_code.alternatives import predicted_alternatives
from shrinking_code.anomalies import input_isolation,outcome_residuals
from shrinking_code.baselines import selection_race,prediction_race,summarize_budget
from shrinking_code.prediction import KNNRegressor

ROOT=Path(__file__).resolve().parents[1]
T=Table.load(ROOT/'data/raw/Apache_AllMeasurements.csv')


class ApacheBatchBTests(unittest.TestCase):
    def test_best_explanation_neighbors_and_pair(self):
        e=Explainer(T); best=e.explain(e.ranked[0][0].number)
        self.assertEqual((best.row.number,best.score,best.goals[0].raw,best.goals[0].squared),(125,0,840,0))
        self.assertEqual([n.explanation.row.number for n in e.neighbors(125)],[73,122,124])
        self.assertTrue(all(n.distance==1/3 for n in e.neighbors(125)))
        a,b,setting,p,q,n=apache.pair_estimate(T)
        self.assertEqual((a.number,b.number,setting,n),(1,34,'handle',190))
        self.assertEqual(q.value-p.value,-150)
        self.assertEqual(b.cells[-1]-a.cells[-1],30)
        self.assertEqual((q.minimum-p.maximum,q.maximum-p.minimum),(-990,270))

    def test_choose_cost_checkpoints_and_hidden_goals(self):
        a,b=train_test_split(T); oracle=HistoricalOracle(b,20)
        original=KNNRegressor.predict
        def guarded(model,query,target):
            self.assertIsNone(query.cells[-1])
            self.assertTrue({r.number for r in model.training.rows}<={r.number for r in a.rows}|set(oracle.inspected))
            return original(model,query,target)
        with patch.object(KNNRegressor,'predict',guarded):
            result=select_candidates(a,oracle.visible,oracle.inspect,20)
        self.assertEqual(len(set(oracle.inspected)),20)
        self.assertEqual(result.prediction_count,590)
        self.assertEqual([result.checkpoint(n).best_row.number for n in (5,10,20)],[151]*3)
        self.assertEqual([result.checkpoint(n).best_row.cells[-1] for n in (5,10,20)],[840]*3)
        changed=Table(b.columns,tuple(Row(r.number,r.cells[:-1]+(-999,)) for r in b.rows))
        other=HistoricalOracle(changed,1)
        first=select_candidates(a,other.visible,other.inspect,1)
        self.assertEqual(first.inspections[0].row.number,result.inspections[0].row.number)
        self.assertEqual(first.inspections[0].predicted_goals,result.inspections[0].predicted_goals)

    def test_actual_fix_universe_target_minimality(self):
        source,target,single,meeting=apache.fix_results(T)
        self.assertEqual((source.number,target),(109,1215))
        self.assertEqual([r.alternative.number for r in single],[63,144,2])
        self.assertEqual([r.improvement for r in single],[900,474,294])
        self.assertEqual(meeting[0].alternative.number,63)
        self.assertEqual(meeting[0].prediction.value,1188)
        a,b=train_test_split(T); ids={r.number for r in b.rows}
        for r in single+meeting:
            self.assertIn(r.alternative.number,ids)
            actual=next(x for x in T.rows if x.number==r.alternative.number)
            self.assertEqual(r.alternative.cells[:-1],actual.cells[:-1])
            self.assertNotIn('base',[c.name for c in r.changed])
            self.assertNotIn(r.alternative.number,{n.number for n in r.prediction.neighbors})
            self.assertTrue({n.number for n in r.prediction.neighbors}<={n.number for n in a.rows})
        self.assertTrue(all(len(r.changed)==1 for r in single))
        self.assertEqual(meeting,tuple(sorted(meeting,key=lambda r:(len(r.changed),r.distance,r.prediction.value,r.alternative.number))))

    def test_predicted_alternatives_known_mean_and_masked_boundary(self):
        t=Table.from_csv(io.StringIO('base,setting,Performance-\n1,0,0\n1,0,0\n1,0,0\n1,1,100\n1,1,100\n1,1,100\n'))
        visible=Table(t.columns,(Row(7,('1','1',None)),Row(8,('1','0',None))))
        r=predicted_alternatives(t,visible,7,'Performance-')[0]
        self.assertEqual((r.source_prediction.value,r.prediction.value,r.improvement),(60,40,20))
        self.assertEqual([(c.name,c.before,c.after) for c in r.changed],[('setting','1','0')])
        self.assertAlmostEqual(r.distance,2**-.5)
        self.assertEqual(r.alternative,visible.rows[1])
        self.assertTrue(r.prediction.value<=45)
        self.assertFalse(r.prediction.value<=30)
        with self.assertRaises(ValueError):predicted_alternatives(t,Table(t.columns,(Row(7,('1','1',999)),)),7,'Performance-')
        with self.assertRaises(ValueError):predicted_alternatives(t,Table(t.columns,(Row(1,('1','1',None)),)),1,'Performance-')

    def test_no_solution_and_no_fallback_claim(self):
        t=Table.from_csv(io.StringIO('base,setting,Performance-\n1,0,100\n1,1,100\n'))
        q=Table(t.columns,(Row(3,('1','0',None)),Row(4,('1','1',None))))
        options=predicted_alternatives(t,q,3,'Performance-')
        self.assertEqual(options[0].improvement,0)
        self.assertFalse(any(r.improvement>0 for r in options))
        self.assertFalse(any(r.prediction.value<=50 for r in options))
        out=io.StringIO()
        with patch('shrinking_code.apache.fix_results',return_value=(T.rows[0],0,(),())),contextlib.redirect_stdout(out):apache.fix(T)
        self.assertEqual(out.getvalue().count('no qualifying solution; no fallback presented'),2)

    def test_fix_predictions_invariant_to_queried_outcomes(self):
        a,b=train_test_split(T); visible=HistoricalOracle(b,0).visible
        first=predicted_alternatives(a,visible,109,'Performance-')
        changed=Table(b.columns,tuple(Row(r.number,r.cells[:-1]+(999999,)) for r in b.rows))
        second=predicted_alternatives(a,HistoricalOracle(changed,0).visible,109,'Performance-')
        self.assertEqual(first,second)

    def test_residual_values_and_signs(self):
        a,b=train_test_split(T); values=outcome_residuals(a,b,'Performance-')[:5]
        self.assertEqual([r.row.number for r in values],[144,130,29,174,176])
        self.assertEqual([r.signed for r in values],[606,474,342,336,336])
        for r in values:self.assertEqual(r.score,abs(r.actual-r.prediction.value))

    def test_isolation_ties_and_y_invariance(self):
        original=input_isolation(T)
        self.assertEqual({r.score for r in original},{1/3})
        self.assertEqual([r.row.number for r in original if r.flagged],[1,2,3,4,5])
        changed=Table(T.columns,tuple(Row(r.number,r.cells[:-1]+(None,)) for r in T.rows))
        self.assertEqual([(r.row.number,r.score,r.neighbors) for r in original],[(r.row.number,r.score,r.neighbors) for r in input_isolation(changed)])
        self.assertEqual(sum(r.row.cells[-1]<=1215 for r in original if r.flagged),2)

    def test_race_exact_rows_and_random_reproducibility(self):
        a,b=train_test_split(T); prediction=prediction_race(a,b,'Performance-')
        self.assertEqual({p.number for p in prediction.pairs},{r.number for r in b.rows})
        self.assertEqual((prediction.wins,prediction.losses,prediction.ties),(35,4,0))
        self.assertAlmostEqual(prediction.evaluation.mae,178.615384615)
        self.assertFalse(prediction.comparison.same)
        race=selection_race(T,budgets=(5,10,20))
        self.assertEqual(race,selection_race(T,budgets=(5,10,20)))
        for c in race.checkpoints:
            values=[t.inspections[c.budget-1].best_score for t in race.random_trials]
            self.assertEqual((c.random_summary,c.match_or_beat_fraction),summarize_budget(c.model_score,values))
        self.assertEqual([round(c.match_or_beat_fraction*30) for c in race.checkpoints],[6,11,20])

    def test_all_thirty_demos_and_previous_hashes(self):
        for skill in ['justify','choose','fix','spot','race']:
            out=subprocess.check_output([sys.executable,'-m','shrinking_code.demo','--domain','apache','--skill',skill],cwd=ROOT,text=True)
            self.assertIn('APACHE-',out)
        hashes=json.loads((ROOT/'tests/fixtures/pre_apache_batch_b_hashes.json').read_text())
        for key,digest in hashes.items():
            domain,skill=key.split('/')
            out=subprocess.check_output([sys.executable,'-m','shrinking_code.demo','--domain',domain,'--skill',skill],cwd=ROOT)
            self.assertEqual(hashlib.sha256(out).hexdigest(),digest,key)
