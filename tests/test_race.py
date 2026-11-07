import hashlib
import io
import json
from pathlib import Path
from random import Random
import subprocess
import sys
import unittest
from unittest.mock import patch

from shrinking_code.baselines import random_selection, selection_race, summarize_budget, prediction_race
from shrinking_code.certification import compare
from shrinking_code.data import Row, Table
from shrinking_code.evaluation import train_test_split, evaluate
from shrinking_code.prediction import KNNRegressor
from shrinking_code.selection import HistoricalOracle, select_candidates

ROOT = Path(__file__).resolve().parents[1]


def table(text):
    return Table.from_csv(io.StringIO(text))


def fixture():
    t = table('Size,Reward+\n0,0\n10,10\n1,2\n2,4\n3,8\n4,10\n')
    return Table(t.columns,t.rows[:2]), Table(t.columns,t.rows[2:])


def random_run(a,b,budget=4,seed=0):
    oracle=HistoricalOracle(b,budget)
    result=random_selection(a,oracle.visible,oracle.inspect,budget,seed)
    return result,oracle


class RaceTests(unittest.TestCase):
    def test_random_seed_order_counts_and_reproducibility(self):
        a,b=fixture()
        r,o=random_run(a,b)
        ids=[3,4,5,6]; Random(0).shuffle(ids)
        self.assertEqual(o.inspected,tuple(ids))
        self.assertEqual(len(set(o.inspected)),4)
        self.assertEqual(r,random_run(a,b)[0])
        self.assertEqual(r,random_run(a,Table(b.columns,tuple(reversed(b.rows))))[0])
        self.assertNotEqual(o.inspected,random_run(a,b,seed=1)[1].inspected)
        self.assertEqual(r.prediction_count,0)

    def test_independent_best_so_far_and_budget_prefix(self):
        a,b=fixture()
        r,o=random_run(a,b)
        best=1
        raw={3:.8,4:.6,5:.2,6:0}
        for d in r.inspections:
            best=min(best,raw[d.row.number])
            self.assertAlmostEqual(d.observed_score,raw[d.row.number])
            self.assertAlmostEqual(d.best_score,best)
        self.assertEqual(r.inspections[:2],random_run(a,b,budget=2)[0].inspections)
        self.assertEqual(random_run(a,b,budget=0)[0].inspections,())
        self.assertEqual(len(random_run(a,b,budget=99)[0].inspections),4)

    def test_hidden_label_changes_cannot_change_random_sequence(self):
        a,b=fixture()
        original,o=random_run(a,b,budget=2)
        selected=set(o.inspected)
        altered=Table(b.columns,tuple(Row(r.number,(r.cells[0],-999)) if r.number not in selected else r for r in b.rows))
        self.assertEqual(original,random_run(a,altered,budget=2)[0])
        all_changed=Table(b.columns,tuple(Row(r.number,(r.cells[0],999)) for r in b.rows))
        self.assertEqual(o.inspected,random_run(a,all_changed,budget=2)[1].inspected)

    def test_model_future_labels_do_not_change_prefix(self):
        a,b=fixture()
        oracle=HistoricalOracle(b,1)
        result=select_candidates(a,oracle.visible,oracle.inspect,1)
        selected=oracle.inspected[0]
        altered=Table(b.columns,tuple(Row(r.number,(r.cells[0],999)) if r.number!=selected else r for r in b.rows))
        other=HistoricalOracle(altered,1)
        self.assertEqual(result,select_candidates(a,other.visible,other.inspect,1))

    def test_random_boundary_validation(self):
        a,b=fixture()
        with self.assertRaisesRegex(ValueError,'only X'):
            random_selection(a,b,lambda _: self.fail('leak'),1,0)
        masked=HistoricalOracle(a,1)
        with self.assertRaisesRegex(ValueError,'overlap'):
            random_selection(a,masked.visible,masked.inspect,1,0)
        with self.assertRaises(ValueError): random_run(a,b,seed=True)
        with self.assertRaises(ValueError): random_run(a,b,budget=-1)

    def test_random_missing_and_fixed_bounds(self):
        a,b=fixture()
        b=Table(b.columns,(Row(3,(1,None)),Row(4,(2,30))))
        r,o=random_run(a,b,budget=2)
        by_id={d.row.number:d for d in r.inspections}
        self.assertIsNone(by_id[3].observed_score)
        self.assertEqual(by_id[4].observed_score,2) # initial 0..10, no clipping
        self.assertEqual(len(o.inspected),2)

    def test_summary_stats_and_match_or_beat_ties(self):
        s,f=summarize_budget(.3,[.1,.3,.5,None])
        self.assertEqual((s.count,s.minimum,s.maximum,s.median),(3,.1,.5,.3))
        self.assertAlmostEqual(s.mean,.3)
        self.assertEqual(f,2/3)
        self.assertIsNone(summarize_budget(None,[.1])[1])
        self.assertIsNone(summarize_budget(.1,[None])[1])

    def test_orchestration_same_data_30_seeds_and_equal_budgets(self):
        t=table('Size,Reward+\n'+''.join(f'{i},{i}\n' for i in range(30)))
        a,b=train_test_split(t)
        seen=[]
        original=random_selection
        def guarded(training,visible,inspect,budget,seed):
            self.assertEqual(training,a)
            self.assertEqual(visible,HistoricalOracle(b,budget).visible)
            self.assertEqual(budget,4)
            seen.append(seed)
            return original(training,visible,inspect,budget,seed)
        with patch('shrinking_code.baselines.random_selection',side_effect=guarded):
            race=selection_race(t,budgets=(1,2,4))
        self.assertEqual(seen,list(range(30)))
        self.assertEqual(race.training_ids,tuple(r.number for r in a.rows))
        self.assertEqual(race.candidate_ids,tuple(r.number for r in b.rows))
        self.assertEqual(len(race.model.inspections),4)
        self.assertTrue(all(len(r.inspections)==4 for r in race.random_trials))
        self.assertGreater(len({tuple(d.row.number for d in r.inspections) for r in race.random_trials}),1)
        for c in race.checkpoints:
            vals=[r.inspections[c.budget-1].best_score for r in race.random_trials]
            self.assertEqual((c.random_summary,c.match_or_beat_fraction),summarize_budget(c.model_score,vals))
        self.assertEqual(race.checkpoints[-1].model_predictions,race.model.prediction_count)

    def test_race_invalid_checkpoints(self):
        a,b=fixture(); t=Table(a.columns,a.rows+b.rows)
        for budgets in [(),(0,),(-1,),(1,1),(40,)]:
            with self.assertRaises(ValueError): selection_race(t,budgets=budgets)
        with self.assertRaises(ValueError): selection_race(t,budgets=(1,),seeds=())

    def test_paired_errors_known_answers_and_percentage(self):
        t=table('Size,Reward+\n0,0\n1,0\n2,0\n3,0\n4,0\n100,100\n0,0\n100,100\n')
        a,b=Table(t.columns,t.rows[:6]),Table(t.columns,t.rows[6:])
        r=prediction_race(a,b,'Reward+')
        self.assertEqual([p.number for p in r.pairs],[7,8])
        self.assertEqual([p.knn_error for p in r.pairs],[0,80])
        self.assertAlmostEqual(r.evaluation.mae,40)
        self.assertAlmostEqual(r.evaluation.baseline_mae,50)
        self.assertAlmostEqual(r.improvement,10)
        self.assertAlmostEqual(r.percentage_improvement,20)
        self.assertEqual((r.wins,r.losses,r.ties),(2,0,0))
        self.assertAlmostEqual(sum(p.difference for p in r.pairs)/2,-10)
        self.assertEqual(r.comparison,compare([0,80],[100/6,100-100/6]))

    def test_ties_zero_baseline_and_insufficient_observations(self):
        a,b=fixture()
        r=prediction_race(a,Table(b.columns,b.rows[:1]),'Reward+')
        self.assertEqual((r.wins,r.losses,r.ties),(0,0,1))
        self.assertIsNone(r.comparison.same)
        t=table('Size,Reward+\n0,5\n1,5\n')
        r=prediction_race(Table(t.columns,t.rows[:1]),Table(t.columns,t.rows[1:]),'Reward+')
        self.assertIsNone(r.percentage_improvement)
        self.assertEqual(r.improvement,0)
        missing=Table(b.columns,(Row(3,(1,None)),))
        with self.assertRaises(ValueError): prediction_race(a,missing,'Reward+')
        with self.assertRaises(ValueError): prediction_race(a,a,'Reward+')

    def test_prediction_labels_not_used_and_training_bounds(self):
        a,b=fixture()
        altered=Table(b.columns,tuple(Row(r.number,(r.cells[0],999)) for r in b.rows))
        first=prediction_race(a,b,'Reward+')
        second=prediction_race(a,altered,'Reward+')
        self.assertEqual([p for _,p in first.evaluation.predictions],[p for _,p in second.evaluation.predictions])
        self.assertNotEqual(first.pairs,second.pairs)
        self.assertEqual(first.evaluation,evaluate(KNNRegressor.fit(a),b,'Reward+'))
        self.assertEqual(first.evaluation.baseline,5)

    def test_cli_and_prior_demos(self):
        out=subprocess.check_output([sys.executable,'-m','shrinking_code.demo','--domain','cars','--skill','race'],cwd=ROOT,text=True)
        for value in ['CAR-19','CAR-20','One deterministic','random seed=29','budget=10','budget=20','budget=40',
                      'knn_wins=','Course descriptive comparison: same=','does not account for paired errors',
                      'not the seven software-development treatments']:
            self.assertIn(value,out)
        self.assertEqual(sum(line.startswith('random seed=') for line in out.splitlines()),30)
        fixture=json.loads((ROOT/'tests/fixtures/pre_race_demo_hashes.json').read_text())
        for skill,digest in fixture['sha256'].items():
            output=subprocess.check_output([sys.executable,'-m','shrinking_code.demo','--domain','cars','--skill',skill],cwd=ROOT)
            self.assertEqual(hashlib.sha256(output).hexdigest(),digest,skill)
