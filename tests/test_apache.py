import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import unittest
from unittest.mock import patch

from shrinking_code import apache
from shrinking_code.data import Row,Table
from shrinking_code.stats import summarize
from shrinking_code.ranking import rank_rows
from shrinking_code.evaluation import train_test_split,evaluate
from shrinking_code.prediction import KNNRegressor
from shrinking_code.certification import compare
from shrinking_code.diagnosis import categorical_pairs,outcome_groups,compare_groups
from shrinking_code.proposals import ProposalStream

ROOT=Path(__file__).resolve().parents[1]
T=Table.load(ROOT/'data/raw/Apache_AllMeasurements.csv')


def table(text):return Table.from_csv(io.StringIO(text))


class ApacheTests(unittest.TestCase):
    def test_schema_summary_and_single_goal_ranking(self):
        self.assertEqual((len(T.rows),len(T.columns)),(192,10))
        self.assertEqual(len(T.role('x')),9)
        self.assertTrue(all(not c.numeric for c in T.role('x')))
        self.assertEqual([(c.name,c.ideal) for c in T.role('y')],[('Performance-',0)])
        self.assertEqual(T.role('ignored'),())
        self.assertEqual(summarize(T,T.columns[0]).frequencies,{'1':192})
        self.assertEqual(len({r.cells[:-1] for r in T.rows}),192)
        self.assertFalse(any(v is None for r in T.rows for v in r.cells))
        with (ROOT/'data/raw/Apache_AllMeasurements.csv').open() as source:
            values=[float(r['Performance-']) for r in csv.DictReader(source)]
        s=summarize(T,T.columns[-1])
        self.assertEqual((s.minimum,s.maximum,s.mean,s.median),(min(values),max(values),statistics.mean(values),statistics.median(values)))
        ranked=rank_rows(T)
        self.assertEqual([r.number for r,_ in ranked[:5]],[125,151,73,91,103])
        for r,score in ranked:self.assertAlmostEqual(score,(values[r.number-1]-840)/1800)

    def test_prediction_baseline_and_labels(self):
        a,b=train_test_split(T); self.assertEqual((len(a.rows),len(b.rows)),(153,39))
        model=KNNRegressor.fit(a); e=evaluate(model,b,apache.TARGET)
        self.assertEqual(model.distance.summaries,(None,)*9)
        mean=statistics.mean(r.cells[-1] for r in a.rows)
        errors=[]
        for row,p in e.predictions:
            # Existing categorical distance is monotone in mismatch count.
            neighbors=sorted(a.rows,key=lambda r:(sum(x!=y for x,y in zip(r.cells[:-1],row.cells[:-1])),r.number))[:5]
            predicted=statistics.mean(r.cells[-1] for r in neighbors)
            self.assertEqual(predicted,p.value)
            self.assertEqual(p,model.predict(apache.masked(row,T),apache.TARGET))
            errors.append(abs(row.cells[-1]-p.value))
        self.assertAlmostEqual(e.mae,statistics.mean(errors))
        self.assertAlmostEqual(e.baseline_mae,statistics.mean(abs(r.cells[-1]-mean) for r in b.rows))
        self.assertAlmostEqual(e.mae,178.615384615)

    def test_observed_one_setting_pair_and_leave_two_out(self):
        a,b,setting,p,q,n=apache.pair_estimate(T)
        self.assertEqual((a.number,b.number,setting,n),(1,34,'handle',190))
        self.assertEqual(sum(a.cells[c.index]!=b.cells[c.index] for c in T.role('x')),1)
        self.assertNotEqual(setting,'base')
        self.assertTrue(all(r.number not in (1,34) for r in p.neighbors+q.neighbors))
        self.assertEqual((p.value,q.value,q.value-p.value),(1308,1158,-150))
        self.assertEqual((q.minimum-p.maximum,q.maximum-p.minimum),(-990,270))
        self.assertEqual(b.cells[-1]-a.cells[-1],30)
        changed=Table(T.columns,tuple(Row(r.number,r.cells[:-1]+(99999,)) if r.number in (1,34) else r for r in T.rows))
        result=apache.pair_estimate(changed)
        self.assertEqual((p,q),result[3:5])

    def test_synthetic_pair_predictions_and_no_pair(self):
        t=table('base,setting,other,Performance-\n1,0,0,999\n1,1,0,-999\n1,0,1,10\n1,1,1,30\n')
        a,b,setting,p,q,n=apache.pair_estimate(t)
        self.assertEqual((a.number,b.number,setting,n),(1,2,'setting',2))
        self.assertEqual((p.value,q.value,p.stddev,q.stddev),(20,20,10,10))
        self.assertIsNone(apache.one_setting_pair(table('a,b,Performance-\n0,0,1\n1,1,2\n')))
        self.assertIsNone(apache.pair_estimate(table('a,Performance-\n0,1\n1,2\n')))

    def test_certify_groups_same_and_small_groups(self):
        for setting in ('keepAlive','accessLog'):
            groups=apache.setting_groups(T,setting)
            self.assertEqual({k:len(v) for k,v in groups.items()},{'0':96,'1':96})
            c=next(c for c in T.columns if c.name==setting)
            self.assertEqual(groups['1'],[r.cells[-1] for r in T.rows if r.cells[c.index]=='1'])
        with patch('shrinking_code.apache.compare',wraps=compare) as used:
            with contextlib.redirect_stdout(io.StringIO()):apache.certify(T)
            self.assertEqual(used.call_count,2)
            for args in used.call_args_list:self.assertFalse(compare(*args.args).same)
        out=io.StringIO()
        with contextlib.redirect_stdout(out):apache.certify(table('keepAlive,accessLog,Performance-\n0,0,1\n1,1,2\n'))
        self.assertIn('same=None',out.getvalue());self.assertIn('Insufficient',out.getvalue())

    def test_proposal_then_completion_shared_component(self):
        t=table('setting,Performance-\n0,10\n1,30\n1,999\n')
        state=ProposalStream(Table(t.columns,t.rows[:2]),('Performance-',),k=1)
        query=apache.masked(t.rows[2],t)
        self.assertEqual(state.assess(query).risks[0].prediction.value,30)
        with self.assertRaises(ValueError):state.assess(t.rows[2])
        self.assertEqual(len(state.inventory.table.rows),2)
        state.complete(t.rows[2]);self.assertEqual(len(state.inventory.table.rows),3)
        self.assertEqual(state.inventory.summaries[-1].maximum,999)

    def test_measured_flow_threshold_before_acceptance(self):
        # First arrival threshold uses only the original training observations.
        a,b=train_test_split(T); observed=[]
        from shrinking_code.streaming import Inventory
        original=Inventory.accept
        def guarded(state,cells,number=None):
            prior={r.number for r in state.table.rows}
            self.assertEqual(prior,{r.number for r in a.rows}|set(observed))
            observed.append(number)
            return original(state,cells,number)
        out=io.StringIO()
        with patch.object(Inventory,'accept',guarded),contextlib.redirect_stdout(out):apache.flow(T)
        self.assertEqual(observed,[r.number for r in b.rows])
        from math import ceil
        q=sorted(r.cells[-1] for r in a.rows)[ceil(.75*len(a.rows))-1]
        self.assertIn(f'arrival={b.rows[0].number} prior_n=153 measured={b.rows[0].cells[-1]} prior_Q75={q}',out.getvalue())
        self.assertIn('accepted=39 alerts=17 final_n=192',out.getvalue())
        self.assertIn('Full-batch verification: PASS',out.getvalue())

    def test_quartiles_and_categorical_diagnosis(self):
        g=outcome_groups(T,apache.TARGET,higher_better=False)
        order=sorted(T.rows,key=lambda r:(r.cells[-1],r.number))
        self.assertEqual(g.poor.rows,tuple(order[-48:]))
        self.assertEqual(g.favorable.rows,tuple(order[:48]))
        for f in compare_groups(g.poor,g.favorable):
            expected=sum(abs(sum(r.cells[f.column.index]==v for r in g.poor.rows)/48-sum(r.cells[f.column.index]==v for r in g.favorable.rows)/48) for v in ('0','1'))/2
            self.assertAlmostEqual(f.separation,expected)
        self.assertNotIn('base',[c.name for c in apache.varying(T)])

    def test_joint_combinations_exclude_constant_and_preserve_missing(self):
        t=table('base,a,b,Performance-\n1,0,0,1\n1,0,1,2\n1,1,0,3\n1,1,1,4\n1,?,1,5\n')
        p=categorical_pairs(t)
        self.assertEqual([c.name for c in p.columns],['pair:a/b'])
        self.assertEqual(p.rows[0].cells,('["0", "0"]',))
        self.assertIsNone(p.rows[-1].cells[0])
        f=compare_groups(Table(p.columns,p.rows[:2]),Table(p.columns,p.rows[2:4]))[0]
        self.assertEqual(f.separation,1)
        p=categorical_pairs(T)
        self.assertEqual(len(p.columns),28)
        c=next(c for c in p.columns if c.name=='pair:keepAlive/handle')
        g=outcome_groups(T,apache.TARGET,higher_better=False)
        self.assertEqual(sum(json.loads(p.rows[r.number-1].cells[c.index])==['1','0'] for r in g.poor.rows),48)
        self.assertEqual(sum(json.loads(p.rows[r.number-1].cells[c.index])==['1','0'] for r in g.favorable.rows),0)

    def test_all_commands_and_previous_hashes(self):
        for skill in ['remember','guess','certify','flow','blame']:
            out=subprocess.check_output([sys.executable,'-m','shrinking_code.demo','--domain','apache','--skill',skill],cwd=ROOT,text=True)
            self.assertIn('APACHE-',out)
        for skill in ['justify','choose','fix','spot','race']:
            result=subprocess.run([sys.executable,'-m','shrinking_code.demo','--domain','apache','--skill',skill],cwd=ROOT,capture_output=True)
            self.assertEqual(result.returncode,2)
        fixture=json.loads((ROOT/'tests/fixtures/pre_apache_demo_hashes.json').read_text())
        for key,digest in fixture.items():
            domain,skill=key.split('/')
            out=subprocess.check_output([sys.executable,'-m','shrinking_code.demo','--domain',domain,'--skill',skill],cwd=ROOT)
            self.assertEqual(hashlib.sha256(out).hexdigest(),digest,key)
