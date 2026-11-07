import contextlib
import csv
import hashlib
import io
import math
from pathlib import Path
from random import Random
import statistics
import subprocess
import sys
import unittest
from unittest.mock import patch

from shrinking_code.data import Row, Table
from shrinking_code.stats import summarize
from shrinking_code.ranking import rank_rows
from shrinking_code.evaluation import train_test_split,evaluate
from shrinking_code.prediction import KNNRegressor
from shrinking_code.certification import compare
from shrinking_code.diagnosis import outcome_groups,compare_groups
from shrinking_code.proposals import ProposalStream
from shrinking_code import nasa

ROOT=Path(__file__).resolve().parents[1]
TABLE=Table.load(ROOT/'data/raw/nasa93dem.csv')


def masked(row,columns):
    return Row(row.number,tuple(row.cells[c.index] if c.role=='x' else None for c in columns))


class NasaRememberTests(unittest.TestCase):
    def test_schema_and_audit(self):
        self.assertEqual((len(TABLE.rows),len(TABLE.columns)),(93,29))
        self.assertEqual(len(TABLE.role('x')),22)
        self.assertTrue(all(not c.numeric for c in TABLE.role('x')))
        self.assertEqual([(c.name,c.ideal) for c in TABLE.role('y')],[('Kloc+',1),('Effort-',0),('Defects-',0)])
        self.assertEqual([c.name for c in TABLE.role('ignored')],['idX','centerX','YearX','MonthsX'])
        self.assertFalse(any(v is None for r in TABLE.rows for v in r.cells))
        self.assertEqual(len({tuple(r.cells[c.index] for c in TABLE.role('x')) for r in TABLE.rows}),55)
        self.assertEqual(sum(len({r.cells[c.index] for r in TABLE.rows})==1 for c in TABLE.columns),8)

    def test_raw_summaries_and_scores(self):
        with (ROOT/'data/raw/nasa93dem.csv').open() as source: raw=list(csv.DictReader(source))
        columns=TABLE.role('y')
        ranges={c.name:[float(r[c.name]) for r in raw] for c in columns}
        for c in columns:
            values=ranges[c.name]; s=summarize(TABLE,c)
            self.assertEqual((s.minimum,s.maximum,s.mean,s.median),(min(values),max(values),statistics.mean(values),statistics.median(values)))
        expected=[]
        for number,row in enumerate(raw,1):
            parts=[]
            for name,ideal in [('Kloc+',1),('Effort-',0),('Defects-',0)]:
                lo,hi=min(ranges[name]),max(ranges[name])
                parts.append(((float(row[name])-lo)/(hi-lo)-ideal)**2)
            expected.append((math.sqrt(sum(parts)/3),number))
        expected.sort()
        for (score,number),(r,actual) in zip(expected,rank_rows(TABLE)):
            self.assertEqual(number,r.number); self.assertAlmostEqual(score,actual)


class NasaGuessTests(unittest.TestCase):
    def test_split_predictions_baseline_and_leakage(self):
        a,b=train_test_split(TABLE); ids=list(range(1,94)); Random(42).shuffle(ids)
        self.assertEqual([r.number for r in a.rows],sorted(ids[:74]))
        self.assertEqual([r.number for r in b.rows],sorted(ids[74:]))
        model=KNNRegressor.fit(a)
        self.assertEqual(model.distance.summaries,(None,)*22)
        for target in nasa.TARGETS:
            c=model.target_column(target); result=evaluate(model,b,target)
            baseline=statistics.mean(r.cells[c.index] for r in a.rows)
            errors=[]
            for row,p in result.predictions:
                # All X are categories: independent Hamming distance, including constants.
                neighbors=sorted(a.rows,key=lambda r:(round(math.sqrt(sum(r.cells[x.index]!=row.cells[x.index] for x in a.role('x'))/22),12),r.number))[:5]
                expected=statistics.mean(r.cells[c.index] for r in neighbors)
                self.assertEqual(p.value,expected)
                self.assertEqual(p,model.predict(masked(row,TABLE.columns),target))
                errors.append(abs(row.cells[c.index]-expected))
            self.assertAlmostEqual(result.mae,statistics.mean(errors))
            self.assertAlmostEqual(result.baseline_mae,statistics.mean(abs(r.cells[c.index]-baseline) for r in b.rows))
        self.assertEqual((len(a.rows),len(b.rows)),(74,19))


class NasaCertifyTests(unittest.TestCase):
    def test_rating_groups_and_comparisons(self):
        for rating,target in [('pcap','Effort-'),('cplx','Defects-')]:
            groups=nasa.rating_groups(TABLE,rating,target)
            c=next(c for c in TABLE.columns if c.name==rating)
            goal=next(c for c in TABLE.columns if c.name==target)
            self.assertEqual(sum(map(len,groups.values())),93)
            for key,values in groups.items():
                self.assertEqual(values,[r.cells[goal.index] for r in TABLE.rows if r.cells[c.index]==key])
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream): nasa.certify(TABLE)
            self.assertIn('same=False',stream.getvalue())
            self.assertIn('fewer than five',stream.getvalue())
        t=Table.from_csv(io.StringIO('pcap,cplx,Effort-,Defects-\nl,h,1,2\nh,l,3,4\n'))
        out=io.StringIO()
        with contextlib.redirect_stdout(out): nasa.certify(t)
        self.assertIn('same=None',out.getvalue())
        self.assertIn('insufficient',out.getvalue())
        with patch('shrinking_code.nasa.compare',wraps=compare) as used:
            with contextlib.redirect_stdout(io.StringIO()): nasa.certify(TABLE)
            self.assertGreater(used.call_count,0)


class NasaFlowTests(unittest.TestCase):
    def test_known_threshold_predictions_and_updates(self):
        t=Table.from_csv(io.StringIO('Size,Effort-,Defects-\n0,0,0\n1,10,10\n2,20,20\n3,30,30\n3,100,100\n4,0,0\n'))
        state=ProposalStream(Table(t.columns,t.rows[:4]),nasa.TARGETS,k=1)
        p=state.assess(masked(t.rows[4],t.columns))
        self.assertEqual(p.reference_count,4)
        self.assertEqual([(r.prediction.value,r.threshold,r.high) for r in p.risks],[(30,20,True)]*2)
        self.assertEqual(len(state.inventory.table.rows),4)
        with self.assertRaisesRegex(ValueError,'X only'): state.assess(t.rows[4])
        state.complete(t.rows[4])
        next_p=state.assess(masked(t.rows[5],t.columns))
        self.assertEqual(next_p.risks[0].threshold,30)
        self.assertEqual(next_p.risks[0].prediction.value,30) # tied X distance chooses original ID 4
        self.assertFalse(next_p.risks[0].high)
        with self.assertRaises(ValueError): state.complete(t.rows[4])

    def test_future_labels_not_available_and_batch_consistency(self):
        a,b=train_test_split(TABLE); state=ProposalStream(a,nasa.TARGETS)
        original=KNNRegressor.predict
        seen=set(r.number for r in a.rows)
        def guarded(model,row,target):
            self.assertTrue(all(row.cells[c.index] is None for c in TABLE.columns if c.role!='x'))
            self.assertEqual({r.number for r in model.training.rows},seen)
            return original(model,row,target)
        with patch.object(KNNRegressor,'predict',guarded):
            for row in b.rows:
                query=masked(row,TABLE.columns)
                changed=Row(row.number,tuple(999999 if c.role=='y' else row.cells[c.index] for c in TABLE.columns))
                self.assertEqual(state.assess(query),state.assess(masked(changed,TABLE.columns)))
                state.complete(row); seen.add(row.number)
        self.assertEqual(state.inventory.table,TABLE)
        self.assertEqual(state.inventory.summaries,tuple(summarize(TABLE,c) for c in TABLE.columns))
        self.assertEqual(state.inventory.accepted_count,19)

    def test_completion_changes_future_predictions(self):
        t=Table.from_csv(io.StringIO('kind,Effort-\na,1\nb,2\nc,100\nc,0\n'))
        state=ProposalStream(Table(t.columns,t.rows[:2]),('Effort-',),k=1)
        query=masked(t.rows[3],t.columns)
        self.assertEqual(state.assess(query).risks[0].prediction.value,1)
        state.complete(t.rows[2])
        self.assertEqual(state.assess(query).risks[0].prediction.value,100)


class NasaBlameTests(unittest.TestCase):
    def test_quartiles_and_category_tv(self):
        for target in nasa.TARGETS:
            c=next(c for c in TABLE.columns if c.name==target)
            order=sorted(TABLE.rows,key=lambda r:(r.cells[c.index],r.number))
            g=outcome_groups(TABLE,target,higher_better=False)
            self.assertEqual(g.poor.rows,tuple(order[-23:])); self.assertEqual(g.favorable.rows,tuple(order[:23]))
            findings=compare_groups(g.poor,g.favorable)
            self.assertTrue(all(not f.column.numeric for f in findings))
            for f in findings:
                values={r.cells[f.column.index] for r in TABLE.rows}
                expected=sum(abs(sum(r.cells[f.column.index]==v for r in g.poor.rows)/23-sum(r.cells[f.column.index]==v for r in g.favorable.rows)/23) for v in values)/2
                self.assertAlmostEqual(f.separation,expected)


class NasaIntegrationTests(unittest.TestCase):
    def test_five_commands_and_rejection_of_later_skills(self):
        for skill in ['remember','guess','certify','flow','blame']:
            out=subprocess.check_output([sys.executable,'-m','shrinking_code.demo','--domain','nasa','--skill',skill],cwd=ROOT,text=True)
            self.assertIn('NASA /',out)
        for skill in ['justify','choose','fix','spot','race']:
            result=subprocess.run([sys.executable,'-m','shrinking_code.demo','--domain','nasa','--skill',skill],cwd=ROOT,capture_output=True)
            self.assertEqual(result.returncode,2)

    def test_cars_outputs_against_step10(self):
        # Capture regression hashes from the pre-transfer run, not current output.
        import json
        hashes=json.loads((ROOT/'tests/fixtures/pre_nasa_demo_hashes.json').read_text())
        for skill,digest in hashes.items():
            out=subprocess.check_output([sys.executable,'-m','shrinking_code.demo','--domain','cars','--skill',skill],cwd=ROOT)
            self.assertEqual(hashlib.sha256(out).hexdigest(),digest,skill)

class TransferMeasurementTests(unittest.TestCase):
    def test_provisional_and_commit_sources(self):
        import tempfile
        from experiments.measure_transfer import measure_transfer
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def git(*args):
                return subprocess.check_output(['git',*args],cwd=root,stdin=subprocess.DEVNULL,stderr=subprocess.PIPE).decode().strip()
            git('init','-q'); git('config','user.name','Test'); git('config','user.email','test@example.invalid')
            git('config','commit.gpgsign','false'); git('config','tag.gpgsign','false')
            (root/'shrinking_code').mkdir()
            (root/'shrinking_code/demo.py').write_text('x=1\n')
            git('add','.'); git('commit','-qm','base'); git('tag','step10')
            (root/'shrinking_code/demo.py').write_text('x=2\ny=3\n')
            (root/'shrinking_code/proposals.py').write_text('# excluded\na=1\n')
            provisional=measure_transfer(root)
            by_path={r['path']:r for r in provisional['files']}
            self.assertEqual((by_path['shrinking_code/demo.py']['added'],by_path['shrinking_code/demo.py']['removed']),(2,1))
            self.assertEqual(by_path['shrinking_code/proposals.py']['added'],1)
            git('add','.'); git('commit','-qm','accepted')
            committed=measure_transfer(root,'HEAD')
            self.assertEqual(provisional['files'],committed['files'])
            (root/'shrinking_code/demo.py').write_text('x=999\n'*10)
            self.assertEqual(committed,measure_transfer(root,'HEAD'))
