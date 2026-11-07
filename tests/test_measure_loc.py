import csv
import io
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.measure_loc import (Git, MeasurementError, category, history, measure,
                                     source_lines, write_results)


class MeasureLOCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'LOC test')
        self.git('config', 'commit.gpgsign', 'false')
        self.git('config', 'tag.gpgsign', 'false')

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE).decode().strip()

    def put(self, path, text):
        p = self.root/path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def commit(self, message='snapshot', tag=None):
        self.git('add', '-A')
        self.git('commit', '-q', '--allow-empty', '-m', message)
        if tag: self.git('tag', tag)
        return self.git('rev-parse', 'HEAD')

    def test_tokens_comments_inline_strings_docstrings(self):
        text = b'# comment\n\nx = 1 # inline\ns = "# string"\n"""doc\n# string content\n\nend"""\n'
        lines, counted = source_lines(text, True)
        self.assertEqual(counted, {3,4,5,6,8})
        self.assertEqual(len(lines), 8)
        with self.assertRaises(MeasurementError): source_lines(b'"""unfinished', True)

    def test_added_modified_deleted_and_consecutive_endpoints(self):
        base = self.commit()
        self.put('shrinking_code/a.py', '# header\nx = 1\n\ny = 2 # inline\n')
        self.put('tests/test_a.py', '# test\nassert True\n')
        self.put('docs/readme.md', '# Heading\n\nProse\n')
        self.put('experiments/script.py', '# infrastructure\nprint(1)\n')
        self.put('data/raw/data.csv', 'x,y\n1,2\n')
        first = self.commit(tag='step1')
        self.put('shrinking_code/a.py', '# changed header\nx = 3\ny = 2 # inline\nz = 4\n')
        self.commit()  # multiple commits, compared endpoints only
        self.put('shrinking_code/b.py', 'b = 5\n')
        second = self.commit(tag='step2')
        (self.root/'shrinking_code/b.py').unlink()
        self.commit(tag='step3')
        rows = measure(self.root, ('One','Two','Three'))
        a,b,c = rows
        self.assertEqual((a['added_impl_loc'],a['removed_impl_loc'],a['total_impl_loc']), (2,0,2))
        self.assertEqual((a['added_test_loc'],a['added_docs_loc'],a['added_infra_loc']), (1,2,1))
        self.assertEqual(a['percent_new_code'], '100.000000')
        self.assertEqual((b['added_impl_loc'],b['removed_impl_loc'],b['total_impl_loc']), (3,1,4))
        self.assertEqual(b['percent_new_code'], '75.000000')
        self.assertEqual((b['added_impl_in_new_files'],b['added_impl_in_existing_files']), (1,2))
        self.assertEqual(b['source_files_changed'],2)
        self.assertEqual((c['added_impl_loc'],c['removed_impl_loc'],c['total_impl_loc']), (0,1,3))
        self.assertEqual([r['base_commit'] for r in rows], [base,first,second])
        self.assertTrue(all(r['reclassified_impl_loc']==0 for r in rows))

    def test_rename_and_rename_with_edit(self):
        self.commit()
        self.put('shrinking_code/old name.py', 'a=1\nb=2\nc=3\nd=4\n')
        self.commit(tag='step1')
        self.git('mv','shrinking_code/old name.py','shrinking_code/new name.py')
        self.commit(tag='step2')
        self.git('mv','shrinking_code/new name.py','shrinking_code/final.py')
        self.put('shrinking_code/final.py','a=1\nb=2\nc=3\nd=5\n')
        self.commit(tag='step3')
        _, rename, edit = measure(self.root, ('A','B','C'))
        self.assertEqual((rename['added_impl_loc'],rename['removed_impl_loc'],rename['total_impl_loc']), (0,0,4))
        self.assertEqual(rename['renamed_source_files'],1)
        self.assertEqual((edit['added_impl_loc'],edit['removed_impl_loc']), (1,1))
        self.assertEqual(edit['renamed_source_files'],1)

    def test_cross_category_rename(self):
        self.commit()
        self.put('shrinking_code/a.py','a=1\n')
        self.commit(tag='step1')
        (self.root/'tests').mkdir()
        self.git('mv','shrinking_code/a.py','tests/test_a.py')
        self.commit(tag='step2')
        r = measure(self.root, ('A','B'))[1]
        self.assertEqual((r['removed_impl_loc'],r['added_test_loc'],r['total_impl_loc']), (1,1,0))
        self.assertEqual(r['percent_new_code'],'')

    def test_unchanged_and_comment_only_edits(self):
        self.commit()
        self.put('shrinking_code/a.py','x=1\n# old\n')
        self.commit(tag='step1')
        self.put('shrinking_code/a.py','x=1\n\n# new\n')
        self.commit(tag='step2')
        self.commit(tag='step3')
        rows=measure(self.root, ('A','B','C'))
        self.assertEqual((rows[1]['added_impl_loc'],rows[1]['removed_impl_loc'],rows[1]['source_files_changed']), (0,0,1))
        self.assertEqual(rows[2]['source_files_changed'],0)

    def test_move_matches_reported_without_claiming_new_logic(self):
        self.commit()
        self.put('shrinking_code/a.py','a=1\nb=2\n')
        self.put('shrinking_code/b.py','c=3\nd=4\n')
        self.commit(tag='step1')
        self.put('shrinking_code/a.py','a=1\n')
        self.put('shrinking_code/b.py','c=3\nd=4\nb=2\n')
        self.commit(tag='step2')
        r=measure(self.root, ('A','B'))[1]
        self.assertEqual((r['added_impl_loc'],r['removed_impl_loc'],r['identical_changed_impl_lines']), (1,1,1))
        self.assertEqual(r['total_impl_loc'],4)

    def test_missing_tags_and_missing_parent(self):
        with self.assertRaises(MeasurementError): measure(self.root, ('A',))
        self.put('shrinking_code/a.py','a=1\n')
        self.commit(tag='step1')
        with self.assertRaisesRegex(MeasurementError,'exactly one parent'): measure(self.root, ('A',))

    def test_duplicate_and_nonancestor_tags(self):
        base=self.commit()
        self.commit(tag='step1')
        self.git('tag','step2')
        with self.assertRaisesRegex(MeasurementError,'same commit'): measure(self.root, ('A','B'))
        self.git('tag','-d','step2')  # Only this throwaway test repository is mutated.
        self.git('checkout','-q','-b','side',base)
        self.commit(tag='step2')
        with self.assertRaises(MeasurementError): measure(self.root, ('A','B'))

    def test_merged_history_rejected(self):
        base=self.commit()
        self.put('shrinking_code/a.py','a=1\n')
        first=self.commit(tag='step1')
        self.git('checkout','-q','-b','side',base)
        self.put('docs/side.md','side\n'); self.commit()
        self.git('checkout','-q','--detach',first)
        self.git('merge','--no-ff','-m','merge','side')
        self.git('tag','step2')
        with self.assertRaisesRegex(MeasurementError,'ambiguous'): measure(self.root, ('A','B'))

    def test_nonrepo_and_noncommit_tag_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(MeasurementError): measure(folder, ('A',))
        self.commit()
        blob=self.git('hash-object','-w','--stdin')
        self.git('tag','step1',blob)
        with self.assertRaises(MeasurementError): measure(self.root, ('A',))

    def test_ten_tags_reproducible_and_worktree_independent(self):
        self.commit()
        for i in range(1,11):
            self.put('shrinking_code/a.py',''.join(f'x{j}={j}\n' for j in range(i)))
            self.commit(tag=f'step{i}')
        self.put('shrinking_code/a.py','this is not the tagged source!\n'*100)
        rows=measure(self.root)
        self.assertEqual([r['added_impl_loc'] for r in rows],[1]*10)
        self.assertEqual([r['total_impl_loc'] for r in rows],list(range(1,11)))
        self.assertEqual(rows[-1]['percent_new_code'],'10.000000')
        out=self.root/'output'
        refs=self.git('show-ref')
        write_results(rows,out)
        first=[p.read_bytes() for p in sorted(out.iterdir())]
        write_results(measure(self.root),out)
        self.assertEqual(first,[p.read_bytes() for p in sorted(out.iterdir())])
        self.assertEqual(refs,self.git('show-ref'))
        records=list(csv.DictReader(io.StringIO((out/'cars_loc.csv').read_text())))
        self.assertEqual(len(records),10)

    def test_category_policy(self):
        self.assertEqual([category(p) for p in ['shrinking_code/a.py','tests/a.py','README.md',
                         'docs/x.md','experiments/a.py','data/raw/a.csv','tests/data.json']],
                         ['impl','test','docs','docs','infra','raw','other'])
