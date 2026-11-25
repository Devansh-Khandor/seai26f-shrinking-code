from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from harness.loc import measure_step
from harness.reply import Reply
from harness.results import append_row
from harness.tag import TagError, tag_name, tag_runs, tree_files
from harness.workspace import apply_step, read_code, step_dir, trial_dir

MODEL = "fake-model"
SKILLS = ("remember", "guess", "certify", "flow", "blame", "justify", "choose", "fix", "spot", "race")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


@unittest.skipIf(shutil.which("git") is None, "git required")
class TestTag(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.repo, self.runs, self.csv = root / "repo", root / "runs", root / "runs.csv"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "Test")
        git(self.repo, "config", "user.email", "test@example.com")
        self.out = []

    def tearDown(self):
        self._tmp.cleanup()

    def fake_trial(self, trial=1, fail=(), stop_after=10, model=MODEL):
        """Write snapshots + rows the way run_step does; steps in `fail` get no reply."""
        t = trial_dir(self.runs, model, "cars", "T1", trial)
        for i in range(1, stop_after + 1):
            prev = read_code(step_dir(t, i - 1)) if i > 1 else {}
            reply = None if i in fail else Reply({f"skill_{i:02d}.py": f"x = {i}\n# note\ny = {i}\n"}, ())
            step = apply_step(t, i, reply)
            row = {"model": model, "domain": "cars", "treatment": "T1", "trial": trial,
                   "skill_index": i, "skill": SKILLS[i - 1],
                   "timestamp": f"2026-11-20T10:{i:02d}:00+00:00", "prompt_template_hash": "abc"}
            if i in fail:
                row["status"] = "parse_failed"
            else:
                loc = measure_step(prev, read_code(step))
                row.update(status="ok", new_loc=loc.new_loc, deleted_loc=loc.deleted_loc,
                           test_loc=loc.test_loc, total_loc=loc.total_loc, pct_new=loc.pct_new)
            append_row(self.csv, row)
        return t

    def tag(self, model=MODEL):
        return tag_runs(model, self.runs, self.csv, repo=self.repo, out=self.out.append)

    def test_tags_each_step_with_its_snapshot(self):
        self.fake_trial()
        self.assertEqual(self.tag(), [("cars", "T1", 1)])
        tags = git(self.repo, "tag", "-l").split()
        self.assertEqual(sorted(tags), sorted(tag_name("T1", "cars", 1, i) for i in range(1, 11)))
        self.assertEqual(tree_files(self.repo, "t1-cars-r1-step3"),
                         {f"skill_0{i}.py": f"x = {i}\n# note\ny = {i}\n" for i in (1, 2, 3)})
        self.assertEqual(tree_files(self.repo, "t1-cars-r1-step1^"), {})
        self.assertIn("(fake-model)", git(self.repo, "log", "-1", "--format=%s", "t1-cars-r1-step5"))

    def test_protocol_prefix_keeps_v1_and_v2_apart(self):
        self.fake_trial()
        tag_runs(MODEL, self.runs, self.csv, repo=self.repo, out=self.out.append, prefix="v2-")
        tags = git(self.repo, "tag", "-l").split()
        self.assertEqual(sorted(tags), sorted(tag_name("T1", "cars", 1, i, "v2-") for i in range(1, 11)))
        self.assertTrue(all(t.startswith("v2-t1-cars-r1-step") for t in tags))
        self.assertEqual(self.tag(), [("cars", "T1", 1)])  # unprefixed (v1) names are a separate set
        self.assertEqual(len(git(self.repo, "tag", "-l").split()), 20)

    def test_no_branch_or_ref_left_behind(self):
        self.fake_trial()
        self.tag()
        refs = git(self.repo, "for-each-ref", "--format=%(refname)").split()
        self.assertTrue(all(r.startswith("refs/tags/") for r in refs), refs)

    def test_rerun_keeps_tags_and_adds_new_trial(self):
        self.fake_trial(1)
        self.tag()
        before = git(self.repo, "rev-parse", "t1-cars-r1-step10")
        self.fake_trial(4)
        self.assertEqual(self.tag(), [("cars", "T1", 4)])
        self.assertEqual(git(self.repo, "rev-parse", "t1-cars-r1-step10"), before)
        self.assertIn("1 already tagged", self.out[-3])

    def test_failed_first_step_carries_forward_without_snapshot(self):
        t = self.fake_trial(fail=(1, 4))
        shutil.rmtree(step_dir(t, 1))  # the results zip drops empty step folders
        self.tag()
        self.assertEqual(tree_files(self.repo, "t1-cars-r1-step1"), {})
        self.assertEqual(tree_files(self.repo, "t1-cars-r1-step4"), tree_files(self.repo, "t1-cars-r1-step3"))

    def test_unfinished_trial_is_skipped(self):
        self.fake_trial(stop_after=7)
        self.assertEqual(self.tag(), [])
        self.assertEqual(git(self.repo, "tag", "-l"), "")
        self.assertTrue(any("not finished, skipped: T1 cars trial 1" in line for line in self.out))

    def test_missing_ok_snapshot_is_an_error(self):
        t = self.fake_trial()
        shutil.rmtree(step_dir(t, 6))
        with self.assertRaises(TagError):
            self.tag()
        self.assertEqual(git(self.repo, "tag", "-l"), "")

    def test_mismatch_with_runs_csv_is_reported(self):
        self.fake_trial()
        self.tag()
        lines = self.csv.read_text(encoding="utf-8").splitlines()
        cols = lines[0].split(",")
        row = lines[3].split(",")
        row[cols.index("new_loc")] = "99"
        lines[3] = ",".join(row)
        self.csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(TagError):
            self.tag()
        self.assertTrue(any("MISMATCH t1-cars-r1-step3" in line for line in self.out))

    def test_partly_tagged_trial_is_an_error(self):
        self.fake_trial()
        self.tag()
        git(self.repo, "tag", "-d", "t1-cars-r1-step7")
        with self.assertRaises(TagError):
            self.tag()

    def test_tags_from_another_model_are_flagged(self):
        self.fake_trial(model="other-model")
        tag_runs("other-model", self.runs, self.csv, repo=self.repo, out=self.out.append)
        self.fake_trial()
        with self.assertRaises(TagError):
            self.tag()


if __name__ == "__main__":
    unittest.main()
