from pathlib import Path
import tempfile
import unittest

from harness.reply import Reply
from harness.workspace import apply_step, read_code, save_exchange, slug, step_dir, trial_dir


class TestWorkspace(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.t = trial_dir(Path(self._tmp.name), "m:1", "cars", "T1", 1)

    def tearDown(self):
        self._tmp.cleanup()

    def test_trial_dir_layout(self):
        self.assertEqual(self.t, Path(self._tmp.name) / "m-1" / "cars" / "T1" / "trial1")

    def test_apply_first_step(self):
        apply_step(self.t, 1, Reply({"skill_01.py": "a\n"}, ()))
        self.assertEqual(read_code(step_dir(self.t, 1)), {"skill_01.py": "a\n"})

    def test_carry_forward_and_modify(self):
        apply_step(self.t, 1, Reply({"skill_01.py": "a\n"}, ()))
        apply_step(self.t, 2, Reply({"skill_02.py": "b\n", "lib/u.py": "c\n"}, ()))
        self.assertEqual(read_code(step_dir(self.t, 2)),
                         {"skill_01.py": "a\n", "skill_02.py": "b\n", "lib/u.py": "c\n"})
        self.assertEqual(read_code(step_dir(self.t, 1)), {"skill_01.py": "a\n"})

    def test_delete(self):
        apply_step(self.t, 1, Reply({"skill_01.py": "a\n"}, ()))
        apply_step(self.t, 2, Reply({}, ("skill_01.py",)))
        self.assertNotIn("skill_01.py", read_code(step_dir(self.t, 2)))

    def test_none_carries_forward(self):
        apply_step(self.t, 1, Reply({"skill_01.py": "a\n"}, ()))
        apply_step(self.t, 2, None)
        self.assertEqual(read_code(step_dir(self.t, 2)), read_code(step_dir(self.t, 1)))

    def test_rebuild_half_written_step(self):
        apply_step(self.t, 1, Reply({"skill_01.py": "a\n"}, ()))
        junk = step_dir(self.t, 2) / "junk.py"
        junk.parent.mkdir(parents=True)
        junk.write_text("x\n")
        apply_step(self.t, 2, None)
        self.assertNotIn("junk.py", read_code(step_dir(self.t, 2)))

    def test_exchange_outside_snapshot(self):
        apply_step(self.t, 1, Reply({"skill_01.py": "a\n"}, ()))
        save_exchange(self.t, 1, "P", "R")
        self.assertTrue((self.t / "step01.prompt.txt").exists())
        self.assertTrue((self.t / "step01.response.txt").exists())
        self.assertFalse(any(p.endswith(".txt") for p in read_code(step_dir(self.t, 1))))

    def test_pycache_skipped(self):
        apply_step(self.t, 1, Reply({"skill_01.py": "a\n"}, ()))
        (step_dir(self.t, 1) / "__pycache__").mkdir()
        (step_dir(self.t, 1) / "__pycache__" / "x.pyc").write_bytes(b"\x00\xff")
        self.assertEqual(list(read_code(step_dir(self.t, 1))), ["skill_01.py"])

    def test_slug(self):
        self.assertEqual(slug("qwen2.5-coder:14b"), "qwen2.5-coder-14b")


if __name__ == "__main__":
    unittest.main()
