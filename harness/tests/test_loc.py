import unittest

from harness.loc import StepLoc, is_test_path, measure_step

TEN = "".join(f"x{i} = {i}\n" for i in range(10))


class TestLoc(unittest.TestCase):
    def test_new_file(self):
        self.assertEqual(measure_step({}, {"a.py": TEN}), StepLoc(10, 0, 0, 10, 100.0))

    def test_one_line_modified(self):
        cur = TEN.replace("x3 = 3", "x3 = 33")
        s = measure_step({"a.py": TEN}, {"a.py": cur})
        self.assertEqual((s.new_loc, s.deleted_loc, s.total_loc), (1, 1, 10))
        self.assertEqual(s.pct_new, 10.0)

    def test_comments_and_blanks_ignored(self):
        s = measure_step({}, {"a.py": "# c\n\nx = 1\n", "b.lua": "-- c\nlocal y = 2\n"})
        self.assertEqual(s.new_loc, 2)

    def test_tests_separate(self):
        s = measure_step({}, {"a.py": "x = 1\n", "tests/test_a.py": "assert 1\nassert 2\n"})
        self.assertEqual((s.new_loc, s.test_loc, s.total_loc), (1, 2, 1))

    def test_unchanged(self):
        self.assertEqual(measure_step({"a.py": TEN}, {"a.py": TEN}).new_loc, 0)

    def test_non_code_ignored(self):
        self.assertEqual(measure_step({}, {"notes.md": "hello\n"}).total_loc, 0)

    def test_deleted_file(self):
        s = measure_step({"a.py": TEN, "b.py": "y = 1\n"}, {"b.py": "y = 1\n"})
        self.assertEqual((s.new_loc, s.deleted_loc, s.total_loc), (0, 10, 1))

    def test_is_test_path(self):
        self.assertTrue(is_test_path("tests/x.py"))
        self.assertTrue(is_test_path("test_skill.py"))
        self.assertTrue(is_test_path("lib/skill_test.lua"))
        self.assertFalse(is_test_path("skill_01.py"))


if __name__ == "__main__":
    unittest.main()
