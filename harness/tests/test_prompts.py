from pathlib import Path
import unittest

from harness.prompts import build_prompt, csv_preview, entry_name, template_hash, VENDOR_DIR
from harness.requirements import load_domain
from harness.treatments import TREATMENTS, Treatment

CARS = load_domain("cars")
CERTIFY = CARS.skills[2]
REUSE = "Reuse and extend the existing skills' code as much as you can."
FRESH = "Write this skill as a fresh, self-contained solution."


def prompt(t, prior=None):
    return build_prompt(CARS, CERTIFY, TREATMENTS[t], prior or {})


class TestPrompts(unittest.TestCase):
    def test_treatment_table(self):
        self.assertEqual(TREATMENTS["T4"], Treatment("T4", "sklearn", True, "python"))
        self.assertEqual(TREATMENTS["T5"].lang, "lua")
        self.assertEqual(sorted(TREATMENTS), ["T1", "T2", "T3", "T4", "T5", "T6"])

    def test_requests_verbatim(self):
        p = prompt("T1")
        for r in CERTIFY.requests:
            self.assertIn(r.id, p)
            self.assertIn(r.text, p)

    def test_reuse_vs_fresh(self):
        self.assertIn(REUSE, prompt("T2"))
        self.assertNotIn(FRESH, prompt("T2"))
        self.assertIn(FRESH, prompt("T1"))
        self.assertNotIn(REUSE, prompt("T1"))

    def test_ezr_background(self):
        t5, t1 = prompt("T5"), prompt("T1")
        for s in ("function same(", 'require("ezr")', "skill_03.lua"):
            self.assertIn(s, t5)
            self.assertNotIn(s, t1)
        self.assertIn("skill_03.py", t1)

    def test_sklearn_background(self):
        self.assertIn("scikit-learn", prompt("T3"))
        self.assertNotIn("scikit-learn", prompt("T1"))

    def test_prior_code_included(self):
        p = prompt("T1", {"skill_01.py": "def f(): pass\n"})
        self.assertIn("FILE: skill_01.py", p)
        self.assertIn("def f(): pass", p)
        self.assertIn("(none yet)", prompt("T1"))

    def test_contract_headers(self):
        p = prompt("T1")
        self.assertIn("== <ID> ==", p)
        self.assertIn("DELETE:", p)

    def test_csv_preview(self):
        lines = csv_preview(CARS.csv_path).splitlines()
        self.assertTrue(lines[0].startswith("Clndrs,Volume"))
        self.assertEqual(len(lines), 6)

    def test_entry_name(self):
        self.assertEqual(entry_name(3, "python"), "skill_03.py")
        self.assertEqual(entry_name(10, "lua"), "skill_10.lua")

    def test_vendor_unmodified(self):
        src = Path.home() / "git/seai26f/src/ezr-lua"
        if not (src / "ezr.lua").exists():
            self.skipTest("course repo not checked out")
        for name in ("ezr.lua", "ezr-lib.lua"):
            self.assertEqual((src / name).read_bytes(), (VENDOR_DIR / name).read_bytes())

    def test_hash_stable(self):
        self.assertEqual(template_hash(), template_hash())
        self.assertEqual(len(template_hash()), 12)


if __name__ == "__main__":
    unittest.main()
