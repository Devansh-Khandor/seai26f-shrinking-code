import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from harness.checks import contract_ok, lua_executable, run_check
from harness.requirements import load_domain
from harness.treatments import TREATMENTS

CARS = load_domain("cars")
CERTIFY = CARS.skills[2]  # CAR-05, CAR-06
IDS = ["CAR-05", "CAR-06"]
GOOD = "== CAR-05 ==\na 1\nb\nc\n== CAR-06 ==\nx 2\ny\nz\n"
PRINT_GOOD = "print(" + repr(GOOD) + ")\n"


class TestContract(unittest.TestCase):
    def test_contract_pass(self):
        self.assertTrue(contract_ok(GOOD, IDS))

    def test_contract_missing_section(self):
        self.assertFalse(contract_ok(GOOD.split("== CAR-06")[0], IDS))

    def test_contract_thin_section(self):
        self.assertFalse(contract_ok("== CAR-05 ==\na 1\nb\nc\n== CAR-06 ==\nok\n", IDS))

    def test_contract_no_number(self):
        self.assertFalse(contract_ok("== CAR-05 ==\na 1\nb\nc\n== CAR-06 ==\na\nb\nc\n", IDS))

    def test_contract_header_with_whitespace(self):
        self.assertTrue(contract_ok("  == CAR-05 ==  \n1\n2\n3\n== CAR-06 ==\n4\n5\n6\n", IDS))


class TestRunCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.step = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        (self.step / name).write_text(text, encoding="utf-8")

    def test_run_python_ok(self):
        self.write("skill_03.py", "import sys\nassert sys.argv[1].endswith('auto93.csv')\n" + PRINT_GOOD)
        r = run_check(self.step, CERTIFY, TREATMENTS["T1"], CARS.csv_path)
        self.assertTrue(r.runs and r.contract, r.stderr)

    def test_run_python_crash(self):
        self.write("skill_03.py", PRINT_GOOD + "raise SystemExit(1)\n")
        r = run_check(self.step, CERTIFY, TREATMENTS["T1"], CARS.csv_path)
        self.assertFalse(r.runs)
        self.assertFalse(r.contract)

    def test_run_timeout(self):
        self.write("skill_03.py", "import time\ntime.sleep(10)\n")
        r = run_check(self.step, CERTIFY, TREATMENTS["T1"], CARS.csv_path, timeout=1)
        self.assertTrue(r.timed_out)
        self.assertFalse(r.runs)

    def test_stdin_closed(self):
        self.write("skill_03.py", "input()\n" + PRINT_GOOD)
        r = run_check(self.step, CERTIFY, TREATMENTS["T1"], CARS.csv_path, timeout=5)
        self.assertFalse(r.timed_out)
        self.assertFalse(r.runs)

    def test_contract_on_long_output(self):
        self.write("skill_03.py", "print('== CAR-05 ==')\nfor i in range(3000): print('row', i, 'x' * 20)\n"
                                  "print('== CAR-06 ==')\nprint('a 1')\nprint('b 2')\nprint('c 3')\n")
        r = run_check(self.step, CERTIFY, TREATMENTS["T1"], CARS.csv_path)
        self.assertTrue(r.contract)
        self.assertLessEqual(len(r.stdout), 20_000)

    def test_output_capped(self):
        self.write("skill_03.py", "import sys\nwhile True: sys.stdout.write('y' * 65536)\n")
        r = run_check(self.step, CERTIFY, TREATMENTS["T1"], CARS.csv_path, timeout=3)
        self.assertFalse(r.runs)
        self.assertTrue(r.output_capped or r.timed_out)

    def test_unicode_output_ok(self):
        self.write("skill_03.py", "print('→ ≥ ✓')\n" + PRINT_GOOD)
        with mock.patch.dict(os.environ, {"PYTHONIOENCODING": "ascii"}):
            r = run_check(self.step, CERTIFY, TREATMENTS["T1"], CARS.csv_path)
        self.assertTrue(r.runs, r.stderr)

    def test_api_keys_hidden(self):
        self.write("skill_03.py", "import os\nassert 'OPENAI_API_KEY' not in os.environ\n" + PRINT_GOOD)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}):
            r = run_check(self.step, CERTIFY, TREATMENTS["T1"], CARS.csv_path)
        self.assertTrue(r.runs, r.stderr)

    def test_missing_entry(self):
        r = run_check(self.step, CERTIFY, TREATMENTS["T1"], CARS.csv_path)
        self.assertFalse(r.runs)
        self.assertFalse(r.timed_out)

    @unittest.skipUnless(shutil.which("lua") or shutil.which("luajit"), "Lua not installed")
    def test_run_lua_uses_vendor(self):
        lua_print = "".join(f"print({line!r})\n".replace("'", '"') for line in GOOD.splitlines())
        self.write("skill_03.lua", 'local ezr = require("ezr")\nprint(ezr.same({1,2,3},{1,2,3}))\n' + lua_print)
        r = run_check(self.step, CERTIFY, TREATMENTS["T5"], CARS.csv_path)
        self.assertTrue(r.runs, r.stderr)
        self.assertTrue(r.contract)
        self.assertIn("true", r.stdout)
        self.assertIsNotNone(lua_executable())


if __name__ == "__main__":
    unittest.main()
