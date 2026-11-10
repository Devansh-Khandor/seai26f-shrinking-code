"""Tier 1 (runs) and tier 2 (output contract) checks on generated code.

The program runs in a throwaway copy of the step snapshot, so files it writes
never become "code" in the snapshot, the next prompt, or the LOC count.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from harness.prompts import VENDOR_DIR, entry_name

MAX_OUTPUT = 20_000          # characters kept in CheckResult / check.txt
CAP_BYTES = 2_000_000        # a program printing more than this is killed
_HEADER = re.compile(r"^== (\S+) ==$")


@dataclass(frozen=True)
class CheckResult:
    runs: bool
    contract: bool
    timed_out: bool
    seconds: float
    stdout: str
    stderr: str
    output_capped: bool = False
    created: tuple = ()      # files the program wrote while running (discarded)


def lua_executable():
    return os.environ.get("HARNESS_LUA") or shutil.which("lua") or shutil.which("luajit")


def python_has_sklearn():
    p = subprocess.run([sys.executable, "-c", "import sklearn"], capture_output=True)
    return p.returncode == 0


def contract_ok(stdout, request_ids):
    sections, current = {}, None
    for line in stdout.splitlines():
        if m := _HEADER.match(line.strip()):
            current = m.group(1)
            sections.setdefault(current, [])
        elif current is not None and line.strip():
            sections[current].append(line)
    for rid in request_ids:
        body = sections.get(rid)
        if not body or len(body) < 3 or not any(re.search(r"\d", l) for l in body):
            return False
    return True


def _env(workdir, lang):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    if lang == "lua":
        env["LUA_PATH"] = env["LUA_PATH_5_4"] = f"{VENDOR_DIR}/?.lua;{workdir}/?.lua;;"
    return env


def _kill_tree(p):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(p.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        p.kill()


def _read(path):
    with open(path, "rb") as f:
        return f.read(CAP_BYTES).decode("utf-8", errors="replace")


def run_check(step, skill, treatment, csv_path, timeout=120.0):
    step = Path(step)
    name = entry_name(skill.index, treatment.lang)
    if not (step / name).is_file():
        return CheckResult(False, False, False, 0.0, "", f"missing entry point {name}")
    if treatment.lang == "lua":
        exe = lua_executable()
        if exe is None:
            return CheckResult(False, False, False, 0.0, "", "no Lua interpreter found")
    else:
        exe = sys.executable
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        shutil.copytree(step, work, ignore=shutil.ignore_patterns("__pycache__"))
        before = {p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()}
        out_path, err_path = Path(tmp) / "stdout", Path(tmp) / "stderr"
        start = time.monotonic()
        timed_out = capped = False
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            p = subprocess.Popen([exe, name, str(csv_path)], cwd=work, env=_env(work, treatment.lang),
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 start_new_session=(os.name != "nt"))
            while p.poll() is None:
                if time.monotonic() - start > timeout:
                    timed_out = True
                elif out_path.stat().st_size + err_path.stat().st_size > CAP_BYTES:
                    capped = True
                if timed_out or capped:
                    _kill_tree(p)
                    p.wait()
                    break
                time.sleep(0.05)
        seconds = time.monotonic() - start
        stdout, stderr = _read(out_path), _read(err_path)
        created = tuple(sorted(p.relative_to(work).as_posix() for p in work.rglob("*")
                               if p.is_file() and "__pycache__" not in p.parts
                               and p.relative_to(work).as_posix() not in before))
    runs = not timed_out and not capped and p.returncode == 0
    contract = runs and contract_ok(stdout, [r.id for r in skill.requests])
    return CheckResult(runs, contract, timed_out, seconds, stdout[:MAX_OUTPUT], stderr[:MAX_OUTPUT],
                       capped, created)
