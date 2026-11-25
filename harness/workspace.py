"""Per-trial directories: one code snapshot per step, prompts/responses beside them."""

from pathlib import Path
import re
import shutil


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]", "-", text)


def trial_dir(runs_root, model, domain, treatment, trial):
    return Path(runs_root) / slug(model) / domain / treatment / f"trial{trial}"


def step_dir(trial, i):
    return Path(trial) / f"step{i:02d}"


def apply_step(trial, i, reply):
    """Build step i from step i-1 plus the reply (None = carry forward)."""
    target, prev = step_dir(trial, i), step_dir(trial, i - 1)
    if target.exists():
        shutil.rmtree(target)
    if i > 1 and prev.is_dir():
        shutil.copytree(prev, target, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        target.mkdir(parents=True)
    if reply is not None:
        for path in reply.deletes:
            victim = target / path
            if victim.is_file():
                victim.unlink()
        for path, text in reply.files.items():
            dest = target / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8", newline="\n")
    return target


def read_code(step):
    step = Path(step)
    out = {}
    if not step.is_dir():
        return out
    for path in sorted(step.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            out[path.relative_to(step).as_posix()] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return out


def save_exchange(trial, i, prompt, response):
    trial = Path(trial)
    trial.mkdir(parents=True, exist_ok=True)
    (trial / f"step{i:02d}.prompt.txt").write_text(prompt, encoding="utf-8")
    if response is not None:
        (trial / f"step{i:02d}.response.txt").write_text(response, encoding="utf-8")
