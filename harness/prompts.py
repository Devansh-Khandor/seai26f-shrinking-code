"""Build the prompt for skill i+1 from the templates in harness/templates/."""

import hashlib
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "templates"
VENDOR_DIR = HERE / "vendor"
EZR_FILES = ("ezr.lua", "ezr-lib.lua")
FENCE_LANG = {".py": "python", ".lua": "lua"}

CLIENTS = {
    "cars": "an independent used-car dealership",
    "nasa": "a software engineering project manager",
    "apache": "an Apache web server administrator",
}


def _t(name):
    return Template((TEMPLATE_DIR / f"{name}.txt").read_text(encoding="utf-8"))


def entry_name(skill_index, lang):
    return f"skill_{skill_index:02d}.{'py' if lang == 'python' else 'lua'}"


def csv_preview(path, rows=5):
    with open(path, encoding="utf-8") as f:
        return "".join(line for _, line in zip(range(rows + 1), f)).rstrip("\n")


def _file_block(path, text):
    lang = FENCE_LANG.get(Path(path).suffix, "")
    return f"FILE: {path}\n```{lang}\n{text.rstrip(chr(10))}\n```"


def _background(treatment):
    if treatment.background != "ezr":
        return _t(f"background_{treatment.background}").substitute()
    files = "\n\n".join(_file_block(n, (VENDOR_DIR / n).read_text(encoding="utf-8")) for n in EZR_FILES)
    return _t("background_ezr").substitute(ezr_files=files)


def build_prompt(domain, skill, treatment, prior_code):
    entry = entry_name(skill.index, treatment.lang)
    prior = "\n\n".join(_file_block(p, t) for p, t in sorted(prior_code.items())) or "(none yet)"
    requests = "\n".join(f"- **{r.id}:** {r.text}" for r in skill.requests)
    contract = _t(f"contract_{treatment.lang}").substitute(
        entry=entry, run_cmd="python" if treatment.lang == "python" else "lua",
        example_header=f"== {skill.requests[0].id} ==")
    return _t("base").substitute(
        client=CLIENTS.get(domain.name, domain.name),
        background=_background(treatment).rstrip("\n"),
        csv_name=domain.csv_path.name,
        csv_preview=csv_preview(domain.csv_path),
        prior_code=prior,
        skill_index=skill.index,
        skill_name=skill.name,
        requests=requests,
        approach=_t("reuse" if treatment.reuse else "fresh").substitute().rstrip("\n"),
        contract=contract.rstrip("\n"),
    )


def template_hash():
    h = hashlib.sha256()
    for path in sorted(TEMPLATE_DIR.glob("*.txt")) + [VENDOR_DIR / n for n in EZR_FILES]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]
