"""Parse the frozen customer requirements into the ten-skill ladder."""

from dataclasses import dataclass
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent

SKILLS = ("remember", "guess", "certify", "flow", "blame",
          "justify", "choose", "fix", "spot", "race")

DOMAIN_FILES = {
    "cars": ("docs/requirements_cars.md", "data/raw/auto93.csv"),
    "nasa": ("docs/requirements_nasa.md", "data/raw/nasa93dem.csv"),
    "apache": ("docs/requirements_apache.md", "data/raw/Apache_AllMeasurements.csv"),
}

_HEADING = re.compile(r"^### (\d+)\. (\w+) — ")
_REQUEST = re.compile(r"^\*\*([A-Z]+-\d+):\*\* (.+)$")


@dataclass(frozen=True)
class Request:
    id: str
    text: str


@dataclass(frozen=True)
class Skill:
    index: int
    name: str
    requests: tuple


@dataclass(frozen=True)
class Domain:
    name: str
    csv_path: Path
    skills: tuple


def parse_requirements(text):
    skills, current = [], None
    for line in text.splitlines():
        line = line.rstrip()
        if m := _HEADING.match(line):
            current = (int(m.group(1)), m.group(2).lower(), [])
            skills.append(current)
        elif (m := _REQUEST.match(line)) and current is not None:
            current[2].append(Request(m.group(1), m.group(2).strip()))
    if [s[1] for s in skills] != list(SKILLS):
        raise ValueError(f"expected skills {SKILLS}, found {[s[1] for s in skills]}")
    for index, name, reqs in skills:
        if len(reqs) != 2:
            raise ValueError(f"skill {name} has {len(reqs)} requests, expected 2")
    return tuple(Skill(i, n, tuple(r)) for i, n, r in skills)


def load_domain(name, repo_root=REPO_ROOT):
    if name not in DOMAIN_FILES:
        raise ValueError(f"unknown domain {name!r}; choose from {sorted(DOMAIN_FILES)}")
    req_path, csv_path = DOMAIN_FILES[name]
    text = (Path(repo_root) / req_path).read_text(encoding="utf-8")
    return Domain(name, Path(repo_root) / csv_path, parse_requirements(text))
