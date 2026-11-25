"""The six LLM treatments (uproj.md, "seven ways up the ladder"; T7 is human).

Protocol v2 (2026-11-17): T5/T6 use the course's Python ezr (vendor/python/ezr.py)
instead of the Lua ezr used for the v1 trials.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Treatment:
    name: str
    background: str  # plain | sklearn | ezr
    reuse: bool
    lang: str        # python (lua was v1's T5/T6)


TREATMENTS = {
    "T1": Treatment("T1", "plain", False, "python"),
    "T2": Treatment("T2", "plain", True, "python"),
    "T3": Treatment("T3", "sklearn", False, "python"),
    "T4": Treatment("T4", "sklearn", True, "python"),
    "T5": Treatment("T5", "ezr", False, "python"),
    "T6": Treatment("T6", "ezr", True, "python"),
}
