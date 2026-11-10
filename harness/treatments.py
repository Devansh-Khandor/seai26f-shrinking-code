"""The six LLM treatments (uproj.md, "seven ways up the ladder"; T7 is human)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Treatment:
    name: str
    background: str  # plain | sklearn | ezr
    reuse: bool
    lang: str        # python | lua


TREATMENTS = {
    "T1": Treatment("T1", "plain", False, "python"),
    "T2": Treatment("T2", "plain", True, "python"),
    "T3": Treatment("T3", "sklearn", False, "python"),
    "T4": Treatment("T4", "sklearn", True, "python"),
    "T5": Treatment("T5", "ezr", False, "lua"),
    "T6": Treatment("T6", "ezr", True, "lua"),
}
