"""Turn an LLM reply into files to write and files to delete.

Accepted (labels may be bold, backticked, or carry a trailing note like "(new)"):
  FILE: <relpath>        followed by one fenced block (the file's full text)
  ### <relpath>          a markdown heading naming a file, then a fenced block
  ```<lang> <relpath>    a fence whose info string names a path (also lang:path, title="path")
  DELETE: <relpath>
<think>...</think> blocks are ignored. Anything else is prose. No files and no
deletes, or an unsafe path, is a ReplyParseError.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath
import re


class ReplyParseError(ValueError):
    pass


@dataclass(frozen=True)
class Reply:
    files: dict
    deletes: tuple


_LABEL = re.compile(r"^[\s*_#>-]*(FILE|DELETE)\s*[*_]*\s*:\s*(.*)$")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
_FENCE = re.compile(r"^\s*(```+|~~~+)(.*)$")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_TOKEN = re.compile(r"[^\s`\"'()\[\]]+")
_PATHLIKE = re.compile(r".*[^./\\]\.[A-Za-z][A-Za-z0-9]*$")
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def _path_token(text, fence=False):
    """First path-like token in text, or None. In fence info, 'lang:a.py' and 'title="a.py"' work."""
    for token in _TOKEN.findall(text):
        pieces = re.split(r"[=:]", token) if fence else [token]
        for piece in pieces:
            piece = piece.strip("*,;:").strip()
            if _PATHLIKE.match(piece):
                return piece
    return None


def _clean_path(raw):
    if raw is None:
        raise ReplyParseError("label names no file path")
    path = raw.replace("\\", "/")
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise ReplyParseError(f"absolute path not allowed: {raw!r}")
    if re.search(r'[\x00-\x1f<>:"|?*]', path):
        raise ReplyParseError(f"illegal character in path: {raw!r}")
    parts = [p for p in PurePosixPath(path).parts if p != "."]
    if not parts or ".." in parts:
        raise ReplyParseError(f"path escapes or is empty: {raw!r}")
    for p in parts:
        if len(p) > 120 or p.split(".")[0].lower() in _RESERVED or p.endswith((" ", ".")):
            raise ReplyParseError(f"unusable file name: {raw!r}")
    return "/".join(parts)


def parse_reply(text):
    if not text:
        raise ReplyParseError("empty reply")
    files, deletes, pending = {}, [], None
    lines = _THINK.sub("", text).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if m := _LABEL.match(line):
            path = _clean_path(_path_token(m.group(2)))
            if m.group(1) == "DELETE":
                deletes.append(path)
            else:
                pending = path
            i += 1
            continue
        if (m := _HEADING.match(line)) and (token := _path_token(m.group(1))):
            pending = _clean_path(token)
            i += 1
            continue
        if m := _FENCE.match(line):
            fence, info = m.group(1), m.group(2).strip()
            path = pending
            if path is None and (token := _path_token(info, fence=True)):
                path = _clean_path(token)
            body, i, closed = [], i + 1, False
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith(fence[:3]) and s.strip(fence[0]) == "":
                    closed = True
                    break
                body.append(lines[i])
                i += 1
            if not closed:
                raise ReplyParseError("unclosed code fence")
            if path is not None:
                files[path] = "\n".join(body) + "\n"
            pending = None
        i += 1
    if not files and not deletes:
        raise ReplyParseError("reply contains no files")
    return Reply(files, tuple(deletes))
