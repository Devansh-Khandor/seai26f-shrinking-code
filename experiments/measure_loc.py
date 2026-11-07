"""Measure tagged Git snapshots without executing historical code or writing Git state."""

import argparse
from collections import Counter
import csv
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tokenize

SKILLS = ('Remember', 'Guess', 'Certify', 'Flow', 'Blame', 'Justify', 'Choose', 'Fix', 'Spot', 'Race')
CATEGORIES = ('impl', 'test', 'docs', 'infra', 'other')


class MeasurementError(ValueError):
    pass


class Git:
    def __init__(self, repo):
        self.repo = Path(repo)

    def run(self, *args):
        result = subprocess.run(['git', '-c', 'core.quotePath=false', *args], cwd=self.repo,
                                env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'LC_ALL': 'C'},
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise MeasurementError(f'git {" ".join(args)}: {result.stderr.decode(errors="replace").strip()}')
        return result.stdout

    def text(self, *args):
        return self.run(*args).decode('utf-8').strip()


def category(path):
    if path.startswith('data/raw/'):
        return 'raw'
    if path.startswith('experiments/'):
        return 'infra'
    if path.startswith('shrinking_code/') and path.endswith('.py'):
        return 'impl'
    if path.startswith('tests/') and path.endswith('.py'):
        return 'test'
    if path.startswith('docs/') or Path(path).name.lower().startswith('readme') and '/' not in path:
        return 'docs'
    return 'other'


def source_lines(blob, python):
    """Return physical text lines and 1-based eligible LOC indices.

    Docstrings/multiline strings count; comment-only/blank lines do not.
    Tokenization distinguishes '#' inside strings from Python comments.
    """
    try:
        encoding = tokenize.detect_encoding(io.BytesIO(blob).readline)[0] if python else 'utf-8'
        text = blob.decode(encoding)
        lines = text.splitlines()
        if not python:
            return lines, {i for i, line in enumerate(lines, 1) if line.strip()}
        eligible = set()
        ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                   tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type not in ignored:
                eligible.update(range(token.start[0], token.end[0] + 1))
        return lines, {i for i in eligible if i <= len(lines) and lines[i-1].strip()}
    except (UnicodeError, SyntaxError, tokenize.TokenError) as error:
        raise MeasurementError(f'Cannot count source: {error}') from error


def history(git, skills=SKILLS):
    milestones = []
    previous = None
    for step, skill in enumerate(skills, 1):
        tag = f'step{step}'
        ref = f'refs/tags/{tag}'
        tag_object = git.text('rev-parse', '--verify', ref)
        commit = git.text('rev-parse', '--verify', ref + '^{commit}')
        if step == 1:
            parents = git.text('rev-list', '--parents', '-n', '1', commit).split()[1:]
            if len(parents) != 1:
                raise MeasurementError('step1 must have exactly one parent commit')
            previous = parents[0]
        if previous == commit:
            raise MeasurementError(f'{tag}: consecutive milestones resolve to the same commit')
        git.run('merge-base', '--is-ancestor', previous, commit)
        if git.text('rev-list', '--merges', f'{previous}..{commit}'):
            raise MeasurementError(f'{tag}: merged history is ambiguous; expected a linear milestone interval')
        milestones.append(dict(step=step, skill=skill, tag=tag, tag_object=tag_object,
                               commit=commit, base_commit=previous))
        previous = commit
    return milestones


def snapshot(git, commit):
    result = {}
    for item in git.run('ls-tree', '-r', '-z', commit).split(b'\0'):
        if not item:
            continue
        metadata, path_bytes = item.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        path = path_bytes.decode('utf-8')
        cat = category(path)
        if cat in ('raw', 'other'):
            continue
        if mode not in ('100644', '100755') or kind != 'blob':
            raise MeasurementError(f'{commit}:{path}: unsupported nonregular measured file')
        lines, eligible = source_lines(git.run('cat-file', 'blob', oid), path.endswith('.py') and cat != 'docs')
        result[path] = (cat, oid, lines, eligible)
    return result


def changed_paths(git, base, commit):
    parts = git.run('diff', '--no-ext-diff', '--no-textconv', '--name-status', '-z',
                    '--find-renames=50%', '-l0', base, commit).decode('utf-8').split('\0')
    i = 0
    while i < len(parts) - 1:
        status, old = parts[i:i+2]
        i += 2
        if status.startswith('R'):
            new = parts[i]; i += 1
        elif status in ('A', 'M', 'D', 'T'):
            new = old
        else:
            raise MeasurementError(f'Unsupported diff status: {status}')
        yield status, None if status == 'A' else old, None if status == 'D' else new


def diff_indices(git, before, after):
    if before is None:
        return set(), set(range(1, len(after[2])+1))
    if after is None:
        return set(range(1, len(before[2])+1)), set()
    patch = git.text('diff', '--no-ext-diff', '--no-textconv', '--no-color', '--no-renames',
                     '--diff-algorithm=myers', '--no-indent-heuristic', '--text', '--unified=0', before[1], after[1])
    removed, added = set(), set()
    old = new = None
    for line in patch.splitlines():
        match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
        if match:
            old, new = map(int, match.groups())
        elif old is not None:
            if line.startswith('-'):
                removed.add(old); old += 1
            elif line.startswith('+'):
                added.add(new); new += 1
            elif line.startswith(' '):
                old += 1; new += 1
    return removed, added


def measure(repo, skills=SKILLS):
    git = Git(repo)
    milestones = history(git, skills)
    rows = []
    before = snapshot(git, milestones[0]['base_commit'])
    for milestone in milestones:
        row = dict(milestone)
        after = snapshot(git, row['commit'])
        for cat in CATEGORIES[:-1]:
            row[f'added_{cat}_loc'] = row[f'removed_{cat}_loc'] = 0
            row[f'total_{cat}_loc'] = sum(len(v[3]) for v in after.values() if v[0] == cat)
        row['source_files_changed'] = row['added_impl_in_existing_files'] = 0
        row['added_impl_in_new_files'] = row['renamed_source_files'] = 0
        additions, deletions = Counter(), Counter()
        for status, old, new in changed_paths(git, row['base_commit'], row['commit']):
            a, b = before.get(old), after.get(new)
            if a is None and b is None:
                continue
            # A category boundary is a removal from one category and addition to another.
            cross = a is not None and b is not None and a[0] != b[0]
            if cross:
                removed, added = set(range(1,len(a[2])+1)), set(range(1,len(b[2])+1))
            else:
                removed, added = diff_indices(git, a, b)
            is_source = any(v and v[0] == 'impl' for v in (a,b))
            row['source_files_changed'] += int(is_source)
            row['renamed_source_files'] += int(is_source and status.startswith('R'))
            for value, indices, direction, counter in ((a,removed,'removed',deletions), (b,added,'added',additions)):
                if value is None:
                    continue
                cat, _, lines, eligible = value
                counted = indices & eligible
                row[f'{direction}_{cat}_loc'] += len(counted)
                if cat == 'impl':
                    counter.update(lines[i-1] for i in counted)
                    if direction == 'added':
                        existing = a is not None and a[0] == 'impl'
                        row['added_impl_in_existing_files' if existing else 'added_impl_in_new_files'] += len(counted)
        row['identical_changed_impl_lines'] = sum((additions & deletions).values())
        old_total = sum(len(v[3]) for v in before.values() if v[0] == 'impl')
        row['base_impl_loc'] = old_total
        row['reclassified_impl_loc'] = row['total_impl_loc'] - old_total - row['added_impl_loc'] + row['removed_impl_loc']
        row['percent_new_code'] = (f"{100*row['added_impl_loc']/row['total_impl_loc']:.6f}"
                                   if row['total_impl_loc'] else '')
        rows.append(row)
        before = after
    return rows


def markdown(rows):
    values = [r['added_impl_loc'] for r in rows]
    n = len(rows)
    xmean, ymean = (n+1)/2, sum(values)/n
    slope = sum((i-xmean)*(v-ymean) for i,v in enumerate(values,1)) / sum((i-xmean)**2 for i in range(1,n+1)) if n>1 else 0
    increases = [str(i+1) for i in range(1,n) if values[i]>values[i-1]]
    out = ['# Cars tagged LOC measurements', '',
           'Regenerate with `python3 experiments/measure_loc.py`. Source: resolved Git tags, not working-tree files.', '',
           '| Step | Skill | Tag | Commit | Added impl | Removed impl | Total impl | New % | Added tests | Added docs | Source files changed |',
           '|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        out.append('| ' + ' | '.join(str(r[k]) for k in ('step','skill','tag','commit','added_impl_loc','removed_impl_loc','total_impl_loc','percent_new_code','added_test_loc','added_docs_loc','source_files_changed')) + ' |')
    out += ['', '## Methodology', '',
        'Step 1 uses its sole parent; later steps use the preceding tag. Tags are peeled to commits; missing tags, repeated commits, non-ancestor steps, and merge-containing intervals fail. Multiple linear commits are supported by endpoint comparison. CSV records both tag-object and commit IDs plus every comparison base.', '',
        'Primary LOC is physical nonblank, noncomment Python lines in shrinking_code/. Python tokenization excludes comment-only lines, retains inline-code lines, and counts docstrings/multiline strings as string statements (including nonblank continuation lines). It does not execute source. Blank string lines are excluded. Tests use the same rule. Documentation (root README* and docs/) counts nonblank text lines. experiments/ is separate infrastructure: Python uses token counting, other files use nonblank text. Raw datasets and other unclassified files (including JSON test fixtures/results) are excluded from LOC.', '',
        'Added LOC counts eligible added lines, never net growth. Deleted LOC is separate. Percentage = 100 × added implementation LOC / total implementation LOC at that tag; an empty denominator is undefined (empty CSV field). Git Myers diffs disable indent heuristics, external diffs, and textconv. Rename detection uses 50% similarity; unchanged renames count zero added/deleted LOC. Crossing category boundaries transfers the file LOC into the new category.', '',
        'CSV separates additions in new versus existing implementation files, renamed source-file counts, and identical added/deleted line matches (possible moves or repeated text). These matches are not subtracted from the primary metric: identical lines do not prove semantic equivalence or a move. Git diffs cannot perfectly distinguish edits, moves, copies, refactoring, and new logic. Replacements count as deletion plus addition. Cross-file/block moves can still count as additions; endpoint diffs omit churn that was added and removed between tags. Git version/rename heuristics can affect ambiguous matches. Source-files-changed counts implementation change records, including renames and comment-only edits.', '',
        'The reclassified_impl_loc column exposes any count change in unchanged physical lines caused by lexical context changes, e.g. multiline strings. It reconciles snapshot totals with added/deleted counts; it is not silently attributed to new code. No semantic functionality or effort metric is inferred.', '',
        '## Growth and descriptive trend', '',
        f"Baseline implementation LOC: {rows[0]['base_impl_loc']}; final: {rows[-1]['total_impl_loc']}; net growth: {rows[-1]['total_impl_loc']-rows[0]['base_impl_loc']}.",
        f"Sum of added implementation LOC: {sum(values)}; removed: {sum(r['removed_impl_loc'] for r in rows)}; lexical reclassification adjustment: {sum(r['reclassified_impl_loc'] for r in rows)}.",
        f"New implementation LOC sequence: {values}.",
        f"Descriptive least-squares slope: {slope:.6f} LOC per step ({'upward' if slope>0 else 'downward' if slope<0 else 'flat'}). Increases over the previous milestone occur at steps: {', '.join(increases) or 'none'}.",
        f"New-code percentage sequence: {[r['percent_new_code'] for r in rows]}.", '',
        'A declining percentage can result from a growing cumulative denominator even when absolute additions rise. Absolute and percentage trends must not be conflated. This one sequential LLM-assisted development history is descriptive, not an independent experimental repetition or statistical support for H1/H2. No same test is applied to dependent milestones. No seven-treatment or cross-domain conclusion follows.', '',
        '## Category totals and traceability', '',
        '| Tag | Base commit | Total tests | Total docs | Added infrastructure | Removed infrastructure | Total infrastructure |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        out.append('| ' + ' | '.join(str(r[k]) for k in ('tag','base_commit','total_test_loc','total_docs_loc','added_infra_loc','removed_infra_loc','total_infra_loc')) + ' |')
    return '\n'.join(out)+'\n'


def write_results(rows, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # No timestamp or machine-specific paths: identical snapshots produce identical artifacts.
    with (destination/'cars_loc.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    (destination/'cars_loc.md').write_text(markdown(rows), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='Default: <repo>/results')
    args = parser.parse_args(argv)
    try:
        rows = measure(args.repo)
        write_results(rows, args.output or args.repo/'results')
    except (MeasurementError, OSError) as error:
        parser.exit(1, f'LOC measurement failed: {error}\n')
    print(f"Measured {len(rows)} tags; final implementation LOC: {rows[-1]['total_impl_loc']}")


if __name__ == '__main__':
    main()
