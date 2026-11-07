"""NASA Batch A transfer cost against step10; working tree provisional by default."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

try:
    from .measure_loc import Git, source_lines, diff_indices
except ImportError:
    from measure_loc import Git, source_lines, diff_indices

SCOPE = {
    'shrinking_code/proposals.py': 'new_shared',
    'shrinking_code/nasa.py': 'nasa_adapter',
    'shrinking_code/demo.py': 'modified_shared',
    'tests/test_nasa.py': 'nasa_tests',
    'tests/test_remember.py': 'test_integration',
    'README.md': 'documentation',
    'experiments/measure_transfer.py': 'measurement_infrastructure',
}


def blob(git, revision, path):
    item = git.run('ls-tree', revision, '--', path).decode().strip()
    if not item:
        return None
    oid = item.split()[2]
    return oid, git.run('cat-file','blob',oid)


def measure_transfer(repo, revision=None):
    git = Git(repo)
    base = git.text('rev-parse','refs/tags/step10^{commit}')
    target = git.text('rev-parse', revision+'^{commit}') if revision else None
    if target:
        git.run('merge-base','--is-ancestor',base,target)
    records = []
    for path, category in SCOPE.items():
        before = blob(git,base,path)
        current = blob(git,target,path) if target else (
            (None,(Path(repo)/path).read_bytes()) if (Path(repo)/path).exists() else None)
        def counted(item):
            if item is None: return None
            lines, eligible = source_lines(item[1],path.endswith('.py'))
            return (category,item[0],lines,eligible)
        a,b = counted(before),counted(current)
        if a is None and b is None: continue
        if target or a is None or b is None:
            removed,added = diff_indices(git,a,b)
        else:
            patch = git.text('diff','--no-ext-diff','--no-textconv','--no-color','--no-renames',
                             '--diff-algorithm=myers','--no-indent-heuristic','--text','--unified=0',base,'--',path)
            removed,added = set(),set()
            old = new = None
            for line in patch.splitlines():
                match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@',line)
                if match: old,new = map(int,match.groups())
                elif old is not None:
                    if line.startswith('-'): removed.add(old); old+=1
                    elif line.startswith('+'): added.add(new); new+=1
                    elif line.startswith(' '): old+=1; new+=1
        records.append(dict(path=path,category=category,added=len(added & b[3]) if b else 0,
                            removed=len(removed & a[3]) if a else 0,total=len(b[3]) if b else 0,
                            sha256=hashlib.sha256(current[1]).hexdigest() if current else None))
    return dict(baseline=base,target=target or 'WORKTREE (provisional)',files=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',help='Accepted commit for finalized measurements; omit for working tree')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    result=measure_transfer(repo,args.revision)
    out=repo/'results'
    out.mkdir(exist_ok=True)
    (out/'nasa_batch_a_loc.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# NASA Batch A transfer LOC','',f"Baseline: `{result['baseline']}` (step10). Target: **{result['target']}**.",'',
           '| File | Category | Added | Removed | Total |','|---|---|---:|---:|---:|']
    for r in result['files']:
        lines.append(f"| {r['path']} | {r['category']} | {r['added']} | {r['removed']} | {r['total']} |")
    lines += ['', 'Counts reuse the Cars nonblank/noncomment policy (including docstrings). Python uses tokenization; documentation counts nonblank text. Added and removed are separate, not net LOC. Modified shared LOC means additions/removals in existing shared dispatch; it is not counted as a new shared algorithm.', '',
              'Scope is explicitly the Batch A files listed above. Existing unchanged shared modules contribute zero additions, not zero reused code. NASA adapter lines include policies and presentation and count as implementation; no duplicated algorithm is discounted as reuse. Generated results and pre-existing Cars measurement infrastructure are excluded. File hashes in the JSON identify measured bytes.', '',
              'Default results are provisional working-tree Git diffs against step10, including new untracked files. Finalize after acceptance with `python3 experiments/measure_transfer.py --revision <accepted-commit>`, which reads Git blobs only. Renames within this explicitly named batch scope are treated as deletion/addition; edits/moves are not mechanically distinguishable from new logic. This is one development sequence, not an independent treatment repetition or evidence for H1/H2. Cars LOC artifacts are not changed.']
    (out/'nasa_batch_a_loc.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
