"""NASA transfer costs: accepted Batch A, provisional Batch B, and cumulative."""

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
    'shrinking_code/hypotheticals.py': 'new_domain_independent',
    'shrinking_code/explanation.py': 'shared_explanation_additions',
    'tests/test_nasa_batch_b.py': 'nasa_tests',
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


def measure_transfer(repo, revision=None, base_ref='step10'):
    git = Git(repo)
    base = git.text('rev-parse',base_ref+'^{commit}')
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


def write_report(repo, result, label):
    out=repo/'results'
    out.mkdir(exist_ok=True)
    stem='nasa_'+label+'_loc'
    (out/(stem+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    lines=['# NASA '+label.replace('_',' ')+' transfer LOC','',
           f"Baseline: `{result['baseline']}`. Target: **{result['target']}**.",'',
           '| File | Category | Added | Removed | Total |','|---|---|---:|---:|---:|']
    for r in result['files']:
        lines.append(f"| {r['path']} | {r['category']} | {r['added']} | {r['removed']} | {r['total']} |")
    impl=[r for r in result['files'] if r['path'].startswith('shrinking_code/')]
    lines += ['',f"Implementation additions: {sum(r['added'] for r in impl)}; removals: {sum(r['removed'] for r in impl)}.",'',
              'Batch A is finalized from the accepted nasa-batch-a Git checkpoint. Batch B is relative to that checkpoint; cumulative transfer is relative to step10. WORKTREE targets are provisional, never labeled finalized. Use --revision <accepted-commit> to finalize Batch B and cumulative reports after acceptance.', '',
              'Counts reuse the Cars nonblank/noncomment policy including docstrings. Python uses tokenization; documentation counts nonblank lines. Added/removed are separate. Shared additions in existing files and new generic capabilities are distinguished from NASA adapter and dispatch. Existing-file additions include edits, not necessarily new logic; block moves/renames can count as additions. Endpoint cumulative counts need not equal the sum of batch additions because edits can replace earlier additions.', '',
              'The explicit scope includes all NASA implementation, tests, README, and transfer measurement source; unchanged shared modules add zero LOC, not zero reuse. Regression JSON fixtures and generated reports are excluded artifacts, not Python LOC. Content hashes identify measured bytes. New untracked scope files are included provisionally. Cars LOC outputs remain unchanged. This is development-history measurement, not an independent treatment experiment.']
    (out/(stem+'.md')).write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',help='Accepted Batch B commit; omit for provisional working tree')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    git=Git(repo)
    accepted=git.text('rev-parse','refs/tags/nasa-batch-a^{commit}')
    for label,revision,base in [('batch_a',accepted,'step10'),('batch_b',args.revision,'nasa-batch-a'),
                                ('cumulative',args.revision,'step10')]:
        result=measure_transfer(repo,revision,base)
        write_report(repo,result,label)
        print(label, result['target'])


if __name__=='__main__':
    main()
