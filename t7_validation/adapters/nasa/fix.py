#!/usr/bin/env python3
"""NASA original T7 Fix -> Jack's NASA-15/16 predicted-repair contract.

Reuse original T7 find_repair and 5-NN predictor; restrict predictions to
historical targets, never the candidate's hidden goals. The inspection
protocol uses every fifth row as proposals, as in Jack's common harness.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / 'human_t7/nasa'


def load(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remember = load('t7_nasa_fix_remember', NASA / 'remember.py')
previous = {name: sys.modules.get(name) for name in ('remember', 'guess')}
try:
    sys.modules['remember'] = remember
    guess = load('t7_nasa_fix_guess', NASA / 'guess.py')
    sys.modules['guess'] = guess
    fix = load('t7_nasa_original_fix', NASA / 'fix.py')
finally:
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def repair_for(projects, history, proposals, features, target):
    """Choose a high predicted-risk held-out proposal with a valid repair.

    The proposal order depends only on history and candidate inputs. No
    candidate's recorded effort/defect targets can affect the selection.
    """
    scored = []
    for proposal in proposals:
        masked = {k: v for k, v in proposal.items() if k not in remember.GOALS}
        baseline = fix.predict_outcomes(masked, history, features)
        scored.append((baseline[target], -proposal['row_number'], masked))
    scored.sort(reverse=True)
    for _, _, proposal in scored:
        baseline, repair, _ = fix.find_repair(
            proposal, history, features, target,
            fix.MAX_EFFORT_INCREASE if target == 'Defects-' else None,
        )
        if repair is not None:
            # Every change comes from original T7 and is a historical value.
            answer = {
                'project': proposal['row_number'],
                'changes': {k: [old, new] for k, (old, new) in repair['changes'].items()},
                f"predicted_{'effort' if target == 'Effort-' else 'defects'}": {
                    'before': baseline[target], 'after': repair['predicted'][target]
                },
            }
            if target == 'Defects-':
                answer['predicted_effort'] = {
                    'before': baseline['Effort-'],
                    'after': repair['predicted']['Effort-'],
                }
            return answer
    raise ValueError(f'No valid original T7 repair among {len(proposals)} proposals for {target}')


def answers(csv_path):
    projects = remember.load_projects(csv_path)
    history = [r for r in projects if r['row_number'] % 5]
    proposals = [r for r in projects if r['row_number'] % 5 == 0]
    features = guess.get_features(projects)
    return (repair_for(projects, history, proposals, features, 'Effort-'),
            repair_for(projects, history, proposals, features, 'Defects-'))


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: fix.py PATH_TO_NASA_CSV')
    effort, defects = answers(Path(sys.argv[1]))
    print('== NASA-15 ==')
    print('5-NN effort-repair prediction using historical project goals only')
    print('Original T7 restricts repairs to observed categorical settings and at most two changes')
    print('ANSWER NASA-15 ' + json.dumps(effort, sort_keys=True))
    print('== NASA-16 ==')
    print('5-NN defect-repair prediction with predicted effort tradeoff')
    print('The repair must reduce predicted defects without increasing effort more than five percent')
    print('ANSWER NASA-16 ' + json.dumps(defects, sort_keys=True))


if __name__ == '__main__':
    main()
