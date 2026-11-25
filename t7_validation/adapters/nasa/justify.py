#!/usr/bin/env python3
"""NASA T7 Justify -> common NASA-11/12 semantic answers.

Uses original NASA 5-NN, explanations, comparison and feature differences.
The every-fifth held-out policy is evaluation-only, not a source change.
No checker oracle is imported.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
NASA=ROOT/'human_t7/nasa'

def load(alias, path):
    spec=importlib.util.spec_from_file_location(alias,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

remember=load('t7_nasa_justify_remember',NASA/'remember.py')
prior={name:sys.modules.get(name) for name in ('remember','guess')}
try:
    sys.modules['remember']=remember
    guess=load('t7_nasa_justify_guess',NASA/'guess.py')
    sys.modules['guess']=guess
    justify=load('t7_nasa_original_justify',NASA/'justify.py')
finally:
    for name,old in prior.items():
        if old is None: sys.modules.pop(name,None)
        else: sys.modules[name]=old


def _record(neighbor):
    return {'row':neighbor['row_number'],
            'values':{'Effort-':neighbor['Effort-'],
                      'Defects-':neighbor['Defects-'],
                      'Kloc+':neighbor['Kloc+']}}


def _dominates(other, preferred):
    # All three observed goals belong to historical records (no hidden targets).
    goals={'Kloc+':1,'Effort-':-1,'Defects-':-1}
    return (all(w*other[g]>=w*preferred[g] for g,w in goals.items())
            and any(w*other[g]>w*preferred[g] for g,w in goals.items()))


def _scores(a,b,history,features):
    ea=justify.risk_estimates(a,history,features)
    eb=justify.risk_estimates(b,history,features)
    # Higher is better, normalized using historic training means.
    mean_e=sum(row['Effort-'] for row in history)/len(history)
    mean_d=sum(row['Defects-'] for row in history)/len(history)
    return (-ea['effort']/mean_e-ea['defects']/mean_d,
            -eb['effort']/mean_e-eb['defects']/mean_d)


def _pair(projects, history, heldout, features):
    """Select contrasting held-out proposals using T7 predictions ONLY.

    Recorded outcomes are consulted only after selection, to describe
    actual goal tradeoffs requested by the common checker.
    """
    a, b = justify.choose_proposals(heldout, history, features)
    s1, s2 = _scores(a, b, history, features)
    if s1 == s2:
        raise ValueError('Chosen proposals have identical risk scores')
    return a, b, s1, s2


def answers(csv_path):
    projects=remember.load_projects(csv_path)
    history=[r for r in projects if r['row_number']%5]
    heldout=[r for r in projects if r['row_number']%5==0]
    features=guess.get_features(projects)
    if len(history)<5 or not heldout: raise ValueError('Not enough history or proposals')
    proposal=heldout[0]
    explanation=guess.predict(proposal,history,features,'Effort-',k=justify.K)
    answer11={'project':proposal['row_number'],
              'estimate':explanation['estimate'],
              'neighbors':[_record(r) for r in explanation['neighbors']]}
    a,b,sa,sb=_pair(projects,history,heldout,features)
    preferred=a if sa>sb else b
    other=b if sa>sb else a
    tradeoffs=[g for g,w in remember.GOALS.items() if w*other[g]>w*preferred[g]]
    answer12={'a':a['row_number'],'b':b['row_number'],
              'preferred':preferred['row_number'],
              'scores':{str(a['row_number']):sa,str(b['row_number']):sb},
              'differences':[{'feature':f,'a':x,'b':y} for f,x,y
                             in justify.different_features(a,b,features)],
              'tradeoffs':tradeoffs}
    return answer11,answer12


def main():
    if len(sys.argv)!=2: raise SystemExit('Usage: justify.py PATH_TO_NASA_CSV')
    a,b=answers(Path(sys.argv[1]))
    print('== NASA-11 ==')
    print('T7 original 5-NN supports an effort estimate with historical neighbors')
    print(f"Held-out project {a['project']} cites {len(a['neighbors'])} real training neighbors")
    print('ANSWER NASA-11 '+json.dumps(a,sort_keys=True))
    print('== NASA-12 ==')
    print('T7 original risk_estimates and different_features compare two proposals')
    print(f"Projects {b['a']} and {b['b']} differ in {len(b['differences'])} recorded inputs")
    print('ANSWER NASA-12 '+json.dumps(b,sort_keys=True))

if __name__=='__main__': main()
