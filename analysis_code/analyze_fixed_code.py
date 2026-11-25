"""Aggregate paired fixed-code policy comparisons from candidate evaluations."""
import collections,csv,hashlib,json,os
from pathlib import Path
import numpy as np
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
R=Path(os.environ.get('PINTRACE_WORKSPACE',Path(__file__).resolve().parents[1]));S=R/'experiments'
def csvwrite(path,records):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def names(lines):return {canonicalize_name(Requirement(x).name) for x in lines}
def fixed():
    p=S/'fixed_code/comparisons.json'
    if not p.exists():return
    cases=json.loads(p.read_text());assert len(cases)==120 and collections.Counter(x['track'] for x in cases)=={'d1':40,'d2':80}
    assert len({(x['track'],x['model'],x['task_id']) for x in cases})==120
    rows=[];prefix=[];drift=[];timings=[]
    for track,label in [('d2','PinTrace / static'),('d1','BigCodeBench / tests')]:
        subset=[x for x in cases if x['track']==track];goal='static_pass' if track=='d2' else 'dynamic_pass';N=len(subset)
        for method in ['original','unpin','neighbor']:
            records=[];recovered=regressed=safe_recovered=0;tested=0
            for x in subset:
                baseline=x['original'];selected=x['neighbor_selected']
                y=x[method] if method!='neighbor' else x['neighbors'][selected-1]['evaluation'] if isinstance(selected,int) else baseline
                assert y['evaluator_revision']=='venv-env-v2'
                assert y['code_sha256']==x['code_sha256'] and names(y['requirements'])==names(x['requirements'])
                for z in x['neighbors']:
                    assert z['evaluation']['evaluator_revision']=='venv-env-v2'
                    assert z['evaluation']['code_sha256']==x['code_sha256'] and names(z['evaluation']['requirements'])==names(x['requirements'])
                if isinstance(selected,int):assert x['neighbors'][selected-1]['evaluation'][goal] and not x['neighbors'][selected-1]['evaluation']['associated']
                records.append(y);recovered+=not baseline[goal] and y[goal];regressed+=baseline[goal] and not y[goal]
                safe_recovered+=not (baseline[goal] and baseline.get('associated') is False) and y[goal] and y.get('associated') is False
                tested+=len(x['neighbors']) if method=='neighbor' else 1
            durations=[]
            for x in subset:
                evaluations=[x[method]] if method!='neighbor' else [v['evaluation'] for v in x['neighbors']]
                durations.append(sum(v.get(stage,{}).get('seconds',0) for v in evaluations for stage in ['resolution','venv','installation','tool_installation','static','installed_packages','dynamic']))
            timings.append({'track':track,'strategy':method,'N':N,'sum_stage_seconds':sum(durations),'median_seconds_per_selected_response':float(np.median(durations)),'p90_seconds_per_selected_response':float(np.quantile(durations,.9))})
            row={'track':track,'population':label,'strategy':method,'N':N,'resolved':sum(y['resolved'] for y in records),'installed':sum(y['installed'] for y in records),'oracle_pass':sum(y[goal] for y in records),'oracle_pass_without_association':sum(y[goal] and y.get('associated') is False for y in records),'recovered':int(recovered),'regressed':int(regressed),'new_joint_successes':int(safe_recovered),'candidate_evaluations':tested}
            rows.append(row)
        for budget in [1,5,10]:
            found=[x for x in subset if isinstance(x['neighbor_selected'],int) and x['neighbor_selected']<=budget]
            prefix.append({'track':track,'N':N,'budget':budget,'verified_replacements':len(found),'new_oracle_passes':sum(not x['original'][goal] for x in found)})
        drift.append({'track':track,'N':N,'static_disagreements':sum(x['historical_static']!=x['original']['static_pass'] for x in subset),'dynamic_disagreements':sum(x['historical_dynamic']!=x['original']['dynamic_pass'] for x in subset) if track=='d1' else None,'association_disagreements':sum(x['historical_associated']!=x['original'].get('associated') for x in subset)})
    summary={'responses':120,'rows':rows,'prefixes':prefix,'timings':timings,'historical_baseline_check':drift,'evaluator_revision':'venv-env-v2','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (S/'fixed_code/summary.json').write_text(json.dumps(summary,indent=2));csvwrite(S/'fixed_code/outcomes.csv',rows);csvwrite(S/'fixed_code/budget_prefixes.csv',prefix);csvwrite(S/'fixed_code/timings.csv',timings)

if __name__=='__main__':fixed()
