"""Aggregate all 145,040 final task-model-condition-runtime observations."""
import collections,csv,json,os
from pathlib import Path
R=Path(os.environ['PINTRACE_WORKSPACE']);OUT=R/'experiments/runtime'
METRICS=['resolver_input','complete_lock','installation_completed','static_pass','dynamic_pass']
def main():
    groups=collections.defaultdict(collections.Counter);seen=set()
    with (OUT/'observations.jsonl').open(encoding='utf-8') as source:
        for line in source:
            row=json.loads(line);key=row['request_id'],row['python_version']
            assert key not in seen;seen.add(key)
            m=row['metrics'];assert m['static_pass']<=m['installation_completed']<=m['complete_lock']<=m['resolver_input']
            g=groups[row['track'],row['mode'],row['model'],row['python_version']]
            g['n']+=1;g.update({k:int(m[k]) for k in METRICS})
    assert len(seen)==145040 and len(groups)==160
    by=[];ag=collections.defaultdict(collections.Counter)
    for (track,mode,model,runtime),counts in sorted(groups.items()):
        assert counts['n']==(813 if track=='d1' else 1000)
        x=dict(track=track,mode=mode,model=model,python_version=runtime,**counts)
        x.update({k+'_pct':100*counts[k]/counts['n'] for k in METRICS});by.append(x)
        ag[track,mode,runtime].update(counts)
    allrows=[]
    for (track,mode,runtime),counts in sorted(ag.items()):
        x=dict(track=track,mode=mode,python_version=runtime,**counts)
        x.update({k+'_pct':100*counts[k]/counts['n'] for k in METRICS});allrows.append(x)
    for name,rows in [('python_robustness_by_model.csv',by),('python_robustness_all_observations.csv',allrows)]:
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Runtime observations:',len(seen))
if __name__=='__main__':main()
