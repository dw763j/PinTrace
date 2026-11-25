"""Paired task-bootstrap analysis of the five-model repeated-generation experiment."""
import collections,csv,hashlib,json,os
from pathlib import Path
import numpy as np
R=Path(os.environ.get('PINTRACE_WORKSPACE',Path(__file__).resolve().parents[1]))
O=R/'experiments/repeated'
MODELS=['qwen/qwen3-30b-a3b-instruct-2507','minimax/minimax-m2.5','openai/gpt-5.4','anthropic/claude-sonnet-4.6','google/gemini-3.1-pro-preview']
SHORT=dict(zip(MODELS,['Qwen3 30B','MiniMax M2.5','GPT-5.4','Claude Sonnet 4.6','Gemini 3.1 Pro']))
def csvwrite(name,records):
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def main():
    data=[json.loads(l) for l in (O/'evaluations.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(data)==len({x['request_id'] for x in data})==2400
    byrepeat=[];paired=[];agreement=[];states=[];directions=[];stages=[];completion_sensitivity=[];rng=np.random.default_rng(20261110);bootstrap_indices={}
    for track,N in [('d2',60),('d1',20)]:
        taskids=sorted({x['task_id'] for x in data if x['track']==track});assert len(taskids)==N
        indices=rng.integers(0,N,size=(10000,N))
        bootstrap_indices[track]=indices
        for model in MODELS:
            cohort=[x for x in data if x['track']==track and x['model']==model]
            assert len(cohort)==N*6 and {x['task_id'] for x in cohort}==set(taskids)
            lookup={(x['task_id'],x['mode'],x['repeat']):x for x in cohort};assert len(lookup)==N*6
            for metric in (['exact','static_pass'] if track=='d2' else ['dynamic_pass']):
                values={mode:np.array([[int(bool(lookup[t,mode,k].get(metric))) for k in [1,2,3]] for t in taskids]) for mode in ['requirements.txt','inline']}
                for mode,mat in values.items():
                    rates=mat.mean(axis=0)*100
                    for i in range(3):byrepeat.append(dict(track=track,model=model,mode=mode,repeat=i+1,metric=metric,numerator=int(mat[:,i].sum()),denominator=N,percentage=float(rates[i])))
                    same=np.all(mat==mat[:,[0]],axis=1)
                    agreement.append(dict(track=track,model=model,mode=mode,metric=metric,agreeing_tasks=int(same.sum()),all_success_tasks=int(np.all(mat==1,axis=1).sum()),all_failure_tasks=int(np.all(mat==0,axis=1).sum()),varying_tasks=int((~same).sum()),tasks=N,min_repeat_pct=float(rates.min()),max_repeat_pct=float(rates.max())))
                delta=(values['inline']-values['requirements.txt']).mean(axis=1)
                samples=delta[indices].mean(axis=1)*100;d=(values['inline']-values['requirements.txt']).mean(axis=0)*100
                paired.append(dict(track=track,model=model,metric=metric,tasks=N,observations_per_condition=N*3,mr_successes=int(values['requirements.txt'].sum()),va_successes=int(values['inline'].sum()),mr_pct=float(values['requirements.txt'].mean()*100),va_pct=float(values['inline'].mean()*100),delta_pp=float(delta.mean()*100),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975))))
                directions.append(dict(track=track,model=model,metric=metric,repeat1_delta_pp=float(d[0]),repeat2_delta_pp=float(d[1]),repeat3_delta_pp=float(d[2]),direction_changes=bool(d.min()<0<d.max())))
                complete=np.array([all(lookup[t,mode,k].get('finish_reason')=='stop' for mode in ['requirements.txt','inline'] for k in [1,2,3]) for t in taskids])
                cm=values['requirements.txt'][complete];cv=values['inline'][complete];nc=int(complete.sum())
                completion_sensitivity.append(dict(track=track,model=model,metric=metric,complete_tasks=nc,excluded_tasks=N-nc,observations_per_condition=nc*3,mr_successes=int(cm.sum()),va_successes=int(cv.sum()),delta_pp=float((cv-cm).mean()*100) if nc else None,repeat1_delta_pp=float((cv-cm).mean(axis=0)[0]*100) if nc else None,repeat2_delta_pp=float((cv-cm).mean(axis=0)[1]*100) if nc else None,repeat3_delta_pp=float((cv-cm).mean(axis=0)[2]*100) if nc else None))
    for model in MODELS:
        for mode in ['requirements.txt','inline']:
            z=[x for x in data if x['model']==model and x['mode']==mode]
            states.append(dict(model=model,mode=mode,N=len(z),normal=sum(x.get('finish_reason')=='stop' for x in z),length=sum(x.get('finish_reason')=='length' for x in z),missing=sum(x['api_status']!='response' for x in z),other_finish=sum(x['api_status']=='response' and x.get('finish_reason') not in ['stop','length'] for x in z)))
            states[-1]['empty_answer']=sum(x.get('content_sha256')==hashlib.sha256(b'').hexdigest() for x in z)
            for track in ['d2','d1']:
                subset=[x for x in z if x['track']==track]
                stage=dict(track=track,model=model,mode=mode,N=len(subset),usable_inputs=sum(bool(x.get('input')) for x in subset),resolved=sum(bool(x.get('resolved')) for x in subset),installed=sum(bool(x.get('installed')) for x in subset),static_pass=sum(bool(x.get('static_pass')) for x in subset),dynamic_pass=sum(bool(x.get('dynamic_pass')) for x in subset) if track=='d1' else None)
                for name in ['resolution','venv','installation','tool_installation','static','dynamic']:
                    stage[name+'_timeouts']=sum(bool((x.get('evaluation') or {}).get(name,{}).get('timeout')) for x in subset)
                stages.append(stage)
    summary=dict(planned_responses=2400,models=SHORT,rows=byrepeat,paired=paired,agreement=agreement,response_states=states,directions=directions,evaluation_stages=stages,completion_sensitivity=completion_sensitivity,bootstrap=dict(seed=20261110,resamples=10000,unit='task',method='percentile',paired=True,repeats_retained=3))
    (O/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for name,records in [('by_repeat.csv',byrepeat),('paired_intervals.csv',paired),('agreement.csv',agreement),('response_states.csv',states),('repeat_directions.csv',directions),('evaluation_stages.csv',stages),('completion_sensitivity.csv',completion_sensitivity)]:csvwrite(name,records)
    print('Repeated generation: 2400 observations, five models, three rounds.')
if __name__=='__main__':main()
