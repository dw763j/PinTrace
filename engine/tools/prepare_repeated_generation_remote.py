"""Freeze paired tasks and original prompts before supplemental generation."""
import argparse, collections, hashlib, json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/data/mine/llm_tpl'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
def rows(p):return [json.loads(l) for l in p.open() if l.strip()]
def h(s):return hashlib.sha256(('20261110|'+s).encode()).hexdigest()
R=a.root;meta=json.loads(a.metadata.read_text())
def raw(track,mode,model):
    entry=next(x for x in meta if x['track']==track and x['mode']==mode and model.lower() in x['path'].lower())
    p=Path(entry['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],str(p)
    return {str(x.get('task_id') or x.get('record_id')):x for x in rows(p)},str(p),hashlib.sha256(p.read_bytes()).hexdigest()
clean={str(x['task_id']) for x in rows(R/'reproduce/resources/revision/pintrace_clean_task_ids.jsonl')}
pin=rows(R/'reproduce/outputs/dataset_builder/so_e2e_balanced/records_import_postdate_ge20200101_parsable_balanced1000_fair.jsonl')
pin={x['record_id']:x for x in pin}
models={'qwen/qwen3-30b-a3b-instruct-2507':'Qwen3-30B-A3B-Instruct-2507','minimax/minimax-m2.5':'MiniMax-M2.5'}
raws={};provenance=[]
for track in ['d2','d1']:
    for mode in ['requirements.txt','inline']:
        for m,label in models.items():
            data,path,sha=raw(track,mode,label);raws[track,mode,m]=data;provenance.append({'track':track,'mode':mode,'model':m,'path':path,'sha256':sha})
selected=[];prompts=[]
for track,n in [('d2',60),('d1',20)]:
    common=set.intersection(*(set(v) for k,v in raws.items() if k[0]==track))
    if track=='d2':common&=clean
    groups=collections.defaultdict(list)
    for tid in common:
        base=raws[track,'inline',next(iter(models))][tid]
        packages=pin[tid].get('target_packages',[]) if track=='d2' else base.get('third_party_libs',[])
        stratum=sorted(str(x).lower() for x in packages)[0] if packages else '(none)'
        groups[stratum].append(tid)
    for v in groups.values():v.sort(key=h)
    order=sorted(groups,key=h);chosen=[]
    depth=0
    while len(chosen)<n:
        for g in order:
            if len(groups[g])>depth:chosen.append((groups[g][depth],g))
            if len(chosen)==n:break
        depth+=1
    for tid,g in chosen:
        selected.append({'track':track,'task_id':tid,'stratum':g,'stratum_population':len(groups[g]),'eligible_population':len(common)})
        for mode in ['requirements.txt','inline']:
            source=[]
            for m in models:
                x=raws[track,mode,m][tid];p=x.get('prompt') if track=='d2' else x.get('augmented_prompt')
                assert p and isinstance(p,str)
                source.append(p)
            assert len(set(source))==1,(track,tid,mode,'Original model prompts differ')
            prompts.append({'track':track,'task_id':tid,'mode':mode,'prompt':source[0],'prompt_sha256':hashlib.sha256(source[0].encode()).hexdigest(),'stratum':g})
requests=[]
for repeat in range(3):
    for p in prompts:
        for model in models:
            x=dict(p,model=model,repeat=repeat+1);x['request_id']=hashlib.sha256((str((p['track'],p['task_id'],p['mode'],model,repeat+1))).encode()).hexdigest()[:24];requests.append(x)
assert len(selected)==80 and len(requests)==960
(a.out/'tasks.json').write_text(json.dumps(selected,indent=2))
(a.out/'source_manifest.json').write_text(json.dumps(provenance,indent=2))
with (a.out/'requests.jsonl').open('w') as f:
    for x in requests:f.write(json.dumps(x)+'\n')
print('Frozen',len(selected),'tasks and',len(requests),'planned calls')
print('Prompt characters min/max',min(len(x['prompt']) for x in prompts),max(len(x['prompt']) for x in prompts))
