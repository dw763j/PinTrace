"""Stratified paired comparisons of constraint policies on archived code."""
import argparse, collections, concurrent.futures, copy, hashlib, json, os, sys
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/data/mine/llm_tpl'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
R=a.root;O=a.out.resolve();O.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(R));os.environ['EXPERIMENT_NOTIFY_ENABLED']='0'
from evaluation_support_remote import evaluate,package_set,unpin,sha
import reanalyze as analysis
analysis.X=O/'evidence';analysis.X.mkdir(exist_ok=True)
if not (analysis.X/'osv').exists():(analysis.X/'osv').symlink_to(R/'resources/osv#pypi#vulns#20260301',target_is_directory=True)
idx=analysis.AdvisoryIndex()
from scripts.run_neighbor_version_experiment import _generate_offset_vectors,_ordered_versions_at_cutoff
def rows(p):return [json.loads(l) for l in p.open() if l.strip()]
def rank(x):return sha('20261110|'+x['track']+'|'+x['model']+'|'+x['task_id'])
E=R/'reproduce/outputs/evaluation'
overlays={(x['track'],x['mode'],x['model'].lower(),str(x['task_id'])):x['record'] for x in rows(E/'overlays.jsonl')}
problems={str(x['task_id']):x for x in rows(R/'resources/BigCodeBench-v0.1.4-local.jsonl')}
selected_path=O/'selected.json'
if selected_path.exists():selected=json.loads(selected_path.read_text())
else:
    groups=collections.defaultdict(list);lineage=[]
    for audit in rows(E/'run_audits.jsonl'):
        if audit['mode']!='inline':continue
        run=Path(audit['run_dir']);m=json.loads((run/'metrics_summary.json').read_text())['model_name'].lower()
        p=run/'m5_compat_records.json';lineage.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        for rec in json.loads(p.read_text()):
            tid=str(rec['task_id']);rec=overlays.get((audit['track'],'inline',m,tid),rec)
            res=rec.get('resolver',{});code=rec.get('code') or ''
            if res.get('status')!='resolved' or not code.strip():continue
            exact={canonicalize_name(x['pypi_name']) for x in rec.get('per_lib',[]) if x.get('is_direct',True) and x.get('resolution_origin')=='model_exact'}
            if not exact:continue
            text=res.get('requirements_in') or ''
            if not text:
                from stages.version_resolver import build_task_requirements
                text='\n'.join(build_task_requirements(rec['extracted_versions']))
            lines=[l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith('#')]
            try:names=package_set(lines)
            except Exception:continue
            if not exact.issubset(set(names)):continue
            graph=rec.get('resolved_environment',[])
            associated=any(idx.match(canonicalize_name(x.get('pypi_name','')),str(x.get('resolved_version',''))) for x in graph if x.get('resolved_version'))
            compat=rec.get('compat_results',{});static=bool((compat.get('ty') or {}).get('is_compatible'));dynamic=(compat.get('bcb_test') or {}).get('status')=='pass'
            success=static if audit['track']=='d2' else dynamic
            stratum=(audit['track'],m,success,associated if audit['track']=='d2' else None)
            groups[stratum].append({'track':audit['track'],'model':m,'task_id':tid,'code':code,'code_sha256':sha(code),'requirements':lines,'exact_packages':sorted(exact),'per_lib':rec.get('per_lib',[]),'historical_static':static,'historical_dynamic':dynamic,'historical_associated':associated,'historical_lock_sha256':res.get('lock_sha256'),'source_path':str(p),'stratum':list(stratum),'stratum_population':0})
    selected=[];pop=[]
    for k,group in sorted(groups.items(),key=str):
        pop.append({'stratum':list(k),'eligible':len(group),'sampled':min(2,len(group))})
        for x in sorted(group,key=rank)[:2]:x['stratum_population']=len(group);selected.append(x)
    selected_path.write_text(json.dumps(selected,indent=2));(O/'sampling.json').write_text(json.dumps(pop,indent=2));(O/'source_manifest.json').write_text(json.dumps(lineage,indent=2))
print('Prepared',len(selected),'responses',dict(collections.Counter(x['track'] for x in selected)),flush=True)
if a.prepare_only:raise SystemExit(0)

def candidates(x):
    # New comparison policy: literal model-exact packages only; no fallback to
    # unversioned dependencies, and no changes to harness-only declarations.
    libs=[]
    for pkg in x['exact_packages']:
        declarations=[Requirement(s) for s in x['requirements'] if canonicalize_name(Requirement(s).name)==pkg]
        versions={s.version for q in declarations for s in q.specifier if s.operator=='==' and '*' not in s.version}
        if len(versions)!=1:continue
        base=next(iter(versions));ordered=_ordered_versions_at_cutoff(pkg,pypi_dirs=[R/'resources/pypi_info',R/'resources/pypi_info_vul_tpls'],reference_date='2026-02-01T00:00:00Z')
        if base not in ordered:continue
        libs.append((pkg,ordered,ordered.index(base)))
    libs=libs[:3];out=[];seen=set()
    for vec in _generate_offset_vectors(len(libs),5,40):
        if any(not 0<=base+offset<len(ordered) for (_,ordered,base),offset in zip(libs,vec)):continue
        replacements={pkg:ordered[base+offset] for (pkg,ordered,base),offset in zip(libs,vec) if offset}
        lines=[]
        from packaging.specifiers import SpecifierSet
        for line in x['requirements']:
            q=Requirement(line);pkg=canonicalize_name(q.name)
            if pkg in replacements:q.specifier=SpecifierSet('=='+replacements[pkg])
            lines.append(str(q))
        sig=tuple(lines)
        if sig in seen:continue
        seen.add(sig);assert package_set(lines)==package_set(x['requirements'])
        out.append({'requirements':lines,'replacements':replacements,'offset':list(vec),'l1':sum(abs(z) for z in vec)})
        if len(out)==10:break
    return out

def process(x):
    uid=sha(str((x['track'],x['model'],x['task_id'])))[:20];path=O/'cases'/uid;path.mkdir(parents=True,exist_ok=True)
    fp=path/'comparison.json'
    if fp.exists():return json.loads(fp.read_text())
    problem=problems.get(x['task_id']) if x['track']=='d1' else None
    def eval_arm(lines,name):return evaluate(x['code'],lines,path/name,root=R,cutoff='2026-02-01T00:00:00Z',bcb_problem=problem,advisory_index=idx)
    base=eval_arm(x['requirements'],'original');bare=eval_arm(unpin(x['requirements'],set(x['exact_packages'])),'unpin')
    result={k:v for k,v in x.items() if k not in ['per_lib','code']};result.update(original=base,unpin=bare,neighbors=[])
    # Accept the first candidate that passes the available engineering oracle
    # and has no matched advisory in its complete resolved graph.
    # The baseline itself is retained at zero search cost when it meets both.
    goal='dynamic_pass' if x['track']=='d1' else 'static_pass'
    result['neighbor_selected']='original' if base.get(goal) and base.get('associated') is False else None
    if result['neighbor_selected'] is None:
        for i,c in enumerate(candidates(x),1):
            y=eval_arm(c['requirements'],'neighbor_'+str(i));result['neighbors'].append(dict(candidate=c,evaluation=y))
            if y.get(goal) and y.get('associated') is False:result['neighbor_selected']=i;break
    fp.write_text(json.dumps(result,indent=2));print(x['track'],x['model'],x['task_id'],'base',base[goal],'unpin',bare[goal],'neighbor',result['neighbor_selected'],flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    all_results=list(pool.map(process,selected))
(O/'comparisons.json').write_text(json.dumps(all_results,indent=2));print('Complete',len(all_results),flush=True)
