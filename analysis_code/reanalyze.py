"""Recompute revision evidence from immutable downloaded experiment records.

Dependencies: Python 3.11+, numpy, pandas, packaging.
Run from any directory: python scripts/reanalyze.py --bootstrap 10000
No API requests, generation, installation, or execution of generated code.
"""
from __future__ import annotations
import os
import argparse, collections, csv, hashlib, json, re, sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from packaging.version import Version, InvalidVersion
from packaging.utils import canonicalize_name, canonicalize_version

ROOT=Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]))
S=ROOT/'inputs/search_archive';X=ROOT/'inputs/registries';OUT=ROOT/'analysis'
CORRECTED=ROOT/'inputs/main'
CUTOFF='2026-02-01T00:00:00+00:00';SEED=20261109
def rows(p):
    with p.open(encoding='utf-8') as f:
        for l in f:
            if l.strip():yield json.loads(l)
def dump(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def csvout(name,data):pd.DataFrame(data).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def model(x):return str(x).lower()
def key(x):return (x['track'],x['mode'],model(x['model']),str(x['task_id']))
def dt(x):
    if not x:return None
    try:
        d=datetime.fromisoformat(str(x).replace('Z','+00:00'))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d
    except ValueError:return None
def ver(x):
    try:return Version(str(x))
    except InvalidVersion:return None
def pair(x):
    version=x.get('specified_version') if x.get('constraint_kind')=='exact' else None
    version=version or x.get('resolved_version') or x.get('constraint_resolved_version') or x.get('version')
    return canonicalize_name(str(x.get('pypi_name') or '')),str(version or '')

def file_hash(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def input_paths(canonical=None,clean_ids=None):
    canonical=Path(canonical) if canonical else (CORRECTED if CORRECTED.exists() else None)
    directory=canonical or S/'outputs/evaluation'
    paths={name:directory/(name+'.jsonl') for name in ['direct_dependencies','task_outcomes']}
    if canonical:
        paths['source_carrier_declarations']=directory/'source_carrier_declarations.jsonl'
        if (directory/'pypi_additional.json').exists():paths['corrected_pypi_additional']=directory/'pypi_additional.json'
    if clean_ids:
        paths['clean_task_ids']=Path(clean_ids)
    elif canonical:
        paths['clean_task_ids']=ROOT/'inputs/assessments/partition/clean_task_ids.jsonl'
    else:
        paths['clean_task_ids']=S/'resources/revision/pintrace_clean_task_ids.jsonl'
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(path)
    return paths,bool(canonical)

def constraint_eligible(row,include_invalid=False):
    if include_invalid and row['constraint_kind']=='invalid':return True
    if row['constraint_kind'] not in ['exact','range']:return False
    if 'frozen_assessment_eligible' in row:return row['frozen_assessment_eligible'] is True
    return row.get('identity_assessable',True) is True and row.get('marker_active',True) is True

def frozen_valid(row):
    return constraint_eligible(row) and row.get('constraint_valid') is True and row.get('resolution_origin') in ['model_exact','model_constrained_resolver']

def task_measurements(task,corrected):
    if corrected:
        fields=['environment_eligible','complete_lock','installation_pass','static_pass','dynamic_pass','response_status']
        if any(field not in task for field in fields):raise ValueError(('Missing canonical outcome field',key(task)))
        result=dict(zip(['input','lock','installation','static','dynamic'],[task[field] is True for field in fields[:5]]))
        assert not result['lock'] or result['input'],key(task)
        assert not result['installation'] or result['lock'],key(task)
        assert not result['static'] or result['installation'],key(task)
        assert not result['dynamic'] or result['installation'],key(task)
        return result
    locked=task['resolver_status']=='resolved'
    return {'input':task['resolver_status'] in ['resolved','failed'],'lock':locked,
            'installation':locked and task['compatibility_layer'] not in ['environment_build','infrastructure'],
            'static':task['ty_compatible'] and locked,'dynamic':task['bcb_status']=='pass'}

def declaration_flags(direct,carriers,enriched):
    result={}
    for kind in ['exact','range']:
        result[kind]=any(row['constraint_kind']==kind for row in carriers)
        result['direct_'+kind]=any(row['constraint_kind']==kind for row in direct)
        result['assessment_'+kind]=any(row['constraint_kind']==kind and constraint_eligible(row) for row in direct)
        result['valid_'+kind]=any(row['constraint_kind']==kind and frozen_valid(row) for row in direct)
        for scope in ['associated','published_by_cutoff','retrospective_only','cve_associated']:
            result[kind+'_'+scope]=any(row['constraint_kind']==kind and row[scope] for row in enriched)
    return result

def all_valid_summary(groups):
    """An active unknown constraint keeps the complete response unassessable."""
    counts=collections.Counter(N=len(groups))
    for group in groups:
        declarations=[row for row in group if row['constraint_kind'] in ['exact','range','invalid'] and row['declaration_source'] in ['manifest','inline_annotation']]
        active=[row for row in declarations if row['constraint_kind']=='invalid' or row.get('marker_active',True) is not False]
        inactive=len(declarations)-len(active)
        counts['inactive_constraints']+=inactive
        counts['responses_with_inactive_constraints']+=bool(inactive)
        counts['S_any']+=bool(active)
        if not active:continue
        unknown=[row for row in active if row['constraint_kind']!='invalid' and not constraint_eligible(row)]
        indeterminate=[row for row in active if constraint_eligible(row) and row.get('constraint_valid') is None]
        counts['responses_with_unknown_identity_or_applicability']+=bool(unknown)
        counts['responses_with_indeterminate_validity']+=bool(indeterminate)
        counts['responses_with_unverified_mapping_validity']+=any(row.get('mapping_validity_unverified') for row in active)
        if unknown or indeterminate:continue
        counts['S_assessable']+=1
        counts['A']+=all(row['constraint_kind']!='invalid' and row.get('constraint_valid') is True for row in active)
    required=['N','S_any','S_assessable','A','inactive_constraints','responses_with_inactive_constraints',
              'responses_with_unknown_identity_or_applicability','responses_with_indeterminate_validity','responses_with_unverified_mapping_validity']
    result={name:counts[name] for name in required}
    result['all_valid_pct']=100*counts['A']/counts['S_assessable'] if counts['S_assessable'] else None
    return result

def assessment_counts(dependencies):
    output=[]
    models=['All observations']+sorted({model(row['model']) for row in dependencies})
    for m in models:
        for mode in ['requirements.txt','inline']:
            for kind in ['exact','range','invalid']:
                group=[row for row in dependencies if row['mode']==mode and row['constraint_kind']==kind and (m=='All observations' or model(row['model'])==m)]
                eligible=[row for row in group if constraint_eligible(row,include_invalid=True)]
                assessed=[row for row in eligible if row.get('constraint_valid') is not None]
                valid=sum(row.get('constraint_valid') is True for row in assessed)
                output.append(dict(model=m,mode=mode,kind=kind,declarations=len(group),eligible=len(eligible),assessed=len(assessed),valid=valid,
                                   invalid=sum(row.get('constraint_valid') is False for row in assessed),indeterminate=len(eligible)-len(assessed),
                                   marker_inactive=sum(row.get('marker_active') is False for row in group if kind!='invalid'),
                                   identity_unknown=sum(row.get('identity_assessable',True) is False for row in group if kind!='invalid'),
                                   validity_pct=100*valid/len(assessed) if assessed else None))
    return output
def ranges_match(v,events):
    """Union of half-open/closed OSV ECOSYSTEM intervals, including multiple intervals."""
    v=ver(v)
    if v is None:return False
    lower=None;opened=False
    for e in events:
        if 'introduced' in e:
            lower=ver(e['introduced']) if e['introduced']!='0' else None;opened=True
        for terminal in ['fixed','last_affected','limit']:
            if terminal in e and opened:
                upper=ver(e[terminal]);above=lower is None or v>=lower
                if upper is not None and above and (v<=upper if terminal=='last_affected' else v<upper):return True
                opened=False
    return opened and (lower is None or v>=lower)

class AdvisoryIndex:
    def __init__(self):
        self.by_pkg=collections.defaultdict(list);self.records={};self.parent={};self.counts=collections.Counter()
        def find(x):
            self.parent.setdefault(x,x)
            if self.parent[x]!=x:self.parent[x]=find(self.parent[x])
            return self.parent[x]
        self.find=find
        for p in sorted((X/'osv').glob('*.json')):
            r=json.loads(p.read_text(encoding='utf-8'));self.counts['records']+=1
            if r.get('withdrawn'):self.counts['withdrawn']+=1;continue
            self.records[r['id']]=r
            ids=[r['id']]+(r.get('aliases') or [])
            find(ids[0])
            for other in ids[1:]:self.parent[find(other)]=find(ids[0])
            for a in r.get('affected',[]):
                pkg=a.get('package',{})
                if pkg.get('ecosystem')!='PyPI':continue
                versions={ver(v) for v in a.get('versions',[]) if ver(v) is not None}
                self.by_pkg[canonicalize_name(pkg.get('name',''))].append((r,a,versions))
        families=collections.defaultdict(list)
        for ident in self.parent:families[find(ident)].append(ident)
        self.canon={i:sorted(ids,key=lambda s:(not s.startswith('CVE-'),s))[0] for ids in families.values() for i in ids}
    @lru_cache(None)
    def match(self,pkg,v):
        found=[];pv=ver(v)
        for r,a,versions in self.by_pkg.get(pkg,[]):
            ok=(pv in versions) if pv is not None else v in a.get('versions',[])
            ok=ok or any(rg.get('type')=='ECOSYSTEM' and ranges_match(v,rg.get('events',[])) for rg in a.get('ranges',[]))
            if ok:found.append(r['id'])
        return tuple(sorted(set(found)))
    def properties(self,ids):
        cutoff=dt(CUTOFF)
        records=[self.records[i] for i in ids]
        published=[dt(r.get('published')) for r in records]
        cv=[i for r in records for i in [r['id']]+(r.get('aliases') or []) if i.startswith('CVE-')]
        return {'associated':bool(ids),'published_by_cutoff':any(d and d<=cutoff for d in published),
                'retrospective_only':bool(ids) and all(d and d>cutoff for d in published),
                'publication_unknown':bool(ids) and not any(published),
                'cve_associated':bool(cv),'advisory_ids':list(ids),'canonical_ids':sorted({self.canon[i] for i in ids})}

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--canonical',type=Path,help='Directory containing corrected canonical JSONL files')
    ap.add_argument('--clean-ids',type=Path,help='PinTrace clean-task JSONL file')
    ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args()
    if args.bootstrap<1:ap.error('--bootstrap must be positive')
    OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
    paths,corrected=input_paths(args.canonical,args.clean_ids)
    deps=list(rows(paths['direct_dependencies']))
    tasks=list(rows(paths['task_outcomes']))
    carriers=list(rows(paths['source_carrier_declarations'])) if corrected else deps
    overrides=json.loads((X/'manifest_mapping_overrides.json').read_text()) if not corrected and (X/'manifest_mapping_overrides.json').exists() else []
    adjudication_path=ROOT/'experiments/constraint_adjudications.json'
    adjudications=json.loads(adjudication_path.read_text()) if adjudication_path.exists() else []
    adjudication_map={(tuple(x['key']),x['ordinal']):x for x in adjudications}
    for o in overrides:
        decision=adjudication_map.get((tuple(o['key']),o['ordinal']))
        if decision:
            assert decision['requirement']==o['raw_requirement']
            o['new_constraint_valid']=decision['constraint_valid']
            o['adjudication_reason']=decision['reason']
    override_map={(tuple(o['key']),o['ordinal']):o for o in overrides}
    for r in deps:
        o=override_map.get((key(r),r['ordinal']))
        if not o:continue
        r['original_pypi_name']=r['pypi_name'];r['pypi_name']=o['new_pypi_name']
        r['mapping_recovered']=True
        if o.get('adjudication_reason'):r['mapping_adjudication_reason']=o['adjudication_reason']
        r['mapping_validity_unverified']=o['new_constraint_valid'] is None and o['old_constraint_valid'] is False
        if o['new_constraint_valid'] is not None:
            r['constraint_valid']=o['new_constraint_valid']
        r['resolved_version']=o['new_resolved_version'];r['constraint_resolved_version']=o['new_constraint_resolved_version']
        r['resolution_origin']=o['new_resolution_origin']
    clean={str(r['task_id']) for r in rows(paths['clean_task_ids'])}
    pin=list(rows(S/'outputs/dataset_builder/so_e2e_balanced/records_import_postdate_ge20200101_parsable_balanced1000_fair.jsonl'))
    assert len(tasks)==36260 and len(clean)==(927 if corrected else 849)
    if not corrected:assert len(deps)==93295
    grouped=collections.defaultdict(list)
    for r in deps:grouped[key(r)].append(r)
    carrier_groups=collections.defaultdict(list)
    for r in carriers:carrier_groups[key(r)].append(r)
    taskmap={key(t):t for t in tasks};assert len(taskmap)==len(tasks)
    assert set(grouped)<=set(taskmap) and set(carrier_groups)<=set(taskmap)
    assert clean<={str(t['task_id']) for t in tasks if t['track']=='d2'}
    idx=AdvisoryIndex();print('Advisory index',dict(idx.counts),flush=True)
    pypi=json.loads((X/'pypi_releases.json').read_text(encoding='utf-8'))
    if (X/'pypi_additional.json').exists():pypi.update(json.loads((X/'pypi_additional.json').read_text(encoding='utf-8')))
    if 'corrected_pypi_additional' in paths:pypi.update(json.loads(paths['corrected_pypi_additional'].read_text(encoding='utf-8')))
    @lru_cache(None)
    def release_state(pkg,v):
        if pkg not in pypi or pypi[pkg].get('missing_cache'):return ('metadata_unavailable',None,[])
        rel=pypi[pkg].get('releases',{});pver=ver(v)
        matching=[vv for vv in rel if vv==v or (pver is not None and ver(vv)==pver)]
        if not matching:return ('absent_release',None,[])
        dates=[dt(f.get('upload_time_iso_8601') or f.get('upload_time')) for vv in matching for f in rel[vv]]
        dates=[d for d in dates if d]
        if not dates:return ('release_without_dated_artifact',None,matching)
        earliest=min(dates)
        return ('after_cutoff' if earliest>dt(CUTOFF) else 'published_by_cutoff',earliest.isoformat(),matching)
    dep_enriched=[];validity_anomalies=[];release_counts=collections.Counter()
    for r in deps:
        if r['track']!='d2' or str(r['task_id']) not in clean:continue
        if not r.get('is_direct') or r.get('constraint_kind') not in ['exact','range','invalid']:continue
        k=r['constraint_kind'];q=dict(r);pkg,v=pair(r)
        valid=frozen_valid(r)
        props=idx.properties(idx.match(pkg,v)) if valid else idx.properties(())
        q.update(props);q['analysis_version']=v;q['normalized_package']=pkg;q['frozen_valid']=valid;q['frozen_assessment_eligible']=constraint_eligible(r)
        if k=='exact' and constraint_eligible(r):
            rs,date,aliases=release_state(pkg,v);q.update(release_state=rs,release_date=date,release_aliases=aliases)
            release_counts[(r['mode'],rs,valid)]+=1
            if valid and rs in ['absent_release','after_cutoff']:validity_anomalies.append({**{x:q.get(x) for x in ['mode','model','task_id','pypi_name','specified_version','resolved_version','constraint_valid','release_state']}})
        dep_enriched.append(q)
    csvout('constraint_assessment_counts.csv',assessment_counts(dep_enriched))
    exclusions=[]
    for row in deps:
        reasons=[]
        if row.get('marker_active') is False:reasons.append('marker_inactive')
        if row.get('identity_assessable') is False and row['constraint_kind']!='invalid':reasons.append('identity_unresolved')
        if reasons:
            exclusions.append({**{field:row.get(field) for field in ['track','mode','model','task_id','declaration_id','constraint_kind','declaration_source','raw_requirement','import_line','pypi_name']},
                               'clean_pintrace':row['track']=='d2' and str(row['task_id']) in clean,'exclusion_reasons':reasons})
    dump('assessment_exclusions.json',exclusions)
    enriched_groups=collections.defaultdict(list)
    for r in dep_enriched:enriched_groups[key(r)].append(r)
    with (OUT/'dependencies_rechecked.jsonl').open('w',encoding='utf-8') as f:
        for r in dep_enriched:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    flags=[];recon=[]
    for t in tasks:
        q=dict(t);g=grouped[key(t)];eg=enriched_groups[key(t)]
        q['model']=model(q['model']);q['task_id']=str(q['task_id']);q['clean']=q['task_id'] in clean
        q.update(task_measurements(t,corrected))
        q.update(declaration_flags(g,carrier_groups[key(t)],eg))
        for kind in ['exact','range']:
            if q['track']=='d2' and q['clean'] and 'has_valid_model_'+kind+'_constraint' in t and q['valid_'+kind]!=bool(t['has_valid_model_'+kind+'_constraint']):
                recon.append({'key':key(t),'kind':kind,'canonical_task':t['has_valid_model_'+kind+'_constraint'],'dependency_or':q['valid_'+kind]})
        flags.append(q)
    # Disagreements must be wholly explained by independently recovered manifest identities.
    assert not recon if corrected else all(any(r.get('mapping_recovered') for r in grouped[tuple(x['key'])]) for x in recon),recon
    dump('validity_flag_corrections.json',recon)
    df=pd.DataFrame(flags);df.to_csv(OUT/'response_flags.csv',index=False,encoding='utf-8-sig')
    d2=df[(df.track=='d2')&df.clean]
    # All observations and per-model tables, with unambiguous numerator/denominator.
    funnel=[]
    for m in ['All observations']+sorted(d2.model.unique()):
        sub=d2 if m=='All observations' else d2[d2.model==m]
        for mode,kind in [('requirements.txt','exact'),('requirements.txt','range'),('inline','exact')]:
            z=sub[sub['mode']==mode];n=len(z);s=int(z['assessment_'+kind].sum());p=int(z['valid_'+kind].sum())
            for scope in ['associated','published_by_cutoff','cve_associated']:
                v=int(z[kind+'_'+scope].sum());assert 0<=v<=p<=s<=n
                funnel.append(dict(model=m,mode=mode,kind=kind,security_scope=scope,N=n,S=s,P=p,V=v,population_pct=100*v/n,conditional_pct=100*v/p if p else None))
    csvout('security_funnels.csv',funnel)
    allvalid=[]
    for m in ['All observations']+sorted(d2.model.unique()):
        for md in ['requirements.txt','inline']:
            keys=[key(t) for t in flags if t['track']=='d2' and t['clean'] and t['mode']==md and (m=='All observations' or t['model']==m)]
            allvalid.append(dict(model=m,mode=md,**all_valid_summary([grouped[k] for k in keys])))
    csvout('all_constraint_validity.csv',allvalid)
    csvout('exact_release_states.csv',[dict(mode=k[0],state=k[1],frozen_valid=k[2],dependency_occurrences=v) for k,v in sorted(release_counts.items())])
    dump('validity_anomalies.json',validity_anomalies)
    # Task-clustered paired bootstrap: the ten models and both conditions travel with a task.
    rng=np.random.default_rng(SEED);boot=[]
    populations=[('PinTrace complete',df[df.track=='d2']),('PinTrace clean',d2),('BigCodeBench',df[df.track=='d1'])]
    for pop,z in populations:
        metrics=['exact','input','lock','installation','static'] if pop!='BigCodeBench' else ['dynamic','static']
        if pop=='PinTrace clean':metrics+=['exact_associated','exact_published_by_cutoff']
        taskids=sorted(z.task_id.unique());mods=sorted(z.model.unique());assert len(mods)==10
        counts=np.zeros((args.bootstrap,len(taskids)),dtype=np.int16)
        for b in range(args.bootstrap):counts[b]=np.bincount(rng.integers(0,len(taskids),len(taskids)),minlength=len(taskids))
        for metric in metrics:
            mat=[]
            for mode in ['requirements.txt','inline']:
                zz=z[z['mode']==mode].pivot(index='task_id',columns='model',values=metric).reindex(index=taskids,columns=mods)
                assert not zz.isna().any().any()
                mat.append(zz.to_numpy(dtype=float))
            mr,va=mat
            for mi,m in [(-1,'All observations')]+list(enumerate(mods)):
                amr=mr.mean(axis=1) if mi==-1 else mr[:,mi];ava=va.mean(axis=1) if mi==-1 else va[:,mi]
                bmr=counts@amr/len(taskids);bva=counts@ava/len(taskids);bd=bva-bmr
                def ci(b):return [float(v)*100 for v in np.quantile(b,[.025,.975])]
                boot.append(dict(population=pop,metric=metric,model=m,tasks=len(taskids),observations_per_condition=len(taskids)*(10 if mi==-1 else 1),
                                 MR_pct=100*amr.mean(),VA_pct=100*ava.mean(),delta_VA_minus_MR_pp=100*(ava-amr).mean(),
                                 MR_ci_low=ci(bmr)[0],MR_ci_high=ci(bmr)[1],VA_ci_low=ci(bva)[0],VA_ci_high=ci(bva)[1],delta_ci_low=ci(bd)[0],delta_ci_high=ci(bd)[1],
                                 both_success=int((mr*va).sum()) if mi==-1 else int((mr[:,mi]*va[:,mi]).sum()),
                                 MR_only=int((mr*(1-va)).sum()) if mi==-1 else int((mr[:,mi]*(1-va[:,mi])).sum()),
                                 VA_only=int((va*(1-mr)).sum()) if mi==-1 else int((va[:,mi]*(1-mr[:,mi])).sum())))
        print('Bootstrap',pop,flush=True)
    csvout('paired_cluster_bootstrap.csv',boot)
    # Package standardization on the original OSV-conditioned candidate pool.
    targets={r['record_id']:sorted(set(canonicalize_name(p) for p in r['target_packages'])) for r in pin}
    pool=json.loads((X/'candidate_population.json').read_text(encoding='utf-8'))['package_incidence_counts']
    weight_rows=[];package_rates=[]
    for mode in ['requirements.txt','inline']:
        z=d2[d2['mode']==mode]
        for metric in ['exact_associated','exact_published_by_cutoff','static']:
            rates=[]
            for pkg,count in sorted(pool.items()):
                ids=[t for t,pkgs in targets.items() if pkg in pkgs and t in clean]
                if not ids:continue
                zz=z[z.task_id.isin(ids)];rate=float(zz[metric].mean())
                rates.append((pkg,count,rate,len(ids)))
                package_rates.append(dict(mode=mode,metric=metric,package=pkg,candidate_task_incidence=count,clean_tasks=len(ids),rate_pct=100*rate))
            total=sum(x[1] for x in rates)
            w=np.array([x[1] for x in rates],float);p=np.array([x[2] for x in rates])
            taskw=collections.Counter()
            for pkg,count,rate,num in rates:
                for tid,pkgs in targets.items():
                    if pkg in pkgs and tid in clean:taskw[tid]+=count/num
            a=np.array(list(taskw.values()),float)
            weight_rows.append(dict(mode=mode,metric=metric,unweighted_response_pct=100*z[metric].mean(),equal_target_package_pct=100*p.mean(),candidate_incidence_standardized_pct=100*np.average(p,weights=w),
                                   represented_target_packages=len(rates),candidate_incidence_coverage=sum(x[1] for x in rates)/sum(pool.values()),kish_effective_tasks=float(a.sum()**2/(a*a).sum()),max_normalized_task_weight=float(a.max()/a.sum())))
    csvout('package_standardization.csv',weight_rows);csvout('target_package_rates.csv',package_rates)
    # Occurrence and unique package-version estimands. Model-written exact and resolver ranges remain separate.
    unique=[];pcounts=[]
    for mode,kind in [('requirements.txt','exact'),('requirements.txt','range'),('inline','exact')]:
        eligible=[r for r in dep_enriched if r['mode']==mode and r['constraint_kind']==kind and r['frozen_valid']]
        pp=collections.defaultdict(list)
        for r in eligible:pp[(r['normalized_package'],canonicalize_version(r['analysis_version']))].append(r)
        unique.append(dict(mode=mode,kind=kind,valid_occurrences=len(eligible),associated_occurrences=sum(r['associated'] for r in eligible),unique_pairs=len(pp),associated_pairs=sum(any(r['associated'] for r in rr) for rr in pp.values()),unique_packages=len({p for p,v in pp})))
        for (p,v),rr in pp.items():pcounts.append(dict(mode=mode,kind=kind,package=p,version=v,occurrences=len(rr),associated=any(r['associated'] for r in rr),models=len({model(r['model']) for r in rr}),tasks=len({r['task_id'] for r in rr})))
    csvout('unique_package_versions.csv',unique);csvout('package_version_counts.csv',pcounts)
    # Context-sensitive validity counts distinguish omitted cutoff releases from absent release keys.
    temporal=[]
    for mode in ['requirements.txt','inline']:
        ee=[r for r in dep_enriched if r['mode']==mode and r['constraint_kind']=='exact' and constraint_eligible(r)]
        for state in sorted({r.get('release_state') for r in ee}):
            rr=[r for r in ee if r.get('release_state')==state]
            temporal.append(dict(mode=mode,state=state,occurrences=len(rr),responses=len({(model(r['model']),r['task_id']) for r in rr}),invalid_occurrences=sum(not r['frozen_valid'] for r in rr)))
    csvout('release_existence_temporal.csv',temporal)
    age=[]
    for r in dep_enriched:
        if r['mode']=='inline' and r['constraint_kind']=='exact' and r['frozen_valid'] and r.get('release_date'):
            age.append(dict(model=model(r['model']),task_id=r['task_id'],package=r['normalized_package'],version=r['analysis_version'],age_months=(dt(CUTOFF)-dt(r['release_date'])).total_seconds()/86400/30.4375))
    csvout('version_age_rechecked.csv',age)
    adf=pd.DataFrame(age,columns=['model','task_id','package','version','age_months'])
    csvout('version_age_summary.csv',[dict(model=m,n=len(z),median_months=z.age_months.median(),q25_months=z.age_months.quantile(.25),q75_months=z.age_months.quantile(.75),q05_months=z.age_months.quantile(.05),q95_months=z.age_months.quantile(.95)) for m,z in adf.groupby('model')])
    source_files={name:{'path':str(path.resolve()),'sha256':file_hash(path)} for name,path in paths.items()}
    for name,path in [('pypi_releases',X/'pypi_releases.json'),('pypi_additional',X/'pypi_additional.json'),('candidate_population',X/'candidate_population.json')]:
        if path.exists():source_files[name]={'path':str(path.resolve()),'sha256':file_hash(path)}
    osv_hashes={path.name:file_hash(path) for path in sorted((X/'osv').glob('*.json'))}
    dump('analysis_manifest.json',{'seed':SEED,'bootstrap_replicates':args.bootstrap,'cluster':'task; fixed panel of ten evaluated models; MR and VA remain paired','cutoff':CUTOFF,'security_snapshot':'2026-03-01','advisory_counts':dict(idx.counts),
         'source_counts':{'responses':len(tasks),'dependencies':len(deps),'source_carrier_declarations':len(carriers),'clean_tasks':len(clean)},
         'corrected_canonical':corrected,'legacy_mapping_overrides_applied':len(overrides),'source_files':source_files,
         'osv_files_sha256':hashlib.sha256(json.dumps(osv_hashes,sort_keys=True).encode()).hexdigest(),'script_sha256':file_hash(Path(__file__)),
         'metric_sources':{'exact_and_range_response_flags':'condition-specific source carriers' if corrected else 'legacy direct dependencies',
                           'security_funnels':'merged model direct dependencies; S requires at least one identity-assessable active constraint, with N retaining the complete response population',
                           'all_constraint_validity':'active model constraints and invalid declarations; S_assessable excludes responses with unresolved identity, applicability, or validity',
                           'environment_outcomes':'explicit canonical boolean fields' if corrected else 'legacy resolver and compatibility states'},
         'validity_OR_task_disagreements':len(recon),'release_validity_anomalies':len(validity_anomalies),'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__})
    print('DONE',OUT,flush=True)

if __name__=='__main__':main()
