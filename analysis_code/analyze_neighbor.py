import os
"""Audit recorded fixed-code substitutions and replay prefixes of the 40-candidate order."""
import argparse,ast,collections,hashlib,json,re,sys
from copy import deepcopy
from pathlib import Path
from functools import lru_cache
from packaging.version import Version,InvalidVersion
from reanalyze import S,X,OUT,AdvisoryIndex,rows,csvout,dump,model,dt
import reanalyze as analysis_module

def ordered_model_archives(directory):
    """Preserve one archive order on case-sensitive and case-insensitive systems."""
    return sorted(directory.glob('*.json'), key=lambda path: path.name.casefold())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--canonical-tasks',type=Path)
    parser.add_argument('--new-search',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if bool(args.qualification) != bool(args.canonical_tasks):
        parser.error('--qualification and --canonical-tasks must be supplied together')
    if args.new_search and not args.qualification:
        parser.error('--new-search requires corrected qualification and canonical tasks')
    if args.out:
        args.out.mkdir(parents=True,exist_ok=True)
        analysis_module.OUT=args.out
    index=AdvisoryIndex();meta=json.loads((X/'pypi_releases.json').read_text())
    if (X/'pypi_additional.json').exists():meta.update(json.loads((X/'pypi_additional.json').read_text()))
    ns={'deepcopy':deepcopy,'Path':Path,'Version':Version,'InvalidVersion':InvalidVersion,'datetime':__import__('datetime').datetime,'Any':object}
    tree=ast.parse((S/'scripts/run_neighbor_version_experiment.py').read_text())
    names=['_is_version_yanked','_normalize_pkg_name','_parse_ref_date','_ordered_versions_at_cutoff','_generate_offset_vectors','_resolve_neighbor_candidates_for_task']
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body,type_ignores=[]),'archived_neighbor_functions','exec'),ns)
    from packaging.utils import canonicalize_name
    ns['_load_pypi_json_from_dirs']=lambda pkg,dirs:meta.get(canonicalize_name(pkg))
    # Preserve the exact archived algorithm, including stable ties and candidate over-generation.
    grouped=collections.defaultdict(list)
    for r in rows(S/'outputs/evaluation/direct_dependencies.jsonl'):
        if r['track']=='d2' and r['mode']=='inline':grouped[(model(r['model']),r['task_id'])].append(r)
    for g in grouped.values():g.sort(key=lambda r:r['ordinal'])
    raw={}
    if not args.qualification:
        for r in rows(X/'raw_security.jsonl'):
            if r['mode']=='inline':raw[(model(r['model']),r['task_id'])]=r
    tasks={}
    task_source=args.canonical_tasks or S/'outputs/evaluation/task_outcomes.jsonl'
    for r in rows(task_source):
        if r['track']=='d2' and r['mode']=='inline':tasks[(model(r['model']),r['task_id'])]=r
    def active(env):
        return any(index.match(canonicalize_name(d.get('pypi_name','')),str(d.get('resolved_version') or d.get('version') or '')) for d in env if d.get('pypi_name'))
    eligible=set()
    qualification={}
    if args.qualification:
        qualification={(model(r['model']),r['task_id']):r for r in rows(args.qualification)}
        assert set(qualification)==set(tasks)
        eligible={k for k,r in qualification.items() if r['eligible']}
    else:
        for k,t in tasks.items():
            if t['resolver_status']=='resolved' and t.get('compatibility_layer')=='environment_build' and any(d.get('resolution_origin')=='model_exact' for d in grouped[k]) and not active(raw[k]['resolved_environment']):eligible.add(k)
    outputs=[];mismatches=[];budget=[];recorded=set();full=collections.Counter();directions=collections.Counter()
    new_search=json.loads(args.new_search.read_text()) if args.new_search else []
    new_keys={(model(r['model']),r['task_id']) for r in new_search}
    assert len(new_keys)==len(new_search)
    for path in ordered_model_archives(S/'outputs/analysis/neighbor/models'):
        data=json.loads(path.read_text());cfg=data['experiment_config']
        m=data['run_selection']['selected_run']['model_name'];m=m.removeprefix('d2_').removesuffix('_blind').lower().replace('meta-llama_llama','meta-llama/llama')
        for t in data['task_results']:
            k=(m,t['task_id']);recorded.add(k)
            if k in new_keys:
                continue
            candidates,cmeta=ns['_resolve_neighbor_candidates_for_task']({'per_lib':grouped[k]},neighbor_n=cfg['neighbor_n'],max_candidates_k=cfg['max_candidates_k'],reference_date=cfg['reference_date'],pypi_dirs=[],specified_only=cfg['specified_only'],max_searchable_libs=cfg['max_searchable_libs'],focused_pypi_names=set(t.get('failed_tpl_from_ty_install_error',[])),focused_neighbor_n=cfg['focused_neighbor_n'])
            if t['generated_candidates']==0:
                candidates=[]
            if len(candidates)!=t['generated_candidates']:
                mismatches.append({'model':m,'task_id':t['task_id'],'stored_candidates':t['generated_candidates'],'reconstructed_candidates':len(candidates)})
            def success_rank(field):
                first=t.get(field)
                if not first:return None
                vec=first['offset_vector']
                ranks=[i+1 for i,c in enumerate(candidates) if c['offset_vector']==vec and c['per_lib_moves']==first['per_lib_moves']]
                if len(ranks)!=1:
                    mismatches.append({'model':m,'task_id':t['task_id'],'field':field,'found_ranks':ranks});return None
                return ranks[0]
            ir=success_rank('first_no_ty_install_error');sr=success_rank('first_ty_compatible')
            first=t.get('first_ty_compatible')
            ds=[int(x['offset']) for x in (first or {}).get('per_lib_moves',[]) if int(x['offset'])]
            moves=[x for x in (first or {}).get('per_lib_moves',[]) if int(x['offset'])]
            moved_model_exact=all(any(canonicalize_name(d['pypi_name'])==canonicalize_name(x['pypi_name']) and d['constraint_kind']=='exact' for d in grouped[k]) for x in moves)
            direction=('newer_only' if all(v>0 for v in ds) else 'older_only' if all(v<0 for v in ds) else 'mixed') if ds else None
            vulnerability=None
            if first:
                vulnerability=active(first.get('vuln_findings',[]))
            row=dict(model=m,task_id=t['task_id'],eligible_under_rematch=k in eligible,baseline_resolver_status=tasks[k]['resolver_status'],baseline_compatibility_layer=tasks[k]['compatibility_layer'],generated=t['generated_candidates'],tested=t['tested_candidate_count'],generation_reason=t['candidate_generation_meta']['reason'],installation_rank=ir,static_rank=sr,static_direction=direction,replacement_has_active_association=vulnerability,at_40_test_limit=t['tested_candidate_count']==40,searchable_libs=t['candidate_generation_meta']['searchable_lib_count'])
            row['static_changed_packages_have_model_exact']=moved_model_exact if first else None
            row['search_evidence']='retained_archived_search'
            exact_packages={canonicalize_name(d['pypi_name']) for d in grouped[k] if d['constraint_kind']=='exact'}
            row['generated_candidates_moving_non_exact']=sum(any(move['offset'] and canonicalize_name(move['pypi_name']) not in exact_packages for move in candidate['per_lib_moves']) for candidate in candidates)
            row['tested_candidates_moving_non_exact']=sum(any(move['offset'] and canonicalize_name(move['pypi_name']) not in exact_packages for move in candidate['per_lib_moves']) for candidate in candidates[:t['tested_candidate_count']])
            if k in eligible and first:
                assert first['candidate_resolution']['status']=='resolved'
                assert first.get('vuln_findings'), 'Missing replacement graph evidence'
                assert moved_model_exact, 'Successful replacement changed a package without model-exact advice'
            outputs.append(row)
    if new_search:
        plans=json.loads((args.new_search.parent/'candidate_plans.json').read_text())
        plans={r['request_id']:r for r in plans}
        for task in new_search:
            k=(model(task['model']),task['task_id'])
            assert k in eligible
            plan=plans[task['request_id']]
            assert task['response_sha256']==qualification[k]['response_sha256']==plan['response_sha256']
            assert task['tested_candidate_count']==len(task['candidates_tested'])
            for rank,tested in enumerate(task['candidates_tested'],1):
                expected=plan['candidates'][rank-1]
                assert tested['candidate_rank']==rank
                for field in ['offset_vector','per_lib_moves','requirements']:
                    assert tested[field]==expected[field],(k,rank,field)
            first=task.get('first_ty_compatible')
            installed=task.get('first_no_ty_install_error')
            moved=[move for move in (first or {}).get('per_lib_moves',[]) if move['offset']]
            offsets=[move['offset'] for move in moved]
            direction=('newer_only' if all(value>0 for value in offsets) else 'older_only' if all(value<0 for value in offsets) else 'mixed') if offsets else None
            moved_model_exact=all(move['pypi_name'] in qualification[k]['model_exact_packages'] for move in moved)
            assert moved_model_exact
            outputs.append(dict(
                model=k[0],task_id=k[1],eligible_under_rematch=True,
                baseline_resolver_status=tasks[k]['resolver_status'],baseline_compatibility_layer=tasks[k]['compatibility_layer'],
                generated=task['generated_candidates'],tested=task['tested_candidate_count'],
                generation_reason=task['candidate_generation_meta']['reason'],
                installation_rank=installed['candidate_rank'] if installed else None,
                static_rank=first['candidate_rank'] if first else None,static_direction=direction,
                replacement_has_active_association=active(first['vuln_findings']) if first else None,
                at_40_test_limit=task['tested_candidate_count']==40,
                searchable_libs=task['candidate_generation_meta']['searchable_lib_count'],
                static_changed_packages_have_model_exact=moved_model_exact if first else None,
                search_evidence='corrected_input_search',
                generated_candidates_moving_non_exact=0,tested_candidates_moving_non_exact=0,
            ))
    assert len({(r['model'],r['task_id']) for r in outputs})==len(outputs)
    csvout('neighbor_task_audit.csv',outputs)
    for cohort in ['recorded','rematched_eligible']:
        z=[x for x in outputs if cohort=='recorded' or x['eligible_under_rematch']]
        for k in [5,10,20,30,40]:
            budget.append({'cohort':cohort,'prefix_budget':k,'N':len(z),'installation_recovered':sum(x['installation_rank'] is not None and x['installation_rank']<=k for x in z),'static_recovered':sum(x['static_rank'] is not None and x['static_rank']<=k for x in z)})
    csvout('neighbor_budget_prefix.csv',budget)
    z=[x for x in outputs if x['eligible_under_rematch']]
    excluded=[x for x in outputs if not x['eligible_under_rematch']]
    per_model=[]
    for m in sorted({x['model'] for x in z}):
        zz=[x for x in z if x['model']==m]
        per_model.append(dict(model=m,N=len(zz),installation_recovered=sum(x['installation_rank'] is not None for x in zz),static_recovered=sum(x['static_rank'] is not None for x in zz)))
    csvout('neighbor_corrected_by_model.csv',per_model)
    report={'archived_cohort_N':len(recorded),'corrected_install_failure_cohort_N':len(eligible),'recorded_still_eligible':len(eligible&recorded),'new_eligible_without_search':sorted(eligible-recorded),'excluded_from_install_failure_cohort':sorted(recorded-eligible),'exclusion_reasons':dict(collections.Counter(x['baseline_resolver_status']+'/'+x['baseline_compatibility_layer'] for x in excluded)),'reconstruction_mismatches':mismatches,'cohort_with_no_candidates':sum(x['generated']==0 for x in z),'cohort_reached_40_test_limit':sum(x['at_40_test_limit'] for x in z),'cohort_generated_40':sum(x['generated']==40 for x in z),'cohort_tested_candidates_total':sum(x['tested'] for x in z),'cohort_generated_candidates_total':sum(x['generated'] for x in z),'cohort_static_direction_counts':dict(collections.Counter(x['static_direction'] for x in z if x['static_rank'])),'cohort_static_replacement_active_association':sum(bool(x['replacement_has_active_association']) for x in z),'eligible_definition':'complete lock, ty installation error, at least one model exact direct constraint, no active association anywhere in baseline lock','budget_definition':'prefix of the reconstructed archived 40-candidate ordering; no new candidate executions'}
    if args.qualification:
        report['qualification_source']=str(args.qualification)
        report['qualification_sha256']=hashlib.sha256(args.qualification.read_bytes()).hexdigest()
        report['task_outcomes_source']=str(task_source)
        report['task_outcomes_sha256']=hashlib.sha256(task_source.read_bytes()).hexdigest()
        report['qualification_population_N']=len(qualification)
        report['recorded_excluded_with_active_association']=sum(bool(qualification[(x['model'],x['task_id'])]['baseline_associations']) for x in excluded)
        report['retained_archived_eligible_N']=sum(r['eligible'] and r['archived_eligible'] for r in qualification.values())
    report['cohort_generated_candidates_moving_non_exact']=sum(r['generated_candidates_moving_non_exact'] for r in z)
    report['cohort_tested_candidates_moving_non_exact']=sum(r['tested_candidates_moving_non_exact'] for r in z)
    report['cohort_responses_with_non_exact_candidate_moves']=sum(r['generated_candidates_moving_non_exact']>0 for r in z)
    if args.new_search:
        covered={(r['model'],r['task_id']) for r in outputs}
        report['new_eligible_without_search']=sorted(eligible-covered)
        report['new_search_responses']=len(new_search)
        report['new_search_source']=str(args.new_search)
        report['new_search_sha256']=hashlib.sha256(args.new_search.read_bytes()).hexdigest()
        report['new_candidate_executions']=sum(r['tested_candidate_count'] for r in new_search)
        report['budget_definition']='Prefixes of the archived candidate order and newly executed corrected-input searches under the same ordering and 40-candidate budget.'
    dump('neighbor_audit.json',report)
    print(json.dumps({k:(len(v) if isinstance(v,list) else v) for k,v in report.items()},indent=2))
    assert not mismatches,'Stored/reconstructed candidate evidence differs; inspect neighbor_audit.json'

if __name__=='__main__':main()
