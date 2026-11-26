"""Export each paper table's input records and verify its displayed numbers."""
import collections,csv,gzip,json,os,re,shutil
from pathlib import Path
from summarize_mr_constraint_profiles import DISPLAY_NAMES

ROOT=Path(__file__).resolve().parents[1]
R=Path(os.environ['PINTRACE_WORKSPACE']);A=R/'analysis';T=R/'tables'
MODELS=list(DISPLAY_NAMES)+['All observations']
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def output(name,rows):
    with (T/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def lookup(rows,**fields):return next(r for r in rows if all(r[k]==v for k,v in fields.items()))
def pct(x):return round(float(x),2)
def numbers(cells):
    return [float(x.replace(',','')) for cell in cells for x in re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',cell)]
def paper_rows(label):
    path=ROOT/'results/reference/tables'/(label.replace(':','_')+'.tex')
    text=path.read_text(encoding='utf-8');active=False;result=[]
    for line in text.splitlines():
        if r'\midrule' in line:active=True;continue
        if r'\bottomrule' in line:break
        if not active or '&' not in line or not line.rstrip().endswith(r'\\'):continue
        if any(x in line for x in ['multicolumn','cmidrule','textbf{Model}']):continue
        cells=line.rstrip().removesuffix(r'\\').split('&')
        result.append([x.strip() for x in cells])
    return result

def main():
    contracts={};sources={}
    date=json.loads((R/'inputs/collection/records_import_postdate_ge20200101.stats.json').read_text())
    parse=json.loads((R/'inputs/collection/records_import_postdate_ge20200101_parsable.stats.json').read_text())
    sample=json.loads((R/'inputs/collection/records_import_postdate_ge20200101_parsable_balanced1000_fair.stats.json').read_text())
    counts=[date['total_records'],date['import_candidates'],date['final_records'],parse['kept_records'],sample['actual_total']]
    output('tab_pintrace_selection',[dict(stage=s,count=n) for s,n in zip(['package_matching','import_screening','date_filtering','parseability','coverage_sampling'],counts)])
    contracts['tab:pintrace_selection']=[[n] for n in counts]
    tasks=read(A/'rq1_model_variation.csv');clean=[json.loads(l) for l in (R/'inputs/assessments/partition/clean_task_ids.jsonl').read_text().splitlines()]
    n=int(lookup(tasks,model='All observations')['mr_N'])//10
    bcb=sum(1 for _ in (R/'inputs/tasks/bigcodebench-task-ids.jsonl').open())
    pops=[dict(population=k,tasks=v,responses_per_condition=v*10) for k,v in [('PinTrace',n),('Clean PinTrace',len(clean)),('BigCodeBench',bcb)]]
    output('tab_task_populations',pops);contracts['tab:task_populations']=[[x['tasks'],x['responses_per_condition']] for x in pops]
    forms=read(A/'rq1_constraint_forms.csv');matrix=[]
    for q in ['MR','VA']:
        z=[r for r in forms if r['condition']==q];d={r['kind']:r for r in z}
        values=[int(z[0]['N'])]
        for k in ['exact','range','bare','invalid']:
            if q=='VA' and k in ['range','invalid']:continue
            values += [int(d[k]['count']),pct(d[k]['percentage'])]
        matrix.append(values)
    output('tab_rq1_constraint_forms',forms);contracts['tab:rq1_constraint_forms']=matrix
    profiles=read(A/'rq1_mr_manifest_profiles.csv');matrix=[]
    for m in MODELS:
        r=lookup(profiles,model=m)
        matrix.append([round(100*int(r[k])/int(r['N']),2) for k in ['bare_only','ranged_only','exact_only','mixed','contains_unparseable','no_manifest_entries']])
    output('tab_rq1_model_variation',profiles);contracts['tab:rq1_model_variation']=matrix
    security=read(A/'security_funnels.csv');complete=read(A/'all_constraint_validity.csv')
    validity=[];matrix=[]
    for mode,kind in [('requirements.txt','exact'),('requirements.txt','range'),('inline','exact')]:
        r=lookup(security,model='All observations',mode=mode,kind=kind,security_scope='published_by_cutoff')
        validity.append(dict(condition=mode,kind=kind,assessable=int(r['S']),passing=int(r['P']),percentage=100*int(r['P'])/int(r['S'])))
    for mode in ['requirements.txt','inline']:
        r=lookup(complete,model='All observations',mode=mode)
        validity.append(dict(condition=mode,kind='all_non_bare',assessable=int(r['S_assessable']),passing=int(r['A']),percentage=float(r['all_valid_pct'])))
    output('tab_rq2_validity',validity);contracts['tab:rq2_validity']=[[r['assessable'],r['passing'],pct(r['percentage'])] for r in validity]
    release=read(A/'exact_release_states.csv');matrix=[]
    for state,valid in [('absent_release',None),('metadata_unavailable',None),('published_by_cutoff',None),('published_by_cutoff','True'),('published_by_cutoff','False'),('after_cutoff',None),('release_without_dated_artifact',None)]:
        matrix.append([sum(int(r['dependency_occurrences']) for r in release if r['mode']==mode and r['state']==state and (valid is None or r['frozen_valid']==valid)) for mode in ['requirements.txt','inline']])
    output('tab_release_states',release);contracts['tab:release_states']=matrix
    matrix=[];data=[]
    for m in MODELS:
        values=[]
        for mode in ['requirements.txt','inline']:
            r=lookup(security,model=m,mode=mode,kind='exact',security_scope='published_by_cutoff')
            values += [int(r['P']),int(r['V']),pct(r['conditional_pct'])];data.append(r)
        matrix.append(values)
    output('tab_rq2_security',data);contracts['tab:rq2_security']=matrix
    boot=read(A/'paired_cluster_bootstrap.csv');matrix=[];data=[]
    for metric in ['input','lock','installation','static']:
        r=lookup(boot,population='PinTrace complete',model='All observations',metric=metric);data.append(r)
        values=[]
        for q in ['MR','VA']:values += [round(float(r[q+'_pct'])*100),pct(r[q+'_pct'])]
        values += [pct(r[k]) for k in ['delta_VA_minus_MR_pp','delta_ci_low','delta_ci_high']];matrix.append(values)
    output('tab_rq3_outcomes',data);contracts['tab:rq3_outcomes']=matrix
    compat=read(A/'rq3_installed_compatibility.csv');matrix=[]
    for m in MODELS:
        values=[]
        for q in ['MR','VA']:
            r=lookup(compat,model=m,condition=q)
            values += [int(r['installed']),int(r['diagnostic_responses']),pct(r['diagnostic_pct_among_installed']),int(r['library_candidates']),pct(r['library_pct_among_installed'])]
        matrix.append(values)
    output('tab_rq3_installed_diagnostics',compat);contracts['tab:rq3_installed_diagnostics']=matrix
    neighbor=read(A/'neighbor_corrected_by_model.csv');matrix=[]
    for m in MODELS:
        rr=neighbor if m=='All observations' else [lookup(neighbor,model=m)]
        n=sum(int(r['N']) for r in rr);ins=sum(int(r['installation_recovered']) for r in rr);sta=sum(int(r['static_recovered']) for r in rr)
        matrix.append([n,ins,round(100*ins/n,2),sta,round(100*sta/n,2)])
    output('tab_neighbor_recovery',neighbor);contracts['tab:neighbor_recovery']=matrix
    fixed=read(R/'experiments/fixed_code/outcomes.csv');matrix=[]
    for track in ['d2','d1']:
        for s in ['original','unpin','neighbor']:
            r=lookup(fixed,track=track,strategy=s);values=[int(r[k]) for k in ['installed','oracle_pass']]
            if s!='original':values += [int(r['recovered']),int(r['regressed'])]
            matrix.append(values+[int(r['oracle_pass_without_association'])])
    output('tab_fixed_code_comparison',fixed);contracts['tab:fixed_code_comparison']=matrix
    runtime=read(R/'experiments/runtime/python_robustness_all_observations.csv');matrix=[]
    for track in ['d2','d1']:
        for metric in ['installation_completed','static_pass' if track=='d2' else 'dynamic_pass']:
            for mode in ['requirements.txt','inline']:
                matrix.append([pct(lookup(runtime,track=track,mode=mode,python_version=v)[metric+'_pct']) for v in ['3.8','3.10','3.12','3.14']])
    output('tab_runtime_sensitivity',runtime);contracts['tab:runtime_sensitivity']=matrix
    manual=json.loads((R/'inputs/manual/summary.json').read_text(encoding='utf-8'))
    with gzip.open(ROOT/'data/inputs/manual/judgments.json.gz','rt',encoding='utf-8') as stream:
        judgments=json.load(stream)['records']
    partition={r['task_id']:r for r in map(json.loads,(R/'inputs/assessments/partition/input_partition.jsonl').read_text(encoding='utf-8').splitlines())}
    input_checks=[partition[r['task_id']] for r in judgments if r['kind']=='partition']
    flagged=[r for r in input_checks if r['archived_partition']=='contaminated']
    false_positives=sum(r['final_partition']=='clean' for r in flagged)
    mr=sum(r['N'] for r in manual['counts_by_response_population'] if r['mode']=='requirements.txt')
    va=sum(r['N'] for r in manual['counts_by_response_population'] if r['mode']=='inline')
    missed=len(manual['missed_bare_manifest_cases'])
    recovered=sum(map(len,manual['recovered_bare_declarations'].values()))
    assert (mr,va,missed,recovered,len(input_checks),len(flagged),false_positives)==(60,60,5,10,40,10,5)
    output('tab_measurement_manual_checks',[
        dict(check='manifest_extraction',unit='responses',sample=mr+va,MR=mr,VA=va,affected=missed,affected_MR_pct=100*missed/mr,recovered_bare_declarations=recovered),
        dict(check='code_and_package_mapping',unit='responses',sample=mr+va,MR=mr,VA=va,affected='',affected_MR_pct='',recovered_bare_declarations=''),
        dict(check='input_version_labels',unit='tasks',sample=len(input_checks),MR='',VA='',affected=false_positives,affected_MR_pct='',recovered_bare_declarations='')])
    contracts['tab:measurement_manual_checks']=[[mr,va,round(100*missed/mr,2)],[mr+va],[len(input_checks),len(input_checks)-len(flagged),len(flagged)]]
    sensitivity=read(A/'extraction_sensitivity.csv')
    selected=[lookup(sensitivity,population=p,metric=k) for p,k in [('PinTrace complete','static'),('PinTrace clean','exact_published_by_cutoff'),('BigCodeBench','dynamic')]]
    output('tab_extraction_sensitivity',selected)
    contracts['tab:extraction_sensitivity']=[[int(r['retained_pairs'])]+[pct(r[k]) for k in ['MR_pct','VA_pct','delta_pp','delta_ci_low','delta_ci_high']] for r in selected]
    dyn=[lookup(boot,population='BigCodeBench',model=m,metric='dynamic') for m in MODELS]
    output('tab_dynamic_by_model',dyn);contracts['tab:dynamic_by_model']=[[pct(r[k]) for k in ['MR_pct','VA_pct','delta_VA_minus_MR_pp','delta_ci_low','delta_ci_high']] for r in dyn]
    by=read(R/'experiments/repeated/by_repeat.csv');agree=read(R/'experiments/repeated/agreement.csv');paired=read(R/'experiments/repeated/paired_intervals.csv')
    order=json.loads((R/'experiments/repeated/analysis_summary.json').read_text())['models']
    for metric,suffix in [('static_pass','static'),('dynamic_pass','tests'),('exact','exact')]:
        matrix=[];data=[]
        for m in order:
            values=[];record={'model':m}
            for mode,q in [('requirements.txt','mr'),('inline','va')]:
                for repeat in [1,2,3]:
                    r=lookup(by,model=m,metric=metric,mode=mode,repeat=str(repeat));values.append(int(r['numerator']));record[f'{q}_r{repeat}']=int(r['numerator'])
            for mode,q in [('requirements.txt','mr'),('inline','va')]:
                r=lookup(agree,model=m,metric=metric,mode=mode);values += [int(r['agreeing_tasks']),int(r['tasks'])];record[q+'_agreeing_tasks']=int(r['agreeing_tasks']);record['tasks']=int(r['tasks'])
            r=lookup(paired,model=m,metric=metric)
            for k in ['delta_pp','ci_low','ci_high']:values.append(pct(r[k]));record[k]=float(r[k])
            matrix.append(values);data.append(record)
        label='tab:five_repeat_'+suffix
        output(label.replace(':','_'),data);contracts[label]=matrix
    report=[]
    for label,expected in contracts.items():
        rows=paper_rows(label)
        # Data rows in the form table have two textual leading columns.
        start=2 if label in ['tab:rq1_constraint_forms','tab:rq2_validity','tab:runtime_sensitivity','tab:extraction_sensitivity'] else 1
        actual=[numbers(row[start:]) for row in rows]
        if label=='tab:pintrace_selection':actual=[numbers(row[-1:]) for row in rows]
        if label=='tab:task_populations':actual=[numbers(row[1:3]) for row in rows]
        assert len(expected)==len(actual),(label,'row count',len(expected),len(actual))
        for i,(want,got) in enumerate(zip(expected,actual)):
            assert len(want)==len(got),(label,i,'column count',want,got)
            assert all(abs(a-b)<0.0051 for a,b in zip(want,got)),(label,i,want,got)
        report.append(dict(label=label,rows=len(expected),numeric_values=sum(map(len,expected)),status='matched'))
    # Qualitative comparisons, model descriptions, and assessment rules are curated inputs.
    for label in ['tab:models','tab:related_comparison','tab:assessment_rules']:
        data=paper_rows(label)
        if data:output(label.replace(':','_'),[dict(row=i+1,**{f'column_{j+1}':x for j,x in enumerate(row)}) for i,row in enumerate(data)])
    (T/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Checked {len(report)} numerical paper tables and {sum(r["numeric_values"] for r in report)} displayed values.')
if __name__=='__main__':main()
