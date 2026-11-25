"""Check headline findings against their units and denominators."""
import collections,csv,json,os
from pathlib import Path
R=Path(os.environ['PINTRACE_WORKSPACE']);A=R/'analysis'
def read(name):
    with (A/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def one(rows,**keys):return next(x for x in rows if all(x[k]==v for k,v in keys.items()))
def main():
    facts=[]
    def rate(name,n,d,expected,source,unit,**scope):
        value=round(100*n/d,2);assert value==expected,(name,value,expected)
        facts.append(dict(claim=name,numerator=n,denominator=d,percentage=value,source=source,unit=unit,**scope))
    forms=read('rq1_model_variation.csv');r=one(forms,model='All observations')
    rate('MR exact-advice incidence',int(r['mr_exact']),int(r['mr_N']),18.58,'rq1_model_variation.csv','PinTrace response')
    rate('VA exact-advice incidence',int(r['va_exact']),int(r['va_N']),81.20,'rq1_model_variation.csv','PinTrace response')
    top=read('rq1_top_three_concentration.csv')
    for q,expected in [('MR',74.44),('VA',67.44)]:
        z=[r for r in top if r['condition']==q]
        rate(q+' top-three share across the ten most frequent packages',sum(int(r['top_three_declarations']) for r in z),sum(int(r['exact_declarations']) for r in z),expected,'rq1_top_three_concentration.csv','source-carrier exact-declaration occurrence')
    rel=read('exact_release_states.csv');invalid=[r for r in rel if r['mode']=='inline' and r['frozen_valid']=='False']
    count=lambda state:sum(int(r['dependency_occurrences']) for r in invalid if r['state']==state)
    total=sum(int(r['dependency_occurrences']) for r in invalid)
    rate('Release absent from PyPI history among unsatisfiable VA exact declarations',count('absent_release'),total,56.10,'exact_release_states.csv','merged exact declaration on clean PinTrace')
    rate('Published by cutoff among unsatisfiable VA exact declarations',count('published_by_cutoff'),total,31.10,'exact_release_states.csv','merged exact declaration on clean PinTrace')
    sec=read('security_funnels.csv')
    for mode,q,pop,cond in [('requirements.txt','MR',9.44,60.81),('inline','VA',38.18,50.05)]:
        r=one(sec,model='All observations',mode=mode,kind='exact',security_scope='published_by_cutoff')
        rate(q+' population advisory association',int(r['V']),int(r['N']),pop,'security_funnels.csv','response in the 927-task clean population',advisory_scope='published_by_release_cutoff')
        rate(q+' conditional advisory association',int(r['V']),int(r['P']),cond,'security_funnels.csv','response with individually satisfiable exact advice',advisory_scope='published_by_release_cutoff')
    boot=read('paired_cluster_bootstrap.csv')
    for population,metric,values in [('PinTrace complete','static',(51.83,36.26)),('BigCodeBench','dynamic',(42.29,16.47))]:
        r=one(boot,population=population,model='All observations',metric=metric)
        for q,expected in zip(['MR','VA'],values):
            d=int(r['observations_per_condition']);n=round(float(r[q+'_pct'])*d/100)
            rate(population+' '+q+' '+metric,n,d,expected,'paired_cluster_bootstrap.csv','task-model response')
    neigh=read('neighbor_corrected_by_model.csv');d=sum(int(r['N']) for r in neigh)
    qualification=R/'inputs/assessments/neighbor/qualification.jsonl'
    with qualification.open(encoding='utf-8') as f:selected=[r for line in f if (r:=json.loads(line))['eligible']]
    assert len(selected)==d==815
    assert all(r['mode']=='inline' and r['complete_lock'] and r['installation_failure'] and r['model_exact_packages'] and not r['baseline_associations'] for r in selected)
    scope=dict(condition='VA',python_version='3.12',population_rule='Complete baseline lock, installation failure, direct model-written exact constraint, and no baseline advisory match.')
    rate('Neighbor installation recovery',sum(int(r['installation_recovered']) for r in neigh),d,62.33,'neighbor_corrected_by_model.csv','selected installation-failing response',**scope)
    rate('Neighbor static recovery',sum(int(r['static_recovered']) for r in neigh),d,52.88,'neighbor_corrected_by_model.csv','selected installation-failing response',**scope)
    concentration=read('rq1_concentration_checked.csv')
    for q,expected in [('MR',86.03),('VA',74.92)]:
        r=one(concentration,package='requests',condition=q)
        rate(q+' requests==2.31.0 share within requests exact advice',int(r['modal_count']),int(r['N']),expected,
            'rq1_concentration_checked.csv','source-carrier exact-declaration occurrence',condition=q,package='requests',version='2.31.0',denominator_scope='Exact advice for requests only')
    counts=collections.defaultdict(collections.Counter)
    with (R/'inputs/main/source_carrier_declarations.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['track']=='d2' and row['rq1_eligible'] and row['constraint_kind']=='exact':
                counts[row['mode']][row.get('pypi_name'),row.get('specified_version')]+=1
    for mode,q,expected,winner in [('requirements.txt','MR',137,('flask','2.3.3')),('inline','VA',628,('numpy','1.24.3'))]:
        assert counts[mode]['numpy','1.24.3']==expected
        assert counts[mode].most_common(1)[0][0]==winner
        facts.append(dict(claim=q+' recurring numpy==1.24.3 advice',type='count',count=expected,condition=q,
            most_frequent_pair=list(winner),most_frequent_pair_count=counts[mode][winner],
            source='inputs/main/source_carrier_declarations.jsonl',unit='source-carrier exact-declaration occurrence on complete PinTrace'))
    with (R/'experiments/runtime/runtime_failure_composition.csv').open(encoding='utf-8-sig') as f:runtime=list(csv.DictReader(f))
    for version,expected in [('3.8',41.45),('3.10',4.07),('3.12',70.33),('3.14',84.50)]:
        r=one(runtime,condition='VA',python_version=version)
        rate('VA responses without an installed environment among nonpasses on Python '+version,
            int(r['no_installed_environment']),int(r['no_test_pass']),expected,'experiments/runtime/runtime_failure_composition.csv',
            'BigCodeBench response without a task-test pass',condition='VA',python_version=version)
    sensitivity=read('extraction_sensitivity.csv')
    for population,metric,pairs,delta in [('PinTrace complete','static',8934,-15.70),('BigCodeBench','dynamic',7594,-25.99),('PinTrace clean','exact_published_by_cutoff',8280,28.71)]:
        r=one(sensitivity,population=population,metric=metric)
        assert int(r['retained_pairs'])==pairs and round(float(r['delta_pp']),2)==delta
        assert float(r['delta_ci_low'])*float(r['delta_ci_high'])>0
        facts.append(dict(claim='Source-exclusion sensitivity: '+population+' '+metric,type='paired_difference',population=population,
            metric=metric,retained_pairs=pairs,estimate_pp=float(r['delta_pp']),ci_low=float(r['delta_ci_low']),ci_high=float(r['delta_ci_high']),
            interval_method='paired task-cluster percentile bootstrap',source='extraction_sensitivity.csv',unit='task-model response pair'))
    (R/'claims.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    print('Verified headline findings:',len(facts))
if __name__=='__main__':main()
