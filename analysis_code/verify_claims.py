"""Check headline findings against their units and denominators."""
import csv,json,os
from pathlib import Path
R=Path(os.environ['PINTRACE_WORKSPACE']);A=R/'analysis'
def read(name):
    with (A/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def one(rows,**keys):return next(x for x in rows if all(x[k]==v for k,v in keys.items()))
def main():
    facts=[]
    def rate(name,n,d,expected,source,unit):
        value=round(100*n/d,2);assert value==expected,(name,value,expected)
        facts.append(dict(claim=name,numerator=n,denominator=d,percentage=value,source=source,unit=unit))
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
        rate(q+' population advisory association',int(r['V']),int(r['N']),pop,'security_funnels.csv','response in the 927-task clean population')
        rate(q+' conditional advisory association',int(r['V']),int(r['P']),cond,'security_funnels.csv','response with individually satisfiable exact advice')
    boot=read('paired_cluster_bootstrap.csv')
    for population,metric,values in [('PinTrace complete','static',(51.83,36.26)),('BigCodeBench','dynamic',(42.29,16.47))]:
        r=one(boot,population=population,model='All observations',metric=metric)
        for q,expected in zip(['MR','VA'],values):
            d=int(r['observations_per_condition']);n=round(float(r[q+'_pct'])*d/100)
            rate(population+' '+q+' '+metric,n,d,expected,'paired_cluster_bootstrap.csv','task-model response')
    neigh=read('neighbor_corrected_by_model.csv');d=sum(int(r['N']) for r in neigh)
    rate('Neighbor installation recovery',sum(int(r['installation_recovered']) for r in neigh),d,62.33,'neighbor_corrected_by_model.csv','selected installation-failing response')
    rate('Neighbor static recovery',sum(int(r['static_recovered']) for r in neigh),d,52.88,'neighbor_corrected_by_model.csv','selected installation-failing response')
    (R/'claims.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    print('Verified headline rates:',len(facts))
if __name__=='__main__':main()
