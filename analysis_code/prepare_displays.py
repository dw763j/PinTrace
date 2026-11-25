import os
"""Produce compact manuscript tables and figure inputs from checked records."""
import argparse,collections,json
import pandas as pd
from reanalyze import ROOT,S,X,OUT,rows,csvout,key
from summarize_mr_constraint_profiles import refresh_mr_profiles
from prepare_rq2_displays import prepare_rq2
from summarize_rq3_compatibility import summarize_compatibility
from analyze_rq3_results import build_evidence as prepare_rq3_results
from prepare_fixed_code_table import build_table as prepare_fixed_code_table

NAMES={'gpt-5.4':'GPT-5.4','claude-sonnet-4-6':'Claude Sonnet 4.6','gemini-3.1-pro-preview':'Gemini 3.1 Pro','deepseek-v3.2':'DeepSeek V3.2','kimi-k2.5':'Kimi K2.5','qwen3.5-397b-a17b':'Qwen3.5 397B','qwen3-235b-a22b-instruct-2507':'Qwen3 235B','qwen3-30b-a3b-instruct-2507':'Qwen3 30B','minimax-m2.5':'MiniMax M2.5','meta-llama/llama-4-scout':'Llama 4 Scout','All observations':'All observations'}
ORDER=list(NAMES)

def prepare_rq1():
    source = ROOT / 'inputs/main/source_carrier_declarations.jsonl'
    declarations = [row for row in rows(source) if row['track'] == 'd2' and row['rq1_eligible']]
    outcomes = [row for row in rows(ROOT / 'inputs/main/task_outcomes.jsonl') if row['track'] == 'd2']
    grouped = collections.defaultdict(list)
    for row in declarations:
        grouped[key(row)].append(row)
    forms = []
    for mode, label in [('requirements.txt', 'MR'), ('inline', 'VA')]:
        counts = collections.Counter(row['constraint_kind'] for row in declarations if row['mode'] == mode)
        for kind in ['exact', 'range', 'bare', 'invalid']:
            forms.append(dict(condition=label, mode=mode, kind=kind, count=counts[kind], N=sum(counts.values()), percentage=100*counts[kind]/sum(counts.values())))
    csvout('rq1_constraint_forms.csv', forms)
    variation = []
    for model in ORDER:
        selected = outcomes if model == 'All observations' else [row for row in outcomes if row['model'] == model]
        result = {'model': model}
        for mode, label in [('requirements.txt', 'mr'), ('inline', 'va')]:
            responses = [row for row in selected if row['mode'] == mode]
            result[label + '_N'] = len(responses)
            for kind in ['exact', 'range', 'bare_only']:
                numerator = 0
                for response in responses:
                    kinds = {row['constraint_kind'] for row in grouped[key(response)]}
                    numerator += kinds == {'bare'} if kind == 'bare_only' else kind in kinds
                result[label + '_' + kind] = numerator
                result[label + '_' + kind + '_pct'] = 100*numerator/len(responses)
        variation.append(result)
    csvout('rq1_model_variation.csv', variation)
    refresh_mr_profiles(ROOT, declarations=declarations, responses=outcomes)
    concentration = []
    exact = [row for row in declarations if row['constraint_kind'] == 'exact' and row.get('pypi_name')]
    top = [package for package, count in collections.Counter(row['pypi_name'] for row in exact).most_common(10)]
    for package in top:
        for mode, label in [('requirements.txt', 'MR'), ('inline', 'VA')]:
            counts = collections.Counter(row['specified_version'] for row in exact if row['pypi_name'] == package and row['mode'] == mode)
            ranked = counts.most_common()
            assert ranked
            concentration.append(dict(package=package, condition=label, N=sum(counts.values()), modal_version=ranked[0][0], modal_count=ranked[0][1], modal_pct=100*ranked[0][1]/sum(counts.values()), rank1=ranked[0][1], rank2=ranked[1][1] if len(ranked)>1 else 0, rank3=ranked[2][1] if len(ranked)>2 else 0, other=sum(count for version,count in ranked[3:])))
    csvout('rq1_concentration_checked.csv', concentration)
    return pd.DataFrame(forms), pd.DataFrame(variation)
def table(name,headers,data):
    path=ROOT/'tables';path.mkdir(exist_ok=True)
    text='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    text+=''.join('| '+' | '.join(map(str,row))+' |\n' for row in data)
    (path/(name+'.md')).write_text(text,encoding='utf-8')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rq1-only', action='store_true')
    args = parser.parse_args()
    forms, v = prepare_rq1()
    if args.rq1_only:
        print('Corrected RQ1 display data generated.')
        return
    flags=pd.read_csv(OUT/'response_flags.csv',low_memory=False)
    boot=pd.read_csv(OUT/'paired_cluster_bootstrap.csv')
    table('rq1_forms',['Condition / carrier','Exact','Range / other nonempty','Bare','Unparseable','Total'],[[mode]+[f'{int(z[z.kind==k]["count"].sum()):,} ({100*z[z.kind==k]["count"].sum()/z.N.iloc[0]:.2f}%)' for k in ['exact','range','bare','invalid']]+[f'{int(z.N.iloc[0]):,}'] for mode,z in forms.groupby('condition')])
    prepare_rq2(ROOT)
    summarize_compatibility(ROOT)
    prepare_rq3_results(ROOT)
    prepare_fixed_code_table(ROOT)
    failures=[]
    for mode,z in flags[flags.track=='d2'].groupby('mode'):
        labels=[]
        for _,r in z.iterrows():
            label='Did not enter joint resolution' if not r['input'] else 'Resolution failure' if not r.lock else 'Installation failure' if not r.installation else 'Static pass' if r['static'] else {'api_mismatch_candidate':'API diagnostic candidate','import_or_initialization':'Import diagnostic candidate','code_or_syntax':'Other code / syntax diagnostic'}.get(r.compatibility_layer,'Other diagnostic')
            labels.append(label)
        for label,n in collections.Counter(labels).items():failures.append(dict(condition='MR' if mode=='requirements.txt' else 'VA',layer=label,count=n,N=len(z)))
    csvout('stage_failure_partition.csv',failures)
    dyn=boot[(boot.population=='BigCodeBench')&(boot.metric=='dynamic')].copy();dyn['order']=dyn.model.map({m:i for i,m in enumerate(ORDER)});dyn=dyn.sort_values('order')
    table('dynamic_by_model',['Model','MR (%)','VA (%)','VA − MR (pp)','95% CI lower (pp)','95% CI upper (pp)'],[[NAMES[r.model],f'{r.MR_pct:.2f}',f'{r.VA_pct:.2f}',f'{r.delta_VA_minus_MR_pp:.2f}',f'{r.delta_ci_low:.2f}',f'{r.delta_ci_high:.2f}'] for _,r in dyn.iterrows()])
    neigh=pd.read_csv(OUT/'neighbor_corrected_by_model.csv');neigh['order']=neigh.model.map({m:i for i,m in enumerate(ORDER)});neigh=neigh.sort_values('order')
    total = neigh[['N','installation_recovered','static_recovered']].sum()
    table('neighbor_models',['Model','Eligible VA responses','Installation recovered','Static pass recovered'],[[NAMES[r.model],int(r.N),f'{r.installation_recovered} ({100*r.installation_recovered/r.N:.2f}%)',f'{r.static_recovered} ({100*r.static_recovered/r.N:.2f}%)'] for _,r in neigh.iterrows()]+[['All observations',int(total.N),f'{int(total.installation_recovered)} ({100*total.installation_recovered/total.N:.2f}%)',f'{int(total.static_recovered)} ({100*total.static_recovered/total.N:.2f}%)']])
    budgets=pd.read_csv(OUT/'neighbor_budget_prefix.csv');budgets=budgets[budgets.cohort=='rematched_eligible']
    table('neighbor_budgets',['Prefix budget','Installation recovered / eligible responses','Static pass recovered / eligible responses'],[[int(r.prefix_budget),f'{r.installation_recovered}/{r.N} ({100*r.installation_recovered/r.N:.2f}%)',f'{r.static_recovered}/{r.N} ({100*r.static_recovered/r.N:.2f}%)'] for _,r in budgets.iterrows()])
    sens=pd.read_csv(OUT/'extraction_sensitivity.csv')
    metric_names={'static':'static pass','dynamic':'test pass','exact_associated':'exact snapshot association','exact_published_by_cutoff':'exact association published by cutoff'}
    table('extraction_sensitivity',['Population / metric','Retained pairs','MR (%)','VA (%)','VA − MR (pp)','95% CI lower (pp)','95% CI upper (pp)'],[[r.population+' / '+metric_names[r.metric],f'{r.retained_pairs:,}',f'{r.MR_pct:.2f}',f'{r.VA_pct:.2f}',f'{r.delta_pp:.2f}',f'{r.delta_ci_low:.2f}',f'{r.delta_ci_high:.2f}'] for _,r in sens.iterrows()])
    print('Tables and display inputs generated.')
if __name__=='__main__':main()
