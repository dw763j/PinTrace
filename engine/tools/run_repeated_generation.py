"""Budget-limited OpenRouter sampling. Credentials are read privately, never saved."""
import argparse, concurrent.futures, datetime, getpass, hashlib, json, os, threading, time, urllib.error, urllib.request
from pathlib import Path

CONFIG={
 'qwen/qwen3-30b-a3b-instruct-2507':{'provider':'nebius/fp8','max_tokens':4096,'max_input_per_million':0.12,'max_output_per_million':0.35},
 'minimax/minimax-m2.5':{'provider':'streamlake','max_tokens':8192,'max_input_per_million':0.35,'max_output_per_million':1.30},
}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--requests',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--budget',type=float,default=7.20);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--pilot',action='store_true');ap.add_argument('--dry-run',action='store_true');ap.add_argument('--retry-failures',action='store_true');a=ap.parse_args()
    jobs=[json.loads(l) for l in a.requests.open(encoding='utf-8') if l.strip()]
    a.out.mkdir(parents=True,exist_ok=True);log=a.out/'responses.jsonl'
    previous=[json.loads(l) for l in log.open(encoding='utf-8') if l.strip()] if log.exists() else []
    done={x['request_id'] for x in previous}
    interruptions=a.out/'interruptions.json'
    interruption_records=json.loads(interruptions.read_text()) if interruptions.exists() else []
    interruption_charge=sum(x['conservative_charge_usd'] for x in interruption_records)
    charged=sum(x.get('budget_charge_usd',0) for x in previous)+interruption_charge
    if a.pilot:
        selected=set()
        for track in ['d2','d1']:
            unique={x['task_id']:x for x in jobs if x['track']==track}
            selected.add((track,next(iter(unique))))
            selected.add((track,max(unique,key=lambda k:len(unique[k]['prompt']))))
        jobs=[x for x in jobs if (x['track'],x['task_id']) in selected and x['repeat']==1]
    pending=[x for x in jobs if x['request_id'] not in done]
    attempts={}
    for r in previous:attempts.setdefault(r['request_id'],[]).append(r)
    if a.retry_failures:
        pending=[x for x in jobs if x['request_id'] in attempts and len(attempts[x['request_id']])<2 and all(r['status'] in ['transport_error','http_error'] for r in attempts[x['request_id']]) and all(r.get('http_status',503)==429 or 500<=r.get('http_status',503)<600 for r in attempts[x['request_id']])]
    config={'models':CONFIG,'temperature':0.7,'top_p':0.95,'provider_fallbacks':False,'budget_usd':a.budget,'request_file_sha256':hashlib.sha256(a.requests.read_bytes()).hexdigest(),'created_at_utc':utc()}
    cp=a.out/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text());assert all(old[k]==config[k] for k in config if k!='created_at_utc'),'Configuration differs from stored run'
    else:cp.write_text(json.dumps(config,indent=2))
    def bound(x):
        c=CONFIG[x['model']]
        return ((len(x['prompt'].encode())+1024)*c['max_input_per_million']+c['max_tokens']*c['max_output_per_million'])/1e6
    print(json.dumps({'pending':len(pending),'previous_calls':len(previous),'previous_budget_charge':charged,'conservative_all_call_bound_usd':sum(bound(x) for x in pending),'pilot':a.pilot}),flush=True)
    if a.dry_run:return
    key=os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter key (hidden): ')
    def request(path,payload=None,timeout=240):
        req=urllib.request.Request('https://openrouter.ai/api/v1/'+path,data=json.dumps(payload).encode() if payload else None,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-OpenRouter-Title':'PinTrace revision sampling'})
        with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
    initial=request('key')['data'];start_usage=float(initial.get('usage') or 0)
    limits={k:initial.get(k) for k in ['limit','limit_remaining','usage','usage_daily','usage_monthly']}
    phase='retry' if a.retry_failures else 'pilot' if a.pilot else 'main'
    (a.out/('account_start_'+phase+'_'+str(len(previous))+'.json')).write_text(json.dumps(dict(limits,checked_at_utc=utc()),indent=2))
    if isinstance(initial.get('limit_remaining'),(int,float)) and initial['limit_remaining']<0.05:raise SystemExit('Insufficient key budget')
    lock=threading.Lock();reserved=0.0;stop=threading.Event();n=0
    def execute(x):
        nonlocal charged,reserved,n
        ceiling=bound(x)
        with lock:
            if stop.is_set() or charged+reserved+ceiling>a.budget:
                stop.set();return
            reserved+=ceiling
        c=CONFIG[x['model']]
        payload={'model':x['model'],'messages':[{'role':'user','content':x['prompt']}],'temperature':0.7,'top_p':0.95,'max_tokens':c['max_tokens'],'stream':False,'provider':{'only':[c['provider']],'allow_fallbacks':False,'require_parameters':True,'max_price':{'prompt':c['max_input_per_million'],'completion':c['max_output_per_million']}}}
        result={k:v for k,v in x.items() if k!='prompt'};result.update(started_at_utc=utc(),attempt=len(attempts.get(x['request_id'],[]))+1,request_parameters={k:v for k,v in payload.items() if k!='messages'},reserved_cost_usd=ceiling)
        with lock:
            with (a.out/'request_starts.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({k:result[k] for k in ['request_id','attempt','started_at_utc','reserved_cost_usd']})+'\n');f.flush()
        t=time.monotonic()
        try:
            body=request('chat/completions',payload)
            result['response']=body;result['status']='response' if body.get('choices') else 'api_error'
            cost=(body.get('usage') or {}).get('cost')
            if isinstance(cost,(float,int)):
                result['actual_cost_usd']=cost;charge=float(cost);result['cost_source']='response.usage.cost'
            else:charge=ceiling;result['cost_source']='conservative_reservation'
            if charge>ceiling+1e-6:stop.set();result['budget_bound_exceeded']=True
            if body.get('error'):stop.set()
        except urllib.error.HTTPError as e:
            result.update(status='http_error',http_status=e.code,error=e.read().decode(errors='replace').replace(key,'[REDACTED]'))
            charge=0.0 if e.code in [400,401,402,403,404,422,429] else ceiling
            result['cost_source']='pre_generation_rejection' if charge==0 else 'conservative_reservation'
            if e.code in [400,401,402,403,404,422]:stop.set()
        except Exception as e:
            result.update(status='transport_error',error=str(e).replace(key,'[REDACTED]'));charge=ceiling;result['cost_source']='conservative_reservation'
        result.update(completed_at_utc=utc(),elapsed_seconds=time.monotonic()-t,budget_charge_usd=charge)
        with lock:
            reserved-=ceiling;charged+=charge;n+=1
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
            ch=(result.get('response') or {}).get('choices') or [{}]
            print(json.dumps({'completed_now':n,'request_id':x['request_id'],'model':x['model'],'status':result['status'],'finish_reason':ch[0].get('finish_reason'),'budget_charge_usd':round(charged,6)}),flush=True)
    # Submit one bounded wave at a time so a budget/auth failure stops promptly.
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for start in range(0,len(pending),a.workers):
            if stop.is_set():break
            list(pool.map(execute,pending[start:start+a.workers]))
            if n and n%40==0:
                now=request('key')['data'];delta=float(now.get('usage') or 0)-start_usage
                if delta+sum(x.get('budget_charge_usd',0) for x in previous)+interruption_charge>a.budget:stop.set()
    final=request('key')['data'];finish={k:final.get(k) for k in ['limit','limit_remaining','usage','usage_daily','usage_monthly']}
    phase='retry' if a.retry_failures else 'pilot' if a.pilot else 'main'
    summary={'phase':phase,'checked_at_utc':utc(),'new_calls':n,'budget_charge_usd':charged,'account_usage_delta_usd':float(final.get('usage') or 0)-start_usage,'stopped_early':stop.is_set(),'account_end':finish}
    (a.out/('summary_'+phase+'.json')).write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
