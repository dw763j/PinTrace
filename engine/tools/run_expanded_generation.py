"""Run a frozen OpenRouter panel with an incremental $50 cap and private credentials.

Each request start is journalled before transmission. Interrupted attempts retain
their full reservation. No output-quality or truncation retry is permitted.
"""
import argparse, concurrent.futures, datetime, getpass, hashlib, json, os, signal, threading, time, urllib.error, urllib.request
from pathlib import Path

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.exists() else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--phase', choices=['pilot', 'main', 'retry'], required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--access-recheck', action='store_true', help='One request per model, only after the user asks to recheck access')
    ap.add_argument('--proxy', help='User-provided HTTP proxy; HTTPS CONNECT retains default TLS verification')
    a = ap.parse_args()
    config = json.loads(a.config.read_text(encoding='utf-8'))
    budget = config['incremental_budget_usd']
    assert 0 < budget <= 50
    models = config['models']
    all_jobs = rows(a.requests)
    assert len(all_jobs) == 1440 and len({x['request_id'] for x in all_jobs}) == 1440
    a.out.mkdir(parents=True, exist_ok=True)
    log = a.out/'responses.jsonl'
    starts_path = a.out/'request_starts.jsonl'
    previous = rows(log)
    starts = rows(starts_path)
    returned = {(r['request_id'], r['attempt']) for r in previous}
    orphaned = [s for s in starts if (s['request_id'], s['attempt']) not in returned]
    charged = sum(r['budget_charge_usd'] for r in previous) + sum(s['reserved_cost_usd'] for s in orphaned)
    attempts = {}
    for s in starts:
        attempts.setdefault(s['request_id'], []).append(s)
    latest = {}
    for r in previous:
        if r['request_id'] not in latest or latest[r['request_id']]['status'] != 'response':
            latest[r['request_id']] = r
    selected = set()
    for track in ['d2', 'd1']:
        unique = {x['task_id']:x for x in all_jobs if x['track'] == track}
        selected.add((track, next(iter(unique))))
        selected.add((track, max(unique, key=lambda k:len(unique[k]['prompt']))))
    jobs = [x for x in all_jobs if a.phase != 'pilot' or ((x['track'],x['task_id']) in selected and x['repeat'] == 1)]
    if a.access_recheck:
        assert a.phase == 'pilot'
        candidates=[x for x in jobs if x['request_id'] in latest and latest[x['request_id']].get('http_status') in [401,402,403]]
        seen=set();jobs=[]
        for x in candidates:
            if x['model'] not in seen:
                jobs.append(x);seen.add(x['model'])
    elif a.phase == 'retry':
        jobs = [x for x in jobs if x['request_id'] in latest and len(attempts.get(x['request_id'], [])) < 2 and (latest[x['request_id']]['status']=='transport_error' or (latest[x['request_id']]['status']=='http_error' and (latest[x['request_id']].get('http_status')==429 or 500<=latest[x['request_id']].get('http_status',0)<600)))]
    else:
        access_restored=(a.out/'access_recheck_passed.json').exists()
        jobs = [x for x in jobs if x['request_id'] not in latest or (access_restored and latest[x['request_id']].get('http_status') in [401,402,403])]
    frozen = dict(config, request_file_sha256=hashlib.sha256(a.requests.read_bytes()).hexdigest())
    fp = a.out/'frozen_config.json'
    if fp.exists():
        assert json.loads(fp.read_text()) == frozen, 'Run configuration changed'
    else:
        fp.write_text(json.dumps(frozen, indent=2))
    def bound(x):
        c = models[x['model']]
        # UTF-8 bytes plus 1024 is a conservative prompt-token reservation for
        # these short text prompts. All visible and reasoning tokens are capped.
        return ((len(x['prompt'].encode())+1024)*c['input_price_per_million'] + (c['parameters']['max_tokens']+32)*c['output_price_per_million'])/1e6
    print(json.dumps(dict(phase=a.phase, pending=len(jobs), previous_attempt_logs=len(previous), unresolved_starts=len(orphaned), budget_charge_usd=charged, all_remaining_maximum_cost_usd=sum(bound(x) for x in jobs))), flush=True)
    if a.dry_run:
        return
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter key (hidden): ')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':a.proxy,'https':a.proxy})) if a.proxy else urllib.request.build_opener()
    def request(path, payload=None):
        req = urllib.request.Request('https://openrouter.ai/api/v1/'+path, data=json.dumps(payload).encode() if payload is not None else None, headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-OpenRouter-Title':'PinTrace repeated generation study'})
        with opener.open(req, timeout=300) as response:
            return json.load(response)
    def account():
        data = request('key')['data']
        safe = {k:data.get(k) for k in ['limit','limit_remaining','usage','usage_daily','usage_monthly']}
        try:
            credit = request('credits')['data']
            safe['account_credit_remaining_usd'] = float(credit['total_credits'])-float(credit['total_usage'])
        except Exception:
            safe['account_credit_remaining_usd'] = None
        safe['checked_at_utc'] = utc()
        return safe
    initial = account()
    start_usage = float(initial.get('usage') or 0)
    start_charge = charged
    (a.out/f'account_start_{a.phase}_{len(previous)}.json').write_text(json.dumps(initial, indent=2))
    print(json.dumps({'account_credit_remaining_usd': initial.get('account_credit_remaining_usd'), 'authorized_incremental_limit_usd':budget}), flush=True)
    lock = threading.Lock()
    reserved = 0.0
    stop = threading.Event()
    stop_reason = []
    n = 0
    def checkpoint(signum, frame):
        stop_reason.append('graceful_checkpoint');stop.set()
    signal.signal(signal.SIGINT,checkpoint)
    def execute(x):
        nonlocal charged, reserved, n
        ceiling = bound(x)
        with lock:
            if stop.is_set():
                return
            if charged + reserved + ceiling > budget:
                stop_reason.append('authorized_budget_reservation_limit'); stop.set(); return
            reserved += ceiling
        c = models[x['model']]
        payload = dict(model=x['model'], messages=[{'role':'user','content':x['prompt']}], stream=False, **c['parameters'], provider={'only':[c['provider']],'allow_fallbacks':False,'require_parameters':True,'max_price':{'prompt':c['input_price_per_million'],'completion':c['output_price_per_million']}})
        attempt = 1 + max([r['attempt'] for r in attempts.get(x['request_id'], [])] or [0])
        result = {k:v for k,v in x.items() if k!='prompt'}
        result.update(attempt=attempt, phase=a.phase, started_at_utc=utc(), reserved_cost_usd=ceiling, network_route='user_provided_http_connect_proxy' if a.proxy else 'default_environment_route', request_parameters={k:v for k,v in payload.items() if k!='messages'})
        with lock:
            with starts_path.open('a',encoding='utf-8') as f:
                f.write(json.dumps({k:result[k] for k in ['request_id','attempt','phase','started_at_utc','reserved_cost_usd']})+'\n'); f.flush(); os.fsync(f.fileno())
        t = time.monotonic()
        try:
            body = request('chat/completions', payload)
            result.update(response=body, status='response' if body.get('choices') else 'api_error')
            cost = (body.get('usage') or {}).get('cost')
            charge = float(cost) if isinstance(cost,(int,float)) else ceiling
            result['cost_source'] = 'response.usage.cost' if isinstance(cost,(int,float)) else 'conservative_reservation'
            if isinstance(cost,(int,float)):
                result['actual_cost_usd'] = cost
            if charge > ceiling+1e-6:
                result['budget_bound_exceeded'] = True; stop_reason.append('reported_cost_exceeds_reservation'); stop.set()
            if body.get('error'):
                stop_reason.append('api_error'); stop.set()
        except urllib.error.HTTPError as e:
            result.update(status='http_error', http_status=e.code, error=e.read().decode(errors='replace').replace(key,'[REDACTED]'))
            charge = 0.0 if e.code in [400,401,402,403,404,422,429] else ceiling
            result['cost_source'] = 'pre_generation_rejection' if charge==0 else 'conservative_reservation'
            if e.code in [400,401,402,403,404,422]:
                stop_reason.append('http_'+str(e.code)); stop.set()
        except Exception as e:
            result.update(status='transport_error', error=str(e).replace(key,'[REDACTED]'))
            charge = ceiling
            result['cost_source'] = 'conservative_reservation'
        result.update(completed_at_utc=utc(), elapsed_seconds=time.monotonic()-t, budget_charge_usd=charge)
        with lock:
            reserved -= ceiling; charged += charge; n += 1
            with log.open('a',encoding='utf-8') as f:
                f.write(json.dumps(result,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
            choice = ((result.get('response') or {}).get('choices') or [{}])[0]
            print(json.dumps(dict(completed_now=n, model=x['model'], status=result['status'], finish_reason=choice.get('finish_reason'), budget_charge_usd=round(charged,6))), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        iterator=iter(jobs);active=set();last_account_check=0
        def fill_slots():
            while len(active)<a.workers and not stop.is_set():
                x=next(iterator,None)
                if x is None:break
                active.add(pool.submit(execute,x))
        fill_slots()
        while active:
            finished,active=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:future.result()
            if n-last_account_check>=60:
                try:
                    now=request('key')['data'];last_account_check=n
                    if float(now.get('usage') or 0)-start_usage+start_charge>budget:
                        stop_reason.append('account_usage_limit');stop.set()
                except Exception:
                    stop_reason.append('account_check_failed');stop.set()
            fill_slots()
    try:
        final = account()
    except Exception:
        final = {}
    summary = dict(phase=a.phase, finished_at_utc=utc(), new_attempts=n, budget_charge_usd=charged, reported_response_cost_usd=sum(r.get('actual_cost_usd',0) for r in rows(log)), stopped_early=stop.is_set(), stop_reasons=sorted(set(stop_reason)), account_end=final, account_usage_delta_usd=(float(final.get('usage') or 0)-start_usage) if final else None)
    if a.access_recheck:
        summary['access_recheck']=True
        new_records=rows(log)[len(previous):]
        if len(new_records)==len(models) and all(r['status']=='response' for r in new_records):
            (a.out/'access_recheck_passed.json').write_text(json.dumps({'checked_at_utc':utc(),'models':sorted(models)},indent=2))
    (a.out/f'summary_{a.phase}_{len(previous)}.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__ == '__main__':
    main()
