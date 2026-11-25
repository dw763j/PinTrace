"""Reevaluate one prepared response on a disposable Linux worker."""
import argparse,gzip,json,os,sys
from pathlib import Path
ENGINE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ENGINE))
from evaluation_support_remote import evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--request-id',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bigcodebench',type=Path,help='The selected BigCodeBench JSONL or JSONL.gz, including tests.')
    a=p.parse_args();record=None
    with a.prepared.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['request_id']==a.request_id:record=r;break
    if record is None:p.error('Request ID absent from the prepared inputs')
    if record['response_status']!='usable':p.error('This response has no complete routine evaluation input')
    problem=None
    if record['track']=='d1':
        if not a.bigcodebench:p.error('BigCodeBench execution requires its original tests')
        opener=gzip.open if a.bigcodebench.suffix=='.gz' else open
        with opener(a.bigcodebench,'rt',encoding='utf-8') as f:
            problem=next(json.loads(l) for l in f if json.loads(l)['task_id']==record['task_id'])
    result=evaluate(record['code'],record['requirements'],a.out,root=ENGINE,cutoff='2026-02-01T00:00:00Z',bcb_problem=problem,files=record['files'],primary_filename=record['primary_filename'])
    print(json.dumps({k:result[k] for k in ['resolved','installed','static_pass','dynamic_pass']}))
if __name__=='__main__':main()
