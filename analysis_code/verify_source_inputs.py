"""Verify response-to-code lineage and export complete environment replay inputs."""
import ast,collections,gzip,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=Path(os.environ['PINTRACE_WORKSPACE'])
sys.path.insert(0,str(ROOT/'engine/tools'))
from manual_response_extraction import extract_response

def key(r):return r['track'],r['mode'],r['model'].lower(),str(r['task_id'])
def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def main():
    choices={key(x):x for x in json.loads((R/'inputs/assessments/source_decisions/source_selection_overrides.json').read_text(encoding='utf-8'))}
    structural={key(x):x for x in json.loads((R/'inputs/assessments/source_decisions/structural_outcomes.json').read_text(encoding='utf-8'))}
    runtime={key(r):r for r in rows(R/'experiments/runtime/observations.jsonl') if r['python_version']=='3.12'}
    assert len(runtime)==36260
    output=R/'prepared/main.jsonl';output.parent.mkdir(exist_ok=True)
    seen=set();counts=collections.Counter()
    source=ROOT/'data/inputs/generation/main.jsonl.gz'
    with gzip.open(source,'rt',encoding='utf-8') as stream,output.open('w',encoding='utf-8') as target:
        for line in stream:
            row=json.loads(line);k=key(row);assert k not in seen;seen.add(k)
            assert hashlib.sha256(row['raw_response'].encode()).hexdigest()==row['response_sha256']
            observation=runtime[k];assert row['response_sha256']==observation['response_sha256']
            extracted=extract_response(row['raw_response'],legacy_code=row['selected_code'])
            if k in choices:
                assert choices[k]['response_sha256']==row['response_sha256'];extracted=choices[k]['result_override']
            if k in structural:
                assert structural[k]['response_sha256']==row['response_sha256']
                extracted.update(code='',files={},primary_filename=None)
            code=extracted['code'];assert hashlib.sha256(code.encode()).hexdigest()==observation['source_code_sha256'],k
            prepared={name:observation[name] for name in ['request_id','track','mode','model','task_id','response_sha256','response_status']}
            prepared.update(code=code,files=extracted['files'],primary_filename=extracted['primary_filename'],requirements=observation['requirements'])
            target.write(json.dumps(prepared,ensure_ascii=False)+'\n');counts[row['track']+'/'+row['mode']]+=1
    assert len(seen)==36260 and seen==set(runtime)
    # Verify the BigCodeBench subset using its supplied library metadata.
    from stdlib_list import stdlib_list
    stdlib=set(stdlib_list('3.12'))
    with gzip.open(ROOT/'data/inputs/tasks/bigcodebench-v0.1.4.jsonl.gz','rt',encoding='utf-8') as f:
        problems=[json.loads(l) for l in f]
    selected={r['task_id'] for r in problems if any(x.strip() not in stdlib for x in ast.literal_eval(r['libs']) if x.strip())}
    ids={r['task_id'] for r in rows(R/'inputs/tasks/bigcodebench-task-ids.jsonl')}
    assert selected==ids and len(selected)==813
    result=dict(main_response_hashes_verified=len(seen),code_hashes_verified=len(seen),bigcodebench_tasks=len(problems),bigcodebench_subset=len(selected),populations=dict(counts),replay_inputs='prepared/main.jsonl')
    (R/'source_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
