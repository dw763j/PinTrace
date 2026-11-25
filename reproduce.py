"""Offline reproduction of the PinTrace paper's analysis, tables, and figures."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def safe_path(root,name):
    target=(root/name).resolve()
    if not target.is_relative_to(root.resolve()):raise ValueError(f'Archive path outside destination: {name}')
    return target

def unpack_tar(path,target):
    with tarfile.open(path) as archive:
        for member in archive:
            if member.isdir():continue
            if not member.isfile():raise ValueError(f'Unsupported archive member: {member.name}')
            dest=safe_path(target,member.name)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,dest.open('wb') as output:shutil.copyfileobj(source,output)

def verify_data():
    manifest=json.loads((ROOT/'config/data_manifest.json').read_text(encoding='utf-8'))
    checked=[]
    for item in manifest:
        records=item.get('parts') or [item]
        combined=hashlib.sha256()
        for record in records:
            p=ROOT/record['path']
            assert p.stat().st_size==record['bytes'],p
            assert digest(p)==record['sha256'],p
            if item['format']=='split-tar.gz':
                with p.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):combined.update(chunk)
            checked.append(record['path'])
        if item['format']=='split-tar.gz':assert combined.hexdigest()==item['sha256']
    return manifest,checked

def hydrate(workspace,manifest,evidence=False):
    for item in manifest:
        if item['group']=='evidence_archive':
            if not evidence:continue
            dest=workspace/'raw_evidence'/item['destination']
            if item['format']=='split-tar.gz':
                joined=workspace/'raw_evidence'/Path(item['path']).name
                joined.parent.mkdir(parents=True,exist_ok=True)
                with joined.open('wb') as out:
                    for part in item['parts']:
                        with (ROOT/part['path']).open('rb') as src:shutil.copyfileobj(src,out)
                unpack_tar(joined,dest)
            else:unpack_tar(ROOT/item['path'],dest)
            continue
        if item['group']=='evidence' and not evidence:continue
        if item['format']=='tar.gz':unpack_tar(ROOT/item['path'],workspace);continue
        p=safe_path(workspace,item['destination']);p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or digest(p)!=item['expanded_sha256']:
            with gzip.open(ROOT/item['path'],'rb') as src,p.open('wb') as out:shutil.copyfileobj(src,out)
        assert digest(p)==item['expanded_sha256'],p
    for d in ['analysis','tables','figures']: (workspace/d).mkdir(exist_ok=True)

def run(workspace,script,*args):
    cmd=[sys.executable,str(ROOT/'analysis_code'/script),*map(str,args)]
    env=dict(os.environ,PINTRACE_WORKSPACE=str(workspace),PYTHONUTF8='1',MPLBACKEND='Agg')
    log=workspace/'logs'/Path(script).with_suffix('.log').name;log.parent.mkdir(exist_ok=True)
    print(f'Running {script}',flush=True)
    with log.open('w',encoding='utf-8') as out:
        p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
    if p.returncode:
        print(log.read_text(encoding='utf-8')[-5000:],file=sys.stderr)
        raise RuntimeError(f'{script} failed; see {log}')

def cell_equal(a,b):
    if a==b:return True
    try:
        a,b=float(a),float(b)
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)
    except (ValueError,TypeError):return False

def compare_csv(expected,actual):
    def read(p):
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);return reader.fieldnames,list(reader)
    eh,er=read(expected);ah,ar=read(actual)
    if set(eh)!=set(ah):return f'columns differ: {eh} / {ah}'
    if len(er)!=len(ar):return f'row count differs: {len(er)} / {len(ar)}'
    # Published outputs preserve deterministic row order; compare all fields.
    for i,(e,a) in enumerate(zip(er,ar),2):
        for k in eh:
            if not cell_equal(e[k],a[k]):return f'row {i}, {k}: {e[k]!r} / {a[k]!r}'
    return None

def check_results(workspace):
    passed=[];failures=[];static=[]
    for family,out in [('analysis',workspace/'analysis'),('fixed_code',workspace/'experiments/fixed_code'),('repeated',workspace/'experiments/repeated'),('runtime',workspace/'experiments/runtime')]:
        for ref in sorted((ROOT/'results/reference'/family).glob('*.csv')):
            actual=out/ref.name
            if not actual.exists():static.append(f'{family}/{ref.name}');continue
            problem=compare_csv(ref,actual)
            (failures if problem else passed).append({'file':f'{family}/{ref.name}','problem':problem})
    figures=[]
    from pypdf import PdfReader
    for ref in sorted((ROOT/'results/reference/figures').glob('*.pdf')):
        actual=workspace/'figures'/ref.name
        assert actual.is_file() and len(PdfReader(actual).pages)==1,actual
        reference_page=PdfReader(ref).pages[0];generated_page=PdfReader(actual).pages[0]
        assert reference_page.extract_text()==generated_page.extract_text(),f'Figure labels differ: {ref.name}'
        figures.append({'file':ref.name,'reference_sha256':digest(ref),'generated_sha256':digest(actual),
            'labels_and_values_match':True,'reference_size_pt':[float(v) for v in reference_page.mediabox[2:]],
            'generated_size_pt':[float(v) for v in generated_page.mediabox[2:]]})
    result=dict(csv_passed=passed,csv_failures=failures,reference_only_files=static,figures=figures)
    (workspace/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assert not failures,f'Numerical differences found; see {workspace / "comparison.json"}'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=ROOT/'build')
    p.add_argument('--verify-only',action='store_true',help='Validate all packaged data hashes without unpacking or computing.')
    p.add_argument('--evidence',action='store_true',help='Also unpack raw responses, tests, commands, locks, and diagnostic evidence.')
    p.add_argument('--bootstrap',type=int,default=10000,help='Use 10000 to reproduce the paper intervals.')
    p.add_argument('--no-figures',action='store_true')
    args=p.parse_args();start=time.monotonic()
    manifest,checked=verify_data();print(f'Verified {len(checked)} packaged data files.',flush=True)
    if args.verify_only:return
    w=args.workspace.resolve();w.mkdir(parents=True,exist_ok=True)
    hydrate(w,manifest,args.evidence)
    run(w,'reanalyze.py','--bootstrap',args.bootstrap)
    run(w,'extraction_sensitivity.py')
    run(w,'analyze_neighbor.py','--qualification',w/'inputs/assessments/neighbor/qualification.jsonl','--canonical-tasks',w/'inputs/main/task_outcomes.jsonl','--new-search',w/'inputs/assessments/neighbor/task_results.json')
    run(w,'analyze_fixed_code.py')
    run(w,'analyze_runtime.py')
    run(w,'prepare_displays.py')
    run(w,'summarize_version_concentration.py')
    run(w,'analyze_repeated.py')
    run(w,'export_tables.py')
    run(w,'verify_source_inputs.py')
    run(w,'verify_claims.py')
    if not args.no_figures:
        run(w,'plot_revision.py')
        run(w,'plot_five_model_generation.py','--summary',w/'experiments/repeated/analysis_summary.json','--out',w/'figures/fig8_repeat_contrasts')
    if args.bootstrap==10000 and not args.no_figures:
        result=check_results(w)
        print(f'Compared {len(result["csv_passed"])} result CSVs and generated {len(result["figures"])} paper figures.',flush=True)
    report=dict(seconds=round(time.monotonic()-start,2),python=sys.version,verified_data_files=len(checked),bootstrap_replicates=args.bootstrap,evidence_unpacked=args.evidence)
    (w/'run.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Results: {w}',flush=True)

if __name__=='__main__':main()
