"""Frozen resolution, installation, static checks, and sandboxed BCB execution."""
import datetime, hashlib, json, os, re, subprocess, tempfile, time
from pathlib import Path, PurePosixPath
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

UV=os.environ.get('PINTRACE_UV', 'uv')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def pins(text):return {canonicalize_name(n):v for n,v in re.findall(r'^([A-Za-z0-9_.-]+)==([^\s;\\]+)',text,re.M)}
def run(cmd, *, env=None, cwd=None, timeout=600):
    start=time.monotonic()
    try:
        p=subprocess.run(cmd,env=env,cwd=cwd,capture_output=True,text=True,timeout=timeout)
        return {'command':list(map(str,cmd)),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.monotonic()-start}
    except subprocess.TimeoutExpired as e:
        def txt(v):return v.decode(errors='replace') if isinstance(v,bytes) else (v or '')
        return {'command':list(map(str,cmd)),'returncode':124,'stdout':txt(e.stdout),'stderr':txt(e.stderr),'seconds':time.monotonic()-start,'timeout':True}

def validate_project_files(code, files, primary_filename):
    if primary_filename not in files or files[primary_filename] != code:
        raise ValueError('Primary source must match the selected code exactly')
    for filename, content in files.items():
        path = PurePosixPath(filename)
        if path.is_absolute() or '..' in path.parts or '\\' in filename or ':' in filename:
            raise ValueError('Project source path must stay inside the evaluation directory')
        if path.suffix not in {'.py', '.pyi', '.wsgi'} or not isinstance(content, str):
            raise ValueError('Project files must contain Python source text')

def write_project_files(directory, files):
    for filename, content in files.items():
        path = directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

def evaluate(code, requirements, out, *, root, cutoff, bcb_problem=None, advisory_index=None, files=None, primary_filename='snippet.py'):
    if files is not None:
        validate_project_files(code, files, primary_filename)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result_path=out/'result.json'
    input_text='\n'.join(requirements)+'\n'
    revision = 'venv-env-v2' if files is None else 'project-files-v1'
    fingerprint_inputs = {'code':code,'requirements':requirements,'cutoff':cutoff,'test':(bcb_problem or {}).get('test'),'python':'3.12','uv':'0.10.7','ty':'0.0.23','bcb_network':'disabled','evaluator_revision':revision}
    if files is not None:
        fingerprint_inputs.update(files=files, primary_filename=primary_filename)
    fingerprint=sha(json.dumps(fingerprint_inputs,sort_keys=True))
    if result_path.exists():
        old=json.loads(result_path.read_text());assert old['input_fingerprint']==fingerprint;return old
    (out/'requirements.in').write_text(input_text);(out/'snippet.py').write_text(code)
    result={'input_fingerprint':fingerprint,'evaluator_revision':revision,'code_sha256':sha(code),'requirements':requirements,'started_at_utc':utc(),'cutoff':cutoff,'python_version':'3.12','python_platform':'x86_64-unknown-linux-gnu','uv_version':'0.10.7','ty_version':'0.0.23','resolved':False,'installed':False,'static_pass':False,'dynamic_pass':False,'bcb_network':'disabled'}
    if files is not None:
        write_project_files(out/'source', files)
        result['source_files'] = {name:sha(content) for name,content in files.items()}
        result['primary_filename'] = primary_filename
    env={k:v for k,v in os.environ.items() if k in ['PATH','HOME','LANG','LC_ALL','SSL_CERT_FILE','SSL_CERT_DIR','REQUESTS_CA_BUNDLE']}
    env.update(UV_CACHE_DIR=str(root/'.cache/uv'),UV_NO_PROGRESS='1',UV_CONCURRENT_BUILDS='1',UV_CONCURRENT_DOWNLOADS='8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
    lock=out/'resolved.txt'
    result['resolution']=run([UV,'pip','compile',str(out/'requirements.in'),'--python-version','3.12','--python-platform','x86_64-unknown-linux-gnu','--exclude-newer',cutoff,'--prerelease','if-necessary-or-explicit','--generate-hashes','--output-file',str(lock)],env=env,timeout=600)
    result['resolved']=result['resolution']['returncode']==0 and lock.exists()
    if result['resolved']:
        locktext=lock.read_text();chosen=pins(locktext);result['lock_sha256']=sha(locktext);result['pins']=chosen
        if advisory_index is not None:
            result['associations']={p:list(ids) for p,v in chosen.items() if (ids:=advisory_index.match(p,v))}
            result['associated']=bool(result['associations'])
        with tempfile.TemporaryDirectory(prefix='cyse_eval_',dir=str(out.parent)) as td:
            td=Path(td);venv=td/'venv';py=venv/'bin/python'
            snippet=td/'snippet.py'
            static_paths = [str(snippet)]
            project_dir = td
            if files is None:
                snippet.write_text(code)
            else:
                project_dir = td/'project'
                write_project_files(project_dir, files)
                static_paths = [str(project_dir/name) for name in sorted(files)]
            result['venv']=run([UV,'venv','--python','3.12',str(venv)],env=env,timeout=120)
            venv_env=dict(env,VIRTUAL_ENV=str(venv),PATH=str(venv/'bin')+':'+env.get('PATH',''))
            if result['venv']['returncode']==0:
                result['installation']=run([UV,'pip','sync','--python',str(py),'--require-hashes','--strict',str(lock)],env=venv_env,timeout=600)
                result['installed']=result['installation']['returncode']==0
            if result['installed']:
                result['tool_installation']=run([UV,'pip','install','--python',str(py),'--no-deps','ty==0.0.23'],env=venv_env,timeout=120)
                if result['tool_installation']['returncode']==0:
                    result['static']=run([str(venv/'bin/ty'),'check','--output-format','gitlab',*static_paths],env=venv_env,cwd=str(project_dir),timeout=120)
                    try:diagnostics=json.loads(result['static']['stdout'] or '[]')
                    except ValueError:diagnostics=['unparsed output']
                    result['static_pass']=result['static']['returncode']==0 and diagnostics==[]
                result['installed_packages']=run([UV,'pip','list','--python',str(py),'--format','json'],env=venv_env,timeout=30)
                if bcb_problem is not None:
                    # Import only the archived runner body. Execute generated code in
                    # an isolated PID/network namespace with the host read-only.
                    from stages.compact_checker import _BCB_RUNNER_SCRIPT
                    runner=td/'runner.py';payload=td/'payload.json';runner.write_text(_BCB_RUNNER_SCRIPT)
                    payload.write_text(json.dumps({'solution':code,'test':bcb_problem['test'],'entry_point':bcb_problem.get('entry_point')}))
                    home=td/'home';home.mkdir();testenv=dict(venv_env,HOME=str(home),PYTHONPATH=str(root) if files is None else str(project_dir)+os.pathsep+str(root))
                    command=['bwrap','--ro-bind','/','/','--dev','/dev','--proc','/proc','--tmpfs','/tmp','--bind',str(td),str(td),'--unshare-net','--unshare-pid','--die-with-parent','--chdir',str(project_dir),'--',str(py),str(runner),str(payload)]
                    result['dynamic']=run(command,env=testenv,timeout=200)
                    if result['dynamic']['returncode']==0:
                        try:result['bcb_result']=json.loads(result['dynamic']['stdout'].strip().splitlines()[-1])
                        except (ValueError,IndexError):result['bcb_result']={'status':'parse_error'}
                    else:result['bcb_result']={'status':'runner_error','returncode':result['dynamic']['returncode']}
                    result['dynamic_pass']=result['bcb_result'].get('status')=='pass'
    result['completed_at_utc']=utc()
    temporary_path=result_path.with_suffix('.tmp')
    temporary_path.write_text(json.dumps(result,indent=2))
    temporary_path.replace(result_path)
    return result

def package_set(lines):return sorted({canonicalize_name(Requirement(x).name) for x in lines})

def unpin(lines, exact_packages):
    result=[]
    for line in lines:
        q=Requirement(line)
        if canonicalize_name(q.name) in exact_packages:
            q.specifier=__import__('packaging.specifiers',fromlist=['SpecifierSet']).SpecifierSet('')
        result.append(str(q))
    assert package_set(result)==package_set(lines)
    return result
