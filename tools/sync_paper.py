"""Synchronize paper reference assets and their checksums without copying the manuscript."""
from __future__ import annotations
import argparse,hashlib,json,re,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def extract_tables(text):
    result={}
    for match in re.finditer(r'\\begin\{(tableorg|table\*?|longtable)\}[\s\S]*?\\end\{\1\}',text):
        label=re.search(r'\\label\{([^}]+)\}',match[0])
        if label:result[label[1]]=(match.start(),match[0])
    for label in re.finditer(r'\\label\{(tab:[^}]+)\}',text):
        if label[1] in result:continue
        start=text.rfind(r'\noindent\begin{minipage}',0,label.start())
        if start<0:raise ValueError(f'No table environment for {label[1]}')
        end=text.index(r'\end{minipage}',label.end())+len(r'\end{minipage}')
        block=text[start:end]
        if r'\captionof{table}' not in block:raise ValueError(f'No table caption for {label[1]}')
        result[label[1]]=(start,block)
    return result

def sync_paper(source,root=ROOT):
    source=source.resolve();text=source.read_text(encoding='utf-8')
    metadata_path=root/'config/manuscript.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    tables=extract_tables(text)
    expected={r['label'] for r in metadata['tables']}
    if set(tables)!=expected:raise ValueError('Table labels differ; update the evidence map and table verification contracts first.')
    figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text)
    if len(figures)!=len(metadata['figures']):raise ValueError('Figure count differs; update the evidence map first.')
    staged=[]
    for name in figures:
        candidates=[source.parent/name,source.parent/'figures'/name]
        path=next((p for p in candidates if p.is_file()),None)
        if path is None:raise FileNotFoundError(name)
        staged.append((path,root/'results/reference/figures'/path.name))
    bibliography=[]
    for names in re.findall(r'\\bibliography\{([^}]+)\}',text):
        for name in names.split(','):
            path=source.parent/(name.strip()+'.bib')
            if not path.is_file():raise FileNotFoundError(path.name)
            bibliography.append(path)
    for path,target in staged:shutil.copyfile(path,target)
    table_records=[]
    for label,(start,block) in sorted(tables.items(),key=lambda item:item[1][0]):
        path=root/'results/reference/tables'/(label.replace(':','_')+'.tex')
        path.write_text(block+'\n',encoding='utf-8',newline='\n')
        table_records.append(dict(label=label,path=path.relative_to(root).as_posix(),source_line=text[:start].count('\n')+1,sha256=digest(path)))
    for path in bibliography:shutil.copyfile(path,root/'results/reference'/path.name)
    metadata.update(sha256=digest(source),main_file=source.name,figures=[target.name for _,target in staged],
        figure_sha256={target.name:digest(target) for _,target in staged},tables=table_records,
        bibliography_sha256={path.name:digest(path) for path in bibliography})
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(f'Synchronized {len(staged)} figures and {len(table_records)} tables for manuscript {metadata["sha256"]}.')

def refresh_inventory(root=ROOT):
    files=[]
    for path in root.rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(root)
        if any(p in {'.git','build','.venv','__pycache__','.pytest_cache','.cache'} or p.startswith('build-') for p in rel.parts):continue
        if rel.as_posix()=='config/repository_manifest.json':continue
        files.append(dict(path=rel.as_posix(),bytes=path.stat().st_size,sha256=digest(path)))
    files.sort(key=lambda row:row['path'])
    metadata=dict(files=files,total_bytes=sum(x['bytes'] for x in files),file_count=len(files),
        manuscript_sha256=json.loads((root/'config/manuscript.json').read_text())['sha256'])
    (root/'config/repository_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(f'Recorded {len(files)} repository file hashes.')

def record_verification(workspace,root=ROOT):
    workspace=workspace.resolve()
    comparison=json.loads((workspace/'comparison.json').read_text(encoding='utf-8'))
    run=json.loads((workspace/'run.json').read_text(encoding='utf-8'))
    if comparison['csv_failures'] or comparison['reference_only_files']:
        raise ValueError('Result comparisons are incomplete or have differences.')
    if run['bootstrap_replicates']!=10000:raise ValueError('Paper verification requires 10000 bootstrap samples.')
    expected=json.loads((root/'config/manuscript.json').read_text(encoding='utf-8'))
    if len(comparison['figures'])!=len(expected['figures']):raise ValueError('Figure verification is incomplete.')
    for figure in comparison['figures']:
        if figure['reference_sha256']!=expected['figure_sha256'][figure['file']]:
            raise ValueError('Comparison used a different figure snapshot.')
    output=root/'results/verification'
    for name in ['comparison.json','source_verification.json','claims.json','run.json']:
        shutil.copyfile(workspace/name,output/name)
    shutil.copyfile(workspace/'tables/verification.json',output/'table_values.json')
    print('Recorded successful reproduction reports.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript',type=Path)
    parser.add_argument('--verification-workspace',type=Path)
    parser.add_argument('--refresh-inventory',action='store_true')
    args=parser.parse_args()
    if not args.manuscript and not args.verification_workspace and not args.refresh_inventory:
        parser.error('Specify --manuscript, --verification-workspace, or --refresh-inventory.')
    if args.manuscript:sync_paper(args.manuscript)
    if args.verification_workspace:record_verification(args.verification_workspace)
    if args.refresh_inventory:refresh_inventory()

if __name__=='__main__':main()
