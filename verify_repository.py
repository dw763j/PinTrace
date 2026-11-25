"""Verify the code, documentation, reference assets, and input inventory."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-index',action='store_true',help='Also verify staged Git blobs before publishing.')
    parser.add_argument('--git',default=shutil.which('git'),help='Git executable for --git-index.')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'config/repository_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        p=ROOT/item['path']
        assert p.stat().st_size==item['bytes'],p
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==item['sha256'],p
    print(f'Verified {len(manifest["files"])} repository files.')
    if args.git_index:
        if not args.git:parser.error('Git is required for --git-index; supply --git.')
        process=subprocess.Popen([args.git,'-C',str(ROOT),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        mismatches=[]
        try:
            for item in manifest['files']:
                process.stdin.write((':'+item['path']+'\n').encode('utf-8'));process.stdin.flush()
                header=process.stdout.readline().decode('ascii').split()
                if not header or header[-1]=='missing':raise ValueError(f'File absent from Git index: {item["path"]}')
                size=int(header[-1]);remaining=size;digest=hashlib.sha256()
                while remaining:
                    chunk=process.stdout.read(min(remaining,1024*1024))
                    if not chunk:raise ValueError('Unexpected end of Git object stream')
                    digest.update(chunk);remaining-=len(chunk)
                if process.stdout.read(1)!=b'\n':raise ValueError('Invalid Git object boundary')
                if size!=item['bytes'] or digest.hexdigest()!=item['sha256']:mismatches.append(item['path'])
        finally:
            process.stdin.close();process.stdout.close();process.wait()
        if mismatches:raise ValueError('Git index differs from the inventory: '+', '.join(mismatches))
        print(f'Verified {len(manifest["files"])} staged Git blobs.')
if __name__=='__main__':main()
