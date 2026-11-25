"""Verify the code, documentation, reference assets, and input inventory."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((ROOT/'config/repository_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        p=ROOT/item['path']
        assert p.stat().st_size==item['bytes'],p
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==item['sha256'],p
    print(f'Verified {len(manifest["files"])} repository files.')
if __name__=='__main__':main()
