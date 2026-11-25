"""Integrity and schema checks independent of external APIs and generated execution."""
import gzip,json,re,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import reproduce

class ArtifactTests(unittest.TestCase):
    def test_archive_paths_stay_in_destination(self):
        root=ROOT/'build/path-test'
        with self.assertRaises(ValueError):reproduce.safe_path(root,'../../escape')
        self.assertTrue(reproduce.safe_path(root,'nested/input.json').is_relative_to(root.resolve()))

    def test_main_population_and_stage_invariants(self):
        seen=set();counts={}
        with gzip.open(ROOT/'data/inputs/main/task_outcomes.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line);k=(r['track'],r['mode'],r['model'],r['task_id'])
                self.assertNotIn(k,seen);seen.add(k)
                counts[r['track'],r['mode']]=counts.get((r['track'],r['mode']),0)+1
                self.assertLessEqual(r['static_pass'],r['installation_pass'])
                self.assertLessEqual(r['dynamic_pass'],r['installation_pass'])
                self.assertLessEqual(r['installation_pass'],r['complete_lock'])
        self.assertEqual(len(seen),36260)
        self.assertEqual(counts,{('d1','inline'):8130,('d1','requirements.txt'):8130,('d2','inline'):10000,('d2','requirements.txt'):10000})

    def test_partition_is_disjoint_and_complete(self):
        def ids(name):
            with gzip.open(ROOT/f'data/inputs/assessments/partition/{name}_task_ids.jsonl.gz','rt') as f:return {str(json.loads(l)['task_id']) for l in f}
        a,b=ids('clean'),ids('contaminated')
        self.assertEqual((len(a),len(b)),(927,73));self.assertFalse(a&b);self.assertEqual(len(a|b),1000)

    def test_all_manuscript_displays_have_sources(self):
        m=json.loads((ROOT/'config/manuscript.json').read_text())
        self.assertEqual(len(m['tables']),19);self.assertEqual(len(m['figures']),8)
        for t in m['tables']:self.assertTrue((ROOT/t['path']).is_file())
        for name in m['figures']:self.assertTrue((ROOT/'results/reference/figures'/name).is_file())

    def test_repository_documentation_and_code_are_english(self):
        for p in ROOT.rglob('*'):
            parts=p.relative_to(ROOT).parts
            if any(x in {'.git','build','.venv','__pycache__'} or x.startswith('build-') for x in parts):continue
            self.assertFalse(re.search(r'[\u3400-\u9fff]',p.name),str(p))
            if p.is_file() and p.suffix in {'.py','.md','.json','.txt','.yml','.yaml','.cff'}:
                self.assertFalse(re.search(r'[\u3400-\u9fff]',p.read_text(encoding='utf-8-sig')),str(p))

    def test_engine_support_modules_are_present(self):
        for name in ['scripts/__init__.py','scripts/notifier_utils.py','scripts/run_neighbor_version_experiment.py']:
            self.assertTrue((ROOT/'engine'/name).is_file(),name)

    def test_neighbor_archives_have_platform_independent_order(self):
        sys.path.insert(0,str(ROOT/'analysis_code'))
        from analyze_neighbor import ordered_model_archives
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['model_GPT.json','model_DeepSeek.json','model_claude.json']:
                (root/name).write_text('{}',encoding='utf-8')
            self.assertEqual([p.name for p in ordered_model_archives(root)],
                ['model_claude.json','model_DeepSeek.json','model_GPT.json'])

if __name__=='__main__':unittest.main()
