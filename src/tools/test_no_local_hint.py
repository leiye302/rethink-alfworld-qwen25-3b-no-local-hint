"""No-hint means same full SFT and ordinary native Teacher, never no-OPSD."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import shared_start
from ablation import contract


class NoHintTests(unittest.TestCase):
    def test_pipeline_requires_shared_model_and_never_auto_retrains_sft(self):
        import run_pair
        with patch.object(run_pair, 'resolve_python', return_value=sys.executable), \
                patch.object(run_pair, 'verify_start', side_effect=ValueError('Missing shared start')), \
                patch.object(run_pair, 'sft_complete', side_effect=AssertionError('Must not bootstrap SFT')):
            with self.assertRaisesRegex(ValueError, 'Missing shared start'):
                run_pair.commands('opsd')
        with patch.object(run_pair, 'resolve_python', return_value=sys.executable), \
                patch.object(run_pair, 'verify_start', return_value={'fixture': 'verified'}), \
                patch.object(run_pair, 'sft_complete', side_effect=AssertionError('Must not bootstrap SFT')):
            self.assertEqual([name for name, _ in run_pair.commands('opsd')],
                             ['training', 'export', 'evaluation'])

    def test_exactly_one_stage2_algorithm_setting_changes(self):
        import yaml
        reference = json.loads((ROOT / 'configs/reference_3b.json').read_text())
        current = yaml.safe_load((ROOT / 'configs/opsd.yaml').read_text())
        self.assertFalse(current['algorithm']['local_hints']['enabled'])
        current['algorithm']['local_hints']['enabled'] = True
        self.assertEqual(current['algorithm'], reference['opsd']['algorithm'])
        self.assertEqual(current['actor_rollout_ref'], reference['opsd']['actor_rollout_ref'])
        self.assertEqual(current['data'], reference['opsd']['data'])
        self.assertEqual(current['env'], reference['opsd']['env'])
        spec = contract()
        self.assertFalse(spec['API_at_training'])
        self.assertEqual(spec['opsd_initialization'], 'shared_full_route_sft_step2000')
        recipe = json.loads((ROOT / 'configs/sft.json').read_text())
        self.assertEqual(recipe['supervision_mode'], 'full')

    @unittest.skipUnless((ROOT / '.deps/sdar/.rethink_setup.json').is_file(), 'Prepare pinned SDAR first')
    def test_actual_native_branch_calls_ordinary_teacher_without_importing_hint_api(self):
        source = ROOT / '.deps/sdar/verl/trainer/ppo/skillsd_ray_trainer.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and 'local_hints' in ast.unparse(node.test)
                    and '_compute_teacher_log_probs' in ast.unparse(node)]
        self.assertEqual(len(branches), 1)
        compiled = compile(ast.fix_missing_locations(ast.Module(body=branches, type_ignores=[])),
                           'native-no-hint-branch', 'exec')
        calls = []
        teacher = object()
        batch = {'actor_marker_present': True}
        trainer = SimpleNamespace(config=SimpleNamespace(algorithm={'local_hints': {'enabled': False}}),
            _compute_teacher_log_probs=lambda value: (calls.append(value), teacher)[1])
        scope = {'self': trainer, 'batch': batch}
        # Execute the real conditional, not a reimplementation of the branch.
        with patch('builtins.__import__', side_effect=AssertionError('No hint import allowed')):
            exec(compiled, scope)
        self.assertIs(scope['teacher_log_probs'], teacher)
        self.assertEqual(calls, [batch])

    def test_launch_does_not_read_credentials_or_offer_paid_calls(self):
        source = (ROOT / 'src/opsd/run.py').read_text(encoding='utf-8')
        self.assertNotIn('allow-api-cost', source)
        self.assertNotIn("os.environ.get('RETHINK_API_KEY')", source)
        self.assertNotIn('from api_worker import', source)
        self.assertNotIn('api_auth.json', source)
        self.assertIn('No-local-hint ablation cannot enable an API', source)
        prepare = (ROOT / 'src/opsd/prepare.py').read_text(encoding='utf-8')
        self.assertIn("'algorithm.local_hints.enabled':False", prepare)
        self.assertIn("'algorithm.local_hints.api_auth_file':None", prepare)

    def test_shared_initialization_is_not_a_public_webshop_or_trigger_release(self):
        value = json.loads((ROOT / 'configs/shared_start.json').read_text())
        self.assertEqual(value['step'], 2000)
        self.assertEqual(value['base_model'], 'Qwen/Qwen2.5-3B-Instruct')
        self.assertEqual(value['supervision'], 'full_suffix_ce_plus_marker_bce')
        self.assertEqual(len(value['files']), 17)
        self.assertEqual(value['files']['sft_stage_manifest.json']['sha256'],
            '59bc680676e9590e9e199b823c956ad3317fb985ea138eac790b589af30bb700')
        self.assertNotIn('training_state.pt', value['files'])
        self.assertEqual(set(value['legal_files']), {'model_license.txt', 'Notice.txt'})


class StartImportTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'configs').mkdir()
        source = root / 'supplied'
        source.mkdir()
        (source / 'config.json').write_text('{}', encoding='utf-8')
        (source / 'model.safetensors').write_bytes(b'fixture-not-model-weights')
        (source / 'sft_stage_manifest.json').write_text('{"fixture":true}', encoding='utf-8')
        files = {path.name: {'bytes': path.stat().st_size, 'sha256': shared_start.sha(path)}
                 for path in source.iterdir()}
        (root / 'configs/shared_start.json').write_text(json.dumps({'files': files}), encoding='utf-8')
        return source, files

    def test_import_is_verified_idempotent_and_excludes_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, files = self.fixture(root)
            (source / 'training_state.pt').write_bytes(b'not-for-release')
            with patch.object(shared_start, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                shared_start.import_start(source)
                shared_start.import_start(source)
                dest = root / 'checkpoints/route_sft/global_step_2000'
                self.assertEqual(shared_start.verify_start(dest), {n: r['sha256'] for n, r in sorted(files.items())})
            self.assertFalse((dest / 'training_state.pt').exists())
            self.assertTrue((dest / 'shared_start_receipt.json').is_file())

    def test_corruption_is_rejected_before_copy_and_existing_files_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.fixture(root)
            (source / 'model.safetensors').write_bytes(b'corrupt')
            with patch.object(shared_start, 'ROOT', root), self.assertRaises(ValueError):
                shared_start.import_start(source)
            self.assertFalse((root / 'checkpoints').exists())

    def test_other_checkpoint_and_unsafe_member_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.fixture(root)
            (source / 'sft_stage_manifest.json').write_text('{"stage":"webshop","step":1500}', encoding='utf-8')
            with patch.object(shared_start, 'ROOT', root), self.assertRaises(ValueError):
                shared_start.verify_start(source)
            for name in ('../outside', '/absolute', '..'):
                with self.assertRaises(ValueError):
                    shared_start.child(source, name)


if __name__ == '__main__':
    unittest.main()
