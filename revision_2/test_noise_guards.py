"""Guardrails against mixing source data, teacher banks, or resumed results."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from run_coalition_noise import checked_resume,compare_manifest,ensure_same,verify_file,sha


class ProvenanceGuards(unittest.TestCase):
    def test_determinism_is_enabled_before_original_runner(self):
        import run_wisdm_noise_deterministic as startup
        calls=[]
        fake_tf=SimpleNamespace(
            __version__='test-runtime',
            keras=SimpleNamespace(__version__='test-keras',utils=SimpleNamespace(
                set_random_seed=lambda seed:calls.append(('seed',seed)))),
            config=SimpleNamespace(experimental=SimpleNamespace(
                enable_op_determinism=lambda:calls.append(('determinism',True)),
                enable_tensor_float_32_execution=lambda value:calls.append(('tf32',value)))))
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict('sys.modules',{'tensorflow':fake_tf}), \
                 patch('sys.argv',['startup','--drive-root',folder,'--preflight-only']), \
                 patch.object(startup.runpy,'run_path',side_effect=lambda *a,**k:calls.append(('delegate',True))):
                startup.main()
            self.assertEqual(calls,[('seed',2026),('determinism',True),('tf32',False),('delegate',True)])
            record=json.loads((Path(folder)/'results/revision-2/coalition-noise/wisdm/deterministic-startup.json').read_text())
            self.assertTrue(record['deterministic_operations'])
            self.assertIn('startup_launcher_sha256',record)

    def test_teacher_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model=Path(folder)/'teacher.keras'
            model.write_bytes(b'frozen-weights')
            expected=sha(model)
            model.write_bytes(b'different-teacher-bank')
            with self.assertRaises(RuntimeError):verify_file(model,expected)

    def test_split_change_is_rejected_before_training(self):
        with self.assertRaises(RuntimeError):
            compare_manifest({'train':{'subjects':[1,2]},'test':{'subjects':[3]}},
                             {'train':{'subjects':[1,3]},'test':{'subjects':[2]}})

    def test_changed_protocol_cannot_reuse_result(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'result.json'
            p.write_text(json.dumps({'status':'complete','identity':'old-teacher-hash','validation_macro_f1':.9}))
            with self.assertRaises(RuntimeError):checked_resume(p,'new-teacher-hash')

    def test_completed_fit_can_resume_but_invalid_score_cannot(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'result.json'
            p.write_text(json.dumps({'status':'complete','identity':'correct','validation_macro_f1':.9}))
            self.assertEqual(checked_resume(p,'correct')['validation_macro_f1'],.9)
            p.write_text(json.dumps({'status':'complete','identity':'correct','validation_macro_f1':float('nan')}))
            with self.assertRaises(RuntimeError):checked_resume(p,'correct')

    def test_epoch_budget_change_cannot_merge(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'protocol.json'
            ensure_same(p,{'epochs':20})
            with self.assertRaises(RuntimeError):ensure_same(p,{'epochs':60})


if __name__=='__main__':unittest.main()
