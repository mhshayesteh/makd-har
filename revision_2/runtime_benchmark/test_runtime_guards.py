"""Fast safety/semantics tests; no neural training or benchmark results."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from run_runtime import (ovo_predict,export_linear,metrics,source_parity,converted_parity,
    linear_choice,timing_plan,timing_summary,atomic_json,verify_files,records,require_gate,digest,canonical_threadpools)

class Guards(unittest.TestCase):
    def test_threadpool_order_is_stable_but_real_changes_are_retained(self):
        pools=[dict(prefix='blas',version='1',num_threads=1),dict(prefix='omp',version=None,num_threads=1),dict(prefix='blas',version='2',num_threads=1)]
        self.assertEqual(canonical_threadpools(pools),canonical_threadpools(pools[::-1]))
        changed=[dict(v) for v in pools];changed[0]['version']='3'
        self.assertNotEqual(canonical_threadpools(pools),canonical_threadpools(changed))
        self.assertEqual(len(canonical_threadpools(pools+[pools[0]])),4)
        changed=[dict(v) for v in pools];changed[0]['num_threads']=2
        with self.assertRaises(RuntimeError):canonical_threadpools(changed)

    def test_libsvm_zero_margin_and_vote_tie(self):
        # Three pairwise classifiers: +,-,+ forms a vote tie -> first class.
        state=dict(classes=np.array([10,20,30]),coef=np.zeros((3,1)),intercept=np.array([1.,-1.,1.]))
        np.testing.assert_array_equal(ovo_predict(np.zeros((1,1)),state),[10])
        # All zero margins choose the second class in each pair -> final class.
        state['intercept'][:]=0
        np.testing.assert_array_equal(ovo_predict(np.zeros((1,1)),state),[30])

    def test_compact_matches_actual_multiclass_svc(self):
        from sklearn.svm import SVC
        rng=np.random.default_rng(19);x=rng.normal(size=(180,12));y=np.arange(180)%6
        model=SVC(kernel='linear',C=.01,break_ties=False).fit(x,y)
        fixture=rng.normal(size=(240,12));state=export_linear(model)
        np.testing.assert_allclose(fixture@state['coef'].T+state['intercept'],model._decision_function(fixture),rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(ovo_predict(fixture,state),model.predict(fixture))

    def test_reject_label_flip_even_if_logits_close(self):
        original=np.array([[1e-7,0.,-1.,-2.,-3.,-4.]])
        changed=original.copy();changed[0,1]=2e-7
        with self.assertRaises(RuntimeError):converted_parity(original,changed)
        with self.assertRaises(AssertionError):converted_parity(original,original+1.)
        self.assertEqual(converted_parity(original,original)['label_disagreements'],0)

    def test_metrics_and_original_tolerance(self):
        y=np.arange(6);p=y.copy();p[-1]=0
        result=metrics(y,p);self.assertAlmostEqual(result['accuracy'],5/6)
        self.assertAlmostEqual(result['macro_f1'],(2/3+4)/6)
        source_parity(result,dict(accuracy=result['accuracy'],macro_f1=result['macro_f1']))
        with self.assertRaises(RuntimeError):source_parity(result,dict(macro_f1=result['macro_f1']+.001))

    def test_linear_choice_uses_only_original_validation(self):
        source={'methods':{'svm':{'tuning_results':[
            dict(params=dict(kernel='rbf',C=100),validation_macro_f1=.99),
            dict(params=dict(kernel='linear',C=.01),validation_macro_f1=.8),
            dict(params=dict(kernel='linear',C=.1),validation_macro_f1=.7)]}}}
        self.assertEqual(linear_choice(source)['params']['C'],.01)

    def test_timing_units_and_deterministic_corpus(self):
        a=timing_plan(400);b=timing_plan(400)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(len(np.unique(a['sample1'])),300)
        self.assertFalse(set(a['warm1'])&set(a['sample1']))
        report=timing_summary([1e6,2e6,3e6],128)
        self.assertEqual(report['median_ms'],2);self.assertEqual(report['samples_per_second'],64000)
        with self.assertRaises(RuntimeError):timing_summary([0,1],1)

    def test_gate_rejects_corruption_and_changed_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'prediction.txt';p.write_text('frozen')
            context=dict(environment_sha256=digest(dict(cpu='fixture')))
            atomic_json(root/'gate.json',dict(status='complete',context=context,files=records(root,[p])))
            require_gate(root,'gate.json',context)
            with self.assertRaises(RuntimeError):require_gate(root,'gate.json',dict(environment_sha256='other'))
            p.write_text('changed')
            with self.assertRaises(RuntimeError):require_gate(root,'gate.json',context)

if __name__=='__main__':unittest.main()
