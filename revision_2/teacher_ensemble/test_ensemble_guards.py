import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from run_ensemble import (predictions,probabilities,metrics,source_weights,parity,validate_preflight,
                          digest,sha,TOLERANCE)


class Guards(unittest.TestCase):
    def test_probability_aggregation_and_temperature(self):
        # Mean probabilities must differ from mean logits for this deliberately unbalanced example.
        logits=np.full((5,1,6),-100.,dtype=float);logits[0,0,:2]=[100,0];logits[1:,0,:2]=[0,2]
        names=list('abcde');result=predictions(logits,names,np.ones(5)/5)
        self.assertEqual(int(result['uniform-T1'][0]),1)
        self.assertEqual(int(logits.mean(axis=0).argmax(axis=1)[0]),0)
        self.assertEqual(int(result['uniform-T4'][0]),0)
        weighted=predictions(logits,names,np.array([.8,.05,.05,.05,.05]))
        self.assertEqual(int(weighted['shapley-T1'][0]),0)
        self.assertEqual(len(result),9)

    def test_stability_and_invalid_inputs(self):
        logits=np.ones((5,2,6))*1e6
        np.testing.assert_allclose(probabilities(logits,1.),1/6)
        pred=predictions(logits,list('abcde'),np.ones(5)/5)
        self.assertTrue(all((values==0).all() for values in pred.values()))
        with self.assertRaises(RuntimeError):probabilities(logits,2.)
        with self.assertRaises(RuntimeError):predictions(logits,list('abcde'),np.array([1,1,1,1,1]))

    def test_original_game_and_weights(self):
        phi=np.array([.001,.002,.003,.004,.005]);z=(phi-phi.mean())/phi.std();w=np.exp(z-z.max());w/=w.sum()
        game={','.join(map(str,c)):.5+sum(phi[list(c)]) for k in range(6) for c in itertools.combinations(range(5),k)}
        source=dict(teacher_order=list('abcde'),shapley=dict(values=phi.tolist(),coalition_values=game),shapley_weights=w.tolist())
        np.testing.assert_allclose(source_weights(source),w,atol=1e-12)
        bad=copy.deepcopy(source);bad['shapley_weights']=[.2]*5
        with self.assertRaises(AssertionError):source_weights(bad)

    def test_metrics_and_parity(self):
        y=np.arange(6);m=metrics(y,y)
        self.assertEqual(m['macro_f1'],1.);self.assertEqual(m['confusion_matrix'],np.eye(6,dtype=int).tolist())
        p=parity(m,m);self.assertEqual(p['tolerance'],1e-4)
        bad=dict(m,accuracy=.99)
        with self.assertRaises(RuntimeError):parity(bad,m)
        with self.assertRaises(RuntimeError):metrics(y,np.array([0,1,2,3,4,6]))

    def test_preflight_identity_and_corruption(self):
        context=dict(source='fixed');names=list('abcde');y=np.arange(6);subjects=np.ones(6,dtype=int)
        logits=np.stack([np.eye(6)*10]*5);m=metrics(y,y)
        source=dict(teacher_order=names,teachers={n:dict(validation=m) for n in names})
        arrays=dict(y_val=y,subjects_val=subjects)
        with tempfile.TemporaryDirectory() as tmp:
            npz=Path(tmp)/'validation-logits.npz';path=Path(tmp)/'preflight.json'
            np.savez_compressed(npz,val_logits=logits,y_val=y,subjects_val=subjects)
            r=dict(context=context,identity=digest(context),passed=True,logits_sha256=sha(npz),
                teacher_validation_parity={n:parity(m,m) for n in names})
            path.write_text(json.dumps(r));validate_preflight(path,npz,context,source,arrays)
            with self.assertRaises(RuntimeError):validate_preflight(path,npz,dict(source='changed'),source,arrays)
            npz.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):validate_preflight(path,npz,context,source,arrays)


if __name__=='__main__':unittest.main()
