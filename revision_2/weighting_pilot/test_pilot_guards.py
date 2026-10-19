"""Scientific safeguards: leakage, normalization, candidate scale and immutable resume identities."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pilot_data import validate_groups,load_roles,part_manifest
from run_pilot import weights_for,select_candidate,CANDIDATES,history_check,checked,digest


class Guards(unittest.TestCase):
    def test_scale_and_uniform(self):
        values=np.array([-.002,-.001,0.,.001,.003]);a=weights_for(values);b=weights_for(100*values)
        np.testing.assert_allclose(a['standardized'],b['standardized'],rtol=0,atol=1e-15)
        self.assertGreater(np.linalg.norm(np.array(a['raw-tau-0p001'])-b['raw-tau-0p001']),.1)
        for name,w in weights_for([0]*5).items():np.testing.assert_allclose(w,[.2]*5)
        with self.assertRaises(RuntimeError):weights_for([float('nan')]*5)

    def test_selection_complete_and_ties(self):
        row=dict.fromkeys(CANDIDATES,.8)
        self.assertEqual(select_candidate([row]*3)['selected_candidate'],'uniform')
        changed=dict(row);changed['raw-tau-0p01']=.81
        self.assertEqual(select_candidate([changed]*3)['selected_candidate'],'raw-tau-0p01')
        with self.assertRaises(RuntimeError):select_candidate([changed]*2)

    def test_split_leakage(self):
        here=Path(__file__).parent;split=json.loads((here/'uci-split.json').read_text())
        # Full metadata from a source snapshot; mutation must be rejected before any training.
        bad=copy.deepcopy(split);bad['groups']['value'].append(bad['groups']['fit'][0])
        labels=[];subjects=[]
        for role,report in split['coverage'].items():
            for subject in report['subjects']:
                for label in range(6):labels.append(label);subjects.append(subject)
        with self.assertRaisesRegex(RuntimeError,'leakage'):validate_groups(np.array(labels),np.array(subjects),bad)

    def test_student_history_and_resume_identity(self):
        task=dict(kind='candidate',epochs=60,seed=11)
        report=dict(history={'train_loss':[1.]*11,'val_macro_f1':[.7]*11},epochs_completed=11,best_epoch=1,
            stop_metrics=dict(macro_f1=.7,accuracy=.7,macro_precision=.7,macro_recall=.7,per_class_f1=[.7]*6))
        history_check(report,task)
        bad=copy.deepcopy(report);bad['epochs_completed']=10
        with self.assertRaises(RuntimeError):history_check(bad,task)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fit.json';payload=dict(status='complete',identity=digest(task),task=task)
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError,'Unsafe cached fit'):checked(path,dict(task,seed=12))

    def test_teacher_checkpoint_history(self):
        task=dict(kind='teacher',epochs=60)
        report=dict(history={'loss':[1.]*12,'val_loss':[.9,.8]+[.8]*10},epochs_completed=12,best_epoch=2)
        history_check(report,task)
        report['best_epoch']=1
        with self.assertRaises(RuntimeError):history_check(report,task)


if __name__=='__main__':unittest.main()
