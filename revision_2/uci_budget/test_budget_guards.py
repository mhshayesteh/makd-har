"""Checks that protect the paired budget comparison and persisted results."""
import json
import tempfile
import unittest
from pathlib import Path
from run_uci_budget import prefix_utility,checked_result


class BudgetGuards(unittest.TestCase):
    def test_later_improvement_cannot_enter_twenty_epoch_utility(self):
        history=dict(train_loss=[1.]*24,val_macro_f1=[.7]*19+[.8,.9,.85,.91,.9])
        self.assertEqual(prefix_utility(history,20)['validation_macro_f1'],.8)
        self.assertEqual(prefix_utility(history,60)['validation_macro_f1'],.91)

    def test_early_stopped_trajectory_has_identical_prefixes(self):
        history=dict(train_loss=[1.]*12,val_macro_f1=[.8]+[.7]*11)
        self.assertEqual(prefix_utility(history,20),prefix_utility(history,60))

    def test_frozen_checkpoint_improvement_threshold(self):
        history=dict(train_loss=[1.]*3,val_macro_f1=[.8,.80000005,.8000002])
        self.assertEqual(prefix_utility(history,2)['best_epoch'],1)
        self.assertEqual(prefix_utility(history,3)['best_epoch'],3)

    def test_invalid_or_truncated_history_is_rejected(self):
        for history in [dict(train_loss=[],val_macro_f1=[]),dict(train_loss=[1],val_macro_f1=[float('nan')]),
                        dict(train_loss=[1,2],val_macro_f1=[.8]),dict(train_loss=[float('inf')],val_macro_f1=[.8])]:
            with self.assertRaises(RuntimeError):prefix_utility(history,20)

    def test_wrong_identity_or_changed_prefix_cannot_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fit.json'
            history=dict(train_loss=[1.],val_macro_f1=[.8])
            result=dict(status='complete',identity='correct',kind='coalition',test_evaluated=False,
                        history=history,budget_utilities={'20':prefix_utility(history,20)})
            path.write_text(json.dumps(result))
            self.assertIsNotNone(checked_result(path,'correct','coalition'))
            with self.assertRaises(RuntimeError):checked_result(path,'changed-budget-or-source','coalition')
            result['budget_utilities']['20']['validation_macro_f1']=.9
            path.write_text(json.dumps(result))
            with self.assertRaises(RuntimeError):checked_result(path,'correct','coalition')

    def test_coalition_test_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fit.json'
            path.write_text(json.dumps(dict(status='complete',identity='correct',kind='coalition',test_evaluated=True)))
            with self.assertRaises(RuntimeError):checked_result(path,'correct','coalition')


if __name__=='__main__':unittest.main()
