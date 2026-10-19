"""Meaningful safeguards for predeclared removal, cached results and checkpoint selection."""
import json
import tempfile
import unittest
from pathlib import Path
from run_teacher_pruning import removal_choice,retained_indices,paired_summary,checked_fit,history_check,digest,sha,validate_parity


def valid_report():
    metrics=dict(accuracy=.8,macro_f1=.8,macro_precision=.8,macro_recall=.8,per_class_f1=[.8]*6)
    return dict(history=dict(train_loss=[1.,.9],val_macro_f1=[.7,.8]),epochs_completed=2,best_epoch=2,
                validation_metrics=metrics,test_metrics=metrics)


class PruningGuards(unittest.TestCase):
    def test_raw_minimum_and_prespecified_tie_order(self):
        source=dict(teacher_order=['a','b','c','d','e'],shapley=dict(values=[.2,-.1,.3,-.1,.0]))
        self.assertEqual(removal_choice(source),1)
        source['shapley']['values']=[.2,.1,.3,.4,.5]
        self.assertEqual(removal_choice(source),1)
        source['shapley']['values'][0]=float('nan')
        with self.assertRaises(RuntimeError):removal_choice(source)

    def test_removal_never_retains_removed_teacher(self):
        self.assertEqual(retained_indices(2),[0,1,3,4])
        self.assertEqual(retained_indices(None),[0,1,2,3,4])
        with self.assertRaises(RuntimeError):retained_indices(5)

    def test_random_policy_averages_all_five_including_chosen(self):
        reports={f'remove-{i}':dict(validation_metrics=dict(macro_f1=v),test_metrics=dict(macro_f1=v))
                 for i,v in enumerate([.6,.7,.8,.9,1.])}
        reports['full']=dict(validation_metrics=dict(macro_f1=.85),test_metrics=dict(macro_f1=.85))
        result=paired_summary(reports,4)['test']
        self.assertAlmostEqual(result['expected_random_removal'],.8)
        self.assertAlmostEqual(result['selected_minus_random'],.2)
        self.assertAlmostEqual(result['selected_minus_full'],.15)
        del reports['remove-0']
        with self.assertRaises(RuntimeError):paired_summary(reports,4)

    def test_incomplete_history_and_wrong_checkpoint_are_rejected(self):
        report=valid_report()
        history_check(report,2)
        with self.assertRaises(RuntimeError):history_check(report,60)
        report['best_epoch']=1
        with self.assertRaises(RuntimeError):history_check(report,2)

    def test_changed_seed_or_model_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);path=folder/'full.json'
            task=dict(training_seed=11000007,epochs=2,context=dict(config=dict(patience=10)))
            model=folder/'full.keras';model.write_bytes(b'fixed-model')
            predictions=folder/'full.npz';predictions.write_bytes(b'fixed-predictions')
            report=valid_report()
            report.update(status='complete',task=task,identity=digest(task),test_evaluated=True,
                model_file=model.name,model_sha256=sha(model),predictions_file=predictions.name,predictions_sha256=sha(predictions))
            path.write_text(json.dumps(report))
            self.assertIsNotNone(checked_fit(path,task))
            with self.assertRaises(RuntimeError):checked_fit(path,dict(task,training_seed=11000008))
            model.write_bytes(b'changed-model')
            with self.assertRaises(RuntimeError):checked_fit(path,task)

    def test_teacher_parity_cannot_relax_tolerance(self):
        source=dict(teacher_order=['a'],teachers=dict(a=dict(validation=dict(accuracy=.9,macro_f1=.8))))
        parity=dict(a=dict(saved=source['teachers']['a']['validation'],reproduced=dict(accuracy=.9,macro_f1=.8),
                           absolute_metric_errors=dict(accuracy=0.,macro_f1=0.),tolerance=1e-4))
        validate_parity(parity,source)
        parity['a']['tolerance']=.01
        with self.assertRaises(RuntimeError):validate_parity(parity,source)


if __name__=='__main__':unittest.main()
