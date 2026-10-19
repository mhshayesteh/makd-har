"""Check interpretation-sensitive arithmetic without fitting or inference."""
import unittest
import numpy as np
from analyze_wisdm_gap import cm_metrics, independent_split, standardize_pair


class GapAuditGuards(unittest.TestCase):
    def test_only_class_prior_changes(self):
        conditional = np.eye(6) * np.arange(1, 7)[:, None] / 10
        conditional[:, -1] += 1 - conditional.sum(1)
        validation = conditional * np.array([10, 20, 30, 40, 50, 60])[:, None]
        test = conditional * np.array([60, 50, 40, 30, 20, 10])[:, None]
        result = standardize_pair(validation, test)
        self.assertAlmostEqual(result['equal_prior_macro_f1_gap_pp'], 0)
        self.assertAlmostEqual(result['within_class_recall_term_pp'], 0)
        self.assertAlmostEqual(result['class_composition_term_pp'], result['raw_accuracy_gap_pp'])

    def test_only_conditional_error_changes(self):
        validation = np.eye(6) * 10
        test = np.eye(6) * 8 + np.roll(np.eye(6), 1, axis=1) * 2
        result = standardize_pair(validation, test)
        self.assertAlmostEqual(result['class_composition_term_pp'], 0)
        self.assertAlmostEqual(result['within_class_recall_term_pp'], 20)
        self.assertAlmostEqual(result['raw_accuracy_gap_pp'], 20)

    def test_participant_observed_class_semantics(self):
        cm = np.diag([10, 20, 0, 0, 0, 0])
        result = cm_metrics(cm)
        self.assertEqual(result['classes_present'], 2)
        self.assertAlmostEqual(result['macro_f1'], 2 / 6)
        self.assertEqual(result['observed_macro_f1'], 1)
        self.assertEqual(result['observed_balanced_accuracy'], 1)

    def test_subject_split_disjoint_and_complete(self):
        subjects = np.repeat(np.arange(1, 37), 6)
        labels = np.tile(np.arange(6), 36)
        result = independent_split(labels, subjects, dict(wisdm_test_fraction=.2, validation_fraction=.2, split_seed=42))
        people = {key: set(subjects[value]) for key, value in result.items()}
        self.assertEqual({key: len(value) for key, value in people.items()}, dict(train=22, validation=7, test=7))
        for key, indices in result.items():
            self.assertEqual(set(labels[indices]), set(range(6)))
            for other in people:
                if key != other:
                    self.assertFalse(people[key] & people[other])
        self.assertEqual(sorted(np.concatenate(list(result.values()))), list(range(len(labels))))


if __name__ == '__main__':
    unittest.main()
