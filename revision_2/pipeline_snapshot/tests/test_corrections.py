import tempfile
import unittest
from pathlib import Path
import numpy as np
from makd_har.data import _iter_wisdm_segments
from makd_har.training import classification_metrics, classification_metrics_from_labels
from run_weighting_pilot import candidate_name, shrink_weights

class CorrectionsTest(unittest.TestCase):
    def parse(self,text):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'raw.txt';path.write_text(text)
            audit={};segments=list(_iter_wisdm_segments(path,audit))
        return segments,audit

    def test_record_syntax(self):
        seg,a=self.parse('21,Walking,100,1,2,3,;21,Walking,200,4,5,6;\n21,Jogging,300,1,2,3;')
        self.assertEqual([len(s[2]) for s in seg],[2,1])
        self.assertEqual(a['accepted_records'],3)
        self.assertEqual(a['trailing_comma_repaired'],1)
        np.testing.assert_array_equal(seg[0][2],[[1,2,3],[4,5,6]])

    def test_clock_policy(self):
        times=[100,100,0,200,1000000200,2000000201,10]
        seg,a=self.parse(''.join(f'1,Walking,{t},1,2,3;' for t in times))
        self.assertEqual([len(s[2]) for s in seg],[5,1,1])
        self.assertEqual(a['duplicate_clock_pairs_retained'],1)
        self.assertEqual(a['long_gap_boundaries'],1)
        self.assertEqual(a['backward_clock_boundaries'],1)

    def test_bad_sensor_breaks_and_subjects_separate(self):
        seg,a=self.parse('1,Walking,1,1,2,3;1,Walking,2,nan,2,3;1,Walking,3,4,5,6;2,Walking,4,1,2,3;')
        self.assertEqual([s[0] for s in seg],[1,1,2])
        self.assertEqual(a['invalid_records'],1)

    def test_labels_are_scored_directly(self):
        true=np.array([0,1,2,0,1,2]);pred=np.array([0,1,2,1,1,2])
        score=classification_metrics_from_labels(true,pred)
        self.assertAlmostEqual(score['accuracy'],5/6)
        self.assertEqual(score,classification_metrics(true,np.eye(3)[pred]))

    def test_weight_shrinkage_candidates(self):
        weights=np.array([.6,.1,.1,.1,.1])
        np.testing.assert_allclose(shrink_weights(weights,0),weights)
        np.testing.assert_allclose(shrink_weights(weights,.5),[.4,.15,.15,.15,.15])
        np.testing.assert_allclose(shrink_weights(weights,1),np.full(5,.2))
        self.assertEqual(candidate_name(60,.5),'shapley-60-shrink-50')

if __name__=='__main__':unittest.main()
