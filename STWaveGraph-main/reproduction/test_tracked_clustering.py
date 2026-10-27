from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from tracked_clustering import cluster_with_record, ClusterExecutionError


class ClusteringRecordTests(unittest.TestCase):
    def setUp(self):
        self.x=np.arange(12,dtype=np.float32).reshape(4,3)

    def success(self,x,**kw):
        self.assertFalse(kw['fallback_to_gmm'])
        return np.array([0,0,1,1]),np.array([[1,0],[1,0],[0,1],[0,1]]),np.zeros((2,3))

    def test_success_records_actual_method(self):
        _,_,_,r=cluster_with_record(self.x,2,backend=self.success)
        self.assertEqual(r['actual_method'],'mclust')
        self.assertFalse(r['fallback_used'])
        self.assertEqual(r['status'],'complete')

    def test_failure_does_not_silently_run_gmm(self):
        calls=[]
        def backend(x,**kw):
            calls.append(kw['method'])
            raise RuntimeError('simulated mclust error')
        with self.assertRaises(ClusterExecutionError) as cm:
            cluster_with_record(self.x,2,backend=backend)
        self.assertEqual(calls,['mclust'])
        self.assertIsNone(cm.exception.record['actual_method'])
        self.assertEqual(cm.exception.record['status'],'failed')
        self.assertFalse(cm.exception.record['fallback_used'])

    def test_explicit_fallback_changes_variant(self):
        calls=[]
        def backend(x,**kw):
            calls.append(kw['method'])
            if kw['method']=='mclust':raise RuntimeError('mclust unavailable')
            return self.success(x,**kw)
        *_,r=cluster_with_record(self.x,2,allow_fallback=True,backend=backend)
        self.assertEqual(calls,['mclust','gmm'])
        self.assertEqual(r['requested_method'],'mclust')
        self.assertEqual(r['actual_method'],'gmm')
        self.assertIn('gmm_fallback',r['variant'])
        self.assertIn('unavailable',r['fallback_reason'])

    def test_failed_fallback_is_not_marked_complete(self):
        def backend(x,**kw):raise RuntimeError('both methods fail')
        with self.assertRaises(ClusterExecutionError) as cm:
            cluster_with_record(self.x,2,allow_fallback=True,backend=backend)
        self.assertEqual(len(cm.exception.record['attempts']),2)
        self.assertEqual(cm.exception.record['status'],'failed')
        self.assertIsNone(cm.exception.record['actual_method'])

    def test_invalid_probability_output_is_rejected(self):
        def backend(x,**kw):
            labels,p,c=self.success(x,**kw)
            p[0]=[0,0]
            return labels,p,c
        with self.assertRaises(ClusterExecutionError):
            cluster_with_record(self.x,2,backend=backend)

    def test_non_finite_input_is_rejected_before_backend(self):
        self.x[0,0]=np.nan
        with self.assertRaises(ValueError):cluster_with_record(self.x,2,backend=self.success)

    def test_invalid_center_shape_is_rejected(self):
        def backend(x,**kw):
            labels,p,c=self.success(x,**kw)
            return labels,p,np.zeros((1,3))
        with self.assertRaises(ClusterExecutionError):
            cluster_with_record(self.x,2,backend=backend)


if __name__=='__main__':unittest.main(verbosity=2)
