"""Synthetic protocol tests; these are not clinical performance measurements."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fusion_protocol import (KEYS, TARGET, calibrated_scores, fit_calibrators, fit_fusion,
    fit_metadata, make_protocol, pair_predictions, paired_patient_bootstrap, select_threshold,
    validate_partition)
from run_support import fit_late_stage, evaluate_predictions, initialise_run


def synthetic_manifest():
    rng = np.random.default_rng(8)
    n = 800
    patient = np.repeat(np.arange(n),2)
    label = (patient%5 == 0).astype(int)
    data = pd.DataFrame({'patient_id':[f'synthetic_{i}' for i in patient],
        'cxr_filename':[f'synthetic_{i}.jpg' for i in range(2*n)],
        'age':rng.integers(18,91,len(patient)), 'sex':np.where(patient%2,'F','M'),
        'slvh':label,'dlv':np.zeros(len(patient),dtype=int),TARGET:label,
        'heart_transplant':(patient%17==0).astype(int),'lung_transplant':0,'pacemaker_or_icd':0})
    return make_protocol(data)


def synthetic_predictions(manifest):
    rng = np.random.default_rng(19)
    frame = manifest[[*KEYS,'role',TARGET]].rename(columns={TARGET:'target'}).copy()
    frame['ml_probability'] = np.clip(.2+.3*frame.target+rng.normal(0,.18,len(frame)),.01,.99)
    frame['dl_probability'] = np.clip(.2+.4*frame.target+rng.normal(0,.18,len(frame)),.01,.99)
    return frame


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = synthetic_manifest()
        cls.predictions = synthetic_predictions(cls.manifest)

    def test_roles_are_disjoint_and_deterministic(self):
        frame = self.manifest
        validate_partition(frame)
        self.assertEqual(frame.groupby('patient_id').role.nunique().max(),1)
        self.assertTrue(frame.loc[frame.split.eq('test'),'role'].eq('historical_test').all())
        self.assertTrue(frame.loc[frame.split.eq('train'),'role'].eq('train').all())
        pd.testing.assert_series_equal(frame.role,make_protocol(frame).role)

    def test_preserves_supplied_old_split(self):
        changed = self.manifest.copy()
        # A supplied old manifest is authoritative, not reconstructed silently.
        ids = changed.patient_id.unique()
        a,b = ids[0],ids[1]
        old_a,old_b = changed.loc[changed.patient_id.eq(a),'split'].iloc[0], changed.loc[changed.patient_id.eq(b),'split'].iloc[0]
        changed.loc[changed.patient_id.eq(a),'split']=old_b
        changed.loc[changed.patient_id.eq(b),'split']=old_a
        result=make_protocol(changed)
        pd.testing.assert_series_equal(changed.split,result.split)

    def test_preparation_rejects_label_leak_and_paths(self):
        broken = self.manifest.copy()
        broken.loc[0,TARGET] = 1-int(broken.loc[0,TARGET])
        with self.assertRaises(ValueError): make_protocol(broken)
        broken=self.manifest.copy();broken.loc[0,'cxr_filename']='../patient.jpg'
        with self.assertRaises(ValueError): make_protocol(broken)

    def test_metadata_never_fits_on_later_roles(self):
        first = fit_metadata(self.manifest)
        changed=self.manifest.copy()
        mask=changed.role.isin(['fusion_fit','calibration','threshold','historical_test'])
        changed.loc[mask,TARGET] = 1-changed.loc[mask,TARGET]
        second=fit_metadata(changed)
        self.assertEqual(first['name'],second['name'])
        np.testing.assert_allclose(first['model'].predict_proba(self.manifest[first['features']]),
                                   second['model'].predict_proba(self.manifest[second['features']]))
        self.assertEqual(first['features'],['age_scaled','sex_binary'])

    def test_pairing_uses_keys_and_rejects_disagreement(self):
        frame=self.predictions.iloc[:20]
        dl=frame[[*KEYS,'target','dl_probability']]
        ml=frame[[*KEYS,'target','ml_probability']].iloc[::-1]
        result=pair_predictions(dl,ml)
        np.testing.assert_allclose(result.ml_probability,frame.ml_probability)
        broken=ml.copy();broken.loc[broken.index[0],'target']=1-broken.iloc[0].target
        with self.assertRaises(ValueError):pair_predictions(dl,broken)
        with self.assertRaises(ValueError):pair_predictions(dl,ml.iloc[:-1])

    def test_fusion_does_not_fit_on_calibration_or_threshold(self):
        frame=self.predictions.copy()
        first=fit_fusion(frame)
        changed=frame.copy()
        mask=changed.role.isin(['calibration','threshold','historical_test'])
        changed.loc[mask,'target']=1-changed.loc[mask,'target']
        second=fit_fusion(changed)
        np.testing.assert_allclose(first.predict_proba(frame[['ml_probability','dl_probability']]),
                                   second.predict_proba(frame[['ml_probability','dl_probability']]))

    def test_calibration_does_not_fit_on_threshold(self):
        frame=self.predictions.copy();fusion=fit_fusion(frame)
        first=fit_calibrators(frame,fusion)
        changed=frame.copy();mask=changed.role.eq('threshold')
        changed.loc[mask,'target']=1-changed.loc[mask,'target']
        second=fit_calibrators(changed,fusion)
        for name,p in calibrated_scores(frame,fusion,first).items():
            np.testing.assert_allclose(p,calibrated_scores(frame,fusion,second)[name])

    def test_threshold_obeys_specificity_even_for_ties(self):
        y=np.array([0,0,0,0,1,1]);p=np.array([.2,.8,.8,.8,.8,.9])
        decision=select_threshold(y,p,minimum_specificity=.8)
        self.assertGreaterEqual(decision['selection_specificity'],.8)
        self.assertEqual(decision['threshold'],.9)

    def test_bootstrap_is_paired_and_reproducible(self):
        frame=self.predictions[self.predictions.role.eq('historical_test')].reset_index(drop=True)
        p=frame.dl_probability.to_numpy()
        args={'frame':frame,'scores':{'DL':p,'Fusion':p},
              'thresholds':{'DL':{'threshold':.5},'Fusion':{'threshold':.5}},'repeats':20}
        first=paired_patient_bootstrap(**args)
        self.assertEqual(first,paired_patient_bootstrap(**args))
        for interval in first['fusion_minus_dl_ci95'].values():self.assertEqual(interval,[0.,0.])

    def test_synthetic_end_to_end_persistence_and_explicit_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            output=initialise_run(Path(temp)/'run',{'synthetic_test_only':True})
            self.manifest.to_csv(output/'private/protocol_manifest.csv',index=False)
            development=self.predictions[~self.predictions.role.isin(['train','historical_test'])]
            bundle=fit_late_stage(development,output)
            test=self.predictions[self.predictions.role.eq('historical_test')]
            rows=evaluate_predictions(test,bundle,output,label='historical_comparison',repeats=20)
            self.assertEqual(set(rows.model),{'ML','DL','Fusion'})
            receipt=json.loads((output/'public/historical_comparison_receipt.json').read_text())
            self.assertTrue(receipt['historical_test_previously_inspected'])
            self.assertEqual(receipt['images'],len(test))
            self.assertTrue((output/'models/fusion_bundle.joblib').exists())
            with self.assertRaises(FileExistsError):
                evaluate_predictions(test,bundle,output,label='historical_comparison',repeats=20)
            with self.assertRaises(ValueError):fit_late_stage(self.predictions,output)

    def test_evaluation_refuses_development_and_partial_historical_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            output=initialise_run(Path(temp)/'run',{'synthetic_test_only':True})
            self.manifest.to_csv(output/'private/protocol_manifest.csv',index=False)
            development=self.predictions[~self.predictions.role.isin(['train','historical_test'])]
            bundle=fit_late_stage(development,output)
            with self.assertRaises(ValueError):
                evaluate_predictions(development,bundle,output,label='historical_comparison',repeats=20)
            historical=self.predictions[self.predictions.role.eq('historical_test')]
            with self.assertRaises(ValueError):
                evaluate_predictions(historical.iloc[:-1],bundle,output,label='historical_comparison',repeats=20)
            self.assertFalse((output/'public/historical_comparison_metrics.csv').exists())


if __name__ == '__main__':unittest.main()
