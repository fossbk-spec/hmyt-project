"""Development-only calibration audit of the Week 12 selected metadata model."""
from pathlib import Path
import sys,json,argparse
import numpy as np,pandas as pd,joblib
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
if not (HERE/'fusion_protocol.py').exists():sys.path.insert(0,str(ROOT/'CheXchoNet_v2/src'))
from fusion_protocol import make_protocol,TARGET,metrics,select_threshold,calibration_diagnostics,paired_patient_bootstrap,write_json,sha256
parser=argparse.ArgumentParser();parser.add_argument('--metadata',required=True,type=Path);parser.add_argument('--run-dir',required=True,type=Path);args=parser.parse_args()
RUN=args.run_dir;PUB=RUN/'public'
src=args.metadata
frame=make_protocol(pd.read_csv(src,dtype={'patient_id':str,'cxr_filename':str}))
bundle=joblib.load(RUN/'private/optimized_metadata.joblib');m=bundle['model'];features=bundle['features']
cal=frame[frame.role.eq('calibration')];th=frame[frame.role.eq('threshold')];val=frame[frame.role.eq('branch_val')]
def raw(x):return m.predict_proba(x[features])[:,1]
def logits(p):return logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1)
c=LogisticRegression(C=1e6,max_iter=2000,random_state=42).fit(logits(raw(cal)),cal[TARGET])
def calibrated(x):return c.predict_proba(logits(raw(x)))[:,1]
threshold=select_threshold(th[TARGET],calibrated(th),minimum_specificity=.8)
p=calibrated(val);original=raw(val)
r=metrics(val[TARGET],p,threshold['threshold']);r['accuracy']=(r['tn']+r['tp'])/r['images']
r['calibration_diagnostics']=calibration_diagnostics(val[TARGET],p)
r['raw_brier']=float(np.mean((original-val[TARGET])**2))
evaluation=val[['patient_id']].copy();evaluation['target']=val[TARGET].to_numpy()
ci=paired_patient_bootstrap(evaluation,{'Metadata':p},{'Metadata':threshold},repeats=500,seed=42)
y,x=calibration_curve(val[TARGET],p,n_bins=5,strategy='quantile')
pd.DataFrame({'mean_prediction':x,'observed_fraction':y}).to_csv(PUB/'metadata_calibration_curve.csv',index=False)
write_json(PUB/'metadata_calibration_audit.json',{'scope':'development branch_val; no independent test and no DL/fusion evaluation',
 'calibrator_fitting_role':'calibration','threshold_fitting_role':'threshold','threshold':threshold,
 'metrics':r,'patient_bootstrap':ci,'limitations':['branch_val previously viewed; interpretation remains developmental',
 'Brier change does not alone prove calibrated probabilities','500 conditional patient bootstrap resamples; training uncertainty not included',
 'Calibration slope/intercept have no CI in this audit; raw predictions and weights remain private'],
 'source_sha256':sha256(__file__),'model_sha256':sha256(RUN/'private/optimized_metadata.joblib')})
joblib.dump({'model_bundle':bundle,'calibrator':c,'threshold':threshold},RUN/'private/calibrated_metadata_bundle.joblib')
print(json.dumps(r,indent=2))
