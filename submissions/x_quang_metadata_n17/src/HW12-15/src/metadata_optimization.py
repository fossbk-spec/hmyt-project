"""Week 12 development experiment; no historical test or image model evaluation.

Run from this directory or install src/fusion_protocol.py next to this script.
Feature groups and hyperparameters are selected by three patient-group folds
within the historical training partition only. branch_val remains development
data and its metrics must not be presented as independent final test results.
"""
from pathlib import Path
import argparse,json,sys,time,itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
if not (HERE/'fusion_protocol.py').exists():
    sys.path.insert(0,str(HERE.parents[1]/'CheXchoNet_v2/src'))
from fusion_protocol import make_protocol,sha256,environment_receipt,write_json,TARGET

def model(kind,param):
    if kind=='LR':
        return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
             LogisticRegression(C=param,class_weight='balanced',max_iter=2000,random_state=42))
    return make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(
        learning_rate=param[0],max_leaf_nodes=param[1],max_iter=120,
        early_stopping=False,l2_regularization=1,random_state=42))

def fit(m,kind,x,y):
    kw={'histgradientboostingclassifier__sample_weight':compute_sample_weight('balanced',y)} if kind=='HGB' else {}
    return m.fit(x,y,**kw)

def main():
    a=argparse.ArgumentParser();a.add_argument('--metadata',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    private=args.output/'private';private.mkdir();public=args.output/'public';public.mkdir()
    frame=make_protocol(pd.read_csv(args.metadata,dtype={'patient_id':str,'cxr_filename':str}))
    tr=frame[frame.role.eq('train')].reset_index(drop=True);val=frame[frame.role.eq('branch_val')]
    cv=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(tr,tr[TARGET],tr.patient_id))
    for x,y in cv:assert not set(tr.iloc[x].patient_id)&set(tr.iloc[y].patient_id)
    feature_sets={'age_sex':['age_scaled','sex_binary'],'age_only':['age_scaled'],'sex_only':['sex_binary']}
    candidates=[('LR',c,f) for f in feature_sets for c in [.1,1.,10.]]
    candidates += [('HGB',p,'age_sex') for p in [(.05,7),(.05,15),(.1,15)]]
    rows=[];start=time.monotonic()
    for kind,param,f in candidates:
        scores=[];features=feature_sets[f]
        for fold,(ii,jj) in enumerate(cv,1):
            m=fit(model(kind,param),kind,tr.loc[ii,features],tr.loc[ii,TARGET])
            p=m.predict_proba(tr.loc[jj,features])[:,1]
            scores.append(float(average_precision_score(tr.loc[jj,TARGET],p)))
        row={'model':kind,'parameter':json.dumps(param),'feature_set':f,'features':','.join(features),
            'cv_mean_ap':float(np.mean(scores)),'cv_std_ap':float(np.std(scores))}
        row.update({f'fold_{i+1}_ap':s for i,s in enumerate(scores)})
        rows.append(row);print(json.dumps(row),flush=True)
    ranked=sorted(rows,key=lambda r:r['cv_mean_ap'],reverse=True);best=ranked[0]
    kind=best['model'];param=json.loads(best['parameter']);features=feature_sets[best['feature_set']]
    fitted=fit(model(kind,param),kind,tr[features],tr[TARGET]);p=fitted.predict_proba(val[features])[:,1]
    joblib.dump({'model':fitted,'features':features,'selection':best},private/'optimized_metadata.joblib')
    # Descriptive permutation importance on branch_val, not an independent test.
    importance=permutation_importance(fitted,val[features],val[TARGET],scoring='average_precision',n_repeats=10,random_state=42,n_jobs=1)
    pd.DataFrame({'feature':features,'ap_decrease_mean':importance.importances_mean,
                  'ap_decrease_std':importance.importances_std}).to_csv(public/'permutation_importance.csv',index=False)
    pd.DataFrame(rows).to_csv(public/'group_cv_search.csv',index=False)
    summary={'measured_on':'2026-11-22','scope':'metadata development only; not V2 DL/fusion/test',
       'selection':'3-fold StratifiedGroupKFold in historical train; candidate feature sets prespecified',
       'search_candidates':len(rows),'training_images':len(tr),'training_patients':int(tr.patient_id.nunique()),
       'branch_val_images':len(val),'branch_val_patients':int(val.patient_id.nunique()),
       'selected':best,'branch_val_ap':float(average_precision_score(val[TARGET],p)),
       'branch_val_roc_auc':float(roc_auc_score(val[TARGET],p)),
       'branch_val_raw_brier':float(brier_score_loss(val[TARGET],p)),
       'elapsed_seconds':time.monotonic()-start,'historical_test_scored':False,
       'permutation_scope':'descriptive AP decrease on branch_val; correlated predictors and repeated images limit interpretation',
       'model_integrated_into_v2_fusion':False,'metadata_sha256':sha256(args.metadata),
       'source_sha256':sha256(Path(__file__)),'environment':environment_receipt()}
    write_json(public/'optimization_receipt.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
