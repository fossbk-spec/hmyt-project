"""Run the V2 metadata stage only. Never score the historical test set."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from fusion_protocol import fit_metadata, make_protocol, sha256, split_summary, write_json
from run_support import initialise_run, artifact_receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--metadata',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--feature-set',default='age_sex',choices=['age_sex','full_retrospective'])
    args=parser.parse_args()
    frame=make_protocol(pd.read_csv(args.metadata,dtype={'patient_id':str,'cxr_filename':str}))
    run=initialise_run(args.output,{'scope':'real-data metadata development only; no image training or test scoring',
        'source_manifest_sha256':sha256(args.metadata),'feature_set':args.feature_set,'seed':42})
    frame.to_csv(run/'private/protocol_manifest.csv',index=False,lineterminator='\n')
    result=fit_metadata(frame,feature_set=args.feature_set)
    joblib.dump(result,run/'models/metadata_bundle.joblib')
    pd.DataFrame(result['comparisons']).to_csv(run/'public/metadata_branch_selection.csv',index=False)
    write_json(run/'public/split_summary.json',split_summary(frame))
    write_json(run/'public/local_stage_receipt.json',{'scope':'metadata only; branch_val selection metrics are not independent test performance',
        'selected_model':result['name'],'features':result['features'],
        'model_sha256':sha256(run/'models/metadata_bundle.joblib'),
        'manifest_sha256':sha256(run/'private/protocol_manifest.csv'),
        'source_sha256':{'fusion_protocol.py':sha256(Path(__file__).with_name('fusion_protocol.py'))},
        'gpu_training_done':False,'fusion_run_done':False,'historical_test_scored':False})
    artifact_receipt(run)
    print(pd.DataFrame(result['comparisons']).to_string(index=False))
    print('Saved real-data metadata stage; no DL/fusion/test scores were computed.')


if __name__=='__main__':main()
