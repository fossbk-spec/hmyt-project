"""Archive existing legacy files without pretending absent evidence can be recovered."""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

EXPECTED = ['metadata_with_patient_split.csv', 'metadata_ml_model.joblib',
            'efficientnet_b0_xray.pt', 'fusion_logistic_regression.joblib',
            'test_predictions.csv', 'test_metrics.csv', 'selected_thresholds.json',
            'dl_training_history.csv']


def inventory_and_archive(project):
    project = Path(project)
    source = project/'outputs_ml_support_dl'
    if not source.is_dir():
        raise FileNotFoundError(f'Legacy output directory not found: {source}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = project/'legacy_evidence_archives'/stamp
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {'archived_utc': stamp, 'source': 'outputs_ml_support_dl',
               'present': {}, 'missing': [], 'historical_runtime_versions': None,
               'provenance': 'These are files found now; archival time is not training time',
               'independent_checkpoint_prediction_consistency_verified': False}
    for name in EXPECTED:
        candidate = source/name
        if not candidate.is_file():
            receipt['missing'].append(name)
            continue
        target = destination/name
        shutil.copyfile(candidate, target)
        digest = hashlib.sha256()
        with target.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                digest.update(block)
        receipt['present'][name] = {'bytes': target.stat().st_size, 'sha256': digest.hexdigest()}
    if 'test_predictions.csv' in receipt['present']:
        with (destination/'test_predictions.csv').open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            rows = sum(1 for _ in reader)
        receipt['prediction_inventory'] = {'columns': columns, 'rows': rows,
                                           'cohort_identity_verified': False}
    (destination/'recovery_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('Archived files:', len(receipt['present']))
    print('Missing files:', receipt['missing'])
    print('Private archive:', destination)
    print('Keep this archive private; do not upload patient-level predictions or models to the class repo.')
    return receipt
