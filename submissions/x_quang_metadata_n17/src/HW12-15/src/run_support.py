"""Drive-friendly persistence and explicit evaluation helpers."""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from fusion_protocol import (FEATURES, KEYS, TARGET, calibrated_scores, calibration_diagnostics,
    environment_receipt, fit_calibrators, fit_fusion, metrics, pair_predictions,
    paired_patient_bootstrap, raw_scores, select_threshold, sha256, write_json)


def dataset_root(project, cache):
    project, cache = Path(project), Path(cache)
    def discover(parent):
        if not parent.exists():
            return None
        for metadata in parent.rglob('metadata.csv'):
            if (metadata.parent/'images').is_dir():
                return metadata.parent
        return None
    existing = discover(cache) or discover(project)
    if existing:
        return existing
    candidates = sorted(project.glob('*.zip'))
    for path in candidates:
        with zipfile.ZipFile(path) as archive:
            if not any(Path(name).name == 'metadata.csv' for name in archive.namelist()):
                continue
            cache.mkdir(parents=True, exist_ok=True)
            for info in archive.infolist():
                dest = (cache/info.filename).resolve()
                if not dest.is_relative_to(cache.resolve()) or '\\' in info.filename:
                    raise ValueError('Unsafe archive path')
                if info.external_attr >> 16 & 0o170000 == 0o120000:
                    raise ValueError('Symlinks are not allowed in dataset archive')
            archive.extractall(cache)
        found = discover(cache)
        if found:
            return found
    raise FileNotFoundError('No dataset with metadata.csv + images/ found; place the official ZIP in the project folder')


def initialise_run(output, config, *, resume=False):
    output = Path(output)
    config_path = output/'run_config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if not resume or previous != config:
            raise ValueError('Run exists or config changed. Use matching RESUME=True or a new RUN_ID')
    else:
        if resume:
            raise FileNotFoundError('Cannot resume a run that has no receipt')
        output.mkdir(parents=True, exist_ok=False)
        write_json(config_path, config)
    for name in ['private', 'models', 'public']:
        (output/name).mkdir(exist_ok=True)
    environment = environment_receipt()
    prior = output/'environment.json'
    if resume and prior.exists() and json.loads(prior.read_text()) != environment:
        raise ValueError('Runtime environment changed; start a new run for reproducible resume')
    write_json(prior, environment)
    pinned = [f'{name}=={version}' for name,version in environment['packages'].items() if version]
    (output/'requirements-runtime.txt').write_text('\n'.join(pinned)+'\n')
    return output


def artifact_receipt(output):
    output = Path(output)
    files = {str(path.relative_to(output)).replace('\\','/'): sha256(path)
             for path in output.rglob('*') if path.is_file() and path.name != 'artifact_receipt.json' and not path.name.endswith('.tmp')}
    write_json(output/'artifact_receipt.json', {'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'artifacts_sha256': files, 'private_files': 'Contains protected record keys, predictions and model artifacts. Do not upload entire run to public GitHub.'})


def fit_late_stage(development_predictions, output, *, seed=42, minimum_specificity=.80):
    frame = development_predictions
    if not set(frame.role).issubset({'fusion_fit','calibration','threshold','branch_val'}):
        raise ValueError('Evaluation rows cannot be used to fit late stages')
    if frame.groupby('patient_id').role.nunique().max() != 1:
        raise ValueError('Patient overlap in late-stage inputs')
    fusion = fit_fusion(frame, seed=seed)
    calibrators = fit_calibrators(frame, fusion, seed=seed)
    tuning = frame[frame.role.eq('threshold')]
    probabilities = calibrated_scores(tuning, fusion, calibrators)
    thresholds = {name: select_threshold(tuning.target, scores, minimum_specificity=minimum_specificity)
                  for name,scores in probabilities.items()}
    bundle = {'fusion': fusion, 'calibrators': calibrators, 'thresholds': thresholds}
    joblib.dump(bundle, Path(output)/'models/fusion_bundle.joblib')
    write_json(Path(output)/'public/selected_thresholds.json', thresholds)
    write_json(Path(output)/'public/stage_roles.json', {
        'branch_training': 'train', 'branch_selection': 'branch_val', 'fusion_fitting': 'fusion_fit',
        'sigmoid_calibration': 'calibration', 'threshold_choice': 'threshold',
        'evaluation': 'No evaluation data used in any fit; historical test is not an untouched external test',
        'natural_prevalence': 'No resampling or balanced weights for fusion/calibration',
        'target_specificity': minimum_specificity})
    artifact_receipt(output)
    return bundle


def evaluate_predictions(frame, bundle, output, *, label, repeats=200, seed=42):
    if label not in {'historical_comparison', 'external_evaluation'}:
        raise ValueError('Explicit evaluation label required')
    output = Path(output)
    manifest = pd.read_csv(output/'private/protocol_manifest.csv',dtype={'patient_id':str,'cxr_filename':str})
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any() or not frame.target.isin([0,1]).all():
        raise ValueError('Invalid evaluation keys or targets')
    if label == 'historical_comparison':
        if 'role' not in frame or not frame.role.eq('historical_test').all():
            raise ValueError('Historical evaluation accepts historical_test rows only')
        expected = manifest[manifest.role.eq('historical_test')][[*KEYS,TARGET]].rename(columns={TARGET:'expected_target'})
        aligned = frame.merge(expected,on=KEYS,how='outer',indicator=True,validate='one_to_one')
        if not aligned['_merge'].eq('both').all() or not aligned.target.eq(aligned.expected_target).all():
            raise ValueError('Historical evaluation must match the entire recorded historical cohort and labels')
    elif frame.cxr_filename.isin(manifest.cxr_filename).any():
        raise ValueError('External image filenames overlap with the development dataset')
    report_path = output/f'public/{label}_metrics.csv'
    if report_path.exists():
        raise FileExistsError('Evaluation already written for this run; do not silently overwrite it')
    scores = calibrated_scores(frame, bundle['fusion'], bundle['calibrators'])
    raw = raw_scores(frame, bundle['fusion'])
    rows = []
    prediction_export = frame[[*KEYS,'target']].copy()
    curves = []
    for name, probability in scores.items():
        rows.append({'model': name, 'evaluation': label,
                     **metrics(frame.target, probability, bundle['thresholds'][name]['threshold']),
                     **calibration_diagnostics(frame.target, probability)})
        prediction_export[f'{name.lower()}_raw'] = raw[name]
        prediction_export[f'{name.lower()}_calibrated'] = probability
        observed, predicted = calibration_curve(frame.target, probability, n_bins=10, strategy='quantile')
        curves.extend({'model': name, 'mean_predicted': float(p), 'observed_fraction': float(o)}
                      for p,o in zip(predicted,observed))
    # Complete statistical calculations before committing the evaluation report.
    ci = paired_patient_bootstrap(frame, scores, bundle['thresholds'], repeats=repeats, seed=seed)
    pd.DataFrame(rows).to_csv(report_path, index=False)
    raw_rows = [{'model':name,'evaluation':label,
                 **metrics(frame.target,probability,.5),**calibration_diagnostics(frame.target,probability)}
                for name,probability in raw.items()]
    pd.DataFrame(raw_rows).to_csv(output/f'public/{label}_uncalibrated_at_05.csv',index=False)
    pd.DataFrame(curves).to_csv(output/f'public/{label}_calibration_curve.csv', index=False)
    prediction_export.to_csv(output/f'private/{label}_predictions.csv', index=False)
    write_json(output/f'public/{label}_ci95.json', ci)
    write_json(output/f'public/{label}_receipt.json', {'label': label, 'images': len(frame),
        'patients': frame.patient_id.nunique(), 'positive_images': int(frame.target.sum()),
        'predictions_sha256': sha256(output/f'private/{label}_predictions.csv'),
        'model_bundle_sha256': sha256(output/'models/fusion_bundle.joblib'),
        'manifest_sha256': sha256(output/'private/protocol_manifest.csv'),
        'historical_test_previously_inspected': label == 'historical_comparison',
        'claim': 'Historical comparison only; cannot establish new unbiased test performance' if label == 'historical_comparison'
                 else 'Independent source must be justified by the supplied external-source receipt; software checks cannot prove independence'})
    artifact_receipt(output)
    return pd.DataFrame(rows)
