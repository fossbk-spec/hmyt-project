"""Patient-separated development and auditable late fusion; no GPU dependency."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata as packages
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, fbeta_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

TARGET = 'composite_slvh_dlv'
FEATURES = ['age_scaled', 'sex_binary', 'heart_transplant', 'lung_transplant', 'pacemaker_or_icd']
KEYS = ['patient_id', 'cxr_filename']
ROLES = ('train', 'branch_val', 'fusion_fit', 'calibration', 'threshold', 'historical_test')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def environment_receipt():
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'torch', 'torchvision', 'pillow']:
        try:
            versions[name] = packages.version(name)
        except packages.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'platform': platform.platform(), 'packages': versions}


def prepare_frame(raw):
    required = {*KEYS, 'age', 'sex', 'slvh', 'dlv', TARGET}
    if not required.issubset(raw.columns):
        raise ValueError(f'Missing required fields: {sorted(required-set(raw.columns))}')
    frame = raw.copy()
    if frame[KEYS].isna().any().any():
        raise ValueError('Missing patient or image key')
    for key in KEYS:
        frame[key] = frame[key].astype(str).str.strip()
    if frame[KEYS].eq('').any().any() or frame.duplicated(KEYS).any():
        raise ValueError('Blank or duplicate examination keys')
    # Keep paths local to the expected images directory.
    if frame.cxr_filename.str.contains(r'[/\\]|^\.{1,2}$', regex=True).any():
        raise ValueError('Image filename must be a basename, not a path')
    frame['age'] = pd.to_numeric(frame.age, errors='coerce')
    frame['sex'] = frame.sex.astype(str).str.upper().str.strip()
    if not frame.age.between(18, 90).all() or not frame.sex.isin(['F', 'M']).all():
        raise ValueError('Invalid age/sex; inspect exclusions explicitly before training')
    for label in ['slvh', 'dlv', TARGET]:
        frame[label] = pd.to_numeric(frame[label], errors='coerce')
        if not frame[label].isin([0, 1]).all():
            raise ValueError(f'Invalid/missing binary label: {label}')
        frame[label] = frame[label].astype(int)
    if not frame[TARGET].eq(frame.slvh | frame.dlv).all():
        raise ValueError('Composite label disagrees with SLVH OR DLV')
    frame['age_scaled'] = (frame.age - 18) / 72
    frame['sex_binary'] = frame.sex.map({'F': 0., 'M': 1.})
    for feature in FEATURES[2:]:
        if feature not in frame:
            frame[feature] = np.nan
        frame[feature] = pd.to_numeric(frame[feature], errors='coerce')
        if not frame[feature].dropna().isin([0, 1]).all():
            raise ValueError(f'Non-binary metadata: {feature}')
    return frame


def validate_partition(frame):
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError('Empty manifest or duplicate keys')
    if frame.role.isna().any() or not set(frame.role).issubset(ROLES):
        raise ValueError('Unrecognised development role')
    if frame.groupby('patient_id').role.nunique().max() != 1:
        raise ValueError('Patient overlap across development roles')
    if set(frame.role) != set(ROLES):
        raise ValueError('At least one required development role is missing')
    for role, subset in frame.groupby('role'):
        if subset[TARGET].nunique() != 2:
            raise ValueError(f'Role {role} needs both classes')


def make_protocol(raw, *, seed=42):
    """Preserve the old train/test assignment; partition original val 25/35/20/20.

    A supplied split column must be the historical manifest, not a new favourable
    random split. Without it, reproduce the original notebook's exact split.
    """
    frame = prepare_frame(raw)
    patients = frame.groupby('patient_id', sort=True)[TARGET].max().reset_index()
    if 'split' not in frame:
        train, holdout = train_test_split(patients, train_size=.70, random_state=seed,
                                         stratify=patients[TARGET])
        val, test = train_test_split(holdout, train_size=.50, random_state=seed+1,
                                    stratify=holdout[TARGET])
        mapping = {**dict.fromkeys(train.patient_id, 'train'), **dict.fromkeys(val.patient_id, 'val'),
                   **dict.fromkeys(test.patient_id, 'test')}
        frame['split'] = frame.patient_id.map(mapping)
    if frame.split.isna().any() or set(frame.split) != {'train', 'val', 'test'}:
        raise ValueError('Historical split must contain train, val and test only')
    if frame.groupby('patient_id').split.nunique().max() != 1:
        raise ValueError('Historical patient leakage')
    val_patients = patients[patients.patient_id.isin(frame.loc[frame.split.eq('val'), 'patient_id'])]
    branch_val, rest = train_test_split(val_patients, train_size=.25, random_state=seed+10,
                                        stratify=val_patients[TARGET])
    fusion_fit, rest = train_test_split(rest, train_size=.35/.75, random_state=seed+11,
                                        stratify=rest[TARGET])
    calibration, threshold = train_test_split(rest, train_size=.5, random_state=seed+12,
                                              stratify=rest[TARGET])
    role_map = {**dict.fromkeys(frame.loc[frame.split.eq('train'), 'patient_id'], 'train'),
                **dict.fromkeys(frame.loc[frame.split.eq('test'), 'patient_id'], 'historical_test')}
    for name, subset in [('branch_val', branch_val), ('fusion_fit', fusion_fit),
                         ('calibration', calibration), ('threshold', threshold)]:
        role_map.update(dict.fromkeys(subset.patient_id, name))
    frame['role'] = frame.patient_id.map(role_map)
    validate_partition(frame)
    return frame


def split_summary(frame):
    return [{'role': role, 'images': len(part), 'patients': part.patient_id.nunique(),
             'positive_images': int(part[TARGET].sum()), 'prevalence': float(part[TARGET].mean())}
            for role, part in frame.groupby('role', sort=True)]


def fit_metadata(frame, *, seed=42, feature_set='age_sex'):
    validate_partition(frame)
    # Age/sex default avoids unverified T0 provenance of report-derived flags.
    features = FEATURES[:2] if feature_set == 'age_sex' else FEATURES if feature_set == 'full_retrospective' else None
    if features is None:
        raise ValueError('Unknown feature_set')
    train = frame[frame.role.eq('train')]
    val = frame[frame.role.eq('branch_val')]
    candidates = {
        'LogisticRegression': make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
            StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=seed)),
        'HistGradientBoosting': make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
            HistGradientBoostingClassifier(learning_rate=.05, max_iter=250, l2_regularization=1.,
                                           early_stopping=False, random_state=seed))}
    fitted, comparisons = {}, []
    for name, model in candidates.items():
        kwargs = {'histgradientboostingclassifier__sample_weight': compute_sample_weight('balanced', train[TARGET])} if name == 'HistGradientBoosting' else {}
        model.fit(train[features], train[TARGET], **kwargs)
        probability = model.predict_proba(val[features])[:, 1]
        comparisons.append({'model': name, 'selection_role': 'branch_val', 'images': len(val),
                            'average_precision': float(average_precision_score(val[TARGET], probability)),
                            'roc_auc': float(roc_auc_score(val[TARGET], probability))})
        fitted[name] = model
    winner = max(comparisons, key=lambda row: row['average_precision'])['model']
    return {'name': winner, 'model': fitted[winner], 'features': features,
            'feature_set': feature_set, 'comparisons': comparisons}


def pair_predictions(dl, ml):
    for label, data, probability in [('DL', dl, 'dl_probability'), ('ML', ml, 'ml_probability')]:
        if not {*KEYS, 'target', probability}.issubset(data.columns):
            raise ValueError(f'Missing {label} prediction columns')
        if data[KEYS].isna().any().any() or data.duplicated(KEYS).any():
            raise ValueError(f'Duplicate/missing {label} keys')
        if not data.target.isin([0, 1]).all():
            raise ValueError(f'Invalid {label} labels')
        if not np.isfinite(data[probability]).all() or not data[probability].between(0, 1).all():
            raise ValueError(f'Invalid {label} probabilities')
    merged = dl.merge(ml[[*KEYS, 'target', 'ml_probability']], on=KEYS, how='outer',
                      validate='one_to_one', indicator=True, suffixes=('', '_ml'))
    if not merged['_merge'].eq('both').all() or not merged.target.eq(merged.target_ml).all():
        raise ValueError('Branch key sets or labels do not match')
    return merged.drop(columns=['_merge', 'target_ml'])


def fit_fusion(frame, *, seed=42):
    fit = frame[frame.role.eq('fusion_fit')]
    if fit.target.nunique() != 2:
        raise ValueError('Fusion fitting needs both classes')
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(fit[['ml_probability', 'dl_probability']], fit.target)
    return model


def fit_calibrators(frame, fusion, *, seed=42):
    cal = frame[frame.role.eq('calibration')]
    raw = raw_scores(cal, fusion)
    if cal.target.nunique() != 2:
        raise ValueError('Calibration needs both classes')
    result = {}
    for name, probability in raw.items():
        # Unweighted sigmoid calibration on disjoint, natural-prevalence rows.
        model = LogisticRegression(C=1e6, max_iter=2000, random_state=seed)
        model.fit(logit(np.clip(probability, 1e-6, 1-1e-6)).reshape(-1, 1), cal.target)
        result[name] = model
    return result


def raw_scores(frame, fusion):
    return {'ML': frame.ml_probability.to_numpy(), 'DL': frame.dl_probability.to_numpy(),
            'Fusion': fusion.predict_proba(frame[['ml_probability', 'dl_probability']])[:, 1]}


def calibrated_scores(frame, fusion, calibrators):
    return {name: calibrators[name].predict_proba(logit(np.clip(score, 1e-6, 1-1e-6)).reshape(-1, 1))[:, 1]
            for name, score in raw_scores(frame, fusion).items()}


def select_threshold(y, probability, *, minimum_specificity=.80):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if not 0 <= minimum_specificity <= 1 or len(np.unique(y)) != 2:
        raise ValueError('Invalid specificity or threshold labels')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid probabilities')
    # Exact candidates and cumulative counts avoid O(n^2) threshold scans.
    order = np.argsort(-probability, kind='stable')
    values, labels = probability[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(values) != 0), len(values)-1]
    tp = np.r_[0, np.cumsum(labels)[ends]]
    fp = np.r_[0, (ends+1)-np.cumsum(labels)[ends]]
    thresholds = np.r_[np.nextafter(float(values[0]), np.inf), values[ends]]
    fn, tn = y.sum()-tp, (len(y)-y.sum())-fp
    specificity = tn / (tn+fp)
    f2 = np.divide(5*tp, 5*tp+4*fn+fp, out=np.zeros(len(tp), dtype=float), where=(5*tp+4*fn+fp)>0)
    valid = np.flatnonzero(specificity >= minimum_specificity)
    best = max(valid, key=lambda i: (f2[i], specificity[i], thresholds[i]))
    return {'threshold': float(thresholds[best]), 'selection_role': 'threshold',
            'minimum_specificity': minimum_specificity, 'selection_specificity': float(specificity[best]),
            'selection_f2': float(f2[best]),
            'note': 'Development constraint, not a clinical recommendation or guarantee on evaluation data'}


def calibration_diagnostics(y, probability):
    y = np.asarray(y, dtype=float)
    x = logit(np.clip(probability, 1e-6, 1-1e-6))
    if len(np.unique(y)) < 2:
        return {'calibration_in_the_large': None, 'calibration_intercept': None, 'calibration_slope': None}
    def objective(theta, slope_fixed=False):
        z = theta[0]+(x if slope_fixed else theta[1]*x)
        residual = expit(z)-y
        gradient = np.array([residual.mean()]) if slope_fixed else np.array([residual.mean(), (residual*x).mean()])
        return float(np.mean(np.logaddexp(0, z)-y*z)), gradient
    fixed = minimize(lambda b: objective(b, True), [0.], jac=True, method='L-BFGS-B', bounds=[(-30,30)])
    joint = minimize(objective, [0.,1.], jac=True, method='L-BFGS-B', bounds=[(-30,30),(-30,30)])
    return {'calibration_in_the_large': float(fixed.x[0]) if fixed.success else None,
            'calibration_intercept': float(joint.x[0]) if joint.success else None,
            'calibration_slope': float(joint.x[1]) if joint.success else None,
            'diagnostic_only': 'Fitted for assessment; does not update the prediction model',
            'bounded_fit': 'Coefficients bounded to [-30,30]; inspect separation/constant predictions before interpretation'}


def metrics(y, probability, threshold):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    pred = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    both = len(np.unique(y)) == 2
    return {'images': len(y), 'positive_images': int(y.sum()), 'threshold': float(threshold),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'specificity': float(tn/(tn+fp)) if tn+fp else None,
            'f1': float(f1_score(y, pred, zero_division=0)),
            'f2': float(fbeta_score(y, pred, beta=2, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, probability)) if both else None,
            'average_precision': float(average_precision_score(y, probability)) if both else None,
            'brier': float(brier_score_loss(y, probability)),
            'log_loss': float(log_loss(y, probability, labels=[0,1])),
            'referral_fraction': float(pred.mean()), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def paired_patient_bootstrap(frame, scores, thresholds, *, repeats=200, seed=42):
    """Resample whole patients with multiplicity, paired across every model."""
    groups = [np.asarray(v) for v in frame.reset_index(drop=True).groupby('patient_id', sort=True).indices.values()]
    rng, samples = np.random.default_rng(seed), {name: [] for name in scores}
    valid = 0
    keys = ['precision', 'recall', 'specificity', 'f1', 'f2', 'roc_auc', 'average_precision', 'brier', 'referral_fraction']
    for _ in range(repeats):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        y = frame.target.to_numpy()[ix]
        if len(np.unique(y)) != 2:
            continue
        valid += 1
        for name, probability in scores.items():
            samples[name].append(metrics(y, np.asarray(probability)[ix], thresholds[name]['threshold']))
    if valid < max(2, int(.8*repeats)):
        raise ValueError('Too few valid patient-bootstrap samples')
    intervals = {name: {key: np.quantile([r[key] for r in records], [.025,.975]).tolist() for key in keys}
                 for name, records in samples.items()}
    differences = {}
    if 'Fusion' in samples and 'DL' in samples:
        differences = {key: np.quantile([f[key]-d[key] for f,d in zip(samples['Fusion'], samples['DL'])],
                                        [.025,.975]).tolist() for key in keys}
    return {'method': 'Paired percentile patient-cluster bootstrap; image-weighted estimand',
            'repeats_requested': repeats, 'repeats_valid': valid, 'seed': seed,
            'ci95': intervals, 'fusion_minus_dl_ci95': differences,
            'limitation': 'Conditional on fitted models; no training variability or guarantee against prior test reuse'}
