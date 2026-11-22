"""EfficientNet training with persistent checkpoints and exact keyed predictions."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from fusion_protocol import TARGET, write_json, sha256


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class ImageRecords(Dataset):
    def __init__(self, frame, images, transform):
        self.frame = frame.reset_index(drop=True)
        self.images, self.transform = Path(images), transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.images / row.cxr_filename) as source:
            tensor = self.transform(source.convert('RGB'))
        return tensor, np.float32(row[TARGET]), str(row.patient_id), str(row.cxr_filename)


def transforms_for_run():
    weights = EfficientNet_B0_Weights.DEFAULT.transforms()
    normalise = transforms.Normalize(mean=weights.mean, std=weights.std)
    # A conservative first protocol revision: same spatial crop in train/eval.
    train = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        transforms.RandomRotation(5), transforms.ToTensor(), normalise])
    evaluation = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalise])
    return train, evaluation


def loader(frame, images, *, batch_size=32, workers=2, training=False, generator=None):
    train_transform, eval_transform = transforms_for_run()
    return DataLoader(ImageRecords(frame, images, train_transform if training else eval_transform),
        batch_size=batch_size, shuffle=training, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=False,
        worker_init_fn=seed_worker, generator=generator)


@torch.inference_mode()
def predict_images(model, frame, images, *, batch_size=32, workers=2):
    model.eval()
    device = next(model.parameters()).device
    records = []
    for tensors, targets, patients, filenames in loader(frame, images, batch_size=batch_size, workers=workers):
        probabilities = torch.sigmoid(model(tensors.to(device, non_blocking=True))).flatten().cpu().numpy()
        records.extend({'patient_id': str(p), 'cxr_filename': str(f), 'target': int(y),
                        'dl_probability': float(score)} for p,f,y,score in zip(patients,filenames,targets,probabilities))
    return pd.DataFrame(records)


def save_torch_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def train_image_model(frame, images, output, *, seed=42, epochs=10, patience=3,
                      batch_size=32, workers=2, learning_rate=1e-4, resume=False):
    output = Path(output)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(seed)
    training, validation = frame[frame.role.eq('train')], frame[frame.role.eq('branch_val')]
    config = {'seed': seed, 'epochs': epochs, 'patience': patience, 'batch_size': batch_size,
              'workers': workers, 'learning_rate': learning_rate, 'architecture': 'EfficientNet-B0',
              'weights': 'IMAGENET1K_V1', 'manifest_sha256': sha256(output/'private/protocol_manifest.csv'),
              'image_transform': 'resize256 center224; train rotation5; ImageNet normalisation',
              'feature_source_sha256': sha256(Path(__file__).with_name('fusion_protocol.py')),
              'image_source_sha256': sha256(__file__)}
    last_path, best_path = output/'models/image_last.pt', output/'models/image_best.pt'
    state = None
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError('Resume requested but image_last.pt does not exist')
        # Load only checkpoints created by this code from your own private run.
        state = torch.load(last_path, map_location='cpu', weights_only=True)
        if state['config'] != config:
            raise ValueError('Checkpoint config/source/split differs; start a separate run')
    elif last_path.exists():
        raise FileExistsError('Existing image run; use resume=True or a new RUN_ID')
    model = efficientnet_b0(weights=None if state else EfficientNet_B0_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    amp = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    positives = int(training[TARGET].sum())
    if positives == 0 or positives == len(training):
        raise ValueError('Image training needs both classes')
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(training)-positives)/positives], device=device))
    history, best_ap, stale, start_epoch = [], -1., 0, 1
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        amp.load_state_dict(state['amp'])
        history, best_ap, stale, start_epoch = state['history'], state['best_ap'], state['stale'], state['epoch']+1
        random.setstate(state['python_rng'])
        nr = state['numpy_rng']
        np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
        torch.set_rng_state(state['torch_rng'])
        generator.set_state(state['loader_rng'])
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    train_loader = loader(training, images, batch_size=batch_size, workers=workers,
                          training=True, generator=generator)
    write_json(output/'image_training_config.json', {**config, 'device': str(device),
               'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
               'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
               'positive_weight': (len(training)-positives)/positives,
               'reproducibility': 'Seeds and RNG states saved; cross-device/version bitwise identity is not guaranteed'})
    for epoch in range(start_epoch, epochs+1):
        if stale >= patience:
            break
        model.train()
        accumulated = 0.
        for tensors, labels, _, _ in train_loader:
            tensors = tensors.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1,1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = criterion(model(tensors), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite image loss; last completed epoch remains saved')
            amp.scale(loss).backward()
            amp.step(optimizer)
            amp.update()
            accumulated += float(loss.detach())*len(tensors)
        predictions = predict_images(model, validation, images, batch_size=batch_size, workers=workers)
        ap = float(average_precision_score(predictions.target, predictions.dl_probability))
        history.append({'epoch': epoch, 'train_loss': accumulated/len(training), 'branch_val_ap': ap})
        if ap > best_ap:
            best_ap, stale = ap, 0
            save_torch_atomic(best_path, {'model': model.state_dict(), 'epoch': epoch, 'branch_val_ap': ap, 'config': config})
        else:
            stale += 1
        nr = np.random.get_state()
        save_torch_atomic(last_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict(), 'epoch': epoch, 'history': history, 'best_ap': best_ap,
            'stale': stale, 'config': config, 'python_rng': random.getstate(),
            'numpy_rng': (nr[0], nr[1].tolist(), nr[2], nr[3], nr[4]),
            'torch_rng': torch.get_rng_state(), 'loader_rng': generator.get_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None})
        pd.DataFrame(history).to_csv(output/'public/dl_training_history.csv', index=False)
        print(f'Epoch {epoch}: loss={history[-1]["train_loss"]:.4f}, branch-val AP={ap:.4f}; checkpoint saved')
    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best['model'])
    write_json(output/'public/image_checkpoint_receipt.json',
               {'best_epoch': best['epoch'], 'branch_val_ap': best['branch_val_ap'],
                'checkpoint_sha256': sha256(best_path), 'role_used_for_selection': 'branch_val'})
    return model
