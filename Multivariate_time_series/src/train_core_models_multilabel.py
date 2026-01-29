#!/usr/bin/env python3
import os
import sys
import yaml
import json
import logging
import platform
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.metrics import average_precision_score

# ---------------------------
# Device selection
# ---------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif (platform.system() == "Darwin") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ---------------------------
# Dataset (multilabel)
# ---------------------------
class RegularDataset(Dataset):
    def __init__(self, demo_path, ts_path, label_cols=None, label_prefix=None, num_labels=None):
        keys = ['subject_id','hadm_id','icustay_id']
        drop = keys + ['timestep']

        demo = pd.read_parquet(demo_path, engine='pyarrow').copy()
        ts   = pd.read_parquet(ts_path,   engine='pyarrow').copy()

        # unify key dtypes as str
        for k in keys:
            if k in demo.columns: demo[k] = demo[k].astype(str)
            if k in ts.columns:   ts[k]   = ts[k].astype(str)

        # infer multilabel columns
        label_cols = self._infer_label_cols(demo, keys, label_cols, label_prefix, num_labels)
        self.label_names = label_cols

        # ensure float32 and fillna
        demo[label_cols] = demo[label_cols].fillna(0).astype('float32')

        # features
        self.feats = [c for c in ts.columns if c not in drop]
        ts[self.feats] = ts[self.feats].replace([np.inf, -np.inf], np.nan)
        med = ts[self.feats].median(numeric_only=True).fillna(0.0).astype('float32')
        ts[self.feats] = ts[self.feats].fillna(med).astype('float32')

        # join & group
        data = ts.merge(demo[keys + label_cols], on=keys, how='inner').sort_values(keys + ['timestep'])
        groups = data.groupby(keys, sort=False)

        self.X, ys = [], []
        for _, g in groups:
            self.X.append(g[self.feats].to_numpy(copy=False, dtype='float32'))
            ys.append(g[label_cols].iloc[0].to_numpy(dtype='float32'))
        self.y = np.stack(ys, axis=0).astype('float32')  # (N, C)

    @staticmethod
    def _infer_label_cols(demo, keys, label_cols, label_prefix, num_labels):
        if label_cols:
            cols = list(label_cols)
            return cols[:int(num_labels)] if num_labels else cols
        if label_prefix:
            cols = [c for c in demo.columns if c.startswith(label_prefix)]
            cols = sorted(cols)
            if num_labels: cols = cols[:int(num_labels)]
            if not cols:
                raise ValueError(f"No columns found with prefix '{label_prefix}'.")
            return cols
        # fallback: binary columns (exclude obvious non-labels)
        exclude = set(keys) | {'timestep','AGE','GENDER','RACE'}
        cand = [c for c in demo.columns if c not in exclude]
        bin_cols = []
        for c in cand:
            vc = pd.unique(demo[c].dropna())
            try_vals = set(map(float, vc))
            if try_vals <= {0.0, 1.0}:
                bin_cols.append(c)
        if not bin_cols:
            raise ValueError("Cannot infer multilabel columns; set task.label_prefix or task.label_cols in config.")
        cols = sorted(bin_cols)
        if num_labels: cols = cols[:int(num_labels)]
        return cols

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_pad(batch):
    xs, ys = zip(*batch)
    n = len(xs)
    d = xs[0].shape[1]
    T = max(x.shape[0] for x in xs)
    xpad = np.zeros((n, T, d), dtype='float32')
    for i, x in enumerate(xs):
        t = x.shape[0]
        xpad[i, :t] = x
    y = np.stack(ys, axis=0).astype('float32')  # (B, C)
    return torch.from_numpy(xpad), torch.from_numpy(y)

# ---------------------------
# Models (heads output C labels)
# ---------------------------
class MLPClassifier(nn.Module):
    """Time-averaged MLP head for variable-length sequences (multilabel)."""
    def __init__(self, D, hidden_sizes, num_labels: int):
        super().__init__()
        in_dim = D
        if isinstance(hidden_sizes, int):
            hidden = hidden_sizes
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, num_labels)
            )
        else:
            layers = []
            prev = in_dim
            for h in (hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]):
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers += [nn.Linear(prev, num_labels)]
            self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, T, D]
        x = x.mean(dim=1)  # [B, D]  global average pooling
        return self.net(x) # [B, C]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, T, D, d_model, nhead, num_layers, num_labels):
        super().__init__()
        self.proj = nn.Linear(D, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, x):  # x: [B, T, D]
        h = self.proj(x)          # [B, T, d]
        h = self.encoder(h)       # [B, T, d]
        h = h.transpose(1, 2)     # [B, d, T]
        h = self.pool(h).squeeze(-1)  # [B, d]
        return self.head(h)       # [B, C]

class LSTMClassifier(nn.Module):
    def __init__(self, T, D, hidden_size, num_layers, bidirectional=False, num_labels: int = 25):
        super().__init__()
        self.lstm = nn.LSTM(input_size=D, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True, bidirectional=bidirectional)
        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(nn.Linear(lstm_out, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_labels))

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        return self.classifier(h)

class RETAIN(nn.Module):
    def __init__(self, T, D, hidden_size, num_labels=25):
        super().__init__()
        self.rnn_alpha = nn.GRU(input_size=D, hidden_size=hidden_size, batch_first=True)
        self.rnn_beta  = nn.GRU(input_size=D, hidden_size=hidden_size, batch_first=True)
        self.attention_alpha = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softmax(dim=1))
        self.attention_beta  = nn.Sequential(nn.Linear(hidden_size, D), nn.Tanh())
        self.embedding = nn.Linear(D, D)
        self.classifier = nn.Linear(D, num_labels)

    def forward(self, x, return_attention=False):
        x_emb = self.embedding(x)
        h_alpha, _ = self.rnn_alpha(torch.flip(x, dims=[1]))
        alpha = self.attention_alpha(h_alpha)           # [B,T,1]
        h_beta, _ = self.rnn_beta(x)
        beta = self.attention_beta(h_beta)              # [B,T,D]
        ctx = (x_emb * beta) * alpha                    # [B,T,D]
        v = ctx.sum(dim=1)                              # [B,D]
        logits = self.classifier(v)
        if return_attention:
            return logits, {'alpha': alpha.squeeze(-1), 'beta': beta}
        return logits

# ---------------------------
# Train / Eval helpers
# ---------------------------

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE, dtype=torch.float32)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_metrics_multilabel(model, loader, return_predictions=False):
    """Compute multilabel metrics.
    Returns dict with: auroc_macro_ovr, auprc_macro_ovr, f1_micro_05, f1_macro_05,
    f1_micro_best, f1_macro_best, best_thr. Optionally (y_true, y_prob).
    """
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys.append(y.numpy())
            ps.append(probs)
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    C = y_true.shape[1]

    # OVR AUROC / AUPRC (only labels with both classes present)
    aurocs, aprs = [], []
    for c in range(C):
        yt = y_true[:, c]
        yp = y_prob[:, c]
        if yt.max() > 0 and yt.min() < 1:
            try:
                aurocs.append(roc_auc_score(yt, yp))
            except Exception:
                pass
            try:
                aprs.append(average_precision_score(yt, yp))
            except Exception:
                pass
    auroc_macro_ovr = float(np.mean(aurocs)) if aurocs else float('nan')
    auprc_macro_ovr = float(np.mean(aprs))   if aprs   else float('nan')

    # F1 @ 0.5
    y_hat_05 = (y_prob >= 0.5).astype(int)
    f1_micro_05 = f1_score(y_true, y_hat_05, average='micro', zero_division=0)
    f1_macro_05 = f1_score(y_true, y_hat_05, average='macro', zero_division=0)

    # Best global threshold (micro-F1)
    thr_grid = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1_micro = 0.5, -1.0
    for thr in thr_grid:
        y_hat = (y_prob >= thr).astype(int)
        f1m = f1_score(y_true, y_hat, average='micro', zero_division=0)
        if f1m > best_f1_micro:
            best_f1_micro, best_thr = f1m, float(thr)
    y_hat_best = (y_prob >= best_thr).astype(int)
    f1_macro_best = f1_score(y_true, y_hat_best, average='macro', zero_division=0)

    metrics = {
        'auroc_macro_ovr': auroc_macro_ovr,
        'auprc_macro_ovr': auprc_macro_ovr,
        'f1_micro_05': float(f1_micro_05),
        'f1_macro_05': float(f1_macro_05),
        'f1_micro_best': float(best_f1_micro),
        'f1_macro_best': float(f1_macro_best),
        'best_thr': float(best_thr),
    }
    if return_predictions:
        return metrics, y_true, y_prob
    return metrics

# ---------------------------
# Logging helpers
# ---------------------------

def setup_logging(log_dir, model_type, timestamp):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model_type}_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger, log_file


def log_config(logger, config, model_type, num_labels):
    logger.info("="*60)
    logger.info(f"Training Configuration for {model_type.upper()} Model")
    logger.info("="*60)
    for k, v in config['models'][model_type].items():
        logger.info(f"  {k}: {v}")
    logger.info("Training:")
    for k, v in config['training'].items():
        logger.info(f"  {k}: {v}")
    logger.info("Data Paths:")
    for k, v in config['data'].items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Num Labels: {num_labels}")
    logger.info(f"Device: {DEVICE}")
    logger.info("="*60)

# ---------------------------
# Main training/eval
# ---------------------------

def train_model(config, model_type, eval_only=False, checkpoint_path=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_file = setup_logging(config['logging']['dir'], model_type, timestamp)
    logger.info(f"Starting {'evaluation' if eval_only else 'training'} for {model_type.upper()} model")
    logger.info(f"Log file: {log_file}")

    # task
    task_cfg = config.get('task', {})
    num_labels = int(task_cfg.get('num_labels', 25))
    label_cols  = task_cfg.get('label_cols', None)
    label_prefix= task_cfg.get('label_prefix', None)

    # types
    cfg_train = config['training']
    cfg_train['epochs'] = int(cfg_train['epochs'])
    cfg_train['batch_size'] = int(cfg_train['batch_size'])
    cfg_train['lr'] = float(cfg_train['lr'])

    if model_type == 'mlp':
        mcfg = config['models']['mlp']
        if 'hidden' in mcfg:
            mcfg['hidden'] = int(mcfg['hidden'])
        elif 'hidden_sizes' in mcfg:
            hs = mcfg['hidden_sizes']
            mcfg['hidden_sizes'] = [int(x) for x in hs] if isinstance(hs, list) else int(hs)
    elif model_type == 'transformer':
        mcfg = config['models']['transformer']
        mcfg['d_model'] = int(mcfg['d_model']); mcfg['nhead'] = int(mcfg['nhead']); mcfg['num_layers'] = int(mcfg['num_layers'])
    elif model_type == 'lstm':
        mcfg = config['models']['lstm']
        mcfg['hidden_size'] = int(mcfg['hidden_size']); mcfg['num_layers'] = int(mcfg['num_layers']); mcfg['bidirectional'] = bool(mcfg['bidirectional'])
    elif model_type == 'retain':
        mcfg = config['models']['retain']
        mcfg['hidden_size'] = int(mcfg['hidden_size'])

    if not eval_only:
        log_config(logger, config, model_type, num_labels)

    # data
    logger.info("Loading datasets...")
    paths = {
        'train': (config['data']['train_demo'], config['data']['train_ts']),
        'val':   (config['data']['val_demo'],   config['data']['val_ts']),
        'test':  (config['data']['test_demo'],  config['data']['test_ts']),
    }
    datasets = {k: RegularDataset(*v, label_cols=label_cols, label_prefix=label_prefix, num_labels=num_labels) for k, v in paths.items()}
    logger.info("Initialised datasets...")
    dataloaders = {k: DataLoader(datasets[k], batch_size=cfg_train['batch_size'], shuffle=(k=='train'),
                                 collate_fn=collate_pad, num_workers=8, pin_memory=True)
                   for k in datasets}
    logger.info("Finished Loading dataloaders...")

    # dims
    x0, y0 = datasets['train'][0]
    T, D = x0.shape; C = y0.shape[0]
    assert C == num_labels, f"num_labels={num_labels}, but dataset has {C}"
    logger.info(f"Data shape: T={T}, D={D}, C={C}, N_train={len(datasets['train'])}, N_val={len(datasets['val'])}, N_test={len(datasets['test'])}")

    # model
    mcfg = config['models'][model_type]
    if model_type == 'mlp':
        model = MLPClassifier(D, mcfg.get('hidden_sizes', mcfg.get('hidden')), num_labels)
    elif model_type == 'transformer':
        model = TimeSeriesTransformer(T, D, mcfg['d_model'], mcfg['nhead'], mcfg['num_layers'], num_labels)
    elif model_type == 'lstm':
        model = LSTMClassifier(T, D, mcfg['hidden_size'], mcfg['num_layers'], mcfg['bidirectional'], num_labels)
    elif model_type == 'retain':
        model = RETAIN(T, D, mcfg['hidden_size'], num_labels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.to(DEVICE)

    # params
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # eval-only
    if eval_only:
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be provided for eval_only mode")
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt: model.load_state_dict(ckpt['model_state_dict'])
            elif 'model' in ckpt:          model.load_state_dict(ckpt['model'])
            else:                           model.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt)

        logger.info("\nEvaluating on all splits:")
        logger.info("-" * 40)
        for split in ['train', 'val', 'test']:
            metrics = eval_metrics_multilabel(model, dataloaders[split])
            logger.info(
                f"{split.upper():5s} - AUROC(OVR,macro): {metrics['auroc_macro_ovr']:.4f} | "
                f"AUPRC(OVR,macro): {metrics['auprc_macro_ovr']:.4f} | "
                f"F1@0.5 micro/macro: {metrics['f1_micro_05']:.4f}/{metrics['f1_macro_05']:.4f} | "
                f"F1@best micro/macro: {metrics['f1_micro_best']:.4f}/{metrics['f1_macro_best']:.4f} | "
                f"best_thr={metrics['best_thr']:.3f}"
            )

        logger.info("\nDetailed Test Set Analysis:")
        logger.info("-" * 40)
        test_metrics, y_true, y_prob = eval_metrics_multilabel(model, dataloaders['test'], return_predictions=True)
        best_thr = test_metrics['best_thr']
        y_hat = (y_prob >= best_thr).astype(int)
        label_names = datasets['test'].label_names
        logger.info("Classification Report @best_thr:")
        logger.info(classification_report(y_true, y_hat, target_names=label_names, zero_division=0))
        return

    # training loss (BCE with per-label pos_weight)
    Y = datasets['train'].y.astype(np.float64)
    pos = Y.sum(axis=0); neg = Y.shape[0] - pos
    pos_weight_np = neg / np.maximum(pos, 1.0)
    pos_weight = torch.tensor(pos_weight_np, device=DEVICE, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"Using BCEWithLogitsLoss; pos_weight min/max: {pos_weight_np.min():.3f}/{pos_weight_np.max():.3f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train['lr'])

    # checkpoints
    ckpt_dir = os.path.join(config['checkpoint']['dir'], f"{model_type}_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {ckpt_dir}")

    best_metric = -float('inf')  # select by AUPRC macro OVR
    best_epoch = 0

    logger.info("\nStarting training...")
    logger.info("-" * 60)
    for epoch in range(1, cfg_train['epochs'] + 1):
        train_loss = train_epoch(model, dataloaders['train'], criterion, optimizer)
        val_metrics = eval_metrics_multilabel(model, dataloaders['val'])
        logger.info(
            f"Epoch {epoch:3d}/{cfg_train['epochs']} | Loss: {train_loss:.4f} | "
            f"Val AUROC(OVR,macro): {val_metrics['auroc_macro_ovr']:.4f} | "
            f"Val AUPRC(OVR,macro): {val_metrics['auprc_macro_ovr']:.4f} | "
            f"F1@0.5 micro/macro: {val_metrics['f1_micro_05']:.4f}/{val_metrics['f1_macro_05']:.4f} | "
            f"F1@best micro/macro: {val_metrics['f1_micro_best']:.4f}/{val_metrics['f1_macro_best']:.4f} | "
            f"best_thr={val_metrics['best_thr']:.3f}"
        )

        # periodic ckpt
        save_every = config['checkpoint'].get('save_every', 5)
        if epoch % save_every == 0 or epoch == cfg_train['epochs']:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                'epoch': int(epoch),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_auroc_macro_ovr': float(val_metrics['auroc_macro_ovr']),
                'val_auprc_macro_ovr': float(val_metrics['auprc_macro_ovr']),
                'val_f1_micro_05': float(val_metrics['f1_micro_05']),
                'val_f1_macro_05': float(val_metrics['f1_macro_05']),
                'val_f1_micro_best': float(val_metrics['f1_micro_best']),
                'val_f1_macro_best': float(val_metrics['f1_macro_best']),
                'val_best_thr': float(val_metrics['best_thr']),
                'train_loss': float(train_loss)
            }, ckpt_path)

        # best selection
        sel = val_metrics['auprc_macro_ovr']
        if np.isfinite(sel) and sel > best_metric:
            best_metric = sel
            best_epoch = epoch
            best_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save({
                'epoch': int(epoch),
                'model': model.state_dict(),
                'val_auroc_macro_ovr': float(val_metrics['auroc_macro_ovr']),
                'val_auprc_macro_ovr': float(val_metrics['auprc_macro_ovr']),
                'val_f1_micro_05': float(val_metrics['f1_micro_05']),
                'val_f1_macro_05': float(val_metrics['f1_macro_05']),
                'val_f1_micro_best': float(val_metrics['f1_micro_best']),
                'val_f1_macro_best': float(val_metrics['f1_macro_best']),
                'val_best_thr': float(val_metrics['best_thr'])
            }, best_path)
            logger.info(f"  -> New best model saved (AUPRC OVR macro: {sel:.4f})")

    logger.info("-" * 60)
    logger.info(f"Training completed. Best epoch: {best_epoch}")

    # final test
    logger.info("\nLoading best model for final evaluation...")
    best_checkpoint = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint['model'])

    test_metrics, y_true, y_prob = eval_metrics_multilabel(model, dataloaders['test'], return_predictions=True)
    best_thr = test_metrics['best_thr']
    y_hat = (y_prob >= best_thr).astype(int)
    label_names = datasets['test'].label_names

    logger.info("="*60)
    logger.info("FINAL TEST SET RESULTS")
    logger.info("="*60)
    logger.info(f"AUROC(OVR,macro): {test_metrics['auroc_macro_ovr']:.4f}")
    logger.info(f"AUPRC(OVR,macro): {test_metrics['auprc_macro_ovr']:.4f}")
    logger.info(f"F1@0.5 micro/macro: {test_metrics['f1_micro_05']:.4f}/{test_metrics['f1_macro_05']:.4f}")
    logger.info(f"F1@best micro/macro: {test_metrics['f1_micro_best']:.4f}/{test_metrics['f1_macro_best']:.4f}  (best_thr={best_thr:.3f})")

    logger.info("\nClassification Report @best_thr:")
    logger.info(classification_report(y_true, y_hat, target_names=label_names, zero_division=0))

    # results.json
    # Save results summary (use macro metrics, same format as other script)
    results = {
        'model_type': model_type,
        'timestamp': timestamp,
        'best_epoch': best_epoch,

        # validation (macro)
        'best_val_auc': float(best_metric),   # Here best_metric is val_auprc_macro_ovr (used for selection)
        'best_val_auprc': float(best_metric),
        'best_val_f1': float(val_metrics['f1_macro_best']),   # Use macro F1

        # test (macro)
        'test_auc': float(test_metrics['auroc_macro_ovr']),
        'test_auprc': float(test_metrics['auprc_macro_ovr']),
        'test_f1': float(test_metrics['f1_macro_best']),      # Keep old key; value is macro F1(best)
        'test_f1_best': float(test_metrics['f1_macro_best']),
        'test_f1_05': float(test_metrics['f1_macro_05']),
        'test_best_thr': float(best_thr),
        # Accuracy is not well-defined for multilabel; omit to keep format consistent

        'checkpoint_dir': ckpt_dir,
        'num_parameters': num_params,
        'config': config['models'][model_type]
    }

    results_file = os.path.join(ckpt_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")
    logger.info("="*60)



def main():
    parser = argparse.ArgumentParser(description='Train ICU Multi-Label Prediction Models')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['mlp', 'transformer', 'lstm', 'retain'], default='transformer', help='Model type to train')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only (no training)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for evaluation')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config.setdefault('task', {})
    config['task'].setdefault('num_labels', 25)

    train_model(config, args.model, args.eval_only, args.checkpoint)


if __name__ == "__main__":
    main()
