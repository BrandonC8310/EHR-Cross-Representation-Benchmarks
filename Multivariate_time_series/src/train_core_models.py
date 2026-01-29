#!/usr/bin/env python3
import os
import sys
import yaml
import json
import logging
import platform
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, classification_report, accuracy_score
from sklearn.metrics import average_precision_score, precision_recall_curve


# Device auto-selection: CUDA > MPS (Mac M1/M2) > CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif (platform.system() == "Darwin") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# —— Dataset Definition (UNCHANGED FROM ORIGINAL) —— #
# class RegularDataset(Dataset):
#     def __init__(self, demo_path, ts_path):
#         demo = pd.read_parquet(demo_path)
#         ts = pd.read_parquet(ts_path)
#         self.labels = demo['label'].values.astype(np.float32)
#         drop = ['subject_id', 'hadm_id', 'icustay_id', 'timestep']
#         self.feats = [c for c in ts.columns if c not in drop]
#         seqs = []
#         for _, r in demo.iterrows():
#             df = ts[
#                 (ts.subject_id == r.subject_id) &
#                 (ts.hadm_id == r.hadm_id) &
#                 (ts.icustay_id == r.icustay_id)
#             ].sort_values('timestep')
#             seqs.append(df[self.feats].values.astype(np.float32))
#         self.X = np.stack(seqs, axis=0)  # (N, T, D)
#         self.y = self.labels              # (N,)

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
class RegularDataset(Dataset):
    def __init__(self, demo_path, ts_path):
        keys = ['subject_id','hadm_id']
        drop = keys + ['timestep', 'icustay_id']

        demo = pd.read_parquet(demo_path, engine='pyarrow')
        # If label has missing values, drop those visits to avoid NaN targets.
        demo = demo.dropna(subset=['label']).copy()
        ts = pd.read_parquet(ts_path, engine='pyarrow')

        self.feats = [c for c in ts.columns if c not in drop]

        # Convert ±Inf to NaN for unified handling.
        ts[self.feats] = ts[self.feats].replace([np.inf, -np.inf], np.nan)

        # Compute per-feature medians for imputation (all-NaN columns become 0).
        med = ts[self.feats].median(numeric_only=True)
        med = med.fillna(0.0).astype('float32')

        # Impute and cast to float32.
        ts[self.feats] = ts[self.feats].fillna(med).astype('float32')

        # Join once, sort, then group by visit.
        data = ts.merge(demo[keys + ['label']], on=keys, how='inner')
        data = data.sort_values(keys + ['timestep'])

        groups = data.groupby(keys, sort=False)
        # Do not stack in __init__; keep as list and pad in batches.
        self.X = [g[self.feats].to_numpy(copy=False, dtype='float32') for _, g in groups]
        self.y = groups['label'].first().to_numpy(dtype='float32')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def collate_pad(batch):
    xs, ys = zip(*batch)
    n = len(xs)
    d = xs[0].shape[1]
    lens = [x.shape[0] for x in xs]
    T = max(lens)
    xpad = np.zeros((n, T, d), dtype='float32')
    mask = np.zeros((n, T), dtype=bool)
    for i, x in enumerate(xs):
        t = x.shape[0]
        xpad[i, :t] = x
        mask[i, :t] = True
    return torch.from_numpy(xpad), torch.from_numpy(np.array(ys, dtype='float32'))




# —— Model Definitions —— #
class MLPClassifier(nn.Module):
    def __init__(self, T, D, hidden_sizes):
        super().__init__()
        # If single value provided, use original architecture
        if isinstance(hidden_sizes, int):
            hidden = hidden_sizes
            self.net = nn.Sequential(
                nn.Flatten(),                    # [T,D] -> [T*D]
                nn.Linear(T*D, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        else:
            # Support for multiple hidden layers
            layers = [nn.Flatten()]
            input_size = T * D
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU()
                ])
                input_size = hidden_size
            layers.append(nn.Linear(input_size, 1))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)      # [B]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, T, D, d_model, nhead, num_layers):
        super().__init__()
        self.proj = nn.Linear(D, d_model)
        # KEEPING ORIGINAL batch_first=False for compatibility
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, T, D] - KEEPING ORIGINAL FORWARD PASS
        B, T, D = x.shape
        h = self.proj(x)               # [B, T, d_model]
        h = h.permute(1, 0, 2)         # [T, B, d_model]
        h = self.encoder(h)            # [T, B, d_model]
        h = h.permute(1, 2, 0)         # [B, d_model, T]
        h = self.pool(h).squeeze(-1)   # [B, d_model]
        return self.head(h).squeeze(-1)

class LSTMClassifier(nn.Module):
    """LSTM model for time series classification"""
    def __init__(self, T, D, hidden_size, num_layers, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=D,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output size depends on bidirectional
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # x: [B, T, D]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate the last hidden states from both directions
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h_last = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_last = h_n[-1, :, :]
        
        # Pass through classifier
        logits = self.classifier(h_last)
        return logits.squeeze(-1)

class RETAIN(nn.Module):
    """
    RETAIN: Reverse Time Attention model for Healthcare
    Choi et al., 2016: "RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism"
    
    This model provides interpretable predictions by learning:
    1. Visit-level attention (which time steps are important)
    2. Variable-level attention (which features are important at each time)
    """
    def __init__(self, T, D, hidden_size, num_classes=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # RNN for visit-level attention (alpha)
        self.rnn_alpha = nn.GRU(
            input_size=D,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False  # Reverse time, so unidirectional
        )
        
        # RNN for variable-level attention (beta)
        self.rnn_beta = nn.GRU(
            input_size=D,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanisms
        self.attention_alpha = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)  # Softmax over time dimension
        )
        
        self.attention_beta = nn.Sequential(
            nn.Linear(hidden_size, D),
            nn.Tanh()  # Variable-level attention weights
        )
        
        # Embedding for input features (optional, helps with sparse features)
        self.embedding = nn.Linear(D, D)
        
        # Final classifier
        self.classifier = nn.Linear(D, num_classes)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor of shape [B, T, D]
            return_attention: If True, return attention weights for interpretability
        
        Returns:
            logits: Predictions of shape [B] (or [B, num_classes] if num_classes > 1)
            attention_weights: (optional) Dictionary with alpha and beta attention weights
        """
        B, T, D = x.shape
        
        # Embed input features
        x_embedded = self.embedding(x)  # [B, T, D]
        
        # Reverse time for alpha RNN (RETAIN's key innovation)
        x_reversed = torch.flip(x, dims=[1])
        
        # Generate visit-level attention (alpha)
        h_alpha, _ = self.rnn_alpha(x_reversed)  # [B, T, hidden_size]
        alpha = self.attention_alpha(h_alpha)    # [B, T, 1]
        
        # Generate variable-level attention (beta)
        h_beta, _ = self.rnn_beta(x)            # [B, T, hidden_size]
        beta = self.attention_beta(h_beta)       # [B, T, D]
        
        # Apply attention mechanisms
        # Element-wise multiply embedded input with variable attention
        context = x_embedded * beta              # [B, T, D]
        
        # Weighted sum over time with visit attention
        context_weighted = context * alpha       # [B, T, D]
        context_vector = torch.sum(context_weighted, dim=1)  # [B, D]
        
        # Final prediction
        logits = self.classifier(context_vector)
        
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        
        if return_attention:
            return logits, {'alpha': alpha.squeeze(-1), 'beta': beta}
        
        return logits

# —— Training/Validation Functions (MINIMAL CHANGES) —— #
def train_epoch(model, loader, criterion, optimizer):
    """Original training function - no changes"""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_metrics(model, loader, return_predictions=False):
    """Return (auroc, auprc, f1_best, f1_05, best_thr) 
       or (auroc, auprc, f1_best, f1_05, best_thr, y_true, y_pred) when return_predictions=True"""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, _mask, y = batch
            else:
                x, y = batch
            x = x.to(DEVICE)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            ys.append(y.numpy())
            ps.append(probs)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    # Safety guard to avoid NaN/Inf during evaluation.
    y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)

    # Threshold-free metrics.
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    # Choose threshold that maximises F1 on PR curve.
    prec, rec, thr = precision_recall_curve(y_true, y_pred)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    if f1s.size:
        best_idx = int(np.nanargmax(f1s))
        best_thr = float(thr[best_idx - 1]) if thr.size and best_idx > 0 else 0.5
        f1_best = float(f1s[best_idx])
    else:
        best_thr, f1_best = 0.5, 0.0

    # F1@0.5
    y_pred_bin_05 = (y_pred >= 0.5).astype(int)
    f1_05 = f1_score(y_true, y_pred_bin_05)

    if return_predictions:
        return auroc, auprc, f1_best, f1_05, best_thr, y_true, y_pred
    return auroc, auprc, f1_best, f1_05, best_thr


# —— NEW: Enhanced Logging Functions —— #
def setup_logging(log_dir, model_type, timestamp):
    """Setup logging with timestamp in filename"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model_type}_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file

def log_config(logger, config, model_type):
    """Log the configuration used for training"""
    logger.info("="*60)
    logger.info(f"Training Configuration for {model_type.upper()} Model")
    logger.info("="*60)
    
    # Log model-specific config
    model_config = config['models'][model_type]
    logger.info(f"Model Configuration ({model_type}):")
    for key, value in model_config.items():
        logger.info(f"  {key}: {value}")
    
    # Log training config
    train_config = config['training']
    logger.info(f"Training Configuration:")
    for key, value in train_config.items():
        logger.info(f"  {key}: {value}")
    
    # Log data paths
    logger.info(f"Data Paths:")
    for key, value in config['data'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"Device: {DEVICE}")
    logger.info("="*60)

# —— Main Training Function —— #
def train_model(config, model_type, eval_only=False, checkpoint_path=None):
    """Main training function with enhanced logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    logger, log_file = setup_logging(config['logging']['dir'], model_type, timestamp)
    logger.info(f"Starting {'evaluation' if eval_only else 'training'} for {model_type.upper()} model")
    logger.info(f"Log file: {log_file}")
    
    # Type conversion for config values
    config['training']['epochs'] = int(config['training']['epochs'])
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['lr'] = float(config['training']['lr'])
    
    # Convert model-specific parameters
    if model_type == 'mlp':
        if 'hidden' in config['models']['mlp']:
            # Support old format
            config['models']['mlp']['hidden'] = int(config['models']['mlp']['hidden'])
        elif 'hidden_sizes' in config['models']['mlp']:
            # Support new format
            sizes = config['models']['mlp']['hidden_sizes']
            if isinstance(sizes, list):
                config['models']['mlp']['hidden_sizes'] = [int(x) for x in sizes]
            else:
                config['models']['mlp']['hidden_sizes'] = int(sizes)
    elif model_type == 'transformer':
        config['models']['transformer']['d_model'] = int(config['models']['transformer']['d_model'])
        config['models']['transformer']['nhead'] = int(config['models']['transformer']['nhead'])
        config['models']['transformer']['num_layers'] = int(config['models']['transformer']['num_layers'])
    elif model_type == 'lstm':
        config['models']['lstm']['hidden_size'] = int(config['models']['lstm']['hidden_size'])
        config['models']['lstm']['num_layers'] = int(config['models']['lstm']['num_layers'])
        config['models']['lstm']['bidirectional'] = bool(config['models']['lstm']['bidirectional'])
    elif model_type == 'retain':
        config['models']['retain']['hidden_size'] = int(config['models']['retain']['hidden_size'])
    
    # Log configuration
    if not eval_only:
        log_config(logger, config, model_type)
    
    # Load data
    logger.info("Loading datasets...")
    paths = {
        'train': (config['data']['train_demo'], config['data']['train_ts']),
        'val': (config['data']['val_demo'], config['data']['val_ts']),
        'test': (config['data']['test_demo'], config['data']['test_ts']),
    }
    
    datasets = {k: RegularDataset(*v) for k, v in paths.items()}
    logger.info("Initialised datasets...")
    # dataloaders = {
    #     k: DataLoader(
    #         datasets[k], 
    #         batch_size=config['training']['batch_size'], 
    #         shuffle=(k == 'train')
    #     )
    #     for k in datasets
    # }
    dataloaders = {
    k: DataLoader(
        datasets[k],
        batch_size=int(config['training']['batch_size']),
        shuffle=(k == 'train'),
        collate_fn=collate_pad,
        num_workers=8,        # Tune based on memory/CPU; start from 0 if needed
        pin_memory=True
    )
    for k in datasets
}
    logger.info("Finished Loading dataloader...")
    # Get data dimensions
    # Infer T,D from first sample (variable-length sequences).
    x0, _ = datasets['train'][0]
    T, D = x0.shape
    logger.info(f"Data shape: T={T}, D={D}, N_train={len(datasets['train'])}, N_val={len(datasets['val'])}, N_test={len(datasets['test'])}")
    
    # Initialize model
    model_config = config['models'][model_type]
    if model_type == 'mlp':
        # Support both old and new config format
        if 'hidden' in model_config:
            model = MLPClassifier(T, D, model_config['hidden'])
        else:
            model = MLPClassifier(T, D, model_config['hidden_sizes'])
    elif model_type == 'transformer':
        model = TimeSeriesTransformer(
            T, D,
            model_config['d_model'],
            model_config['nhead'],
            model_config['num_layers']
        )
    elif model_type == 'lstm':
        model = LSTMClassifier(
            T, D,
            model_config['hidden_size'],
            model_config['num_layers'],
            model_config['bidirectional']
        )
    elif model_type == 'retain':
        model = RETAIN(
            T, D,
            model_config['hidden_size']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Eval-only mode
    if eval_only:
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be provided for eval_only mode")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        logger.info("\nEvaluating on all splits:")
        logger.info("-" * 40)
        
        # Evaluate on all splits
        for split in ['train', 'val', 'test']:
            auc, auprc, f1_best, f1_05, best_thr = eval_metrics(model, dataloaders[split])
            logger.info(f"{split.upper():5s} - AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | "
                        f"F1(best): {f1_best:.4f} @thr={best_thr:.3f} | F1@0.5: {f1_05:.4f}")

        
        # Detailed test set analysis
        logger.info("\nDetailed Test Set Analysis:")
        logger.info("-" * 40)
        test_auc, test_f1, test_acc, y_true, y_pred = eval_metrics(
            model, dataloaders['test'], return_predictions=True
        )
        y_pred_bin = (y_pred >= 0.5).astype(int)
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_true, y_pred_bin, 
                                         target_names=['Survived', 'Died']))
        return
    
    # Training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    # criterion = nn.BCEWithLogitsLoss()
    pos = float((datasets['train'].y > 0.5).sum())
    neg = float(len(datasets['train']) - pos)
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"Using BCEWithLogitsLoss pos_weight={pos_weight.item():.2f}")

    # Checkpoint directory with timestamp
    ckpt_dir = os.path.join(
        config['checkpoint']['dir'], 
        f"{model_type}_{timestamp}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {ckpt_dir}")
    
    # Training loop
    best_val_auc = 0.0
    best_val_auprc = 0.0     # Used to select the best checkpoint
    best_val_f1 = 0.0
    best_epoch = 0
    
    logger.info("\nStarting training...")
    logger.info("-" * 60)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        train_loss = train_epoch(model, dataloaders['train'], criterion, optimizer)
        
        # Validate
        val_auc, val_auprc, val_f1_best, val_f1_05, val_best_thr = eval_metrics(model, dataloaders['val'])

        # Log metrics
        logger.info(
            f"Epoch {epoch:3d}/{config['training']['epochs']} | "
            f"Loss: {train_loss:.4f} | "
            f"Val AUROC: {val_auc:.4f} | "
            f"Val AUPRC: {val_auprc:.4f} | "
            f"Val F1(best): {val_f1_best:.4f} @thr={val_best_thr:.3f} | "
            f"Val F1@0.5: {val_f1_05:.4f}"
        )


        # Save checkpoint periodically
        save_every = config['checkpoint'].get('save_every', 0)
        if save_every != 0:
            if epoch % save_every == 0 or epoch == config['training']['epochs']:
                ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
                torch.save({
                    'epoch': int(epoch),
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_auc': float(val_auc),
                    'val_auprc': float(val_auprc),        # added
                    'val_f1_best': float(val_f1_best),    # added
                    'val_f1_05': float(val_f1_05),        # added
                    'val_best_thr': float(val_best_thr),  # added
                    'train_loss': float(train_loss)
                }, ckpt_path)

        
        # Save best model (AUPRC as primary metric)
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_val_auc = val_auc          # Keep as record
            best_val_f1 = val_f1_best       # Keep as record
            best_epoch = epoch
            best_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save({
                'epoch': int(epoch),
                'model': model.state_dict(),
                'val_auc': float(val_auc),
                'val_auprc': float(val_auprc),
                'val_f1_best': float(val_f1_best),
                'val_f1_05': float(val_f1_05),
                'val_best_thr': float(val_best_thr)
            }, best_path)
            logger.info(f"  -> New best model saved (AUPRC: {val_auprc:.4f})")

    
    logger.info("-" * 60)
    logger.info(f"Training completed. Best epoch: {best_epoch}")
    logger.info(f"Best Validation - AUROC: {best_val_auc:.4f} | F1: {best_val_f1:.4f}")
    
    # Load best model for final test
    logger.info("\nLoading best model for final evaluation...")
    best_checkpoint = torch.load(
    os.path.join(ckpt_dir, "best_model.pt"),
    map_location=DEVICE,
    weights_only=False  # Important
)

    model.load_state_dict(best_checkpoint['model'])
    
    # Final test evaluation
    test_auc, test_auprc, test_f1_best, test_f1_05, test_best_thr, y_true, y_pred = eval_metrics(
        model, dataloaders['test'], return_predictions=True
    )

    logger.info("="*60)
    logger.info("FINAL TEST SET RESULTS")
    logger.info("="*60)
    logger.info(f"Test AUROC: {test_auc:.4f}")
    logger.info(f"Test AUPRC: {test_auprc:.4f}")
    logger.info(f"Test F1(best): {test_f1_best:.4f} @thr={test_best_thr:.3f}")
    logger.info(f"Test F1@0.5: {test_f1_05:.4f}")

    # Classification reports under two thresholds
    from sklearn.metrics import classification_report, accuracy_score
    y_pred_bin_best = (y_pred >= test_best_thr).astype(int)
    y_pred_bin_05   = (y_pred >= 0.5).astype(int)

    test_acc_best = accuracy_score(y_true, y_pred_bin_best)
    test_acc_05   = accuracy_score(y_true, y_pred_bin_05)

    logger.info("\nClassification Report @best_thr:")
    logger.info(classification_report(y_true, y_pred_bin_best, target_names=['Survived', 'Died']))

    logger.info("\nClassification Report @0.5:")
    logger.info(classification_report(y_true, y_pred_bin_05, target_names=['Survived', 'Died']))

    
    
    
    
    # Save results summary
    results = {
    'model_type': model_type,
    'timestamp': timestamp,
    'best_epoch': best_epoch,
    'best_val_auc': float(best_val_auc),
    'best_val_auprc': float(best_val_auprc),   # Primary model selection metric
    'best_val_f1': float(best_val_f1),

    'test_auc': float(test_auc),
    'test_auprc': float(test_auprc),           # Added
    'test_f1': float(test_f1_best),            # Keep old key; value is F1(best)
    'test_f1_best': float(test_f1_best),       # added
    'test_f1_05': float(test_f1_05),           # added
    'test_best_thr': float(test_best_thr),     # added
    'test_accuracy': float(test_acc_best),     # Keep old key; accuracy under best_thr.
    'test_accuracy_05': float(test_acc_05),    # Optional

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
    exit(0)

def main():
    parser = argparse.ArgumentParser(description='Train ICU Mortality Prediction Models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, 
                       choices=['mlp', 'transformer', 'lstm', 'retain'],
                       default='transformer', help='Model type to train')
    parser.add_argument('--eval-only', action='store_true',
                       help='Run evaluation only (no training)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run training or evaluation
    train_model(config, args.model, args.eval_only, args.checkpoint)

if __name__ == "__main__":
    main()
