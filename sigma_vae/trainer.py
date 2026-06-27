#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 02:42:20 2026

@author: marzab
"""
"""
Training Functions for Conditional Multi-Modal VAE
Single unified trainer with frozen_gate option.

CHANGES FROM ORIGINAL:
- train_epoch: constructs covariates c, passes to model(x_dict, c)
- train_epoch: criterion call simplified (no age_true, sex_true)
- train_epoch: removed age/sex from sums dict
- evaluate: constructs covariates c, passes to model(x_dict, c)
- evaluate: criterion call simplified
- train_model: removed age/sex from history
- extract_latents: constructs c, passes to model, returns mu/logvar for W-score
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, List
import pandas as pd


# ==============================================================================
# 1. UTILITIES (UNCHANGED)
# ==============================================================================

def save_checkpoint(model, optimizer, epoch: int, path: str, extra: Optional[Dict] = None):
    """Save a checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'model_full': model,
        'optimizer_state': optimizer.state_dict(),
        'active_dims': model.get_active_dims(),
        'active_indices': model.get_active_indices(),
        'gate_frozen': model.gate.is_frozen
    }
    if extra:
        checkpoint.update(extra)
    
    torch.save(checkpoint, path)
    print(f"💾 Saved checkpoint: {path} (epoch {epoch}, active_dims={checkpoint['active_dims']})")


def load_checkpoint(model, path: str, optimizer=None, device: str = 'cpu') -> Dict:
    """Load a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    print(f"✓ Loaded: {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


def check_for_nan(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check if tensor contains NaN or Inf."""
    if tensor is None:
        return False
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️ Inf detected in {name}")
        return True
    return False


# ==============================================================================
# 2. CORE TRAINING FUNCTIONS
# ==============================================================================

def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: str,
    check_nan: bool = False
) -> Dict[str, float]:
    """Train for one epoch. Returns dict of averaged metrics.
    
    CHANGES:
    - Constructs covariates c from age and sex
    - Passes c to model: model(x_dict, c)
    - Simplified criterion call (no age_true, sex_true)
    - Removed age/sex from sums
    """
    model.train()
    
    # CHANGED: removed 'age' and 'sex' from sums
    sums = {'total': 0.0, 'recon': 0.0, 'sigreg': 0.0, 'l0': 0.0, 'kl': 0.0}
    n_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Pop age and sex from batch
        age_true = batch.pop('age', None)
        sex_true = batch.pop('sex', None)
        
        # Move data to device
        x_dict = {name: x.to(device) for name, x in batch.items()}
        if age_true is not None:
            age_true = age_true.to(device)
        if sex_true is not None:
            sex_true = sex_true.to(device)
        
        # ADDED: Construct covariates tensor [batch, 2]
        c = torch.stack([age_true, sex_true], dim=-1)
        
        # NaN check on inputs
        if check_nan and batch_idx == 0:
            for name, x in x_dict.items():
                if check_for_nan(x, f"input.{name}"):
                    raise ValueError(f"NaN in input: {name}")
            if check_for_nan(c, "covariates"):
                raise ValueError("NaN in covariates")
        
        # CHANGED: Pass covariates to model
        output = model(x_dict, c)
        
        # NaN check on output
        if check_nan and batch_idx == 0:
            if check_for_nan(output.get('z'), "z"):
                raise ValueError("NaN in latent z")
        
        # CHANGED: Simplified criterion call (no age_true, sex_true)
        # losses = criterion(x_dict, output, model)
        losses = criterion(x_dict, output,age_true,sex_true, model)
        
        total_loss = losses['total']
        
        if check_nan and check_for_nan(total_loss, "total_loss"):
            print("Loss breakdown:", {k: v.item() for k, v in losses.items()})
            raise ValueError("NaN in loss")
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate metrics
        sums['total'] += total_loss.item()
        sums['recon'] += losses['recon'].item()
        sums['sigreg'] += losses.get('sigreg', torch.zeros(())).item()
        sums['l0'] += losses.get('l0', torch.zeros(())).item()
        sums['kl'] += losses.get('kl', torch.zeros(())).item()
        # REMOVED: age and sex accumulation
        n_batches += 1
    
    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str
) -> float:
    """Evaluate model, return average total loss.
    
    CHANGES:
    - Constructs covariates c from age and sex
    - Passes c to model: model(x_dict, c)
    - Simplified criterion call
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Pop age and sex from batch
        age_true = batch.pop('age', None)
        sex_true = batch.pop('sex', None)
        
        # Move data to device
        x_dict = {name: x.to(device) for name, x in batch.items()}
        if age_true is not None:
            age_true = age_true.to(device)
        if sex_true is not None:
            sex_true = sex_true.to(device)
        
        # ADDED: Construct covariates tensor
        c = torch.stack([age_true, sex_true], dim=-1)
        
        # CHANGED: Pass covariates to model
        output = model(x_dict, c)
        
        # CHANGED: Simplified criterion call
        # losses = criterion(x_dict, output, model)
        losses = criterion(x_dict, output,age_true,sex_true, model)
        
        total_loss += losses['total'].item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


# ==============================================================================
# 3. UNIFIED TRAINER
# ==============================================================================

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer,
    device: str = 'cuda',
    num_epochs: int = 100,
    # L0 schedule (ignored if gate is frozen)
    l0_warmup_epochs: int = 20,
    l0_rampup_epochs: int = 20,
    # Temperature annealing
    temp_anneal_start: int = None,
    temp_anneal_epochs: int = 100,
    temp_final: float = 0.1,
    # Checkpointing
    save_path: Optional[str] = None,
    # Scheduling
    scheduler=None,
    early_stop_patience: int = 30,
    min_epochs_before_stopping: int = 0,
    log_every: int = 5
) -> Dict[str, List]:
    """
    Unified training loop for Conditional MMVAE.
    
    Automatically detects gate state from model.gate.is_frozen:
    - If frozen: disables L0, skips dimension tracking
    - If unfrozen: applies L0 warmup/rampup schedule
    
    CHANGES:
    - Removed 'age' and 'sex' from history dict
    
    Args:
        model: Conditional MMVAE model (gate state auto-detected)
        train_loader: Training data
        val_loader: Validation data
        criterion: Loss function
        optimizer: Optimizer
        device: 'cuda' or 'cpu'
        num_epochs: Maximum epochs
        l0_warmup_epochs: Epochs before L0 starts (ignored if gate frozen)
        l0_rampup_epochs: Epochs to ramp L0 to target (ignored if gate frozen)
        temp_anneal_start: Epoch to start temperature annealing
        temp_anneal_epochs: Epochs over which to anneal temperature
        temp_final: Final temperature value
        save_path: Path to save best checkpoint
        scheduler: LR scheduler (optional)
        early_stop_patience: Stop if no improvement for this many epochs
        min_epochs_before_stopping: Don't stop or save before this epoch
        log_every: Print progress every N epochs
    
    Returns:
        History dict with losses per epoch
    """
    model.to(device)
    criterion.to(device)
    
    # --- Auto-detect mode from model state ---
    frozen_gate = model.gate.is_frozen
    
    if frozen_gate:
        criterion.weights.l0 = 0.0
        target_l0 = 0.0
        phase_name = "Frozen Gate Fine-tuning"
    else:
        target_l0 = criterion.weights.l0
        phase_name = "Gate Learning"
    
    # --- History ---
    # CHANGED: removed 'age' and 'sex'
    history = {
        'train_loss': [], 'val_loss': [],
        'recon': [], 'sigreg': [], 'l0': [], 'kl': []
    }
    if not frozen_gate:
        history['active_dims'] = []
    
    best_val_loss = float('inf')
    last_active_dim = int(1000)
    epochs_no_improve = 0
    
    # --- Print header ---
    print(f"\n{'='*60}")
    print(f"Training: {phase_name}")
    print(f"{'='*60}")
    print(f"Device: {device} | Epochs: {num_epochs}")
    if frozen_gate:
        print(f"Active dims: {model.get_active_dims()} (fixed)")
    else:
        print(f"L0 warmup: {l0_warmup_epochs} epochs | L0 rampup: {l0_rampup_epochs} epochs")
        print(f"Target L0 weight: {target_l0}")
        if temp_anneal_start is not None:
            print(f"Temp annealing: starts at epoch {temp_anneal_start}, "
                  f"over {temp_anneal_epochs} epochs, final={temp_final}")
    print(f"{'='*60}\n")
    
    # --- Training loop ---
    for epoch in range(num_epochs):
        
        # L0 schedule (only if learning gate)
        if not frozen_gate:
            if epoch < l0_warmup_epochs:
                criterion.weights.l0 = 0.0
            else:
                progress = min(1.0, (epoch - l0_warmup_epochs) / max(l0_rampup_epochs, 1))
                criterion.weights.l0 = target_l0 * progress
        
        # Temperature annealing (only if learning gate)
        if not frozen_gate and temp_anneal_start is not None:
            if epoch >= temp_anneal_start:
                temp_progress = min(1.0, (epoch - temp_anneal_start) / max(temp_anneal_epochs, 1))
                new_temp = 0.67 - (0.67 - temp_final) * temp_progress
                model.gate.temperature = new_temp
        
        # Train
        try:
            metrics = train_epoch(
                model, train_loader, criterion, optimizer, device,
                check_nan=(epoch == 0)
            )
        except ValueError as e:
            print(f"\n❌ Training stopped: {e}")
            break
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Track history
        history['train_loss'].append(metrics['total'])
        history['val_loss'].append(val_loss)
        history['recon'].append(metrics['recon'])
        history['sigreg'].append(metrics['sigreg'])
        history['l0'].append(metrics['l0'])
        history['kl'].append(metrics['kl'])
        # REMOVED: age and sex history tracking
        
        if not frozen_gate:
            history['active_dims'].append(model.get_active_dims())
        
        if model.get_active_dims() < last_active_dim:
            epochs_no_improve = 0
            last_active_dim = model.get_active_dims()
            
        # Save best (after min_epochs)
        if val_loss < best_val_loss and epoch >= min_epochs_before_stopping:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if save_path:
                save_checkpoint(model, optimizer, epoch, save_path)
        else:
            if epoch >= min_epochs_before_stopping:
                epochs_no_improve += 1
        
        # Log
        if (epoch + 1) % log_every == 0 or epoch == 0:
            dims_str = f"Dims: {model.get_active_dims():2d}/{model.latent_dim}" if not frozen_gate else ""
            l0w_str = f"L0w: {criterion.weights.l0:.4f}" if not frozen_gate else ""
            temp_str = f"T: {model.gate.temperature:.3f}" if not frozen_gate and temp_anneal_start else ""
            
            print(f"Ep {epoch+1:3d}/{num_epochs} | "
                  f"Train: {metrics['total']:.3f} | Val: {val_loss:.3f} | "
                  f"Recon: {metrics['recon']:.3f} | SIG: {metrics['sigreg']:.4f} | "
                  f"{dims_str} {l0w_str} {temp_str}".strip())
        
        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"\n⏹ Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n✓ Training complete. Active dims: {model.get_active_dims()}")
    return history


# ==============================================================================
# 4. LATENT EXTRACTION
# ==============================================================================

@torch.no_grad()
def extract_latents(
    model: nn.Module,
    dataloader,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """Extract latent representations from model.
    
    CHANGES:
    - Constructs covariates c from age and sex
    - Passes c to model
    - Returns mu and logvar (for W-score computation)
    - Removed age_pred and sex_pred
    """
    model.eval()
    model.to(device)
    
    latents = []
    mus = []
    logvars = []
    
    for batch in tqdm(dataloader, desc="Extracting latents"):
        # Pop age and sex, but keep them for covariates
        age = batch.pop('age').to(device)
        sex = batch.pop('sex').to(device)
        x_dict = {k: v.to(device) for k, v in batch.items()}
        
        # Construct covariates
        c = torch.stack([age, sex], dim=-1)
        
        # Forward pass with covariates
        out = model(x_dict, c)
        
        # Collect outputs
        latents.append(out['z'].cpu().numpy())
        mus.append(out['mu'].cpu().numpy())
        logvars.append(out['logvar'].cpu().numpy())
    
    return {
        'z': np.vstack(latents),
        'mu': np.vstack(mus),
        'logvar': np.vstack(logvars)
    }


def compute_wscores(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """
    Compute W-scores (uncertainty-aware deviation metric).
    
    W_ij = mu_ij / sqrt(1 + sigma_ij^2)
    
    Args:
        mu: Posterior means [N, K]
        logvar: Posterior log-variances [N, K]
    
    Returns:
        W-scores [N, K]
    """
    sigma_sq = np.exp(logvar)
    return mu / np.sqrt(1 + sigma_sq)


def save_latents_to_csv(
    model: nn.Module,
    dataloader,
    metadata: Dict[str, np.ndarray],
    output_path: str,
    device: str = 'cuda'
) -> pd.DataFrame:
    """Extract latents and save with metadata to CSV.
    
    CHANGES:
    - Computes and saves W-scores alongside raw latents
    - Removed age_pred and sex_pred
    """
    results = extract_latents(model, dataloader, device)
    z = results['z']
    mu = results['mu']
    logvar = results['logvar']
    
    # Compute W-scores
    wscores = compute_wscores(mu, logvar)
    
    # Create DataFrame with latent dimensions
    latent_cols = [f'latent_{i}' for i in range(z.shape[1])]
    df = pd.DataFrame(z, columns=latent_cols)
    
    # Add W-score columns
    wscore_cols = [f'wscore_{i}' for i in range(wscores.shape[1])]
    for i, col in enumerate(wscore_cols):
        df[col] = wscores[:, i]
    
    # Add metadata
    df.insert(0, 'subject_id', metadata['subject_id'])
    df['age'] = metadata['age']
    df['sex'] = metadata['sex']
    df['diagnosis'] = metadata.get('diagnosis')
    df['dataset_name'] = metadata.get('dataset_name')
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved latents: {output_path}")
    print(f"  Shape: {df.shape} | Active dims: {model.get_active_dims()}")
    
    return df