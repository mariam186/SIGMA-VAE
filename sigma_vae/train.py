#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 16:41:44 2026

@author: marzab
"""
"""
Training Script for Decorrelated Multi-Modal VAE
Edit the CONFIG section below and run with F5 in VSCode.
"""

# %%
import os
import sys
import json
# Make intra-package imports work regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# ▼▼▼ CONFIGURATION - EDIT HERE ▼▼▼
# =============================================================================

# --- Data Paths ---
DATA_PATHS = {
    'subcortical': './data/subcortical_data.csv',
    'cortical': './data/cortical_data.csv',
    'surface': './data/surface_data.csv'
}

TEST_PATHS = {
    'subcortical': './data/subcortical_data_test.csv',
    'cortical': './data/cortical_data_test.csv',
    'surface': './data/surface_data_test.csv'
}

EXTERNAL_PATHS = {
    'subcortical': './data/subcortical_data_external_combined_ids.csv',
    'cortical': './data/cortical_data_external_combined_ids.csv',
    'surface': './data/surface_data_external_combined_ids.csv'
}

OUTPUT_DIR = './results/sigma_03__5_beta_01'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- Model Architecture ---
config = dict(
    # ABLATION: choose one of 'kl', 'kl_l0', 'sigreg', 'full'
    # kl      -> KL reg, no sparsity  (Vanilla MVAE)
    # kl_l0   -> KL reg, with L0
    # sigreg  -> SIGReg, no sparsity
    # full    -> SIGReg + L0          (SIGMA-VAE, default)
    MODEL_VARIANT = 'kl',
    LATENT_DIM = 5,
    ENCODER_HIDDEN = [512,256, 128],
    DECODER_HIDDEN = [128, 256,512],
    
    # --- Loss Weights ---
    RECON_WEIGHT_1 = 10,
    SIGREG_WEIGHT_1 = .75,
    L0_WEIGHT = 5,
    DEC_WEIGHT = 0,

    RECON_WEIGHT_2 = 10,
    SIGREG_WEIGHT_2 = 150,

    # KL weights for ablation variants (KL operates at very different scale than SIGReg)
    KL_WEIGHT_1 = 0.010,   # standard beta=1 for Phase 1
    KL_WEIGHT_2 = 0.010,   # keep same for Phase 2 (SIGReg uses 3000, KL doesn't need that)
    
    # --- Training ---
    BATCH_SIZE = 128,
    LEARNING_RATE = 1e-3,
    VAL_SPLIT = 0.05,
    
    
    # Phase 1: Structure Discovery
    P1_EPOCHS = 10000,
    L0_WARMUP = 50,
    L0_RAMPUP = 50,
    MIN_EPOCHS_BEFORE_STOPPING = 500,
    EARLY_STOP = 15,
    
    TEMP_ANNEAL_START = 200,#L0_WARMUP + L0_RAMPUP  # start after L0 is fully ramped
    TEMP_ANNEAL_EPOCHS = 250 ,                   # anneal over 200 epochs
    TEMP_FINAL = 0.07,                            # final temperature before freeze
    
    # Phase 2: Fine-tuning
    P2_EPOCHS = 1000,
    P2_LEARNING_RATE = 5e-3)
locals().update(config)
OUTPUT_DIR_VARIANT = OUTPUT_DIR + f'_{MODEL_VARIANT}'
os.makedirs(OUTPUT_DIR_VARIANT, exist_ok=True)

json.dump(config, open(OUTPUT_DIR_VARIANT+'/config.json', 'w'), indent=2)

# =============================================================================
# ▲▲▲ END CONFIGURATION ▲▲▲
# =============================================================================

# %%
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from models import ConditionalMMVAE
from losses import DecorrelatedVAELoss, LossWeights
from trainer import train_model, extract_latents, save_checkpoint
from data import DataProcessor, create_dataloaders, MultiModalDataset

# %%
print("\n" + "="*70)
print("SIGMA-VAE: Sketched Isotropic Gaussian Multi-View Autoencoder")
print("="*70)



# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# %%
# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n" + "-"*70)
print("STEP 1: Loading Data")
print("-"*70)

processor = DataProcessor(output_dir=OUTPUT_DIR, scaler_type='robust')
data_dict = processor.load_train_data(DATA_PATHS, val_split=VAL_SPLIT)
train_loader, val_loader = create_dataloaders(data_dict, batch_size=BATCH_SIZE)
processor.save()

# %%
# =============================================================================
# 2. CREATE MODEL
# =============================================================================
print("\n" + "-"*70)
print("STEP 2: Creating Model")
print("-"*70)

feature_dims = data_dict['metadata']['feature_dims']
print(f"Input dims: {feature_dims}")
print(f"Latent dim: {LATENT_DIM}")

model = ConditionalMMVAE(
    input_dims=feature_dims,
    latent_dim=LATENT_DIM,
    encoder_hidden=ENCODER_HIDDEN,
    decoder_hidden=DECODER_HIDDEN,
    covariate_dim=2  # age + sex
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# ABLATION VARIANT SETUP
# Sets regularization and sparsity based on MODEL_VARIANT
# =============================================================================
print(f"\nModel variant: {MODEL_VARIANT}")

USE_SIGREG = MODEL_VARIANT in ('sigreg', 'full')
USE_L0     = MODEL_VARIANT in ('kl_l0', 'full')

# For no-L0 variants: freeze all gates open immediately (no sparsity training)
if not USE_L0:
    model.gate.freeze_all_active()


LATENT_DIR = OUTPUT_DIR_VARIANT + '/latents/'
os.makedirs(LATENT_DIR, exist_ok=True)
print(f"  SIGReg: {USE_SIGREG} | L0 sparsity: {USE_L0}")
print(f"  Output: {OUTPUT_DIR_VARIANT}")

# Save config to variant-specific directory (so each ablation has its own config)
json.dump(config, open(os.path.join(OUTPUT_DIR_VARIANT, 'config.json'), 'w'), indent=2)

# %%
# =============================================================================
# 3. PHASE 1: STRUCTURE DISCOVERY
# =============================================================================
print("\n" + "="*70)
print("PHASE 1: STRUCTURE DISCOVERY")
print("="*70)

weights = LossWeights(
    recon=RECON_WEIGHT_1,
    sigreg=SIGREG_WEIGHT_1 if USE_SIGREG else 0.0,
    kl=KL_WEIGHT_1 if not USE_SIGREG else 0.0,  # KL scale is different from SIGReg
    l0=L0_WEIGHT if USE_L0 else 0.0,
    decorr=DEC_WEIGHT
)

criterion = DecorrelatedVAELoss(latent_dim=LATENT_DIM, weights=weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, min_lr=5e-6)

history_p1 = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=P1_EPOCHS,
    l0_warmup_epochs=L0_WARMUP,
    l0_rampup_epochs=L0_RAMPUP,
    min_epochs_before_stopping=MIN_EPOCHS_BEFORE_STOPPING,
    early_stop_patience=EARLY_STOP,
    temp_anneal_start=TEMP_ANNEAL_START,
    temp_anneal_epochs=TEMP_ANNEAL_EPOCHS,
    temp_final=TEMP_FINAL,
    scheduler=scheduler,
    save_path=os.path.join(OUTPUT_DIR_VARIANT, 'best_model_p1.pth')
)

# %
checkpoint=torch.load(OUTPUT_DIR_VARIANT+'/best_model_p1.pth',map_location=device,weights_only=False)
model=checkpoint['model_full']

# %
# =============================================================================
# 4. FREEZE GATES
# =============================================================================
print("\n" + "="*70)
print("FREEZING LATENT STRUCTURE")
print("="*70)

if USE_L0:
    # L0 variants: freeze based on learned alpha values
    model.gate.freeze()
else:
    # No-L0 variants: gate was already frozen all-active before Phase 1.
    # Calling freeze() here would zero out all dims (sigmoid(0)=0.5 is not > 0.5).
    # Re-call freeze_all_active() to be safe after checkpoint reload.
    model.gate.freeze_all_active()
active_dims = model.get_active_dims()
print(f"Locked: {active_dims} active dimensions")

# 
# =============================================================================s
# 5. PHASE 2: FINE-TUNING
# =============================================================================
print("\n" + "="*70)
print(f"PHASE 2: FINE-TUNING ({active_dims} dims)")
print("="*70)
weights_p2 = LossWeights(
    recon=RECON_WEIGHT_2,
    sigreg=SIGREG_WEIGHT_2 if USE_SIGREG else 0.0,
    kl=KL_WEIGHT_2 if not USE_SIGREG else 0.0,
    l0=L0_WEIGHT if USE_L0 else 0.0,
)
P2_LEARNING_RATE = 5e-6

criterion = DecorrelatedVAELoss(latent_dim=LATENT_DIM, weights=weights_p2)
# Fresh optimizer with lower LR, excluding frozen gate
optimizer_p2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=P2_LEARNING_RATE,
    weight_decay=1e-5
)

# Trainer auto-detects frozen gate and disables L0
history_p2 = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,  # Reuse same criterion, L0 auto-disabled
    optimizer=optimizer_p2,
    device=device,
    num_epochs=P2_EPOCHS,
    early_stop_patience=EARLY_STOP,
    min_epochs_before_stopping=MIN_EPOCHS_BEFORE_STOPPING,
    save_path=os.path.join(OUTPUT_DIR_VARIANT, 'best_model_p2.pth')
)

# Save final model
save_checkpoint(
    model, optimizer_p2, P2_EPOCHS,
    path=os.path.join(OUTPUT_DIR_VARIANT, 'final_model.pth')
)

print("\n✓ Training complete!")

# 
# =============================================================================
# 6. LOAD BEST MODEL & EVALUATE
# =============================================================================
print("\n" + "-"*70)
print("STEP 6: Loading Best Model & Extracting Latents")
print("-"*70)

# from trainer import load_checkpoint«

# torch.serialization.add_safe_globals([DecorrelatedMMVAE])
# load_checkpoint(model, os.path.join(OUTPUT_DIR, 'best_model_p2.pth'), device=device)

# Quick validation evaluation

results = extract_latents(model, val_loader, device)
z_val = results['z']
mu_val = results['mu']
logvar_val = results['logvar']

print(f"Latents shape: {z_val.shape}")
print(f"Active dims: {model.get_active_dims()}/{LATENT_DIM}")

# Compute W-scores
from trainer import compute_wscores
wscores_val = compute_wscores(mu_val, logvar_val)
print(f"W-scores shape: {wscores_val.shape}")
print(f"W-score range: [{wscores_val.min():.3f}, {wscores_val.max():.3f}]")

# Check independence
active_idx = model.get_active_indices()
z_active = z_val[:, active_idx]
if z_active.shape[1] > 1:
    corr_matrix = np.corrcoef(z_active.T)
    off_diag = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
    print(f"\nLatent Independence:")
    print(f"  Mean off-diag corr: {np.mean(np.abs(off_diag)):.3f}")
    print(f"  Max off-diag corr: {np.max(np.abs(off_diag)):.3f}")

from diagnostics import check_disentanglement

disentangle_results = check_disentanglement(
    z=z_val,
    age=data_dict['age_val'],
    sex=data_dict['sex_val'],
    active_indices=active_idx,
    cv_folds=5,
    verbose=True
)

# 
# =============================================================================
# 7. EXTRACT LATENTS FOR ALL DATASETS
# =============================================================================
print("\n" + "-"*70)
print("STEP 7: Extracting Latents for All Datasets")
print("-"*70)

from evaluation import Evaluator
from trainer import save_latents_to_csv

evaluator = Evaluator(model, device, output_dir=OUTPUT_DIR_VARIANT + '/evaluation')

DATASETS = {
    "train": DATA_PATHS,
    "test": TEST_PATHS,
    "external": EXTERNAL_PATHS
}

for split_name, split_paths in DATASETS.items():
    print(f"\n=== Processing {split_name} dataset ===")
    
    split_dict = processor.load_test_data(split_paths)
    dataset = MultiModalDataset(split_dict['data'], split_dict['age'], split_dict['sex'])
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Save latents
    latent_path = os.path.join(LATENT_DIR, f'latents_{split_name}.csv')
    save_latents_to_csv(model, loader, split_dict, latent_path, device)
    
    # Evaluate
    evaluator.evaluate(loader, split_dict['age'], split_dict['sex'], split_name)

#
# =============================================================================
# 8. TRIM LATENT FILES TO ACTIVE DIMS ONLY
# =============================================================================
print("\n" + "-"*70)
print("STEP 8: Trimming Latent Files")
print("-"*70)

import pandas as pd

active_indices = model.get_active_indices()
latent_cols = [f'latent_{i}' for i in active_indices]
wscore_cols = [f'wscore_{i}' for i in active_indices]
keep_cols = ['subject_id', 'age', 'sex', 'dataset_name', 'diagnosis'] + latent_cols + wscore_cols

for split_name in DATASETS.keys():
    latent_path = os.path.join(LATENT_DIR, f'latents_{split_name}.csv')
    df = pd.read_csv(latent_path)
    df = df[keep_cols]
    df.to_csv(latent_path, index=False)
    print(f"  Trimmed {split_name}: {df.shape}")

print("\n" + "="*70)
print("PIPELINE COMPLETE")
print(f"Results saved to: {OUTPUT_DIR_VARIANT}")
print("="*70)
# %%