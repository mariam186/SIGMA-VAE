#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 04:23:31 2025

@author: marzab
"""
"""
Latent Interpretation for DMVAE + Normative Model
=================================================
Single script to run - update paths in CONFIG section.

Computes:
1. Jacobian sensitivity (∂decoder/∂z) using healthy test subjects
2. Latent traversals at reference ages using normative centiles
3. Visualization of top features per latent
"""

# %%
import os
import sys

sys.path.append('./code/') 
# =============================================================================
# CONFIG - UPDATE THESE PATHS
# =============================================================================

MODEL_PATH = './results/dmvae_output/best_model_p2.pth'
PROCESSOR_PATH = './results/dmvae_output/processor.pkl'

# Latent data - use TEST for Jacobian (will filter to healthy)
LATENT_TEST_PATH = './results/dmvae_output/latents/latents_test.csv'

# Healthy diagnosis labels in your data (add all that apply)
HEALTHY_LABELS = ['HC', 'healthy', 'control', 'CN', 'Control']

# Normative centiles (from nm_blr.py)
CENTILE_LOW = './results/dmvae_output/nm_blr/results/centile_0.025.csv'
CENTILE_MEDIAN = './results/dmvae_output/nm_blr/results/centile_0.50.csv'
CENTILE_HIGH = './results/dmvae_output/nm_blr/results/centile_0.975.csv'

# Output
OUTPUT_DIR = './results/interpretation/'

# Reference ages for traversal
REFERENCE_AGES = [.30, .50, .70]

# Device
DEVICE = 'cuda'  # 'cuda', 'cpu', or 'mps'

# =============================================================================
# IMPORTS
# =============================================================================

# %%
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

os.makedirs(OUTPUT_DIR, exist_ok=True)
fig_dir = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# %%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_full_z_from_row(row, active_indices, latent_dim):
    """
    Convert row with latent_1, latent_2, ... to full z vector.
    
    Centile files have: latent_1, latent_2, ..., latent_N (renamed from active dims)
    These map to: active_indices[0], active_indices[1], ..., active_indices[N-1]
    
    Args:
        row: Series/dict with latent_1, latent_2, ... values
        active_indices: Original indices in full z space [3, 7, 12, ...]
        latent_dim: Total latent dimensions (e.g., 20)
    
    Returns:
        z: Full latent vector (latent_dim,) with zeros for inactive dims
    """
    z = np.zeros(latent_dim)
    
    for i, orig_idx in enumerate(active_indices):
        col_name = f'latent_{i + 1}'
        if col_name in row.index:
            z[orig_idx] = row[col_name]
    
    return z


def compute_jacobian_at_point(model, z, modality, device):
    """Compute Jacobian ∂decoder/∂z at a specific point."""
    z_tensor = torch.FloatTensor(z).unsqueeze(0).requires_grad_(True).to(device)
    decoder = model.decoders[modality]
    output = decoder(z_tensor)
    
    jacobian = []
    for i in range(output.shape[1]):
        grad = torch.autograd.grad(
            output[0, i], z_tensor,
            retain_graph=True
        )[0]
        jacobian.append(grad.cpu().detach().numpy().flatten())
    
    return np.array(jacobian)  # (n_features, latent_dim)


def compute_average_jacobian(model, z_samples, modality, active_indices, device, n_samples=100):
    """
    Average Jacobian over multiple samples.
    
    Returns Jacobian only for active dimensions: (n_features, n_active)
    """
    if len(z_samples) > n_samples:
        idx = np.random.choice(len(z_samples), n_samples, replace=False)
        z_samples = z_samples[idx]
    
    jacobians = []
    for z in tqdm(z_samples, desc=f"Jacobian [{modality}]"):
        J_full = compute_jacobian_at_point(model, z, modality, device)
        # Extract only active columns
        J_active = J_full[:, active_indices]
        jacobians.append(J_active)
    
    return np.mean(jacobians, axis=0)  # (n_features, n_active)


def decode_latent(model, z, device):
    """Decode a full latent vector to all modalities."""
    z_tensor = torch.FloatTensor(z).unsqueeze(0).to(device)
    with torch.no_grad():
        recons = {}
        for modality in model.modality_names:
            recons[modality] = model.decoders[modality](z_tensor).cpu().numpy().squeeze()
    return recons


def plot_top_features(sensitivity_df, modality, latent_col, active_idx_original, 
                      n_top=15, save_path=None):
    """Plot top features by absolute sensitivity."""
    values = sensitivity_df[latent_col].values
    features = sensitivity_df.index.tolist()
    
    sorted_idx = np.argsort(np.abs(values))[::-1][:n_top]
    top_features = [features[i] for i in sorted_idx]
    top_values = values[sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, max(6, n_top * 0.4)))
    colors = ['#d73027' if v < 0 else '#4575b4' for v in top_values]
    ax.barh(range(len(top_values)), top_values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(top_values)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel('Sensitivity (∂feature/∂latent)')
    ax.set_title(f'{modality.title()} - {latent_col} (original idx: {active_idx_original})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# %%
# =============================================================================
# 1. LOAD MODEL
# =============================================================================
print("\n" + "="*70)
print("1. LOADING MODEL")
print("="*70)
from model import DecorrelatedMMVAE, Loss
torch.serialization.add_safe_globals([DecorrelatedMMVAE])
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

if 'model_full' in checkpoint:
    model = checkpoint['model_full']
    model.load_state_dict(checkpoint['model'])
else:
    from model import DecorrelatedMMVAE

    checkpoint=torch.load(os.path.join('./results/dmvae_output/final_model.pth'), map_location=DEVICE,weights_only=False)
    model=checkpoint['model_full']
    checkpoint=torch.load(os.path.join(MODEL_PATH), map_location=DEVICE,weights_only=False)
    model.load_state_dict(checkpoint['model'])

model.to(DEVICE)
model.eval()

# Get active indices
active_indices = checkpoint.get('active_indices', model.get_active_indices())
n_active = len(active_indices)
latent_dim = model.latent_dim

print(f"Total latent dims: {latent_dim}")
print(f"Active latent dims: {n_active}")
print(f"Active indices: {active_indices}")

# Load processor for feature names
with open(PROCESSOR_PATH, 'rb') as f:
    proc_state = pickle.load(f)

feature_names = proc_state['feature_names']
scalers = proc_state['scalers']

print(f"\nModalities:")
for mod, names in feature_names.items():
    print(f"  {mod}: {len(names)} features")

# %%
# =============================================================================
# 2. LOAD HEALTHY TEST SUBJECTS
# =============================================================================
print("\n" + "="*70)
print("2. LOADING HEALTHY TEST SUBJECTS")
print("="*70)

test_df = pd.read_csv(LATENT_TEST_PATH)
print(f"Total test subjects: {len(test_df)}")

# Check diagnosis column
if 'diagnosis' in test_df.columns:
    print(f"Diagnosis values: {test_df['diagnosis'].unique()}")
    healthy_test = test_df[test_df['diagnosis'].isin(HEALTHY_LABELS)]
    print(f"Healthy test subjects: {len(healthy_test)}")
else:
    print("WARNING: No 'diagnosis' column found, using all subjects")
    healthy_test = test_df

# Build full z vectors from the renamed columns (latent_1, latent_2, ...)
print("\nBuilding full z vectors...")
z_samples = []
for _, row in healthy_test.iterrows():
    z_full = build_full_z_from_row(row, active_indices, latent_dim)
    z_samples.append(z_full)

z_samples = np.array(z_samples)
print(f"z_samples shape: {z_samples.shape}")  # (n_healthy, latent_dim)

# %%
# =============================================================================
# 3. COMPUTE JACOBIAN SENSITIVITY
# =============================================================================
print("\n" + "="*70)
print("3. COMPUTING JACOBIAN SENSITIVITY")
print("="*70)

# Column names for output: latent_1 (idx=3), latent_2 (idx=7), etc.
latent_col_names = [f'latent_{i+1}' for i in range(n_active)]
latent_col_with_orig = [f'latent_{i+1}_idx{active_indices[i]}' for i in range(n_active)]

sensitivity_dict = {}

for modality in model.modality_names:
    print(f"\nProcessing {modality}...")
    
    J_active = compute_average_jacobian(
        model, z_samples, modality, active_indices, DEVICE, n_samples=100
    )
    
    # Create DataFrame
    df = pd.DataFrame(
        J_active,
        columns=latent_col_names,
        index=feature_names.get(modality, [f'feat_{i}' for i in range(J_active.shape[0])])
    )
    
    sensitivity_dict[modality] = df
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, f'sensitivity_{modality}.csv')
    df.to_csv(save_path)
    print(f"  Saved: {save_path}")
    print(f"  Shape: {df.shape}")

# Save index mapping for reference
index_map = pd.DataFrame({
    'column_name': latent_col_names,
    'original_index': active_indices
})
index_map.to_csv(os.path.join(OUTPUT_DIR, 'latent_index_mapping.csv'), index=False)
print(f"\nSaved latent index mapping")

# %%
# =============================================================================
# 4. VISUALIZE SENSITIVITY
# =============================================================================
print("\n" + "="*70)
print("4. GENERATING SENSITIVITY PLOTS")
print("="*70)

for modality, df in sensitivity_dict.items():
    for i, col in enumerate(df.columns):
        save_path = os.path.join(fig_dir, f'sensitivity_{modality}_{col}.png')
        plot_top_features(
            df, modality, col, 
            active_idx_original=active_indices[i],
            n_top=15, 
            save_path=save_path
        )
    print(f"  {modality}: saved {len(df.columns)} plots")

# %%
# =============================================================================
# 5. SUMMARY HEATMAP
# =============================================================================
print("\n" + "="*70)
print("5. CREATING SUMMARY HEATMAP")
print("="*70)

for modality, df in sensitivity_dict.items():
    # Get top features (by max absolute sensitivity across all latents)
    max_sens = df.abs().max(axis=1)
    top_features = max_sens.nlargest(30).index
    df_subset = df.loc[top_features]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 12))
    vmax = np.percentile(np.abs(df_subset.values), 95)
    
    sns.heatmap(
        df_subset,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax, vmax=vmax,
        ax=ax,
        cbar_kws={'label': 'Sensitivity', 'shrink': 0.5}
    )
    ax.set_title(f'{modality.title()} Sensitivity (Top 30 features)')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Brain Feature')
    plt.tight_layout()
    
    save_path = os.path.join(fig_dir, f'heatmap_{modality}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# %%
# =============================================================================
# 6. LATENT TRAVERSALS (Using Normative Centiles)
# =============================================================================
print("\n" + "="*70)
print("6. LATENT TRAVERSAL ANALYSIS")
print("="*70)

try:
    centile_median = pd.read_csv(CENTILE_MEDIAN)
    centile_low = pd.read_csv(CENTILE_LOW)
    centile_high = pd.read_csv(CENTILE_HIGH)
    HAS_CENTILES = True
    print("Loaded normative centiles")
except Exception as e:
    HAS_CENTILES = False
    print(f"Could not load centiles: {e}")
    print("Skipping traversal analysis")

if HAS_CENTILES:
    traversal_dir = os.path.join(OUTPUT_DIR, 'traversals')
    os.makedirs(traversal_dir, exist_ok=True)
    
    for ref_age in REFERENCE_AGES:
        for sex in ['Female', 'Male']:
            print(f"\n  Traversal at age={ref_age}, sex={sex}")
            
            # Find matching row (or closest age)
            mask = (centile_median['age'] == ref_age) & (centile_median['sex'] == sex)
            if mask.sum() == 0:
                closest = centile_median.loc[
                    centile_median['sex'] == sex, 'age'
                ].iloc[(centile_median.loc[centile_median['sex'] == sex, 'age'] - ref_age).abs().argmin()]
                mask = (centile_median['age'] == closest) & (centile_median['sex'] == sex)
                print(f"    Using closest age: {closest}")
            
            row_median = centile_median[mask].iloc[0]
            row_low = centile_low[mask].iloc[0]
            row_high = centile_high[mask].iloc[0]
            
            # Build baseline z
            baseline_z = build_full_z_from_row(row_median, active_indices, latent_dim)
            baseline_decoded = decode_latent(model, baseline_z, DEVICE)
            
            # Traverse each active latent
            effects_data = {mod: {} for mod in model.modality_names}
            
            for i, orig_idx in enumerate(active_indices):
                col = f'latent_{i + 1}'
                
                if col not in row_low.index or col not in row_high.index:
                    continue
                
                # Low traversal
                z_low = baseline_z.copy()
                z_low[orig_idx] = row_low[col]
                decoded_low = decode_latent(model, z_low, DEVICE)
                
                # High traversal
                z_high = baseline_z.copy()
                z_high[orig_idx] = row_high[col]
                decoded_high = decode_latent(model, z_high, DEVICE)
                
                # Store effects
                for mod in model.modality_names:
                    effects_data[mod][f'{col}_low'] = decoded_low[mod] - baseline_decoded[mod]
                    effects_data[mod][f'{col}_high'] = decoded_high[mod] - baseline_decoded[mod]
            
            # Save effects
            for mod in model.modality_names:
                df_effects = pd.DataFrame(
                    effects_data[mod],
                    index=feature_names.get(mod, [f'feat_{i}' for i in range(len(list(effects_data[mod].values())[0]))])
                )
                save_path = os.path.join(traversal_dir, f'traversal_age{ref_age}_{sex}_{mod}.csv')
                df_effects.to_csv(save_path)
            
            print(f"    Saved traversal effects")

# %%
# =============================================================================
# 7. INTERPRETATION TABLE
# =============================================================================
print("\n" + "="*70)
print("7. CREATING INTERPRETATION TABLE")
print("="*70)

rows = []
for i in range(n_active):
    col = f'latent_{i + 1}'
    orig_idx = active_indices[i]
    
    row = {
        'latent': col,
        'original_index': orig_idx
    }
    
    for mod, df in sensitivity_dict.items():
        if col in df.columns:
            values = df[col].values
            features = df.index.tolist()
            
            # Top 3 positive
            top_pos_idx = np.argsort(values)[-3:][::-1]
            top_pos = [f"{features[k]} ({values[k]:.3f})" 
                      for k in top_pos_idx if values[k] > 0]
            
            # Top 3 negative
            top_neg_idx = np.argsort(values)[:3]
            top_neg = [f"{features[k]} ({values[k]:.3f})" 
                      for k in top_neg_idx if values[k] < 0]
            
            row[f'{mod}_positive'] = '; '.join(top_pos) if top_pos else '-'
            row[f'{mod}_negative'] = '; '.join(top_neg) if top_neg else '-'
    
    rows.append(row)

interpretation_df = pd.DataFrame(rows)
save_path = os.path.join(OUTPUT_DIR, 'interpretation_table.csv')
interpretation_df.to_csv(save_path, index=False)
print(f"Saved: {save_path}")

print("\n" + "-"*70)
print("Interpretation Summary:")
print("-"*70)
print(interpretation_df.to_string())

# %%
print("\n" + "="*70)
print("COMPLETE")
print(f"Results saved to: {OUTPUT_DIR}")
print("="*70)
