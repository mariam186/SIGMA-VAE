#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 04:42:49 2025

@author: marzab
"""
"""
Brain Surface Visualization for FreeSurfer Parcellations
=========================================================
Maps latent sensitivities to Desikan-Killiany atlas for visualization.

Supports:
- Cortical surface plots (fsaverage)
- Subcortical glass brain
- Combined multi-panel figures for papers

Requires: nilearn, matplotlib
"""
import os
import sys

sys.path.append('./code/') 
#%%
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting, datasets, surface
from nilearn import image as nli

# Import your modules
from model import DecorrelatedMMVAE
from data import DataProcessor
import pandas as pd
import numpy as np

def expand_latents(df, active_indices, total_latents):
    # Metadata columns
    target_latents=[f"latent_{i}" for i in active_indices]
    meta_cols = ['age', 'sex']
    out = df[meta_cols].copy()

    # Create full latent space filled with zeros
    for i in range(total_latents):
        out[f'latent_{i}'] = 0.0

    # Get existing latent columns in order
    existing_latents = sorted(
        [c for c in df.columns if c.startswith('latent_')],
        key=lambda x: int(x.split('_')[1])
    )

    # Map existing latents to target positions
    for src, tgt in zip(existing_latents, target_latents):
        out[tgt] = df[src].values

    return out

class BrainVisualizer:
    def __init__(self, model, processor_path, centile_dir, device='cuda'):
        self.device = device

        self.model=model
        self.model.eval()
        self.latent_dim=model.latent_dim
        # Get active indices for reference
        self.active_indices = model.get_active_indices()
        
        # 2. Load Processor (for inverse scaling)
        print(f"Loading processor from {processor_path}...")
        self.processor = DataProcessor()
        self.processor.load(processor_path)
        
        # 3. Load Centiles (Forward Model Results)
        # Expecting files like: centile_0.5.csv, centile_0.025.csv, etc.
        self.centile_data = {}
        for c in ['0.025', '0.50', '0.975']:
            path = os.path.join(centile_dir, 'centile_'+c+'.csv')
            if os.path.exists(path):
                df_raw= pd.read_csv(path)
                
                
                self.centile_data[c] = expand_latents(df_raw, self.active_indices, self.latent_dim)
                print(f"Loaded centile {c}: {len(self.centile_data[c])} rows")
            else:
                print(f"Warning: Centile file not found: {path}")

    def get_latent_vector(self, age, sex, target_latent_idx=None, target_centile=None):
        """
        Construct a latent vector z for a specific age/sex.
        Base: All latents at 50th centile (Median).
        Perturbation: If target_latent_idx is set, move that specific latent to target_centile.
        """
        # 1. Get Baseline (Median / 50th centile)
        df_50 = self.centile_data['0.50']
        
        # Find row closest to requested age (and matching sex if separated in BLR)
        # Assuming df has 'age' column. If sex is separate, filter by it.
        # Here we just find closest age average.
        idx = (df_50['age'] - age).abs().idxmin()
        row_50 = df_50.iloc[idx]
        
        # Construct base Z vector
        # Note: Your model expects a full latent vector. 
        # Ensure 'latent_0', 'latent_1'... columns match model expectation.
        
        z_values = []
        for i in range(self.model.latent_dim):
            col_name = f'latent_{i}' # Or whatever your column naming is in centile csv
            if col_name in row_50:
                z_values.append(row_50[col_name])
            else:
                z_values.append(0.0) # Fallback for inactive dims
        
        z = np.array(z_values, dtype=np.float32)
        
        # 2. Apply Perturbation (Traverse one latent)
        if target_latent_idx is not None and target_centile is not None:
            df_target = self.centile_data[target_centile]
            row_target = df_target.iloc[idx]
            
            col_name = f'latent_{target_latent_idx}'
            if col_name in row_target:
                print(f"  Perturbing {col_name}: {z[target_latent_idx]:.3f} -> {row_target[col_name]:.3f}")
                z[target_latent_idx] = row_target[col_name]
        
        return torch.tensor(z).unsqueeze(0).to(self.device)

    def decode_brain(self, z):
        """Decode latent z into raw physical units (mm, thickness, etc)."""
        with torch.no_grad():
            # Use the model's decoders
            # Note: We pass z directly. If your model gates z inside forward, 
            # we might need to be careful. Ideally, pass z to specific decoders.
            
            # If your model structure requires Gating, apply it (or skip if analysing raw z)
            # Usually for analysis, we want to see the effect of z directly.
            
            recons = {}
            for mod_name, decoder in self.model.decoders.items():
                out = decoder(z).cpu().numpy()
                
                # Inverse Scale to get real units
                scaler = self.processor.scalers[mod_name]
                out_raw = scaler.inverse_transform(out)
                
                # Convert to Series with feature names
                feat_names = self.processor.feature_names[mod_name]
                recons[mod_name] = pd.Series(out_raw.flatten(), index=feat_names)
                
        return recons

    def plot_cortical_map(self, feature_series, title, threshold=None):
        """
        Map ROI tabular data to a cortical surface for visualization.
        Requires nilearn and fsaverage.
        """
        # Fetch fsaverage
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
        
        # 1. Map ROIs to Surface Vertices
        # We need a parcellation map (e.g., Desikan-Killiany)
        # This assumes your CSV columns match standard FreeSurfer names
        parcellation = datasets.fetch_atlas_surf_destrieux() # Or desikan_killiany
        
        # This is a placeholder logic. In reality, you need to match 
        # your feature_series.index (e.g., "lh_precuneus_thickness") 
        # to the labels in the atlas.
        
        # Example using simple string matching if you have standard names:
        # Create an empty map
        lh_map = np.zeros(fsaverage['pial_left'].shape[0])
        # ... (Matching logic required here depending on your specific column names) ...
        
        print("Note: ROI-to-Vertex mapping requires matching your column names to atlas labels.")
        print("Skipping actual plotting in this snippet to avoid errors.")
        
        # If you have the map:
        # plotting.plot_surf_stat_map(
        #     fsaverage.infl_left, map_data, hemi='left',
        #     title=title, threshold=threshold
        # )

    def visualize_latent_effect(self, age, latent_idx, save_path=None):
        """
        Main function:
        1. Decode Median (50th)
        2. Decode Target (2.5th)
        3. Compute Difference
        4. Plot
        """
        print(f"\n--- Visualizing Latent {latent_idx} at Age {age} ---")
        
        # 1. Get Latents
        z_med = self.get_latent_vector(age, 'Male', None, None) # 50th baseline
        z_low = self.get_latent_vector(age, 'Male', latent_idx, '0.025') # "Atrophy" direction?
        
        # 2. Decode
        rec_med = self.decode_brain(z_med)
        rec_low = self.decode_brain(z_low)
        
        # 3. Compute Difference (Low Centile - Median)
        # Negative values = Features that are smaller in the 'Low' condition (Atrophy)
        diffs = {}
        for mod in rec_med.keys():
            diffs[mod] = rec_low[mod] - rec_med[mod]
            
            # Print top 5 affected regions
            print(f"\nTop changes in {mod}:")
            print(diffs[mod].abs().sort_values(ascending=False).head(5))
            
        # 4. Plotting (Example for simple bar chart if brain plot fails)
        # In a real scenario, pass 'diffs' to the plot_cortical_map function
        fig, axes = plt.subplots(1, len(diffs), figsize=(15, 5))
        if len(diffs) == 1: axes = [axes]
        
        for ax, (mod, diff) in zip(axes, diffs.items()):
            # Plot top 10 regions
            top_diff = diff.abs().sort_values(ascending=False).head(10)
            # Re-sign them
            top_diff = diff[top_diff.index]
            
            top_diff.plot(kind='barh', ax=ax, color='salmon')
            ax.set_title(f"{mod}: 2.5th vs 50th Centile")
            ax.set_xlabel("Difference (mm/vol)")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        plt.show()

# --- USAGE ---
# Edit paths to match your folder structure
if __name__ == "__main__":
    MODEL_PATH = './results/dmvae_output/final_model.pth'
    PROC_PATH = './results/dmvae_output/processor.pkl'
    NM_RES_DIR = './results/dmvae_output/nm_blr/results' # Folder containing centile_X.csv
    
    torch.serialization.add_safe_globals([DecorrelatedMMVAE])
    device='cuda'
    checkpoint=torch.load(os.path.join('./results/dmvae_output/final_model.pth'), map_location=device,weights_only=False)
    MODEL=checkpoint['model_full']
    checkpoint=torch.load(os.path.join('./results/dmvae_output/best_model_p2.pth'), map_location=device,weights_only=False)
    MODEL.load_state_dict(checkpoint['model'])
    viz = BrainVisualizer(MODEL, PROC_PATH, NM_RES_DIR)
    
    # Analyze Latent 0 at Age 70
    viz.visualize_latent_effect(age=.5, latent_idx=19, save_path='latent_0_age70.png')
    
    # You can loop through all active latents
    # for idx in viz.active_indices:
    #     viz.visualize_latent_effect(age=70, latent_idx=idx)
    
