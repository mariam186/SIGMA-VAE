#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 04:45:58 2026

@author: marzab
"""
"""
Visualization and Analysis Functions for Conditional MVAE
Supports train/test/external datasets with automatic saving.

CHANGES FROM ORIGINAL:
- Model now requires covariates c = [age, sex] in forward pass
- Removed age/sex prediction metrics (model no longer predicts these)
- Added W-score computation and visualization
- Added disentanglement diagnostic integration
- Updated all extraction functions to pass covariates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
import os
import json


class Evaluator:
    """
    Comprehensive model evaluation with saving capabilities.
    Handles train, test, and external datasets.
    
    Updated for Conditional MVAE:
    - Passes covariates to model
    - Computes W-scores
    - Runs disentanglement diagnostics
    """
    
    def __init__(self, model, device, output_dir):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate(self, dataloader, age_true, sex_true, dataset_name='test'):
        """
        Full evaluation: reconstruction + latent quality + disentanglement.
        
        Args:
            dataloader: DataLoader for the dataset
            age_true: True age values (normalized)
            sex_true: True sex values (0/1)
            dataset_name: 'train', 'val', 'test', or 'external'
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Extract latents and reconstructions
        results = self._extract_all(dataloader)
        latents = results['latents']
        recon_results = results['reconstructions']
        
        # Compute metrics
        metrics = {
            'dataset': dataset_name,
            'n_samples': len(age_true),
            'reconstruction': {},
            'latent_quality': {},
            'wscores': {},
            'disentanglement': {}
        }
        
        # 1. Reconstruction metrics per modality
        print(f"\n--- Reconstruction ---")
        for mod_name, res in recon_results.items():
            mse = mean_squared_error(res['true'], res['pred'])
            r2 = r2_score(res['true'], res['pred'])
            feature_mse = np.mean((res['true'] - res['pred']) ** 2, axis=0)
            
            metrics['reconstruction'][mod_name] = {
                'mse': float(mse),
                'r2': float(r2),
                'feature_mse_min': float(np.min(feature_mse)),
                'feature_mse_max': float(np.max(feature_mse)),
                'feature_mse_mean': float(np.mean(feature_mse))
            }
            print(f"  {mod_name}: MSE={mse:.4f}, R²={r2:.4f}")
        
        # 2. Latent quality
        print(f"\n--- Latent Quality ---")
        z = latents['z']
        mu = latents['mu']
        logvar = latents['logvar']
        active_idx = self.model.get_active_indices()
        
        metrics['latent_quality']['total_dims'] = z.shape[1]
        metrics['latent_quality']['active_dims'] = len(active_idx)
        metrics['latent_quality']['active_indices'] = active_idx
        
        if len(active_idx) > 1:
            z_active = z[:, active_idx]
            corr_matrix = np.corrcoef(z_active.T)
            off_diag = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
            
            metrics['latent_quality']['mean_abs_correlation'] = float(np.mean(np.abs(off_diag)))
            metrics['latent_quality']['max_abs_correlation'] = float(np.max(np.abs(off_diag)))
            
            print(f"  Active dims: {len(active_idx)}/{z.shape[1]}")
            print(f"  Mean |off-diag corr|: {np.mean(np.abs(off_diag)):.3f}")
            print(f"  Max |off-diag corr|: {np.max(np.abs(off_diag)):.3f}")
        
        # 3. W-scores
        print(f"\n--- W-Scores ---")
        wscores = self._compute_wscores(mu, logvar)
        
        metrics['wscores']['mean'] = float(np.mean(wscores))
        metrics['wscores']['std'] = float(np.std(wscores))
        metrics['wscores']['min'] = float(np.min(wscores))
        metrics['wscores']['max'] = float(np.max(wscores))
        
        # Per-dimension stats for active dims
        if len(active_idx) > 0:
            wscores_active = wscores[:, active_idx]
            metrics['wscores']['per_dim_mean'] = [float(m) for m in np.mean(wscores_active, axis=0)]
            metrics['wscores']['per_dim_std'] = [float(s) for s in np.std(wscores_active, axis=0)]
        
        print(f"  W-score range: [{np.min(wscores):.3f}, {np.max(wscores):.3f}]")
        print(f"  W-score mean: {np.mean(wscores):.3f} ± {np.std(wscores):.3f}")
        
        # 4. Disentanglement diagnostic
        print(f"\n--- Disentanglement Check ---")
        try:
            from diagnostics import check_disentanglement
            disentangle_results = check_disentanglement(
                z=z,
                age=age_true,
                sex=sex_true,
                active_indices=active_idx,
                cv_folds=5,
                verbose=True
            )
            metrics['disentanglement'] = {
                'age_r': float(disentangle_results['age_r']),
                'age_r2': float(disentangle_results['age_r2']),
                'sex_acc': float(disentangle_results['sex_acc']),
                'sex_chance': float(disentangle_results['sex_chance']),
                'age_disentangled': disentangle_results['age_disentangled'],
                'sex_disentangled': disentangle_results['sex_disentangled'],
                'fully_disentangled': disentangle_results['fully_disentangled']
            }
        except ImportError:
            print("  Warning: diagnostics.py not found, skipping disentanglement check")
            metrics['disentanglement'] = {'error': 'diagnostics module not found'}
        
        # Save metrics
        self._save_metrics(metrics, dataset_name)
        
        # Generate plots
        self._generate_plots(
            recon_results, z, mu, logvar, wscores,
            age_true, sex_true, active_idx, dataset_name
        )
        
        return metrics
    
    def _extract_all(self, dataloader):
        """Extract latents, reconstructions, mu, and logvar."""
        self.model.eval()
        self.model.to(self.device)
        
        true_data = {name: [] for name in self.model.modality_names}
        pred_data = {name: [] for name in self.model.modality_names}
        latents = []
        mus = []
        logvars = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting"):
                # Keep age and sex for covariates
                age = batch.pop('age').to(self.device)
                sex = batch.pop('sex').to(self.device)
                
                # Construct covariates
                c = torch.stack([age, sex], dim=-1)
                
                x_dict = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with covariates
                out = self.model(x_dict, c)
                
                # Latents
                latents.append(out['z'].cpu().numpy())
                mus.append(out['mu'].cpu().numpy())
                logvars.append(out['logvar'].cpu().numpy())
                
                # Reconstructions
                for name in self.model.modality_names:
                    true_data[name].append(x_dict[name].cpu().numpy())
                    pred_data[name].append(out['recons'][name].cpu().numpy())
        
        # Aggregate reconstructions
        recon_results = {}
        for name in self.model.modality_names:
            recon_results[name] = {
                'true': np.vstack(true_data[name]),
                'pred': np.vstack(pred_data[name])
            }
        
        return {
            'latents': {
                'z': np.vstack(latents),
                'mu': np.vstack(mus),
                'logvar': np.vstack(logvars)
            },
            'reconstructions': recon_results
        }
    
    def _compute_wscores(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """
        Compute W-scores (uncertainty-aware deviation metric).
        
        W_ij = mu_ij / sqrt(1 + sigma_ij^2)
        """
        sigma_sq = np.exp(logvar)
        return mu / np.sqrt(1 + sigma_sq)
    
    def _save_metrics(self, metrics, dataset_name):
        """Save metrics as JSON and CSV."""
        # JSON (full metrics)
        json_path = os.path.join(self.output_dir, f'{dataset_name}_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved: {json_path}")
        
        # CSV (summary row for easy comparison)
        summary = {
            'dataset': dataset_name,
            'n_samples': metrics['n_samples'],
            'active_dims': metrics['latent_quality']['active_dims'],
            'mean_latent_corr': metrics['latent_quality'].get('mean_abs_correlation', np.nan),
            'max_latent_corr': metrics['latent_quality'].get('max_abs_correlation', np.nan),
            'wscore_mean': metrics['wscores']['mean'],
            'wscore_std': metrics['wscores']['std'],
        }
        
        # Add disentanglement metrics
        if 'age_r' in metrics['disentanglement']:
            summary['disentangle_age_r'] = metrics['disentanglement']['age_r']
            summary['disentangle_sex_acc'] = metrics['disentanglement']['sex_acc']
            summary['fully_disentangled'] = metrics['disentanglement']['fully_disentangled']
        
        # Add per-modality reconstruction
        for mod_name, mod_metrics in metrics['reconstruction'].items():
            summary[f'{mod_name}_r2'] = mod_metrics['r2']
            summary[f'{mod_name}_mse'] = mod_metrics['mse']
        
        # Append to summary CSV
        csv_path = os.path.join(self.output_dir, 'performance_summary.csv')
        df_new = pd.DataFrame([summary])
        
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            # Remove old entry for same dataset
            df_existing = df_existing[df_existing['dataset'] != dataset_name]
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        
        df.to_csv(csv_path, index=False)
        print(f"✓ Summary updated: {csv_path}")
    
    def _generate_plots(self, recon_results, z, mu, logvar, wscores,
                        age_true, sex_true, active_idx, dataset_name):
        """Generate and save all plots."""
        fig_dir = os.path.join(self.output_dir, f'{dataset_name}_figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # 1. Reconstruction scatter per modality
        for mod_name, res in recon_results.items():
            self._plot_reconstruction(res['true'], res['pred'], mod_name, 
                                     dataset_name, fig_dir)
        
        # 2. Latent correlation matrix
        if len(active_idx) > 1:
            self._plot_latent_correlations(z[:, active_idx], dataset_name, fig_dir)
        
        # 3. W-score distribution
        self._plot_wscore_distribution(wscores, active_idx, dataset_name, fig_dir)
        
        # 4. Disentanglement plots (latent vs age/sex)
        self._plot_disentanglement(z, age_true, sex_true, active_idx, dataset_name, fig_dir)
        
        print(f"✓ Figures saved: {fig_dir}/")
    
    def _plot_reconstruction(self, true, pred, mod_name, dataset_name, fig_dir):
        """Reconstruction scatter plot."""
        t = true.flatten()
        p = pred.flatten()
        
        if len(t) > 10000:
            idx = np.random.choice(len(t), 10000, replace=False)
            t, p = t[idx], p[idx]
        
        r2 = r2_score(true, pred)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(t, p, alpha=0.1, s=1, c='blue')
        
        min_val, max_val = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_title(f'{dataset_name.upper()} - {mod_name}\nR² = {r2:.3f}')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'recon_{mod_name}.png'), dpi=150)
        plt.close()
    
    def _plot_latent_correlations(self, z_active, dataset_name, fig_dir):
        """Latent correlation matrix."""
        corr = np.corrcoef(z_active.T)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label='Correlation')
        
        ax.set_title(f'{dataset_name.upper()} - Latent Correlations')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Latent Dimension')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'latent_correlations.png'), dpi=150)
        plt.close()
    
    def _plot_wscore_distribution(self, wscores, active_idx, dataset_name, fig_dir):
        """W-score distribution plot."""
        if len(active_idx) == 0:
            return
        
        wscores_active = wscores[:, active_idx]
        n_dims = min(len(active_idx), 12)  # Plot at most 12 dims
        
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
        axes = axes.flatten()
        
        for i in range(n_dims):
            ax = axes[i]
            w = wscores_active[:, i]
            
            ax.hist(w, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', lw=2)
            ax.axvline(-2, color='orange', linestyle=':', lw=1.5)
            ax.axvline(2, color='orange', linestyle=':', lw=1.5)
            
            ax.set_title(f'Dim {active_idx[i]}\nμ={np.mean(w):.2f}, σ={np.std(w):.2f}')
            ax.set_xlabel('W-score')
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for i in range(n_dims, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{dataset_name.upper()} - W-Score Distributions', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'wscore_distributions.png'), dpi=150)
        plt.close()
    
    def _plot_disentanglement(self, z, age, sex, active_idx, dataset_name, fig_dir):
        """Plot latent dimensions vs age and sex to visualize disentanglement."""
        if len(active_idx) == 0:
            return
        
        z_active = z[:, active_idx]
        n_dims = min(len(active_idx), 6)  # Plot at most 6 dims
        
        # --- Latent vs Age ---
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_dims):
            ax = axes[i]
            ax.scatter(age * 100, z_active[:, i], alpha=0.3, s=10, c='steelblue')
            
            r, p = pearsonr(age, z_active[:, i])
            ax.set_title(f'Dim {active_idx[i]} (R={r:.3f})')
            ax.set_xlabel('Age (years)')
            ax.set_ylabel(f'z_{active_idx[i]}')
            ax.grid(True, alpha=0.3)
            
            # Color title red if correlated
            if abs(r) > 0.15:
                ax.set_title(f'Dim {active_idx[i]} (R={r:.3f}) ⚠️', color='red')
        
        for i in range(n_dims, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{dataset_name.upper()} - Latent vs Age (check disentanglement)', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'latent_vs_age.png'), dpi=150)
        plt.close()
        
        # --- Latent vs Sex ---
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_dims):
            ax = axes[i]
            
            z_male = z_active[sex == 1, i]
            z_female = z_active[sex == 0, i]
            
            ax.boxplot([z_female, z_male], labels=['Female', 'Male'])
            ax.set_title(f'Dim {active_idx[i]}')
            ax.set_ylabel(f'z_{active_idx[i]}')
            ax.grid(True, alpha=0.3)
            
            # Check if distributions differ
            from scipy.stats import ttest_ind
            _, p_val = ttest_ind(z_female, z_male)
            if p_val < 0.01:
                ax.set_title(f'Dim {active_idx[i]} (p={p_val:.2e}) ⚠️', color='red')
        
        for i in range(n_dims, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{dataset_name.upper()} - Latent vs Sex (check disentanglement)', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'latent_vs_sex.png'), dpi=150)
        plt.close()


# ==============================================================================
# STANDALONE PLOTTING FUNCTIONS (for quick use)
# ==============================================================================

def analyze_reconstructions(model, dataloader, device='cuda'):
    """Analyze reconstruction quality per modality.
    
    Updated for Conditional MVAE - passes covariates to model.
    """
    model.eval()
    model.to(device)
    
    true_data = {name: [] for name in model.modality_names}
    pred_data = {name: [] for name in model.modality_names}
    
    print("\nExtracting Reconstructions...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Keep age/sex for covariates
            age = batch.pop('age').to(device)
            sex = batch.pop('sex').to(device)
            
            # Construct covariates
            c = torch.stack([age, sex], dim=-1)
            
            x_dict = {k: v.to(device) for k, v in batch.items()}
            
            # Forward with covariates
            out = model(x_dict, c)
            recons = out['recons']
            
            for name in model.modality_names:
                true_data[name].append(x_dict[name].cpu().numpy())
                pred_data[name].append(recons[name].cpu().numpy())
    
    results = {}
    for name in model.modality_names:
        t = np.vstack(true_data[name])
        p = np.vstack(pred_data[name])
        
        mse = mean_squared_error(t, p)
        r2 = r2_score(t, p)
        feature_mse = np.mean((t - p) ** 2, axis=0)
        
        results[name] = {
            'true': t,
            'pred': p,
            'global_mse': mse,
            'global_r2': r2,
            'feature_mse': feature_mse
        }
        
        print(f"\n--- {name} ---")
        print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
        print(f"  Feature MSE: min={np.min(feature_mse):.4f}, max={np.max(feature_mse):.4f}")

    return results


def plot_reconstruction_scatter(results, modality_name):
    """Plot predicted vs true values."""
    if modality_name not in results:
        print(f"Modality {modality_name} not found")
        return
        
    t = results[modality_name]['true'].flatten()
    p = results[modality_name]['pred'].flatten()
    
    if len(t) > 10000:
        idx = np.random.choice(len(t), 10000, replace=False)
        t, p = t[idx], p[idx]
        
    plt.figure(figsize=(8, 8))
    plt.scatter(t, p, alpha=0.1, s=1, c='blue')
    
    min_val = min(t.min(), p.min())
    max_val = max(t.max(), p.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    
    plt.title(f"{modality_name}\nR²: {results[modality_name]['global_r2']:.3f}")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_errors(results, modality_name):
    """Bar plot of per-feature MSE."""
    mse_vals = results[modality_name]['feature_mse']
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(mse_vals)), mse_vals, color='orange', alpha=0.7)
    plt.title(f"Per-Feature MSE: {modality_name}")
    plt.xlabel("Feature Index")
    plt.ylabel("MSE")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training curves."""
    n_plots = 3
    if 'active_dims' in history:
        n_plots = 4
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction
    axes[1].plot(history['recon'], label='Recon')
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.3)
    
    # SIGReg
    axes[2].plot(history['sigreg'], label='SIGReg')
    axes[2].set_title('SIGReg Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True, alpha=0.3)
    
    # Active dims (if available)
    if 'active_dims' in history:
        axes[3].plot(history['active_dims'])
        axes[3].set_title('Active Dimensions')
        axes[3].set_xlabel('Epoch')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved: {save_path}")
    plt.show()


def plot_latent_correlations(z, active_indices=None, save_path=None):
    """Plot correlation matrix of latents."""
    if active_indices is not None and len(active_indices) > 0:
        z = z[:, active_indices]
    
    corr = np.corrcoef(z.T)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Latent Correlation Matrix')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Dimension')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved: {save_path}")
    plt.show()
    
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    print(f"Mean |correlation|: {np.mean(np.abs(off_diag)):.3f}")
    print(f"Max |correlation|: {np.max(np.abs(off_diag)):.3f}")


def plot_wscore_summary(wscores, active_indices=None, save_path=None):
    """Plot summary of W-scores across all dimensions."""
    if active_indices is not None and len(active_indices) > 0:
        wscores = wscores[:, active_indices]
        dim_labels = [f'z_{i}' for i in active_indices]
    else:
        dim_labels = [f'z_{i}' for i in range(wscores.shape[1])]
    
    # Compute stats
    means = np.mean(wscores, axis=0)
    stds = np.std(wscores, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean ± std
    x = np.arange(len(means))
    axes[0].bar(x, means, yerr=stds, capsize=3, alpha=0.7, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dim_labels, rotation=45)
    axes[0].set_title('W-Score Mean ± Std per Dimension')
    axes[0].set_ylabel('W-Score')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Violin plot
    axes[1].violinplot(wscores, positions=x, showmeans=True)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].axhline(-2, color='orange', linestyle=':', alpha=0.7)
    axes[1].axhline(2, color='orange', linestyle=':', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dim_labels, rotation=45)
    axes[1].set_title('W-Score Distribution per Dimension')
    axes[1].set_ylabel('W-Score')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved: {save_path}")
    plt.show()