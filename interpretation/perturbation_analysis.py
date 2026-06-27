"""
Anatomical Interpretation via Perturbation Analysis

For each active latent dimension, this script:
1. Creates a reference point (z=0)
2. Perturbs one dimension to extreme values (+/- 3 SD)
3. Decodes to get reconstructed brain features
4. Computes difference maps showing regional effects

Generates maps for both Female and Male at specified age.
"""
# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import os, sys
warnings.filterwarnings('ignore')

# allow importing the installed package from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ee_models

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATHS = {
    'subcortical': '../data/example/subcortical_example.csv',
    'cortical': '../data/example/cortical_example.csv',
    'surface': '../data/example/surface_example.csv'
}
df = pd.read_csv(DATA_PATHS['cortical'])
col = df.columns
cort_col = col.drop(['subject_id', 'age', 'sex', 'site', 'dataset_name', 'diagnosis','specific_diagnosis'])

df = pd.read_csv(DATA_PATHS['subcortical'])
col = df.columns
sub_col = col.drop(['subject_id', 'age', 'sex', 'site', 'dataset_name', 'diagnosis','specific_diagnosis','non-WM-hypointensities', '3rd-Ventricle', '4th-Ventricle' , '5th-Ventricle',       'Left-WM-hypointensities', 'Right-WM-hypointensities',
       'Left-non-WM-hypointensities', 'Right-non-WM-hypointensities'])

df = pd.read_csv(DATA_PATHS['surface'])
col = df.columns
sur_col = col.drop(['subject_id', 'age', 'sex', 'site', 'dataset_name', 'diagnosis','specific_diagnosis'])


class Config:
    """Configuration for perturbation analysis."""
    
    # Model checkpoint path
    CHECKPOINT_PATH = '../pretrained_models/sigma_vae_full/final_model.pth'
    
    # Output directory
    OUTPUT_DIR = './results/perturbation_maps_eenm_04/'
    
    # Active dimensions (from L0 gating) - UPDATE THIS
    ACTIVE_DIMS = [3, 6, 7, 23, 28]
    
    # Perturbation magnitude (in standard deviations)
    PERTURBATION_MAGNITUDE = 3.0
    
    # Reference age (actual years) - normalized as age/100
    REF_AGE_RAW = 26
    
    # Feature names for each modality (optional, for better labeling)
    FEATURE_NAMES = {
        'cortical': cort_col,
        'subcortical': sub_col,
        'surface': sur_col
    }
    
    # Model architecture (must match training)
    INPUT_DIMS = {
        'cortical': 68,
        'subcortical': 33,
        'surface': 68
    }
    LATENT_DIM = 32
    ENCODER_HIDDEN = [512, 256, 128]
    DECODER_HIDDEN = [128, 256, 512]
    
    # Visualization
    FIGSIZE = (16, 10)
    CMAP = 'RdBu_r'
    TOP_K_FEATURES = 15


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path: str, config: Config, device: str = 'cpu'):
    """Rebuild the model from the checkpoint's state_dict (rename-safe).

    Also syncs config.ACTIVE_DIMS to the checkpoint's stored active indices so the
    perturbation uses the dimensions the L0 gate actually kept.
    """
    from sigma_vae.loader import load_sigma_model
    model, ckpt = load_sigma_model(
        checkpoint_path,
        input_dims=config.INPUT_DIMS,
        latent_dim=config.LATENT_DIM,
        encoder_hidden=config.ENCODER_HIDDEN,
        decoder_hidden=config.DECODER_HIDDEN,
        device=device,
    )
    if ckpt.get('active_indices') is not None:
        config.ACTIVE_DIMS = list(ckpt['active_indices'])
        print(f"  active dims from checkpoint: {config.ACTIVE_DIMS}")
    print(f"✓ Loaded model from {checkpoint_path}")
    return model


# =============================================================================
# PERTURBATION ANALYSIS
# =============================================================================

def create_perturbation_maps(
    model: nn.Module,
    active_dims: List[int],
    magnitude: float = 3.0,
    reference_age: float = 0.26,
    reference_sex: float = 0.0,
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Generate perturbation maps for each active dimension.
    
    For each dimension k:
        Δx = D(z_k = magnitude, z_¬k = 0; c) - D(z = 0; c)
    
    Args:
        model: Trained model
        active_dims: List of active dimension indices
        magnitude: Perturbation magnitude in SD
        reference_age: Normalized age (age/100), e.g., 26 → 0.26
        reference_sex: Sex encoding (0=female, 1=male)
        device: Device to run on
    
    Returns:
        Dictionary mapping dimension index to perturbation maps per modality
    """
    model.eval()
    model.to(device)
    
    results = {}
    
    # Create conditioning vector [age, sex]
    c_ref = torch.tensor([[reference_age, reference_sex]], device=device, dtype=torch.float32)
    
    with torch.no_grad():
        # Reference: z = 0
        z_ref = torch.zeros(1, model.latent_dim, device=device)
        
        # Apply gate (frozen mode will use fixed mask)
        z_ref_gated, _ = model.gate(z_ref, training=False)
        
        # Decode reference WITH conditioning
        ref_recons = {}
        for name in model.modality_names:
            ref_recons[name] = model.decoders[name](z_ref_gated, c_ref).cpu().numpy().squeeze()
        
        # Perturb each active dimension
        for dim_idx in active_dims:
            print(f"  Processing dimension {dim_idx}...")
            
            # Positive perturbation
            z_pos = torch.zeros(1, model.latent_dim, device=device)
            z_pos[0, dim_idx] = magnitude
            z_pos_gated, _ = model.gate(z_pos, training=False)
            
            # Negative perturbation
            z_neg = torch.zeros(1, model.latent_dim, device=device)
            z_neg[0, dim_idx] = -magnitude
            z_neg_gated, _ = model.gate(z_neg, training=False)
            
            # Decode perturbations WITH conditioning
            pos_recons = {}
            neg_recons = {}
            for name in model.modality_names:
                pos_recons[name] = model.decoders[name](z_pos_gated, c_ref).cpu().numpy().squeeze()
                neg_recons[name] = model.decoders[name](z_neg_gated, c_ref).cpu().numpy().squeeze()
            
            # Compute difference maps
            results[dim_idx] = {
                'positive': {name: pos_recons[name] - ref_recons[name] for name in model.modality_names},
                'negative': {name: neg_recons[name] - ref_recons[name] for name in model.modality_names},
                'symmetric': {name: (pos_recons[name] - neg_recons[name]) / 2 for name in model.modality_names}
            }
    
    return results, ref_recons


# =============================================================================
# VISUALIZATION
# =============================================================================

def get_display_name(dim_idx: int, active_dims: List[int]) -> str:
    """Convert original dimension index to display name (W0, W1, ...)."""
    display_idx = active_dims.index(dim_idx)
    return f"W{display_idx}"


def plot_perturbation_heatmap(
    results: Dict,
    active_dims: List[int],
    modality: str,
    feature_names: Optional[List[str]] = None,
    top_k: int = 15,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    title_suffix: str = ''
):
    """Plot heatmap showing top affected features for each dimension."""
    n_dims = len(active_dims)
    
    # Collect all symmetric perturbations for this modality
    all_effects = []
    for dim_idx in active_dims:
        effects = results[dim_idx]['symmetric'][modality]
        all_effects.append(effects)
    
    effects_matrix = np.array(all_effects)
    n_features = effects_matrix.shape[1]
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(n_features)]
    else:
        feature_names = list(feature_names)
    
    # Find top-K most affected features
    max_abs_effect = np.max(np.abs(effects_matrix), axis=0)
    top_indices = np.argsort(max_abs_effect)[-top_k:][::-1]
    
    # Extract subset
    effects_subset = effects_matrix[:, top_indices]
    feature_subset = [feature_names[i] for i in top_indices]
    
    # Create display names for dimensions
    dim_labels = [get_display_name(d, active_dims) + f" (d{d})" for d in active_dims]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        effects_subset.T,
        xticklabels=dim_labels,
        yticklabels=feature_subset,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        ax=ax,
        cbar_kws={'label': 'Effect (Δ from reference)'}
    )
    
    ax.set_xlabel('Latent Dimension', fontsize=12)
    ax.set_ylabel('Brain Feature', fontsize=12)
    title = f'Perturbation Effects: {modality.title()}\n(Top {top_k} most affected features)'
    if title_suffix:
        title += f' - {title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_path}")
    
    plt.close()
    
    return effects_matrix, top_indices


def plot_dimension_profiles(
    results: Dict,
    active_dims: List[int],
    feature_names_dict: Dict[str, Optional[List[str]]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12),
    title_suffix: str = ''
):
    """Plot bar charts showing top positive/negative effects per dimension."""
    n_dims = len(active_dims)
    modalities = list(results[active_dims[0]]['symmetric'].keys())
    
    fig, axes = plt.subplots(n_dims, len(modalities), figsize=figsize)
    if n_dims == 1:
        axes = axes.reshape(1, -1)
    
    for i, dim_idx in enumerate(active_dims):
        display_name = get_display_name(dim_idx, active_dims)
        
        for j, modality in enumerate(modalities):
            ax = axes[i, j]
            
            effects = results[dim_idx]['symmetric'][modality]
            n_features = len(effects)
            
            # Get feature names
            if feature_names_dict.get(modality) is not None:
                feat_names = list(feature_names_dict[modality])
            else:
                feat_names = [f"F{k}" for k in range(n_features)]
            
            # Top 5 positive and negative
            sorted_idx = np.argsort(effects)
            top_neg = sorted_idx[:5]
            top_pos = sorted_idx[-5:][::-1]
            
            # Combine and plot
            selected_idx = np.concatenate([top_pos, top_neg])
            selected_effects = effects[selected_idx]
            selected_names = [feat_names[k] for k in selected_idx]
            
            colors = ['#d73027' if e > 0 else '#4575b4' for e in selected_effects]
            
            ax.barh(range(len(selected_idx)), selected_effects, color=colors)
            ax.set_yticks(range(len(selected_idx)))
            ax.set_yticklabels(selected_names, fontsize=8)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlabel('Effect')
            
            if i == 0:
                ax.set_title(modality.title(), fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{display_name} (dim {dim_idx})', fontsize=10, fontweight='bold')
    
    main_title = 'Top Features Affected by Each Latent Dimension'
    if title_suffix:
        main_title += f' - {title_suffix}'
    plt.suptitle(main_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_path}")
    
    plt.close()


def plot_combined_perturbation_figure(
    results: Dict,
    active_dims: List[int],
    feature_names_dict: Dict[str, Optional[List[str]]],
    output_path: Optional[str] = None,
    top_k: int = 10,
    title_suffix: str = ''
):
    """Create a single publication-ready figure with perturbation maps."""
    modalities = list(results[active_dims[0]]['symmetric'].keys())
    n_mods = len(modalities)
    
    fig, axes = plt.subplots(1, n_mods, figsize=(6*n_mods, 8))
    if n_mods == 1:
        axes = [axes]
    
    for idx, modality in enumerate(modalities):
        ax = axes[idx]
        
        # Collect effects
        all_effects = []
        for dim_idx in active_dims:
            effects = results[dim_idx]['symmetric'][modality]
            all_effects.append(effects)
        
        effects_matrix = np.array(all_effects)
        n_features = effects_matrix.shape[1]
        
        # Get feature names
        if feature_names_dict.get(modality) is not None:
            feat_names = list(feature_names_dict[modality])
        else:
            feat_names = [f"F{k}" for k in range(n_features)]
        
        # Top-K features
        max_abs_effect = np.max(np.abs(effects_matrix), axis=0)
        top_indices = np.argsort(max_abs_effect)[-top_k:][::-1]
        
        effects_subset = effects_matrix[:, top_indices]
        feature_subset = [feat_names[i] for i in top_indices]
        
        # Dimension labels
        dim_labels = [get_display_name(d, active_dims) for d in active_dims]
        
        # Heatmap
        sns.heatmap(
            effects_subset.T,
            xticklabels=dim_labels,
            yticklabels=feature_subset,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            ax=ax,
            cbar_kws={'label': 'Δ', 'shrink': 0.6},
            annot_kws={'size': 8}
        )
        
        ax.set_xlabel('W-score Dimension', fontsize=11)
        ax.set_ylabel('Brain Region', fontsize=11)
        ax.set_title(f'{modality.title()}', fontsize=12, fontweight='bold')
    
    main_title = 'Anatomical Interpretation: Perturbation Analysis (±3 SD)'
    if title_suffix:
        main_title += f'\n{title_suffix}'
    plt.suptitle(main_title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"✓ Saved {output_path}")
    
    plt.close()


def plot_sex_comparison(
    results_female: Dict,
    results_male: Dict,
    active_dims: List[int],
    feature_names_dict: Dict[str, Optional[List[str]]],
    output_path: Optional[str] = None,
    top_k: int = 10,
    age: int = 26,
    sd: int = 3
):
    """Create side-by-side comparison of female vs male perturbation maps."""
    modalities = list(results_female[active_dims[0]]['symmetric'].keys())
    n_mods = len(modalities)
    
    fig, axes = plt.subplots(2, n_mods, figsize=(6*n_mods, 14))
    if n_mods == 1:
        axes = axes.reshape(2, 1)
    
    for row_idx, (sex_label, results) in enumerate([('Female', results_female), ('Male', results_male)]):
        for col_idx, modality in enumerate(modalities):
            ax = axes[row_idx, col_idx]
            
            # Collect effects
            all_effects = []
            for dim_idx in active_dims:
                effects = results[dim_idx]['symmetric'][modality]
                all_effects.append(effects)
            
            effects_matrix = np.array(all_effects)
            n_features = effects_matrix.shape[1]
            
            # Get feature names
            if feature_names_dict.get(modality) is not None:
                feat_names = list(feature_names_dict[modality])
            else:
                feat_names = [f"F{k}" for k in range(n_features)]
            
            # Top-K features
            max_abs_effect = np.max(np.abs(effects_matrix), axis=0)
            top_indices = np.argsort(max_abs_effect)[-top_k:][::-1]
            
            effects_subset = effects_matrix[:, top_indices]
            feature_subset = [feat_names[i] for i in top_indices]
            
            # Dimension labels
            dim_labels = [get_display_name(d, active_dims) for d in active_dims]
            
            # Heatmap
            sns.heatmap(
                effects_subset.T,
                xticklabels=dim_labels,
                yticklabels=feature_subset,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                ax=ax,
                vmin=-1.5, vmax=1.5,
                cbar_kws={'label': 'Δ', 'shrink': 0.6},
                annot_kws={'size': 8}
            )
            
            ax.set_xlabel('W-score Dimension', fontsize=10)
            ax.set_ylabel('Brain Region', fontsize=10)
            
            if row_idx == 0:
                ax.set_title(f'{modality.title()}', fontsize=12, fontweight='bold')
            
            # Add sex label on left
            if col_idx == 0:
                ax.text(-0.3, 0.5, sex_label, transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='center', rotation=90)
    
    plt.suptitle(f'Perturbation Analysis: Female vs Male (Age {age}, ±{sd} SD)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"✓ Saved {output_path}")
    
    plt.close()


def plot_dimension_profiles_both_sexes(
    all_results: Dict,
    active_dims: List[int],
    feature_names_dict: Dict[str, Optional[List[str]]],
    output_path: Optional[str] = None
):
    """Plot bar charts comparing male vs female effects for each dimension."""
    results_female = all_results['female']['results']
    results_male = all_results['male']['results']
    
    n_dims = len(active_dims)
    modalities = list(results_female[active_dims[0]]['symmetric'].keys())
    
    fig, axes = plt.subplots(n_dims, len(modalities), figsize=(5*len(modalities), 3*n_dims))
    if n_dims == 1:
        axes = axes.reshape(1, -1)
    if len(modalities) == 1:
        axes = axes.reshape(n_dims, 1)
    
    for i, dim_idx in enumerate(active_dims):
        display_name = get_display_name(dim_idx, active_dims)
        
        for j, modality in enumerate(modalities):
            ax = axes[i, j]
            
            effects_f = results_female[dim_idx]['symmetric'][modality]
            effects_m = results_male[dim_idx]['symmetric'][modality]
            n_features = len(effects_f)
            
            # Get feature names
            if feature_names_dict.get(modality) is not None:
                feat_names = list(feature_names_dict[modality])
            else:
                feat_names = [f"F{k}" for k in range(n_features)]
            
            # Top 5 by combined max effect
            combined_max = np.maximum(np.abs(effects_f), np.abs(effects_m))
            top_idx = np.argsort(combined_max)[-5:][::-1]
            
            # Bar positions
            x = np.arange(len(top_idx))
            width = 0.35
            
            ax.bar(x - width/2, effects_f[top_idx], width,
                   label='Female', color='#e377c2', alpha=0.8)
            ax.bar(x + width/2, effects_m[top_idx], width,
                   label='Male', color='#17becf', alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels([feat_names[k] for k in top_idx], rotation=45, ha='right', fontsize=8)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_ylabel('Effect (Δ)')
            
            if i == 0:
                ax.set_title(modality.title(), fontsize=11, fontweight='bold')
                ax.legend(fontsize=8, loc='upper right')
            if j == 0:
                ax.text(-0.25, 0.5, f'{display_name}\n(dim {dim_idx})',
                        transform=ax.transAxes, fontsize=10, fontweight='bold',
                        va='center', ha='right')
    
    plt.suptitle('Top Features by Latent Dimension: Female vs Male',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_path}")
    
    plt.close()


def save_perturbation_tables(
    results: Dict,
    active_dims: List[int],
    feature_names_dict: Dict[str, Optional[List[str]]],
    output_dir: str,
    suffix: str = ''
):
    """Save perturbation effects as CSV tables."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    modalities = list(results[active_dims[0]]['symmetric'].keys())
    
    for modality in modalities:
        # Collect effects
        all_effects = []
        for dim_idx in active_dims:
            effects = results[dim_idx]['symmetric'][modality]
            all_effects.append(effects)
        
        effects_matrix = np.array(all_effects).T
        n_features = effects_matrix.shape[0]
        
        # Get feature names
        if feature_names_dict.get(modality) is not None:
            feat_names = list(feature_names_dict[modality])
        else:
            feat_names = [f"feature_{k}" for k in range(n_features)]
        
        # Dimension labels
        dim_labels = [f"W{active_dims.index(d)}_d{d}" for d in active_dims]
        
        # Create DataFrame
        df = pd.DataFrame(effects_matrix, index=feat_names, columns=dim_labels)
        df.index.name = 'feature'
        
        # Add summary columns
        df['max_abs_effect'] = np.max(np.abs(effects_matrix), axis=1)
        df['dominant_dim'] = [dim_labels[np.argmax(np.abs(row))] for row in effects_matrix]
        
        # Sort by max effect
        df = df.sort_values('max_abs_effect', ascending=False)
        
        # Save
        output_path = output_dir / f'perturbation_{modality}{suffix}.csv'
        df.to_csv(output_path)
        print(f"✓ Saved {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_perturbation_analysis(config: Config):
    """Run full perturbation analysis pipeline for both sexes."""
    
    print("\n" + "="*70)
    print("PERTURBATION ANALYSIS FOR ANATOMICAL INTERPRETATION")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {config.CHECKPOINT_PATH}...")
    model = load_model(config.CHECKPOINT_PATH, config, device)
    
    # Normalize age: age_normalized = age / 100.0
    normalized_age = config.REF_AGE_RAW / 100.0
    print(f"\nReference age: {config.REF_AGE_RAW} years → normalized: {normalized_age:.3f}")
    
    # Generate perturbation maps for both sexes
    all_results = {}
    
    for sex_label, sex_value in [('female', 0.0), ('male', 1.0)]:
        print(f"\n{'='*50}")
        print(f"Generating maps for {sex_label.upper()} (age={config.REF_AGE_RAW})")
        print(f"{'='*50}")
        print(f"Perturbation magnitude = ±{config.PERTURBATION_MAGNITUDE} SD")
        
        results, ref_recons = create_perturbation_maps(
            model,
            config.ACTIVE_DIMS,
            config.PERTURBATION_MAGNITUDE,
            reference_age=normalized_age,
            reference_sex=sex_value,
            device=device
        )
        
        all_results[sex_label] = {'results': results, 'ref_recons': ref_recons}
        
        # Generate visualizations for this sex
        print(f"\nGenerating visualizations for {sex_label}...")
        
        # 1. Combined figure (publication-ready)
        plot_combined_perturbation_figure(
            results,
            config.ACTIVE_DIMS,
            config.FEATURE_NAMES,
            output_path=str(output_dir / f'fig_perturbation_{sex_label}.png'),
            top_k=config.TOP_K_FEATURES,
            title_suffix=f'{sex_label.title()}, Age {config.REF_AGE_RAW}'
        )
        
        # 2. Individual modality heatmaps
        for modality in model.modality_names:
            plot_perturbation_heatmap(
                results,
                config.ACTIVE_DIMS,
                modality,
                feature_names=config.FEATURE_NAMES.get(modality),
                top_k=config.TOP_K_FEATURES,
                output_path=str(output_dir / f'perturbation_{modality}_{sex_label}.png'),
                title_suffix=f'{sex_label.title()}, Age {config.REF_AGE_RAW}'
            )
        
        # 3. Dimension profiles
        plot_dimension_profiles(
            results,
            config.ACTIVE_DIMS,
            config.FEATURE_NAMES,
            output_path=str(output_dir / f'dimension_profiles_{sex_label}.png'),
            title_suffix=f'{sex_label.title()}, Age {config.REF_AGE_RAW}'
        )
        
        # 4. Save tables
        save_perturbation_tables(
            results,
            config.ACTIVE_DIMS,
            config.FEATURE_NAMES,
            str(output_dir),
            suffix=f'_{sex_label}'
        )
    
    # 5. Generate comparison figure (male vs female)
    print(f"\n{'='*50}")
    print("Generating MALE vs FEMALE comparison...")
    print(f"{'='*50}")
    
    plot_sex_comparison(
        all_results['female']['results'],
        all_results['male']['results'],
        config.ACTIVE_DIMS,
        config.FEATURE_NAMES,
        output_path=str(output_dir / 'fig_perturbation_sex_comparison.png'),
        top_k=config.TOP_K_FEATURES,
        age=config.REF_AGE_RAW,
        sd=config.PERTURBATION_MAGNITUDE
    )
    
    # 6. Dimension profiles for both sexes
    plot_dimension_profiles_both_sexes(
        all_results,
        config.ACTIVE_DIMS,
        config.FEATURE_NAMES,
        output_path=str(output_dir / 'dimension_profiles_comparison.png')
    )
    
    print("\n" + "="*70)
    print("✓ PERTURBATION ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - fig_perturbation_female.png/pdf")
    print("  - fig_perturbation_male.png/pdf")
    print("  - fig_perturbation_sex_comparison.png/pdf")
    print("  - perturbation_<modality>_<sex>.png")
    print("  - dimension_profiles_<sex>.png")
    print("  - dimension_profiles_comparison.png")
    print("  - perturbation_<modality>_<sex>.csv")
    
    return all_results


# =============================================================================
# ENTRY POINT
# =============================================================================
# %%
if __name__ == "__main__":
    
    # =========================================================================
    # CONFIGURATION - UPDATE THESE FOR YOUR DATA
    # =========================================================================
    
    # Path to trained model checkpoint
    Config.CHECKPOINT_PATH = './results/eenm_04/best_model_p2.pth'
    
    # Output directory for figures and tables
    Config.OUTPUT_DIR = './results/perturbation_maps_eenm_04/'
    
    # Active dimensions from your L0 gating results (original indices)
    Config.ACTIVE_DIMS = [3, 6, 7, 23, 28]
    
    # Perturbation magnitude (in standard deviations)
    Config.PERTURBATION_MAGNITUDE = 2.0
    
    # Reference age (actual years)
    # Age normalization: normalized_age = age / 100.0
    # So age 26 → 0.26
    Config.REF_AGE_RAW = 26
    
    # Number of top features to display in figures
    Config.TOP_K_FEATURES = 15
    
    # Model architecture (must match training)
    Config.INPUT_DIMS = {
        'cortical': 68,
        'subcortical': 33,
        'surface': 68
    }
    Config.LATENT_DIM = 32
    
    # =========================================================================
    
    all_results = run_perturbation_analysis(Config)
# %%