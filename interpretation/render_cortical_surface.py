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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import nilearn
    from nilearn import plotting, datasets, surface
    from nilearn.plotting import plot_surf_stat_map, view_surf
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False
    print("WARNING: nilearn not installed. Run: pip install nilearn")


# =============================================================================
# FREESURFER LABEL MAPPINGS
# =============================================================================

# Desikan-Killiany atlas label indices (for nilearn's fetch_atlas_surf_destrieux or custom)
# These map region names to the label indices in the fsaverage annotation
DK_LABEL_MAP = {
    # Left hemisphere (indices vary by atlas version)
    'lh_bankssts': 1001, 'lh_caudalanteriorcingulate': 1002,
    'lh_caudalmiddlefrontal': 1003, 'lh_cuneus': 1005,
    'lh_entorhinal': 1006, 'lh_fusiform': 1007,
    'lh_inferiorparietal': 1008, 'lh_inferiortemporal': 1009,
    'lh_isthmuscingulate': 1010, 'lh_lateraloccipital': 1011,
    'lh_lateralorbitofrontal': 1012, 'lh_lingual': 1013,
    'lh_medialorbitofrontal': 1014, 'lh_middletemporal': 1015,
    'lh_parahippocampal': 1016, 'lh_paracentral': 1017,
    'lh_parsopercularis': 1018, 'lh_parsorbitalis': 1019,
    'lh_parstriangularis': 1020, 'lh_pericalcarine': 1021,
    'lh_postcentral': 1022, 'lh_posteriorcingulate': 1023,
    'lh_precentral': 1024, 'lh_precuneus': 1025,
    'lh_rostralanteriorcingulate': 1026, 'lh_rostralmiddlefrontal': 1027,
    'lh_superiorfrontal': 1028, 'lh_superiorparietal': 1029,
    'lh_superiortemporal': 1030, 'lh_supramarginal': 1031,
    'lh_frontalpole': 1032, 'lh_temporalpole': 1033,
    'lh_transversetemporal': 1034, 'lh_insula': 1035,
    # Right hemisphere
    'rh_bankssts': 2001, 'rh_caudalanteriorcingulate': 2002,
    'rh_caudalmiddlefrontal': 2003, 'rh_cuneus': 2005,
    'rh_entorhinal': 2006, 'rh_fusiform': 2007,
    'rh_inferiorparietal': 2008, 'rh_inferiortemporal': 2009,
    'rh_isthmuscingulate': 2010, 'rh_lateraloccipital': 2011,
    'rh_lateralorbitofrontal': 2012, 'rh_lingual': 2013,
    'rh_medialorbitofrontal': 2014, 'rh_middletemporal': 2015,
    'rh_parahippocampal': 2016, 'rh_paracentral': 2017,
    'rh_parsopercularis': 2018, 'rh_parsorbitalis': 2019,
    'rh_parstriangularis': 2020, 'rh_pericalcarine': 2021,
    'rh_postcentral': 2022, 'rh_posteriorcingulate': 2023,
    'rh_precentral': 2024, 'rh_precuneus': 2025,
    'rh_rostralanteriorcingulate': 2026, 'rh_rostralmiddlefrontal': 2027,
    'rh_superiorfrontal': 2028, 'rh_superiorparietal': 2029,
    'rh_superiortemporal': 2030, 'rh_supramarginal': 2031,
    'rh_frontalpole': 2032, 'rh_temporalpole': 2033,
    'rh_transversetemporal': 2034, 'rh_insula': 2035,
}

# Common FreeSurfer naming variations
NAME_NORMALIZATIONS = {
    'thickness': '', 'area': '', 'volume': '', 'meancurv': '',
    '_thickness': '', '_area': '', '_volume': '', '_meancurv': '',
    '.thickness': '', '.area': '', '.volume': '', '.meancurv': '',
    'ctx-lh-': 'lh_', 'ctx-rh-': 'rh_', 'ctx_lh_': 'lh_', 'ctx_rh_': 'rh_',
    'lh.': 'lh_', 'rh.': 'rh_', 'lh-': 'lh_', 'rh-': 'rh_',
}


def normalize_feature_name(name: str) -> str:
    """Normalize FreeSurfer feature name to standard format."""
    normalized = name.lower()
    for old, new in NAME_NORMALIZATIONS.items():
        normalized = normalized.replace(old, new)
    return normalized.strip('_')


def parse_freesurfer_name(name: str) -> Tuple[str, str, str]:
    """
    Parse FreeSurfer feature name into components.
    
    Returns:
        (hemisphere, region, measure) e.g., ('lh', 'superiorfrontal', 'thickness')
    """
    normalized = name.lower()
    
    # Determine measure type
    measure = 'unknown'
    for m in ['thickness', 'area', 'volume', 'meancurv']:
        if m in normalized:
            measure = m
            break
    
    # Determine hemisphere
    if 'lh' in normalized or 'left' in normalized:
        hemi = 'lh'
    elif 'rh' in normalized or 'right' in normalized:
        hemi = 'rh'
    else:
        hemi = 'unknown'
    
    # Extract region
    region = normalize_feature_name(normalized)
    for prefix in ['lh_', 'rh_']:
        region = region.replace(prefix, '')
    
    return hemi, region, measure


# =============================================================================
# SURFACE VISUALIZATION
# =============================================================================

class BrainSurfaceVisualizer:
    """
    Create publication-quality brain surface visualizations.
    """
    
    def __init__(self, output_dir: str = './brain_figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if HAS_NILEARN:
            # Load fsaverage surfaces
            self.fsaverage = datasets.fetch_surf_fsaverage()
            print("Loaded fsaverage surfaces")
        else:
            self.fsaverage = None
            print("nilearn not available - using fallback visualizations")
    
    def map_values_to_parcellation(
        self,
        feature_names: List[str],
        values: np.ndarray,
        hemisphere: str = 'lh'
    ) -> np.ndarray:
        """
        Map region-level values to vertex-level for surface plotting.
        
        Args:
            feature_names: List of FreeSurfer feature names
            values: Corresponding sensitivity values
            hemisphere: 'lh' or 'rh'
            
        Returns:
            Vertex-level array for surface plotting
        """
        if not HAS_NILEARN:
            return values
        
        # Load parcellation
        atlas = datasets.fetch_atlas_surf_destrieux()
        
        if hemisphere == 'lh':
            parcellation = atlas['map_left']
        else:
            parcellation = atlas['map_right']
        
        # Create vertex-level values
        vertex_values = np.zeros_like(parcellation, dtype=float)
        
        for feat, val in zip(feature_names, values):
            hemi, region, _ = parse_freesurfer_name(feat)
            
            if hemi != hemisphere:
                continue
            
            # Find matching label
            key = f'{hemi}_{region}'
            if key in DK_LABEL_MAP:
                label_idx = DK_LABEL_MAP[key]
                # Map to vertices
                vertex_values[parcellation == label_idx] = val
        
        return vertex_values
    
    def plot_sensitivity_surface(
        self,
        sensitivity_df: pd.DataFrame,
        feature_names: List[str],
        latent_name: str,
        title: str = None,
        cmap: str = 'RdBu_r',
        threshold: float = None,
        save_name: str = None,
        views: List[str] = ['lateral', 'medial']
    ):
        """
        Plot sensitivity values on brain surface.
        
        Args:
            sensitivity_df: DataFrame with sensitivity values
            feature_names: List of feature names
            latent_name: Column name in sensitivity_df
            title: Plot title
            cmap: Colormap
            threshold: Only show values above this (absolute)
            save_name: Filename to save
            views: List of views to show
        """
        if not HAS_NILEARN:
            print("nilearn not available, using fallback")
            return self._plot_fallback(sensitivity_df, feature_names, latent_name, save_name)
        
        values = sensitivity_df[latent_name].values
        
        # Create figure with both hemispheres
        n_views = len(views)
        fig, axes = plt.subplots(2, n_views, figsize=(6*n_views, 8),
                                 subplot_kw={'projection': '3d'})
        
        if n_views == 1:
            axes = axes.reshape(-1, 1)
        
        vmax = np.percentile(np.abs(values), 95)
        
        for col, view in enumerate(views):
            for row, hemi in enumerate(['lh', 'rh']):
                ax = axes[row, col]
                
                # Get surface
                if hemi == 'lh':
                    surf_mesh = self.fsaverage['infl_left']
                    bg_map = self.fsaverage['sulc_left']
                else:
                    surf_mesh = self.fsaverage['infl_right']
                    bg_map = self.fsaverage['sulc_right']
                
                # Map values
                vertex_vals = self.map_values_to_parcellation(
                    feature_names, values, hemi
                )
                
                # Apply threshold
                if threshold:
                    vertex_vals[np.abs(vertex_vals) < threshold] = 0
                
                try:
                    plotting.plot_surf_stat_map(
                        surf_mesh,
                        stat_map=vertex_vals,
                        hemi=hemi,
                        view=view,
                        cmap=cmap,
                        symmetric_cbar=True,
                        vmax=vmax,
                        bg_map=bg_map,
                        darkness=0.5,
                        axes=ax,
                        colorbar=False
                    )
                except Exception as e:
                    print(f"Surface plot failed for {hemi}-{view}: {e}")
                    ax.text(0.5, 0.5, f'{hemi} {view}\n(plot failed)', 
                           ha='center', va='center', transform=ax.transAxes)
                
                ax.set_title(f'{hemi.upper()} - {view}')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax))
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5, aspect=30)
        cbar.set_label('Sensitivity')
        
        if title:
            fig.suptitle(title, fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        return fig
    
    def _plot_fallback(self, sensitivity_df, feature_names, latent_name, save_name):
        """Fallback visualization without nilearn."""
        values = sensitivity_df[latent_name].values
        
        # Parse into left/right
        left_data, right_data = [], []
        
        for feat, val in zip(feature_names, values):
            hemi, region, _ = parse_freesurfer_name(feat)
            if hemi == 'lh':
                left_data.append((region, val))
            elif hemi == 'rh':
                right_data.append((region, val))
        
        # Create side-by-side bar plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        for ax, data, title in [(ax1, left_data, 'Left'), (ax2, right_data, 'Right')]:
            if not data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue
                
            regions, vals = zip(*sorted(data, key=lambda x: abs(x[1]), reverse=True)[:20])
            colors = ['#d73027' if v < 0 else '#4575b4' for v in vals]
            
            ax.barh(range(len(vals)), vals, color=colors, alpha=0.8)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(regions, fontsize=9)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlabel('Sensitivity')
            ax.set_title(f'{title} Hemisphere (Top 20)')
            ax.invert_yaxis()
        
        plt.suptitle(f'Cortical Sensitivity to {latent_name}', fontsize=14)
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_multi_latent_grid(
        self,
        sensitivity_df: pd.DataFrame,
        feature_names: List[str],
        latent_names: List[str] = None,
        save_name: str = 'latent_grid.png',
        figsize_per_latent: Tuple[float, float] = (8, 3)
    ):
        """
        Create a grid showing all latents on brain surfaces.
        """
        if latent_names is None:
            latent_names = sensitivity_df.columns.tolist()
        
        n_latents = len(latent_names)
        fig, axes = plt.subplots(n_latents, 4, 
                                 figsize=(figsize_per_latent[0]*2, 
                                         figsize_per_latent[1]*n_latents))
        
        if n_latents == 1:
            axes = axes.reshape(1, -1)
        
        for i, latent in enumerate(latent_names):
            values = sensitivity_df[latent].values
            vmax = np.percentile(np.abs(values), 95)
            
            for j, (hemi, view) in enumerate([
                ('lh', 'lateral'), ('lh', 'medial'),
                ('rh', 'lateral'), ('rh', 'medial')
            ]):
                ax = axes[i, j]
                
                if HAS_NILEARN:
                    try:
                        surf_mesh = self.fsaverage[f'infl_{hemi.replace("lh","left").replace("rh","right")}']
                        vertex_vals = self.map_values_to_parcellation(feature_names, values, hemi)
                        
                        plotting.plot_surf_stat_map(
                            surf_mesh, stat_map=vertex_vals,
                            hemi=hemi, view=view, cmap='RdBu_r',
                            symmetric_cbar=True, vmax=vmax,
                            axes=ax, colorbar=False
                        )
                    except:
                        ax.text(0.5, 0.5, f'{hemi}-{view}', ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, f'{hemi}-{view}', ha='center', va='center')
                
                if i == 0:
                    ax.set_title(f'{hemi.upper()} {view}')
            
            axes[i, 0].set_ylabel(latent, rotation=90, fontsize=10)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, save_name)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()


# =============================================================================
# SUBCORTICAL VISUALIZATION
# =============================================================================

def plot_subcortical_glass_brain(
    feature_names: List[str],
    values: np.ndarray,
    title: str = 'Subcortical Sensitivity',
    output_path: str = None,
    cmap: str = 'RdBu_r'
):
    """
    Plot subcortical structures on a glass brain.
    """
    if not HAS_NILEARN:
        print("nilearn required for glass brain plots")
        return
    
    # MNI coordinates for subcortical structures
    SUBCORT_MNI = {
        'Left-Lateral-Ventricle': (-15, -10, 15),
        'Left-Inf-Lat-Vent': (-25, -40, -10),
        'Left-Cerebellum-White-Matter': (-15, -55, -30),
        'Left-Cerebellum-Cortex': (-25, -60, -35),
        'Left-Thalamus': (-10, -20, 5),
        'Left-Thalamus-Proper': (-10, -20, 5),
        'Left-Caudate': (-12, 10, 10),
        'Left-Putamen': (-25, 5, 0),
        'Left-Pallidum': (-18, -2, -2),
        'Left-Hippocampus': (-25, -25, -10),
        'Left-Amygdala': (-22, -5, -18),
        'Left-Accumbens-area': (-10, 10, -8),
        'Left-VentralDC': (-10, -15, -10),
        'Right-Lateral-Ventricle': (15, -10, 15),
        'Right-Inf-Lat-Vent': (25, -40, -10),
        'Right-Cerebellum-White-Matter': (15, -55, -30),
        'Right-Cerebellum-Cortex': (25, -60, -35),
        'Right-Thalamus': (10, -20, 5),
        'Right-Thalamus-Proper': (10, -20, 5),
        'Right-Caudate': (12, 10, 10),
        'Right-Putamen': (25, 5, 0),
        'Right-Pallidum': (18, -2, -2),
        'Right-Hippocampus': (25, -25, -10),
        'Right-Amygdala': (22, -5, -18),
        'Right-Accumbens-area': (10, 10, -8),
        'Right-VentralDC': (10, -15, -10),
        'Brain-Stem': (0, -25, -30),
    }
    
    # Match features to coordinates
    coords = []
    matched_values = []
    matched_names = []
    
    for name, val in zip(feature_names, values):
        for subcort_name, coord in SUBCORT_MNI.items():
            # Flexible matching
            if subcort_name.lower().replace('-', '') in name.lower().replace('-', '').replace('_', ''):
                coords.append(coord)
                matched_values.append(val)
                matched_names.append(subcort_name)
                break
    
    if not coords:
        print("No subcortical structures matched")
        return
    
    coords = np.array(coords)
    matched_values = np.array(matched_values)
    
    # Create glass brain plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vmax = np.percentile(np.abs(matched_values), 95)
    
    for ax, display_mode in zip(axes, ['x', 'y', 'z']):
        plotting.plot_markers(
            matched_values,
            coords,
            node_cmap=cmap,
            node_size=np.abs(matched_values) * 100 + 50,
            display_mode=display_mode,
            axes=ax,
            node_vmin=-vmax,
            node_vmax=vmax
        )
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()


# =============================================================================
# COMBINED FIGURE FOR PAPER
# =============================================================================

def create_paper_figure(
    sensitivity_dict: Dict[str, pd.DataFrame],
    feature_names: Dict[str, List[str]],
    latent_idx: int,  # 1-indexed
    output_dir: str,
    title: str = None
):
    """
    Create a combined figure suitable for publication.
    
    Layout:
    - Top row: Cortical surface views (lateral/medial, both hemispheres)
    - Middle: Subcortical bar chart
    - Bottom: Surface feature bar chart
    """
    latent_name = f'latent_{latent_idx}'
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create gridspec
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 0.8, 0.8], hspace=0.3, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # Top: Cortical surfaces (if nilearn available)
    # -------------------------------------------------------------------------
    if HAS_NILEARN and 'cortical' in sensitivity_dict:
        visualizer = BrainSurfaceVisualizer(output_dir)
        cortical_vals = sensitivity_dict['cortical'][latent_name].values
        cortical_features = feature_names.get('cortical', [])
        vmax = np.percentile(np.abs(cortical_vals), 95)
        
        for i, (hemi, view) in enumerate([
            ('lh', 'lateral'), ('lh', 'medial'),
            ('rh', 'lateral'), ('rh', 'medial')
        ]):
            ax = fig.add_subplot(gs[0, i], projection='3d')
            
            try:
                surf_mesh = visualizer.fsaverage[f'infl_{hemi.replace("lh","left").replace("rh","right")}']
                vertex_vals = visualizer.map_values_to_parcellation(
                    cortical_features, cortical_vals, hemi
                )
                
                plotting.plot_surf_stat_map(
                    surf_mesh, stat_map=vertex_vals,
                    hemi=hemi, view=view, cmap='RdBu_r',
                    symmetric_cbar=True, vmax=vmax,
                    axes=ax, colorbar=False
                )
                ax.set_title(f'{hemi.upper()} {view}', fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f'{hemi}-{view}', ha='center', va='center')
    else:
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.5, 'Cortical\n(nilearn required)', 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # -------------------------------------------------------------------------
    # Middle: Subcortical bar chart
    # -------------------------------------------------------------------------
    if 'subcortical' in sensitivity_dict:
        ax_sub = fig.add_subplot(gs[1, :2])
        
        subcort_vals = sensitivity_dict['subcortical'][latent_name].values
        subcort_features = feature_names.get('subcortical', [f'feat_{i}' for i in range(len(subcort_vals))])
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(subcort_vals))[::-1][:15]
        
        sorted_features = [subcort_features[i] for i in sorted_idx]
        sorted_vals = subcort_vals[sorted_idx]
        
        colors = ['#d73027' if v < 0 else '#4575b4' for v in sorted_vals]
        ax_sub.barh(range(len(sorted_vals)), sorted_vals, color=colors, alpha=0.8)
        ax_sub.set_yticks(range(len(sorted_vals)))
        ax_sub.set_yticklabels(sorted_features, fontsize=9)
        ax_sub.axvline(x=0, color='black', linewidth=0.5)
        ax_sub.invert_yaxis()
        ax_sub.set_xlabel('Sensitivity')
        ax_sub.set_title('Subcortical Regions (Top 15)', fontsize=11)
    
    # -------------------------------------------------------------------------
    # Middle right: Surface bar chart
    # -------------------------------------------------------------------------
    if 'surface' in sensitivity_dict:
        ax_surf = fig.add_subplot(gs[1, 2:])
        
        surf_vals = sensitivity_dict['surface'][latent_name].values
        surf_features = feature_names.get('surface', [f'feat_{i}' for i in range(len(surf_vals))])
        
        sorted_idx = np.argsort(np.abs(surf_vals))[::-1][:15]
        sorted_features = [surf_features[i] for i in sorted_idx]
        sorted_vals = surf_vals[sorted_idx]
        
        colors = ['#d73027' if v < 0 else '#4575b4' for v in sorted_vals]
        ax_surf.barh(range(len(sorted_vals)), sorted_vals, color=colors, alpha=0.8)
        ax_surf.set_yticks(range(len(sorted_vals)))
        ax_surf.set_yticklabels(sorted_features, fontsize=9)
        ax_surf.axvline(x=0, color='black', linewidth=0.5)
        ax_surf.invert_yaxis()
        ax_surf.set_xlabel('Sensitivity')
        ax_surf.set_title('Surface Features (Top 15)', fontsize=11)
    
    # -------------------------------------------------------------------------
    # Bottom: Cortical bar chart by hemisphere
    # -------------------------------------------------------------------------
    if 'cortical' in sensitivity_dict:
        cortical_vals = sensitivity_dict['cortical'][latent_name].values
        cortical_features = feature_names.get('cortical', [])
        
        # Split by hemisphere
        left_data, right_data = [], []
        for feat, val in zip(cortical_features, cortical_vals):
            hemi, region, _ = parse_freesurfer_name(feat)
            if hemi == 'lh':
                left_data.append((region[:20], val))  # Truncate long names
            elif hemi == 'rh':
                right_data.append((region[:20], val))
        
        for idx, (data, title_text) in enumerate([(left_data, 'Left Cortex'), 
                                                   (right_data, 'Right Cortex')]):
            ax = fig.add_subplot(gs[2, idx*2:(idx+1)*2])
            
            if data:
                data_sorted = sorted(data, key=lambda x: abs(x[1]), reverse=True)[:12]
                regions, vals = zip(*data_sorted)
                
                colors = ['#d73027' if v < 0 else '#4575b4' for v in vals]
                ax.barh(range(len(vals)), vals, color=colors, alpha=0.8)
                ax.set_yticks(range(len(vals)))
                ax.set_yticklabels(regions, fontsize=9)
                ax.axvline(x=0, color='black', linewidth=0.5)
                ax.invert_yaxis()
            
            ax.set_xlabel('Sensitivity')
            ax.set_title(f'{title_text} (Top 12)', fontsize=11)
    
    # Title
    fig.suptitle(title or f'Brain Sensitivity to {latent_name}', fontsize=14, y=0.98)
    
    # Save
    save_path = os.path.join(output_dir, f'paper_figure_{latent_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == '__main__':
    print("""
    Usage:
    ------
    from brain_visualization import BrainSurfaceVisualizer, create_paper_figure
    
    # Load your sensitivity results
    sensitivity_dict = {
        'subcortical': pd.read_csv('sensitivity_subcortical.csv', index_col=0),
        'cortical': pd.read_csv('sensitivity_cortical.csv', index_col=0),
        'surface': pd.read_csv('sensitivity_surface.csv', index_col=0)
    }
    
    feature_names = {
        'subcortical': sensitivity_dict['subcortical'].index.tolist(),
        'cortical': sensitivity_dict['cortical'].index.tolist(),
        'surface': sensitivity_dict['surface'].index.tolist()
    }
    
    # Create paper figure for latent 1
    create_paper_figure(
        sensitivity_dict, 
        feature_names, 
        latent_idx=1,
        output_dir='./figures/'
    )
    
    # Or use the surface visualizer directly
    viz = BrainSurfaceVisualizer('./figures/')
    viz.plot_sensitivity_surface(
        sensitivity_dict['cortical'],
        feature_names['cortical'],
        'latent_1',
        title='Cortical Sensitivity to Latent 1'
    )
    """)
