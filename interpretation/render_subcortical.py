"""
plot_perturbation_subcortical.py
---------------------------------
Maps FreeSurfer aseg subcortical ROIs onto brain slices.
Uses nilearn display objects → image buffers → composited figure.

Requirements:
    pip install nilearn nibabel matplotlib pandas mne
"""

import os
import io
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from nilearn import plotting

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CSV_PATH    = "./results/perturbation_maps/perturbation_subcortical_male.csv"
OUTPUT_PATH = "./results/brain_maps/brain_subcortical_perturbation.png"

# Which W columns to plot ("all" or list e.g. ["W0_d3", "W1_d6"])
COLS_TO_PLOT = "all"

# MNE fsaverage path
MNE_FSAVERAGE = os.path.expanduser(
    "~/mne_data/MNE-fsaverage-data/fsaverage"
)

# Views per row: list of (display_mode, cut_coords) tuples
# "ortho" = 3-panel (sag/cor/ax), "z" = axial slices
VIEWS = [
    ("ortho", None),
    ("z",     [-20, 0, 20]),
]

CMAP_DIV = "cold_hot"
CMAP_SEQ = "YlOrRd"
BG_COLOR = "#111118"
DPI      = 180
# ─────────────────────────────────────────────────────────────────────────────


ASEG_LABELS = {
    "Left-Lateral-Ventricle":        4,
    "Right-Lateral-Ventricle":      43,
    "Left-Cerebellum-Cortex":        8,
    "Right-Cerebellum-Cortex":      47,
    "Right-Cerebellum-White-Matter":46,
    "Brain-Stem":                   16,
    "Left-Cerebellum-White-Matter":  7,
    "Left-Inf-Lat-Vent":             5,
    "Right-Inf-Lat-Vent":           44,
    "Left-choroid-plexus":          31,
    "Right-choroid-plexus":         63,
    "Left-Pallidum":                13,
    "Left-VentralDC":               28,
    "Right-VentralDC":              60,
    "Right-Pallidum":               52,
    "Right-Hippocampus":            53,
    "Left-Hippocampus":             17,
    "Left-Amygdala":                18,
    "CC_Anterior":                 255,
    "Right-Amygdala":               54,
    "Left-Thalamus":                10,
    "Right-Thalamus":               49,
    "WM-hypointensities":           77,
    "CC_Central":                  252,
    "Left-Caudate":                 11,
    "Right-Caudate":                50,
    "Left-Putamen":                 12,
    "CC_Posterior":                251,
    "Left-Accumbens-area":          26,
    "Right-Putamen":                51,
    "CC_Mid_Anterior":             254,
    "CC_Mid_Posterior":            253,
    "Right-Accumbens-area":         58,
}


def load_mne_files(mne_dir):
    aseg_path = os.path.join(mne_dir, "mri", "aseg.mgz")
    t1_path   = os.path.join(mne_dir, "mri", "T1.mgz")
    if not os.path.exists(aseg_path):
        raise FileNotFoundError(
            f"aseg.mgz not found at {aseg_path}\n"
            "Run: import mne; mne.datasets.fetch_fsaverage()"
        )
    bg = nib.load(t1_path) if os.path.exists(t1_path) else None
    return nib.load(aseg_path), bg


def build_stat_volume(aseg_img, parcel_values):
    data = np.asarray(aseg_img.dataobj, dtype=np.float32)
    stat = np.full_like(data, np.nan)
    for roi, label_id in ASEG_LABELS.items():
        if roi in parcel_values:
            stat[data == label_id] = float(parcel_values[roi])
    return nib.Nifti1Image(stat, aseg_img.affine, aseg_img.header)


def render_to_image(stat_img, bg_img, display_mode, cut_coords,
                    cmap, vmin, vmax):
    """Render a nilearn plot to an RGBA numpy array via an in-memory buffer."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 2.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    plotting.plot_stat_map(
        stat_img,
        bg_img=bg_img,
        display_mode=display_mode,
        cut_coords=cut_coords,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        annotate=False,
        draw_cross=False,
        black_bg=True,
        axes=ax,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    img_array = mpimg.imread(buf)
    buf.close()
    return img_array


def main():
    df = pd.read_csv(CSV_PATH).set_index("feature")

    w_cols = [c for c in df.columns if c.startswith("W") and df[c].dtype != object]
    cols   = w_cols if COLS_TO_PLOT == "all" else COLS_TO_PLOT

    print("Loading MNE fsaverage aseg ...")
    aseg_img, bg_img = load_mne_files(MNE_FSAVERAGE)

    # Per-column colour scaling
    col_cmaps, col_vmins, col_vmaxs = {}, {}, {}
    for col in cols:
        vals     = df[col].dropna().values.astype(float)
        vmax_abs = np.percentile(np.abs(vals), 99)
        div      = vals.min() < 0
        col_cmaps[col] = CMAP_DIV if div else CMAP_SEQ
        col_vmins[col] = -vmax_abs if div else float(vals.min())
        col_vmaxs[col] = vmax_abs
        print(f"  {col}: [{col_vmins[col]:.3f}, {col_vmaxs[col]:.3f}]")

    n_view_cols = len(VIEWS)
    n_rows      = len(cols)

    fig = plt.figure(figsize=(12, 3.2 * n_rows), facecolor=BG_COLOR)
    grid = fig.add_gridspec(
        n_rows, n_view_cols + 1,
        width_ratios=[10] * n_view_cols + [0.4],
        hspace=0.08, wspace=0.03,
    )

    for row_idx, col in enumerate(cols):
        parcel_values = df[col].to_dict()
        stat_img = build_stat_volume(aseg_img, parcel_values)
        cmap = col_cmaps[col]
        vmin = col_vmins[col]
        vmax = col_vmaxs[col]

        for panel_idx, (display_mode, cut_coords) in enumerate(VIEWS):
            img_arr = render_to_image(
                stat_img, bg_img, display_mode, cut_coords,
                cmap, vmin, vmax
            )
            ax = fig.add_subplot(grid[row_idx, panel_idx])
            ax.imshow(img_arr)
            ax.axis("off")
            ax.set_facecolor(BG_COLOR)

            if row_idx == 0:
                label = "Orthographic" if display_mode == "ortho" else "Axial slices"
                ax.set_title(label, color="#aaaaaa", fontsize=9, pad=4)

        # Row label
        label_ax = fig.add_subplot(grid[row_idx, 0])
        label_ax.text(-0.04, 0.5, col,
                      transform=label_ax.transAxes,
                      fontsize=11, color="white", va="center", ha="right",
                      fontweight="bold", rotation=90)
        label_ax.axis("off")

        # Per-row colorbar
        cax = fig.add_subplot(grid[row_idx, -1])
        sm  = plt.cm.ScalarMappable(
            cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.set_ticks([vmin, 0, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])

        print(f"  [{row_idx+1}/{n_rows}] {col} done")

    fig.suptitle(
        "SIGMA-VAE Perturbation — Subcortical (Male)",
        color="white", fontsize=13, fontweight="bold", y=1.002,
    )

    plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    print(f"\nSaved → {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()