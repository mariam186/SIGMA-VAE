# Interpretation — Fig. 3 (perturbation analysis)

Fig. 3 is produced in two stages: **generate** perturbation maps from a trained model,
then **render** them on the brain. Run from inside this folder.

## 1. Generate — `perturbation_analysis.py`
Loads a checkpoint, perturbs each active latent dimension by ±SD, decodes through every
view's decoder, and writes one CSV per modality
(`perturbation_subcortical_*.csv`, `perturbation_cortical_*.csv`, `perturbation_surface_*.csv`).

- Edit the `Config` block: `CHECKPOINT_PATH` (default
  `../pretrained_models/sigma_vae_full/final_model.pth`) and `OUTPUT_DIR`.
- The model is rebuilt from the checkpoint's `model_state` via `sigma_vae.loader`
  (no pickled-object dependency), and `ACTIVE_DIMS` is taken automatically from the
  checkpoint's stored `active_indices` — so you don't hand-set which dimensions survived L0.
- The three example CSVs are only read to recover column/ROI names; point them at your own
  view files if your ROI set differs.

```bash
python perturbation_analysis.py
```

## 2. Render
- **`render_subcortical.py`** — maps the subcortical (aseg) perturbation CSV onto MNI
  slices. This is the panel **shown** in the paper.
- **`render_cortical_surface.py`** — maps the cortical thickness / surface-area
  perturbation CSVs onto fsaverage surfaces. These are the cortical/surface effects
  **described but not shown** in the paper.

Set each script's `CSV_PATH`/`OUTPUT_PATH` to the CSVs produced in step 1.

## Requirements
`pip install nilearn nibabel mne` (in addition to the repo's `requirements.txt`).
The renderers need FreeSurfer/`fsaverage` templates: the subcortical script expects
`~/mne_data/.../fsaverage/mri/aseg.mgz` (fetch with `import mne; mne.datasets.fetch_fsaverage()`),
and the cortical script uses nilearn's `fetch_surf_fsaverage()`. These templates are not
redistributed here.

> The subcortical panel is the published one; the cortical/surface renderers are provided
> so the full Fig. 3 analysis is reproducible.
