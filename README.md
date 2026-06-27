# SIGMA-VAE

**Sketched Isotropic Gaussian Multi-View Autoencoder for Normative Modeling**
Zabihi et al., MICCAI 2026.

SIGMA-VAE is a multi-view normative model for structural brain MRI. It combines
(i) **Sketched Isotropic Gaussian Regularization (SIGReg)** for a hole-free,
Euclidean-valid latent geometry, (ii) **L0 sparsity gating** for automatic discovery
of a minimal, interpretable set of latent dimensions, and (iii) an **uncertainty-aware
W-score** that down-weights deviations by posterior variance. Individual atypicality is
quantified as the L2 norm of the active W-score vector.

## Repository structure

```
sigma_vae/            # core package (the method in the paper)
  models.py           # ConditionalMMVAE: per-view encoders/decoders, PoE fusion, L0 gate
  losses.py           # SIGReg + L0 + reconstruction + variance terms
  trainer.py          # two-phase training, latent extraction, W-scores, CSV export
  data.py             # multi-view CSV loading, subject alignment, scaling
  diagnostics.py      # check_disentanglement: latent vs age/sex independence
  evaluation.py       # Evaluator: per-view reconstruction, calibration, W-score figures
  train.py            # end-to-end script: edit CONFIG, run
data/example/         # synthetic CSVs + generator showing the expected input schema
pretrained_models/    # final trained weights for all four ablations (reference)
interpretation/       # Fig. 3 perturbation / brain-rendering scripts (see note below)
```

## Installation

```bash
git clone https://github.com/mariam186/sigma-vae.git
cd sigma-vae
pip install -r requirements.txt
```

## Input data format

Three CSV files, one per view: **subcortical volumes**, **cortical thickness**,
**cortical surface area**. One row per subject. The three views are matched on
`subject_id` (inner join), so a subject must appear in all three files.

A runnable, synthetic example with the exact schema is in `data/example/`
(regenerate with `python data/example/make_example_data.py`).

**Metadata columns**

| Column         | Required | Used for |
|----------------|----------|----------|
| `subject_id`   | yes      | aligning the three views (inner join) |
| `age`          | yes      | covariate (auto-normalized as `age/100`) |
| `sex`          | yes      | covariate (`M/F`, `male/female`, `1/0` all accepted) |
| `diagnosis`    | test/eval only | group labels for evaluation (e.g. `healthy`/`patient`) |
| `dataset_name` | test/eval only | site/cohort labels for evaluation |
| `site`, `specific_diagnosis` | optional | dropped if present |

**Feature columns:** every remaining numeric column is treated as an ROI feature for
that view. The model trained in the paper uses 33 subcortical volumes, 68 cortical
thickness parcels, and 68 surface-area parcels (FreeSurfer v7.3,
Desikan–Killiany + aseg).

**Per-view cleaning.** `DataProcessor` (in `data.py`) drops the metadata columns above
plus a per-view list of non-ROI columns. By default, for the **subcortical** view it
drops global/ventricular measures that are not normative ROIs:
`non-WM-hypointensities, 3rd-Ventricle, 4th-Ventricle, 5th-Ventricle, eTIV,
EstimatedTotalIntraCranialVol`. To change what is excluded for a view, edit
`self.modality_drops` in `data.py`. Constant (all-zero) columns are dropped
automatically; features are scaled with a `RobustScaler` fit on the training set.

## Training

Edit the `CONFIG` block at the top of `sigma_vae/train.py`, set the data paths, then:

```bash
python sigma_vae/train.py
```

The four paper variants are selected with a single switch, `MODEL_VARIANT`:

| `MODEL_VARIANT` | Regularization | Sparsity | = |
|-----------------|----------------|----------|---|
| `kl`     | KL      | –  | Vanilla multi-view VAE |
| `kl_l0`  | KL      | L0 | KL + sparsity |
| `sigreg` | SIGReg  | –  | SIGReg, no sparsity |
| `full`   | SIGReg  | L0 | **SIGMA-VAE** |

Training is two-phase (structure discovery with active L0 + temperature annealing,
then fine-tuning with frozen gates). For the paper configuration set the raw latent
size `LATENT_DIM = 32`; L0 then selects the active dimensions (K\*=5 for the full model).
The decorrelation weight `DEC_WEIGHT` is `0`, so the optimized objective matches Eq. 2
in the paper.

## Pretrained models

`pretrained_models/` contains the final weights and config for all four ablations
(`sigma_vae_full`, `sigreg`, `kl_l0`, `kl_baseline`), plus each run's
`performance_summary.csv`. Load a checkpoint with the matching class:

```python
import torch
from sigma_vae.models import ConditionalMMVAE
ckpt = torch.load("pretrained_models/sigma_vae_full/final_model.pth",
                  map_location="cpu", weights_only=False)
```

> Weights are ~15 MB each. For GitHub, track `*.pth` with **Git LFS**, or attach them
> to a GitHub Release / Zenodo (a DOI is convenient to cite in the paper).

## Interpretation (Fig. 3)

`interpretation/` holds the perturbation and brain-surface rendering scripts used for the
anatomical figure. They are provided **for reference**: they currently depend on the
bundled `interpretation/model.py` definitions and on FreeSurfer template files
(fsaverage surfaces, a reference volume) that you must supply locally. Adapting them to
load the released `ConditionalMMVAE` checkpoints requires pointing them at
`sigma_vae.models`.

## Notes on data sharing

The example CSVs are synthetic. The released artifacts contain no subject-level data
(`pretrained_models/` holds only weights, configs, and aggregate metrics). 




