# Interpretation (Fig. 3) — reference scripts

These reproduce the latent-perturbation analysis and subcortical brain rendering.
They are **not** part of the turnkey training pipeline:

- `latent_interpretation.py` and `brain_vis.py` import the bundled `model.py`
  (`DecorrelatedMMVAE`) and register it for checkpoint loading. To use them with the
  released `ConditionalMMVAE` checkpoints in `../pretrained_models/`, point the imports
  at `sigma_vae.models.ConditionalMMVAE`.
- `brain_vis.py` / `brain_maps.py` need FreeSurfer template files (fsaverage surfaces and
  a reference volume) on your machine; these are not redistributed here.

Treat this folder as a starting point for the anatomical figure rather than a fixed script.
