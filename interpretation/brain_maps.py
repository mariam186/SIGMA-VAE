#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 06:54:34 2025

@author: marzab
"""
from nilearn import datasets, plotting
import numpy as np

fsavg = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
from nilearn.datasets import fetch_atlas_surf_destrieux

atlas = fetch_atlas_surf_destrieux()



feature_series=viz.decode_brain(z_med)
cortical=feature_series['cortical']
lh_values = cortical[cortical.index.str.startswith("lh_")]
rh_values = cortical[cortical.index.str.startswith("rh_")]

lh_values.index = lh_values.index.str.replace("lh_", "")
rh_values.index = rh_values.index.str.replace("rh_", "")

lh_map = np.zeros(len(atlas['map_left']))
rh_map = np.zeros(len(atlas['map_right']))

labels = atlas['labels']

# LEFT hemisphere
for region, value in lh_values.items():
    matches = [i for i, lab in enumerate(labels) if region in lab.lower()]
    if not matches:
        print(f"Warning: no match for {region}")
        continue
    for lab_id in matches:
        lh_map[atlas['map_left'] == lab_id] = value

# RIGHT hemisphere
for region, value in rh_values.items():
    matches = [i for i, lab in enumerate(labels) if region in lab.lower()]
    if not matches:
        print(f"Warning: no match for {region}")
        continue
    for lab_id in matches:
        rh_map[atlas['map_right'] == lab_id] = value
plotting.plot_surf_stat_map(
    fsavg.infl_left,
    lh_map,
    hemi='left',
    view='lateral',
    colorbar=True,
    title='Left hemisphere'
)

plotting.plot_surf_stat_map(
    fsavg.infl_right,
    rh_map,
    hemi='right',
    view='lateral',
    colorbar=True,
    title='Right hemisphere'
)
