"""
Generate small SYNTHETIC example data with the exact schema SIGMA-VAE expects.
No real subjects — values are random. Run:  python make_example_data.py
Produces three view CSVs (one row per subject, aligned by `subject_id`).
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
N = 30  # subjects

# Desikan-Killiany cortical regions (34 per hemisphere -> 68 columns)
DK = ["bankssts","caudalanteriorcingulate","caudalmiddlefrontal","cuneus","entorhinal",
"fusiform","inferiorparietal","inferiortemporal","isthmuscingulate","lateraloccipital",
"lateralorbitofrontal","lingual","medialorbitofrontal","middletemporal","parahippocampal",
"paracentral","parsopercularis","parsorbitalis","parstriangularis","pericalcarine",
"postcentral","posteriorcingulate","precentral","precuneus","rostralanteriorcingulate",
"rostralmiddlefrontal","superiorfrontal","superiorparietal","superiortemporal","supramarginal",
"frontalpole","temporalpole","transversetemporal","insula"]
cortical_cols = [f"{h}_{r}" for h in ("lh","rh") for r in DK]   # 68

# aseg subcortical volumes (kept) + columns that data.py drops for the subcortical view
subcort_keep = [f"{s}-{r}" for s in ("Left","Right") for r in
    ("Lateral-Ventricle","Cerebellum-Cortex","Thalamus-Proper","Caudate","Putamen",
     "Pallidum","Hippocampus","Amygdala","Accumbens-area","VentralDC")] + ["Brain-Stem","CSF"]
subcort_drop = ["non-WM-hypointensities","3rd-Ventricle","4th-Ventricle","5th-Ventricle",
                "eTIV","EstimatedTotalIntraCranialVol"]   # auto-dropped by DataProcessor

# shared metadata (identical across the three views so they align on subject_id)
meta = pd.DataFrame({
    "subject_id": [f"sub-{i:04d}" for i in range(N)],
    "age": rng.integers(8, 85, N).astype(float),
    "sex": rng.choice(["M","F"], N),
    "site": rng.choice(["siteA","siteB"], N),
    "dataset_name": rng.choice(["EXAMPLE1","EXAMPLE2"], N),
    "diagnosis": rng.choice(["healthy","patient"], N, p=[0.7,0.3]),
})

def view(feature_cols, scale):
    df = meta.copy()
    for c in feature_cols:
        df[c] = rng.normal(scale, scale*0.1, N).round(3)
    return df

view(subcort_keep + subcort_drop, 4000).to_csv("subcortical_example.csv", index=False)
view(cortical_cols, 2.5).to_csv("cortical_example.csv", index=False)     # thickness (mm)
view(cortical_cols, 2200).to_csv("surface_example.csv", index=False)     # area (mm^2)
print("Wrote subcortical_example.csv, cortical_example.csv, surface_example.csv")
