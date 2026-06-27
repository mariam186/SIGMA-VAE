#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 03:18:00 2026

@author: marzab
"""
"""
Disentanglement Diagnostics for Conditional MVAE

Verifies that latent space z is independent of covariates (age, sex)
by attempting to predict them using simple linear classifiers.

If conditioning works correctly:
- Age prediction from z should have R ≈ 0 (no correlation)
- Sex prediction from z should have accuracy ≈ 50% (chance level)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from typing import Dict, Tuple, Optional


def check_disentanglement(
    z: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    active_indices: Optional[list] = None,
    cv_folds: int = 5,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Check if latent space z is disentangled from age and sex.
    
    Uses cross-validated linear models to predict age/sex from z.
    If conditioning worked, predictions should be at chance level.
    
    Args:
        z: Latent representations [N, K]
        age: Age values [N,] (normalized or raw)
        sex: Sex values [N,] (0/1)
        active_indices: If provided, only use these latent dimensions
        cv_folds: Number of cross-validation folds
        verbose: Print results
    
    Returns:
        Dict with metrics:
            - age_r: Pearson correlation for age prediction
            - age_r2: R² score for age prediction
            - sex_acc: Accuracy for sex prediction
            - sex_chance: Chance-level accuracy (majority class)
    """
    # Use only active dimensions if specified
    if active_indices is not None and len(active_indices) > 0:
        z = z[:, active_indices]
    
    # Standardize latents for fair comparison
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    
    results = {}
    
    # --- Age prediction (regression) ---
    age_model = Ridge(alpha=1.0)
    
    # Cross-validated predictions
    age_pred = cross_val_predict(age_model, z_scaled, age, cv=cv_folds)
    
    # Compute correlation
    age_r, age_p = pearsonr(age, age_pred)
    age_r2 = 1 - np.sum((age - age_pred)**2) / np.sum((age - age.mean())**2)
    age_mae = np.mean(np.abs(age - age_pred))
    
    results['age_r'] = age_r
    results['age_r2'] = age_r2
    results['age_mae'] = age_mae
    results['age_p'] = age_p
    
    # --- Sex prediction (classification) ---
    sex_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Cross-validated accuracy
    sex_scores = cross_val_score(sex_model, z_scaled, sex, cv=cv_folds, scoring='accuracy')
    sex_acc = sex_scores.mean()
    sex_std = sex_scores.std()
    
    # Chance level (majority class)
    sex_chance = max(sex.mean(), 1 - sex.mean())
    
    results['sex_acc'] = sex_acc
    results['sex_std'] = sex_std
    results['sex_chance'] = sex_chance
    
    # --- Interpretation ---
    # Good disentanglement: |age_r| < 0.1, sex_acc close to sex_chance
    age_disentangled = abs(age_r) < 0.15
    sex_disentangled = abs(sex_acc - sex_chance) < 0.05
    
    results['age_disentangled'] = age_disentangled
    results['sex_disentangled'] = sex_disentangled
    results['fully_disentangled'] = age_disentangled and sex_disentangled
    
    if verbose:
        print("\n" + "="*60)
        print("DISENTANGLEMENT DIAGNOSTIC")
        print("="*60)
        print(f"Latent dims used: {z.shape[1]}")
        print(f"Samples: {z.shape[0]}")
        print("-"*60)
        
        print("\nAge Prediction (Ridge Regression):")
        print(f"  Pearson R: {age_r:.3f} (p={age_p:.2e})")
        print(f"  R² score:  {age_r2:.3f}")
        print(f"  MAE:       {age_mae:.4f}")
        if age_disentangled:
            print(f"  ✓ PASSED: |R| < 0.15 → Age info removed from z")
        else:
            print(f"  ✗ FAILED: |R| >= 0.15 → z still encodes age")
        
        print("\nSex Prediction (Logistic Regression):")
        print(f"  Accuracy:  {sex_acc:.1%} ± {sex_std:.1%}")
        print(f"  Chance:    {sex_chance:.1%}")
        if sex_disentangled:
            print(f"  ✓ PASSED: Accuracy ≈ chance → Sex info removed from z")
        else:
            print(f"  ✗ FAILED: Accuracy > chance → z still encodes sex")
        
        print("-"*60)
        if results['fully_disentangled']:
            print("✓ DISENTANGLEMENT SUCCESSFUL")
            print("  Latent space z is independent of age and sex.")
        else:
            print("⚠ DISENTANGLEMENT INCOMPLETE")
            print("  Consider: stronger conditioning, more training, or architecture changes.")
        print("="*60)
    
    return results


def check_disentanglement_svm(
    z: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    active_indices: Optional[list] = None,
    cv_folds: int = 5,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Same as check_disentanglement but using SVM models.
    
    SVMs with RBF kernel can capture non-linear relationships,
    so this is a stricter test of disentanglement.
    """
    # Use only active dimensions if specified
    if active_indices is not None and len(active_indices) > 0:
        z = z[:, active_indices]
    
    # Standardize latents
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    
    results = {}
    
    # --- Age prediction (SVM regression) ---
    age_model = SVR(kernel='rbf', C=1.0)
    age_pred = cross_val_predict(age_model, z_scaled, age, cv=cv_folds)
    
    age_r, age_p = pearsonr(age, age_pred)
    results['age_r_svm'] = age_r
    results['age_p_svm'] = age_p
    
    # --- Sex prediction (SVM classification) ---
    sex_model = SVC(kernel='rbf', C=1.0, random_state=42)
    sex_scores = cross_val_score(sex_model, z_scaled, sex, cv=cv_folds, scoring='accuracy')
    sex_acc = sex_scores.mean()
    sex_chance = max(sex.mean(), 1 - sex.mean())
    
    results['sex_acc_svm'] = sex_acc
    results['sex_chance'] = sex_chance
    
    if verbose:
        print("\nSVM Diagnostic (stricter, non-linear):")
        print(f"  Age R (SVR):      {age_r:.3f}")
        print(f"  Sex Acc (SVC):    {sex_acc:.1%} (chance: {sex_chance:.1%})")
    
    return results