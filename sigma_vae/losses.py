#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss Functions for Conditional Multi-Modal VAE
SIGReg implementation + Decorrelation Loss for better disentanglement

CHANGES FROM ORIGINAL:
- Added decorrelation_loss function
- Added decorr weight to LossWeights
- Added decorr loss to forward pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass


# ==============================================================================
# 1. SIGREG - EXACT LEJEPA IMPLEMENTATION (UNCHANGED)
# ==============================================================================

class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (LeJEPA).
    
    Exact implementation from the paper pseudocode.
    
    Reference: LeJEPA (Balestriero & LeCun, 2025)
    """
    
    def __init__(self, latent_dim: int, num_slices: int = 256):
        """
        Args:
            latent_dim: Dimension of latent space
            num_slices: Number of random projection directions
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_slices = max(latent_dim * 8, 64)  # e.g. 40 for D=5, 256 for D=32
        
        # Integration points for characteristic function
        t = torch.linspace(-5, 5, 17)
        self.register_buffer('t', t)
        
        # Theoretical CF for N(0,1) - also used as Gaussian window weight
        exp_f = torch.exp(-0.5 * t ** 2)
        self.register_buffer('exp_f', exp_f)
        
        # Track step for reproducible projections (optional)
        self.register_buffer('step', torch.tensor(0))
    
    def forward(self, x: torch.Tensor, global_step: Optional[int] = None) -> torch.Tensor:
        """
        Compute SIGReg loss.
        
        Args:
            x: [N, D] latent representations
            global_step: Optional step for reproducible projections across devices
        Returns:
            Scalar SIGReg loss
        """
        N, D = x.shape
        
        if N < 4:
            return x.new_zeros(())
        
        device = x.device
        
        # Slice sampling - use step for reproducibility if provided
        if global_step is not None:
            g = torch.Generator(device=device)
            g.manual_seed(global_step)
            A = torch.randn(D, self.num_slices, generator=g, device=device)
        else:
            A = torch.randn(D, self.num_slices, device=device)
        
        # Normalize projection directions
        A = A / A.norm(p=2, dim=0)
        
        # Project: [N, D] @ [D, M] -> [N, M]
        # Then multiply by integration points t: [N, M, 1] * [T] -> [N, M, T]
        x_proj = x @ A  # [N, M]
        x_t = x_proj.unsqueeze(2) * self.t  # [N, M, T]
        
        # Empirical characteristic function (complex exponential)
        # ecf = E[exp(i * t * x)] averaged over samples
        ecf = torch.exp(1j * x_t).mean(dim=0)  # [M, T]
        
        # Weighted L2 distance from theoretical CF
        # err = |ecf - exp_f|^2 * exp_f (Gaussian window weighting)
        err = (ecf - self.exp_f).abs().square() * self.exp_f  # [M, T]
        
        # Integrate using trapezoidal rule
        T_stat = torch.trapezoid(err, self.t, dim=1)  # [M]
        
        # Average over slices
        return T_stat.mean()


# ==============================================================================
# 2. DECORRELATION LOSS (NEW)
# ==============================================================================

def decorrelation_loss(z: torch.Tensor, age: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
    """
    Penalize correlation between latent z and covariates (age, sex).
    
    This explicitly encourages disentanglement by minimizing:
        L_decorr = |corr(z, age)|² + |corr(z, sex)|²
    
    Args:
        z: Latent representations [N, latent_dim]
        age: Age values [N] (should be normalized)
        sex: Sex values [N] (binary 0/1)
    
    Returns:
        Scalar loss (sum of squared correlations)
    """
    # Normalize z across samples (per dimension)
    z_norm = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-8)
    
    # Normalize age
    age_norm = (age - age.mean()) / (age.std() + 1e-8)
    
    # Normalize sex (even though binary)
    sex_norm = (sex - sex.mean()) / (sex.std() + 1e-8)
    
    # Compute correlations: corr = E[z_norm * covariate_norm]
    # Shape: [latent_dim]
    corr_age = (z_norm * age_norm.unsqueeze(1)).mean(0)
    corr_sex = (z_norm * sex_norm.unsqueeze(1)).mean(0)
    
    # Loss = mean squared correlation (penalize any correlation)
    loss = corr_age.pow(2).mean() + corr_sex.pow(2).mean()
    
    return loss


# ==============================================================================
# 3. LOSS CLASSES
# ==============================================================================

@dataclass
class LossWeights:
    """Loss weights for Conditional MVAE."""
    recon: float = 1.0
    sigreg: float = 1.0
    l0: float = 0.01
    logvar: float = 1e-4
    decorr: float = 0
    kl: float = 0.0  # ABLATION: use KL instead of SIGReg when > 0


class ConditionalVAELoss(nn.Module):
    """
    Combined loss for Conditional Multi-Modal VAE.
    
    Components:
    - Reconstruction loss (MSE)
    - SIGReg: Isotropic Gaussian regularization (LeJEPA)
    - L0: Sparsity on latent dimensions
    - Logvar: Variance stabilization
    - Decorr: Decorrelation from age/sex (NEW)
    """
    
    def __init__(
        self, 
        latent_dim: int,
        weights: Optional[LossWeights] = None,
        num_slices: int = 256
    ):
        """
        Args:
            latent_dim: Dimension of latent space
            weights: Loss weight configuration
            num_slices: Number of random projections for SIGReg
        """
        super().__init__()
        self.weights = weights or LossWeights()
        self.global_step = 0
        
        # Initialize SIGReg module
        self.sigreg = SIGRegLoss(
            latent_dim, 
            num_slices=num_slices
        ) if self.weights.sigreg > 0 else None
        
    def reconstruction_loss(self, x_dict: Dict, recons: Dict) -> torch.Tensor:
        """Compute mean MSE reconstruction loss across modalities."""
        total = sum(
            F.mse_loss(recons[name], x_dict[name], reduction='mean')
            for name in recons.keys()
        )
        return total
    
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor],
        model_output: Dict[str, torch.Tensor],
        age: torch.Tensor,
        sex: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        NEW: Added age and sex parameters for decorrelation loss
        
        Args:
            x_dict: Input features per modality
            model_output: Output dict from model forward pass
            age: Age values [N] (normalized)
            sex: Sex values [N] (0/1)
            model: Model instance (for L0 gate access)
        
        Returns:
            Dict with 'total' and individual loss components as tensors.
        """
        device = model_output['z_raw'].device
        losses = {}
        total = torch.zeros((), device=device)
        
        # 1. Reconstruction
        recon_loss = self.reconstruction_loss(x_dict, model_output['recons'])
        losses['recon'] = recon_loss
        total = total + self.weights.recon * recon_loss
        
        # 2a. SIGReg (Isotropic Gaussian regularization)
        if self.sigreg is not None and self.weights.sigreg > 0:
            sigreg_loss = self.sigreg(model_output['z_raw'], self.global_step)
            losses['sigreg'] = sigreg_loss
            total = total + self.weights.sigreg * sigreg_loss
            self.global_step += 1

        # 2b. KL divergence (ABLATION: used instead of SIGReg)
        if self.weights.kl > 0:
            mu = model_output['mu']
            logvar = model_output['logvar']
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            losses['kl'] = kl_loss
            total = total + self.weights.kl * kl_loss
        
        # 3. Logvar stabilization
        logvar = model_output['logvar']
        logvar_mean = logvar.mean()
        logvar_loss = logvar_mean.pow(2)
        losses['logvar'] = logvar_loss
        total = total + self.weights.logvar * logvar_loss
        
        # 4. L0 regularization (sparsity)
        if model is not None and hasattr(model, 'gate'):
            l0_loss = model.gate.get_expected_l0()
            losses['l0'] = l0_loss
            total = total + self.weights.l0 * l0_loss
        
        # 5. Decorrelation (NEW)
        if self.weights.decorr > 0:
            decorr_loss = decorrelation_loss(model_output['z_raw'], age, sex)
            losses['decorr'] = decorr_loss
            total = total + self.weights.decorr * decorr_loss

        losses['total'] = total
        return losses
    
    def get_metrics(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Convert loss tensors to floats for logging."""
        return {k: v.item() for k, v in losses.items()}


# Backwards compatibility alias
DecorrelatedVAELoss = ConditionalVAELoss