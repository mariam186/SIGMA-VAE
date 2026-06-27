#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 16:45:46 2026

@author: marzab
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 02:40:09 2026

@author: marzab
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Union


# ==============================================================================
# 1. HELPER MODULES
# ==============================================================================

class LearnableLatentGate(nn.Module):
    """L0 Regularization via Hard Concrete Distribution."""
    
    def __init__(self, latent_dim: int, temperature: float = 0.67):
        super().__init__()
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.log_alpha = nn.Parameter(torch.zeros(latent_dim))
        
        self.gamma = -0.05
        self.zeta = 1.05
        self.limit_term = math.log(-self.gamma / self.zeta)
        # Precompute constant for get_expected_l0
        self.temp_limit = self.temperature * self.limit_term
        self.is_frozen = False
        
        # Initialize buffers so they always exist (for checkpoint compatibility)
        self.register_buffer('fixed_mask', torch.ones(latent_dim))
        self.register_buffer('fixed_mask_expanded', torch.ones(1, latent_dim))
        
    def freeze_all_active(self):
        """ABLATION: Freeze all gates as active (no sparsity). Bypasses L0 entirely."""
        self.is_frozen = True
        self.log_alpha.requires_grad = False
        with torch.no_grad():
            self.fixed_mask.fill_(1.0)
            self.fixed_mask_expanded.fill_(1.0)
        print(f"\n❄️ GATE FROZEN (all active): {self.latent_dim} dimensions kept open")

    def freeze(self):
        """Permanently lock gate mask based on learned alpha."""
        self.is_frozen = True
        self.log_alpha.requires_grad = False
        
        with torch.no_grad():
            final_mask = (torch.sigmoid(self.log_alpha) > 0.5).float()
            
            # Update existing buffers (not register new ones)
            self.fixed_mask.copy_(final_mask)
            self.fixed_mask_expanded.copy_(final_mask.unsqueeze(0))
        
        active_count = final_mask.sum().item()
        print(f"\n❄️ GATE FROZEN! Active Dimensions locked at: {int(active_count)}")
        return final_mask

    def forward(self, z: torch.Tensor, training: bool = True):
        # FROZEN MODE - use pre-expanded mask
        if self.is_frozen:
            return z * self.fixed_mask_expanded, self.fixed_mask

        # TRAINING MODE: Stochastic Hard Concrete
        if training:
            u = torch.rand_like(self.log_alpha)
            u = torch.clamp(u, 1e-6, 1.0 - 1e-6) 
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.temperature)
            s = s * (self.zeta - self.gamma) + self.gamma
            gate = torch.clamp(s, 0, 1)
        # INFERENCE MODE
        else:
            gate = (torch.sigmoid(self.log_alpha) > 0.5).float()
        
        return z * gate.unsqueeze(0), gate
        
    def get_active_dims(self) -> int:
        if self.is_frozen:
            return int(self.fixed_mask.sum().item())
        with torch.no_grad():
            return int((torch.sigmoid(self.log_alpha) > 0.5).sum().item())
    
    def get_active_indices(self, as_tensor: bool = False) -> Union[List[int], torch.Tensor]:
        """Return indices of active dimensions."""
        if self.is_frozen:
            indices = torch.where(self.fixed_mask > 0.5)[0]
        else:
            with torch.no_grad():
                indices = torch.where(torch.sigmoid(self.log_alpha) > 0.5)[0]
        return indices if as_tensor else indices.tolist()
    
    def get_expected_l0(self):
        if self.is_frozen:
            return self.fixed_mask.sum()
        # Use precomputed temp_limit
        prob_active = torch.sigmoid(self.log_alpha - self.temp_limit)
        return prob_active.sum()


# ==============================================================================
# 2. ENCODER & DECODER
# ==============================================================================

class Encoder(nn.Module):
    """Conditional encoder with feature attention.
    
    Takes input features x and covariates c, produces posterior parameters.
    Covariates are concatenated after feature attention, before backbone.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, 
                 covariate_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.covariate_dim = covariate_dim
        
        # Feature Attention (operates on x only, not covariates)
        self.attention_output = nn.Linear(input_dim // 2, input_dim)
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            self.attention_output,
            nn.Sigmoid()
        )

        # Backbone (takes concatenated [x_weighted; c])
        layers = []
        prev = input_dim + covariate_dim  # <-- conditioned input
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h, bias=False), 
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ])
            prev = h
        self.backbone = nn.Sequential(*layers)
        
        # Latent Projections
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        self.apply(self._init_weights)
        self._init_attention_output()
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _init_attention_output(self):
        """Initialize attention output layer to start with gates open."""
        nn.init.normal_(self.attention_output.weight, mean=0, std=0.01)
        nn.init.constant_(self.attention_output.bias, 2.0)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple:
        """
        Args:
            x: Input features [batch, input_dim]
            c: Covariates [batch, covariate_dim] (e.g., [age, sex])
        
        Returns:
            mu: Posterior mean [batch, latent_dim]
            logvar: Posterior log-variance [batch, latent_dim]
        """
        # Feature attention on x
        attention_weights = self.feature_attention(x)
        x_weighted = x * attention_weights
        
        # Concatenate covariates
        x_cond = torch.cat([x_weighted, c], dim=-1)
        
        # Backbone
        h = self.backbone(x_cond)
        
        # Latent projections
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10, 10)
        
        return mu, logvar


class Decoder(nn.Module):
    """Conditional decoder.
    
    Takes latent z and covariates c, reconstructs input features.
    """
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int,
                 covariate_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.covariate_dim = covariate_dim
        
        layers = []
        prev = latent_dim + covariate_dim  # <-- conditioned input
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h, bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent representation [batch, latent_dim]
            c: Covariates [batch, covariate_dim]
        
        Returns:
            Reconstructed features [batch, output_dim]
        """
        z_cond = torch.cat([z, c], dim=-1)
        return self.decoder(z_cond)


# ==============================================================================
# 3. MAIN MODEL
# ==============================================================================

class ConditionalMMVAE(nn.Module):
    """
    Conditional Multi-Modal VAE with:
    - Demographic conditioning (age, sex) in encoder and decoder
    - Product of Experts fusion
    - L0 gating for dimensionality discovery
    - SIGReg for isotropic latent space
    
    The latent space z captures deviations from the normative trajectory
    defined by covariates c. At z=0, the model reconstructs the expected
    anatomy for a healthy individual of given age and sex.
    """
    
    def __init__(
        self, 
        input_dims: Dict[str, int], 
        latent_dim: int = 64,
        encoder_hidden: List[int] = [256, 128],
        decoder_hidden: List[int] = [128, 256],
        covariate_dim: int = 2  # age + sex
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.covariate_dim = covariate_dim
        
        # Store sorted order once
        self.modality_names = sorted(input_dims.keys())
        self.input_dims = input_dims
        self.num_modalities = len(self.modality_names)
        
        # Conditional Encoders
        self.encoders = nn.ModuleDict({
            name: Encoder(dim, encoder_hidden, latent_dim, covariate_dim) 
            for name, dim in input_dims.items()
        })
        
        # Conditional Decoders
        self.decoders = nn.ModuleDict({
            name: Decoder(latent_dim, decoder_hidden, dim, covariate_dim) 
            for name, dim in input_dims.items()
        })
        
        # Gating Mechanism
        self.gate = LearnableLatentGate(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z from q(z|x,c) using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def product_of_experts(self, mus: torch.Tensor, logvars: torch.Tensor):
        """
        Product-of-Experts Fusion with standard normal prior.
        
        Args:
            mus: Stacked means [num_modalities, batch, latent_dim]
            logvars: Stacked logvars [num_modalities, batch, latent_dim]
        
        Returns:
            fused_mu: [batch, latent_dim]
            fused_logvar: [batch, latent_dim]
        """
        vars = torch.exp(logvars) + 1e-7  # [M, B, L]
        precisions = 1.0 / vars           # [M, B, L]
        
        # Prior: N(0, I) contributes precision=1, precision*mu=0
        prior_precision = torch.ones_like(precisions[0])  # [B, L]
        
        fused_precision = prior_precision + precisions.sum(dim=0)  # [B, L]
        fused_var = 1.0 / fused_precision                         # [B, L]
        fused_mu = fused_var * (mus * precisions).sum(dim=0)       # [B, L]
        fused_logvar = torch.log(fused_var + 1e-7)                 # [B, L]
        
        return fused_mu, fused_logvar

    def forward(self, x_dict: Dict[str, torch.Tensor], c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_dict: Dict of modality tensors {name: [batch, dim]}
            c: Covariates [batch, covariate_dim] (e.g., [age_normalized, sex])
        
        Returns:
            Dict containing:
                - recons: Reconstructed features per modality
                - z: Gated latent representation
                - z_raw: Pre-gated latent
                - mu: Posterior mean
                - logvar: Posterior log-variance
                - gate: Gate values
        """
        batch_size = None
        device = None
        
        # Count available modalities and get batch info
        available_modalities = []
        for name in self.modality_names:
            if name in x_dict and x_dict[name] is not None:
                available_modalities.append(name)
                if batch_size is None:
                    batch_size = x_dict[name].size(0)
                    device = x_dict[name].device
        
        if not available_modalities:
            raise ValueError("No data provided in x_dict")
        
        num_available = len(available_modalities)
        
        # Pre-allocate tensors for encoding results
        mus = torch.empty(num_available, batch_size, self.latent_dim, device=device)
        logvars = torch.empty(num_available, batch_size, self.latent_dim, device=device)
        
        # Encode all modalities (conditioned on c)
        for i, name in enumerate(available_modalities):
            mu, logvar = self.encoders[name](x_dict[name], c)
            mus[i] = mu
            logvars[i] = logvar
        
        # Fuse (vectorized PoE)
        if num_available > 1:
            mu_z, logvar_z = self.product_of_experts(mus, logvars)
        else:
            mu_z, logvar_z = mus[0], logvars[0]
            
        # Reparameterize
        z = self.reparameterize(mu_z, logvar_z)
        
        # Apply Gate
        z_gated, gate_val = self.gate(z, training=self.training)
        
        # Decode (conditioned on c)
        recons = {}
        for name in available_modalities:
            recons[name] = self.decoders[name](z_gated, c)
                
        return {
            'recons': recons,
            'z': z_gated,
            'z_raw': z,
            'mu': mu_z,
            'logvar': logvar_z,
            'gate': gate_val
        }
    
    def decode(self, z: torch.Tensor, c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode from latent space (for perturbation analysis).
        
        Args:
            z: Latent representation [batch, latent_dim]
            c: Covariates [batch, covariate_dim]
        
        Returns:
            Dict of reconstructed features per modality
        """
        recons = {}
        for name in self.modality_names:
            recons[name] = self.decoders[name](z, c)
        return recons
    
    def get_active_dims(self) -> int:
        return self.gate.get_active_dims()
    
    def get_active_indices(self, as_tensor: bool = False) -> Union[List[int], torch.Tensor]:
        return self.gate.get_active_indices(as_tensor=as_tensor)