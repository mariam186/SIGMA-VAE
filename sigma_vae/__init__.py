"""SIGMA-VAE: Sketched Isotropic Gaussian Multi-View Autoencoder for Normative Modeling."""
from .data import DataProcessor, MultiModalDataset, create_dataloaders
from .models import ConditionalMMVAE
from .losses import DecorrelatedVAELoss, LossWeights
from .trainer import train_model, extract_latents, compute_wscores, save_latents_to_csv, save_checkpoint
from .diagnostics import check_disentanglement
from .evaluation import Evaluator
__version__ = "1.0.0"
