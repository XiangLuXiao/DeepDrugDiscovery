"""
Hybrid molecular VAE architecture with GRU-based feature fusion for fingerprint learning.

Implements configurable encoder/decoder stacks, beta scheduling, and dropout/batchnorm support.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MolecularVAE(nn.Module):
    """
    A Hybrid Variational Autoencoder (VAE) for molecular fingerprints with optional features.
    
    This VAE architecture is designed to learn meaningful latent representations of molecular
    data by combining molecular fingerprints with optional auxiliary features through a
    GRU-based fusion mechanism. The model supports variable-depth encoder/decoder stacks
    and includes modern training stabilization techniques.
    """
    
    def __init__(self, args):
        """
        Initialize the Molecular VAE with specified architecture parameters.
        """
        super(MolecularVAE, self).__init__()
        self.param = args
        
        # Extract and store architecture parameters
        self.fingerprint_size = self.param['fingerprint_size']
        self.feature_size = self.param.get('feature_size', 0)
        self.latent_size = self.param['latent_size']
        self.hidden_size = self.param['hidden_size']
        self.dropout = self.param['dropout']
        
        # Input processing layers
        self.fingerprint_encoder = nn.Linear(self.fingerprint_size, self.hidden_size)
        
        # Optional feature processing for hybrid input
        if self.feature_size > 0:
            self.feature_encoder = nn.Linear(self.feature_size, self.hidden_size)
            self.grucell = nn.GRUCell(self.hidden_size, self.hidden_size)
        
        # Encoder and decoder stacks with configurable depth
        self.encoder = self._build_network_stack(
            self.param.get('encoder_n_layers', 2), 
            self.hidden_size
        )
        self.decoder = self._build_network_stack(
            self.param.get('decoder_n_layers', 1), 
            self.latent_size
        )
        
        # Latent space parameterization layers
        self.fc_mu = nn.Linear(self.hidden_size, self.latent_size)
        self.fc_logvar = nn.Linear(self.hidden_size, self.latent_size)
        
        # Output reconstruction layer
        self.output_layer = nn.Linear(self.hidden_size, self.fingerprint_size)

    def _build_network_stack(self, n_layers: int, input_size: int) -> nn.Sequential:
        """
        Build a multi-layer neural network stack with batch normalization and dropout.
        
        Creates a sequence of linear layers with batch normalization, ReLU activation,
        and dropout for training stability. The first layer adapts from input_size to
        hidden_size, while subsequent layers maintain hidden_size dimensions.
        """
        layers = []
        
        # First layer: input_size → hidden_size
        layers.extend([
            nn.Linear(input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ])
        
        # Subsequent layers: hidden_size → hidden_size
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            
        return nn.Sequential(*layers)
        
    def _process_inputs(self, fingerprint: torch.Tensor, feature: torch.Tensor = None) -> torch.Tensor:
        """
        Process and optionally fuse fingerprint and auxiliary feature inputs.
        
        Encodes the molecular fingerprint through a linear layer and optionally
        fuses it with auxiliary features using a GRU cell for enhanced representation.
        """
        # Encode fingerprint to hidden space
        fp_embed = F.relu(self.fingerprint_encoder(fingerprint))
        
        # Optional feature fusion using GRU cell
        if feature is not None and self.feature_size > 0:
            feature_embed = F.relu(self.feature_encoder(feature))
            # GRU-based fusion: fingerprint as input, features as hidden state
            combined_features = self.grucell(fp_embed, feature_embed)
            return combined_features
        else:
            return fp_embed

    def encode(self, fingerprint: torch.Tensor, feature: torch.Tensor = None) -> tuple:
        """
        Encode input data into latent space parameters (mean and log-variance).
        
        Processes the input through the feature fusion mechanism and encoder stack
        to produce the parameters of the latent distribution.
        """
        processed_input = self._process_inputs(fingerprint, feature)
        h = self.encoder(processed_input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Apply the reparameterization trick for sampling from the latent distribution.
        
        Enables backpropagation through stochastic sampling by expressing the sample
        as a deterministic function of the parameters and random noise.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors back to reconstructed fingerprints.
        
        Processes latent representations through the decoder stack to reconstruct
        the original fingerprint space.
        """
        h = self.decoder(z)
        reconstruction = self.output_layer(h)
        return torch.sigmoid(reconstruction)

    def forward(self, fingerprint: torch.Tensor, feature: torch.Tensor = None, 
               output_embedding: bool = False) -> tuple:
        """
        Complete forward pass through the VAE.
        
        Performs encoding, optional sampling, and decoding. Can return either
        reconstruction outputs for training or latent embeddings for inference.
        """
        mu, logvar = self.encode(fingerprint, feature)
        
        # Return embeddings only for inference
        if output_embedding:
            return mu
            
        # Full forward pass for training
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar


def vae_loss_function(reconstruction: torch.Tensor, target: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> tuple:
    """
    Calculate the VAE loss combining reconstruction loss and KL divergence.
    
    Implements the Evidence Lower Bound (ELBO) objective with optional beta weighting
    for controlled disentanglement in Beta-VAE formulation.
    
    Args:
        reconstruction (torch.Tensor): Reconstructed fingerprints from decoder
        target (torch.Tensor): Original input fingerprints (ground truth)
        mu (torch.Tensor): Latent distribution means from encoder
        logvar (torch.Tensor): Latent distribution log-variances from encoder
        beta (float, optional): Weighting factor for KL divergence term. 
                               Defaults to 1.0 (standard VAE)
                               
    Returns:
        tuple: (total_loss, reconstruction_loss, kl_loss) where:
            - total_loss (torch.Tensor): Combined weighted loss for optimization
            - reconstruction_loss (torch.Tensor): Binary cross-entropy reconstruction loss
            - kl_loss (torch.Tensor): Kullback-Leibler divergence regularization loss
            
    Mathematical Details:
        - Reconstruction Loss: BCE(reconstruction, target)
        - KL Loss: -0.5 * mean(1 + logvar - mu² - exp(logvar))
        - Total Loss: reconstruction_loss + beta * kl_loss
        
    Note:
        Beta values > 1.0 encourage disentanglement but may reduce reconstruction quality.
        Beta values < 1.0 prioritize reconstruction over regularization.
    """
    # Reconstruction loss using binary cross-entropy for binary fingerprints
    reconstruction_loss = F.binary_cross_entropy(reconstruction, target, reduction='mean')
    
    # KL divergence loss: regularizes latent space to approximate standard normal
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Combined loss with beta weighting
    total_loss = reconstruction_loss + beta * kl_loss
    
    return total_loss, reconstruction_loss, kl_loss
