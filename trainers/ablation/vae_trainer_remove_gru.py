"""
Ablation training framework for molecular VAE variants without GRU-based fusion.

Maintains beta scheduling, logging, checkpointing, and GPU-accelerated optimization.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

from models.ablation.molecular_vae_ablation_remove_gru import MolecularVAE, vae_loss_function
from datasets.molecularNN_dataset import prepare_vae_dataset
from utils.utils import setup_logging


class VAETrainer:
    """
    Trainer class for Molecular VAE with simple feature fusion.
    
    This trainer handles the complete training pipeline including data preparation,
    model initialization, training/validation loops, checkpointing, and metrics tracking.
    It supports beta scheduling for KL divergence, early stopping, learning rate scheduling,
    and comprehensive logging. This version uses simple fusion methods instead of GRU.
    
    Attributes:
        config (dict): Training configuration dictionary
        device (torch.device): Computation device (CPU/GPU)
        model (MolecularVAE): The molecular VAE model
        optimizer (torch.optim.Optimizer): Model optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        logger (logging.Logger): Training logger
        output_dir (str): Output directory for results and checkpoints
        
    Training Metrics:
        - train_losses, val_losses: Total loss tracking
        - train_recon_losses, val_recon_losses: Reconstruction loss tracking
        - train_kl_losses, val_kl_losses: KL divergence loss tracking
        - learning_rates: Learning rate schedule tracking
    """
    
    def __init__(self, config):
        """
        Initialize the VAE trainer with configuration parameters.
        """
        self.config = config
        self.device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Validate fusion method
        fusion_method = config['model_params'].get('fusion_method', 'concat')
        valid_methods = ['concat', 'add', 'fingerprint_only']
        if fusion_method not in valid_methods:
            raise ValueError(f"fusion_method must be one of {valid_methods}, got {fusion_method}")
        
        # Initialize model
        self.model = MolecularVAE(config['model_params']).to(self.device)
        print(self.model)
        print(f"Using fusion method: {fusion_method}")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-5)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config['scheduler'].get('factor', 0.5), 
            patience=config['scheduler'].get('patience', 10)
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
        self.val_recon_losses = []
        self.val_kl_losses = []
        self.learning_rates = []  
        
        # Feature normalization parameters
        self.feature_normalization = None
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = config.get('output_dir', f'vae_results_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.output_dir, 'training.log')
        self.logger = setup_logging(log_file)
        
        # Save config
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def prepare_data(self, smiles_list):
        """
        Prepare molecular fingerprint and feature data for training.
        This method now uses the dataset module with custom Molecule structure.
        
        Args:
            smiles_list (list): List of SMILES strings
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        self.logger.info("Using dataset module with custom Molecule structure...")
        self.logger.info(f"Feature fusion method: {self.config['model_params'].get('fusion_method', 'concat')}")
        
        # Use the dataset module function for processing
        train_loader, val_loader, normalization_params = prepare_vae_dataset(
            smiles_list=smiles_list,
            config=self.config,
            logger=self.logger
        )
        
        # Store normalization parameters
        self.feature_normalization = normalization_params
        
        # Update model's feature size if encoding was applied
        if normalization_params and 'encoded_feature_size' in normalization_params:
            old_feature_size = self.config['model_params']['feature_size']
            new_feature_size = normalization_params['encoded_feature_size']
            self.config['model_params']['feature_size'] = new_feature_size
            
            # Log the update
            self.logger.info(f"Updating model feature size from {old_feature_size} to {new_feature_size}")
            self.logger.info(f"Target reconstruction size: {self.config['model_params']['fingerprint_size']} (fingerprints only)")
            
            # Reinitialize model with updated feature size
            self.model = MolecularVAE(self.config['model_params']).to(self.device)
            
            # Reinitialize optimizer with new model parameters
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training'].get('weight_decay', 1e-5)
            )
            
            # Reinitialize scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=self.config['scheduler'].get('factor', 0.5), 
                patience=self.config['scheduler'].get('patience', 10)
            )
        
        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch):
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch number
            
        Returns:
            tuple: (avg_total_loss, avg_recon_loss, avg_kl_loss)
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        # Improved beta scheduling with max cap
        beta_start = self.config['vae'].get('beta_start', 0.0)
        beta_increment = self.config['vae'].get('beta_increment', 0.001)
        beta_max = self.config['vae'].get('beta_max', 1.0)
        
        # Calculate beta value with cap
        beta_value = min(beta_max, beta_start + epoch * beta_increment)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        
        for batch_idx, (fingerprint, feature, target) in enumerate(progress_bar):
            try:
                fingerprint = fingerprint.to(self.device)
                feature = feature.to(self.device) if feature is not None else None
                target = target.to(self.device)
                
                # Debug: Check tensor shapes and values (only first batch of first epoch)
                if batch_idx == 0 and epoch == 0:  # Only log first batch of first epoch
                    self.logger.info(f"Batch shapes - FP: {fingerprint.shape}, Target: {target.shape}")
                    if feature is not None:
                        self.logger.info(f"Feature shape: {feature.shape}")
                        self.logger.info(f"Value ranges - FP: [{fingerprint.min():.3f}, {fingerprint.max():.3f}], "
                                    f"Feat: [{feature.min():.3f}, {feature.max():.3f}], "
                                    f"Target: [{target.min():.3f}, {target.max():.3f}]")
                    else:
                        self.logger.info(f"Value ranges - FP: [{fingerprint.min():.3f}, {fingerprint.max():.3f}], "
                                    f"Target: [{target.min():.3f}, {target.max():.3f}]")
                    self.logger.info(f"Fusion method: {self.config['model_params'].get('fusion_method', 'concat')}")
                
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstruction, mu, logvar = self.model(fingerprint, feature)
                
                # Debug: Check reconstruction (only first batch of first epoch)
                if batch_idx == 0 and epoch == 0:
                    self.logger.info(f"Reconstruction shape: {reconstruction.shape}")
                    self.logger.info(f"Reconstruction range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]")
                    self.logger.info(f"Latent mu range: [{mu.min():.3f}, {mu.max():.3f}]")
                    self.logger.info(f"Latent logvar range: [{logvar.min():.3f}, {logvar.max():.3f}]")
                
                # Calculate loss with fixed KL calculation
                loss, recon_loss, kl_loss = vae_loss_function(
                    reconstruction, target, mu, logvar, beta=beta_value
                )
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(f"Invalid loss at batch {batch_idx}: total={loss}, recon={recon_loss}, kl={kl_loss}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                            self.config['training'].get('gradient_clip', 1.0))
                
                self.optimizer.step()
                
                # Accumulate losses (only if batch succeeded)
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1
                
                # Update progress bar with more reasonable KL values
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'KL': f'{kl_loss.item():.4f}',
                    'Beta': f'{beta_value:.4f}'
                })
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Handle case where no batches succeeded
        if num_batches == 0:
            self.logger.error("No batches completed successfully!")
            return 0.0, 0.0, 0.0
        
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_total_loss, avg_recon_loss, avg_kl_loss
    
    def validate_epoch(self, val_loader):
        """
        Validate the model for one epoch.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            tuple: (avg_total_loss, avg_recon_loss, avg_kl_loss)
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for fingerprint, feature, target in val_loader:
                try:
                    fingerprint = fingerprint.to(self.device)
                    feature = feature.to(self.device) if feature is not None else None
                    target = target.to(self.device)
                    
                    # Forward pass
                    reconstruction, mu, logvar = self.model(fingerprint, feature)
                    
                    # Calculate loss
                    loss, recon_loss, kl_loss = vae_loss_function(
                        reconstruction, target, mu, logvar, beta=1.0
                    )
                    
                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    # Accumulate losses
                    total_loss += loss.item()
                    total_recon_loss += recon_loss.item()
                    total_kl_loss += kl_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Validation error: {e}")
                    continue
        
        # Handle case where no batches succeeded
        if num_batches == 0:
            self.logger.error("No validation batches completed successfully!")
            return 0.0, 0.0, 0.0
        
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_total_loss, avg_recon_loss, avg_kl_loss

    def train(self, train_loader, val_loader):
        """
        Main training loop.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Feature fusion method: {self.config['model_params'].get('fusion_method', 'concat')}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            # Training
            train_loss, train_recon, train_kl = self.train_epoch(train_loader, epoch)
            
            # Check if training failed completely
            if train_loss == 0.0 and train_recon == 0.0 and train_kl == 0.0:
                self.logger.error(f"Training failed at epoch {epoch+1}, stopping early")
                break
            
            # Validation
            val_loss, val_recon, val_kl = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_recon_losses.append(train_recon)
            self.train_kl_losses.append(train_kl)
            self.val_recon_losses.append(val_recon)
            self.val_kl_losses.append(val_kl)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr']) # Record LR for this epoch
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train Recon: {train_recon:.4f}, Train KL: {train_kl:.4f}, "
                f"Val Recon: {val_recon:.4f}, Val KL: {val_kl:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.config['checkpointing'].get('save_best', True):
                    self.save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['checkpointing'].get('save_freq', 10) == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if patience_counter >= self.config['early_stopping'].get('patience', 20):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.logger.info("Training completed!")
        
        # Save and Plot training curves
        self.plot_training_curves()

    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Add feature normalization parameters if they exist
        if self.feature_normalization is not None:
            checkpoint['feature_normalization'] = self.feature_normalization
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'best_model.pth'))
            self.logger.info(f"Best model saved at epoch {epoch+1}")
        
        torch.save(checkpoint, os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    def plot_training_curves(self):
        """
        Plot and save training curves and their underlying data.
        
        Creates visualizations of training progress and exports all metrics
        to a CSV file for further analysis.
        """
        # Gather all metrics into a dictionary
        metrics_data = {
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_recon_loss': self.train_recon_losses,
            'val_recon_loss': self.val_recon_losses,
            'train_kl_loss': self.train_kl_losses,
            'val_kl_loss': self.val_kl_losses,
            'learning_rate': self.learning_rates
        }
        
        # Create a pandas DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Save the DataFrame to a CSV file
        metrics_filepath = os.path.join(self.output_dir, 'training_metrics.csv')
        df.to_csv(metrics_filepath, index=False)
        self.logger.info(f"Training metrics saved to {metrics_filepath}")
        
        # Create training curve plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Reconstruction Loss
        axes[0, 1].plot(self.train_recon_losses, label='Train Recon Loss / Bit', color='blue')
        axes[0, 1].plot(self.val_recon_losses, label='Val Recon Loss / Bit', color='red')
        axes[0, 1].set_title('Reconstruction Loss per Bit')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL Divergence Loss
        axes[1, 0].plot(self.train_kl_losses, label='Train KL Loss', color='blue')
        axes[1, 0].plot(self.val_kl_losses, label='Val KL Loss', color='red')
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(self.learning_rates)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info("Training curves saved!")
