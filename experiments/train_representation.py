#!/usr/bin/env python3
"""
Training entry point for the molecular VAE representation model.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

from trainers.vae_trainer import VAETrainer
from utils.utils import load_yaml
import pandas as pd
import gc

def main():
    """Main function to run the VAE training process."""
    
    print("=" * 50)
    print("Training Molecular Representation by Molecular VAE")
    print("=" * 50)
    
    # Load configuration from YAML file
    config_path = 'config/default.yaml'
    config = load_yaml(config_path)
    if config is None:
        print("Failed to load configuration file.")
        return

    # Add output directory with timestamp
    output_dir = f'model_checkpoints/Molecular_VAE_Representation'
    config['output_dir'] = output_dir
    
    # Load training dataset
    try:
        df = pd.read_csv('data/processed_data/candidate_compounds_library_data.csv')
        smiles_list = df['Smiles_unify'].tolist()
        
        # Clear dataframe to save memory
        del df
        gc.collect()
        
    except FileNotFoundError:
        print("Error: The dataset file 'candidate_compounds_library_data.csv' was not found.")
        return
    
    print(f"\nTraining configuration:")
    print(f"  Device: {config['hardware']['device']}")
    print(f"  Latent size: {config['model_params']['latent_size']}")
    print(f"  Hidden size: {config['model_params']['hidden_size']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Dataset size: {len(smiles_list)} molecules")
    
    # Initialize trainer
    trainer = VAETrainer(config)
    
    # Prepare data
    print("\nPreparing data...")
    
    train_loader, val_loader = trainer.prepare_data(smiles_list)
    
    # Clear SMILES list to save memory
    del smiles_list
    gc.collect()
    
    # Train model
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Model and results saved in: {config['output_dir']}")
    print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Please check your environment and dependencies.")
        import traceback
        traceback.print_exc()
