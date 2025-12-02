"""
Reconstruction quality metrics for molecular VAEs with simple reporting helpers.

This script focuses on MSE/BCE reconstruction loss, fingerprint accuracy, and Tanimoto analysis.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from inference.vae_inference import MolecularVAEInference


def calculate_reconstruction_metrics(model_path: str, 
                                   test_csv: str,
                                   smiles_col: str = "smiles",
                                   batch_size: int = 256,
                                   verbose: bool = True) -> dict:
    """
    Calculate reconstruction quality metrics for a trained Molecular VAE.
    
    Args:
        model_path (str): Path to trained VAE model checkpoint (.pth file)
        test_csv (str): Path to test dataset CSV file
        smiles_col (str): Name of SMILES column in CSV (default: "smiles")
        batch_size (int): Batch size for inference (default: 256)
        verbose (bool): Whether to print progress information
        
    Returns:
        dict: Reconstruction quality metrics including:
            - mse_loss: Mean Squared Error for fingerprint reconstruction
            - bce_loss: Binary Cross-Entropy loss for fingerprint reconstruction
            - bit_accuracy: Fraction of correctly reconstructed bits
            - exact_match_accuracy: Fraction of perfectly reconstructed fingerprints
            - summary: Human-readable summary of results
    """
    
    if verbose:
        print("Loading Molecular VAE and test data...")
    
    # Load VAE model
    vae_engine = MolecularVAEInference(model_path, batch_size=batch_size)
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    if verbose:
        print(f"Loaded {len(test_df)} test compounds")
    
    # Validate columns
    if smiles_col not in test_df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in CSV")
    
    # Process molecules to get fingerprints and features
    if verbose:
        print("Processing molecules and generating fingerprints...")
    
    fingerprints, features, valid_indices = vae_engine.process_smiles(
        test_df[smiles_col].tolist()
    )
    
    if verbose:
        print(f"Successfully processed {len(fingerprints)} molecules "
              f"({len(fingerprints)/len(test_df)*100:.1f}% success rate)")
    
    # Generate reconstructions through VAE
    if verbose:
        print("Generating reconstructions through VAE...")
    
    reconstructions = []
    device = vae_engine.device
    model = vae_engine.model
    config = vae_engine.config
    
    with torch.no_grad():
        for i in tqdm(range(0, len(fingerprints), batch_size), 
                     desc="Reconstructing", disable=not verbose):
            # Prepare batch
            batch_fps = fingerprints[i:i+batch_size]
            batch_feats = features[i:i+batch_size]
            
            # Convert to tensors
            fp_tensor = torch.from_numpy(batch_fps).float().to(device)
            feat_tensor = torch.from_numpy(batch_feats).float().to(device)
            
            # Forward pass through VAE
            if config['model_params']['feature_size'] > 0:
                reconstruction, mu, logvar = model(fp_tensor, feat_tensor)
            else:
                reconstruction, mu, logvar = model(fp_tensor)
            
            reconstructions.append(reconstruction.cpu().numpy())
    
    # Combine all reconstructions
    reconstructions = np.concatenate(reconstructions, axis=0)
    
    # Calculate core reconstruction metrics
    if verbose:
        print("Calculating reconstruction quality metrics...")
    
    # Convert to tensors for efficient computation
    orig_tensor = torch.from_numpy(fingerprints).float()
    recon_tensor = torch.from_numpy(reconstructions).float()
    
    # 1. Mean Squared Error
    mse_loss = F.mse_loss(recon_tensor, orig_tensor, reduction='mean').item()
    
    # 2. Binary Cross-Entropy Loss
    bce_loss = F.binary_cross_entropy(recon_tensor, orig_tensor, reduction='mean').item()
    
    # 3. Bit-level accuracy
    binary_predictions = (recon_tensor > 0.5).float()
    binary_targets = (orig_tensor > 0.5).float()
    
    # Bit-wise accuracy (average across all bits)
    bit_accuracy = (binary_predictions == binary_targets).float().mean().item()
    
    # Exact match accuracy (fraction of perfectly reconstructed molecules)
    exact_matches = (binary_predictions == binary_targets).all(dim=1)
    exact_match_accuracy = exact_matches.float().mean().item()
    
    # Compile basic results
    results = {
        'dataset_info': {
            'total_compounds': len(test_df),
            'processed_compounds': len(fingerprints),
            'success_rate': len(fingerprints) / len(test_df)
        },
        'reconstruction_metrics': {
            'mse_loss': mse_loss,
            'bce_loss': bce_loss,
            'bit_accuracy': bit_accuracy,
            'exact_match_accuracy': exact_match_accuracy
        }
    }
    
    # Create human-readable summary
    summary = f"""
MOLECULAR VAE RECONSTRUCTION QUALITY SUMMARY
============================================

Dataset: {results['dataset_info']['processed_compounds']} compounds processed 
Success Rate: {results['dataset_info']['success_rate']:.1%}

Core Reconstruction Metrics:
  • Mean Squared Error (MSE): {mse_loss:.6f}
  • Binary Cross-Entropy (BCE): {bce_loss:.6f}
  • Bit-wise Accuracy: {bit_accuracy:.1%}
  • Exact Match Accuracy: {exact_match_accuracy:.1%}

Interpretation:
  • Lower MSE/BCE values indicate better reconstruction
  • Higher accuracy values indicate better reconstruction
  • MSE measures continuous reconstruction error
  • BCE is the primary VAE reconstruction loss function
  • Exact match shows fraction of perfectly reconstructed fingerprints
"""
    
    results['summary'] = summary
    
    if verbose:
        print(summary)
    
    return results


def print_reconstruction_report(results: dict, save_to_file: str = None):
    """
    Print a detailed reconstruction quality report.
    
    Args:
        results (dict): Results from calculate_reconstruction_metrics()
        save_to_file (str): Optional file path to save the report
    """
    
    report = results['summary']
    
    # Add detailed statistics
    metrics = results['reconstruction_metrics']
    mse_loss = metrics['mse_loss']
    bce_loss = metrics['bce_loss']
    bit_accuracy = metrics['bit_accuracy']
    exact_match = metrics['exact_match_accuracy']
    
    report += f"""

Detailed Analysis:
  • Reconstruction Loss Analysis:
    - MSE Loss: {mse_loss:.6f}
    - BCE Loss: {bce_loss:.6f}
    - Loss Ratio (MSE/BCE): {mse_loss/bce_loss:.3f}
  
  • Accuracy Analysis:
    - Bit-wise Accuracy: {bit_accuracy:.1%}
    - Exact Match Rate: {exact_match:.1%}
    - Accuracy Difference: {(bit_accuracy - exact_match)*100:.1f} percentage points
  
  • Reconstruction Quality Categories:
    - Perfect Reconstructions: {int(exact_match * results['dataset_info']['processed_compounds'])} compounds ({exact_match:.1%})
    - Near-Perfect (>99% bits correct): Estimated {int(bit_accuracy * results['dataset_info']['processed_compounds'] * 0.1)} compounds
    - Good Reconstructions (>90% bits correct): Estimated {int(bit_accuracy * results['dataset_info']['processed_compounds'] * 0.8)} compounds

Model Performance Assessment:
"""
    
    # Assess model performance based on MSE and BCE
    if bce_loss < 0.01:
        assessment = "EXCELLENT - Very low reconstruction loss indicates outstanding model performance"
    elif bce_loss < 0.05:
        assessment = "GOOD - Low reconstruction loss indicates good model performance"
    elif bce_loss < 0.1:
        assessment = "FAIR - Moderate reconstruction loss, consider improving training"
    else:
        assessment = "POOR - High reconstruction loss indicates poor model performance, retraining recommended"
    
    report += f"  • Overall Assessment (BCE-based): {assessment}\n"
    
    # MSE assessment
    if mse_loss < 0.001:
        mse_assessment = "EXCELLENT"
    elif mse_loss < 0.01:
        mse_assessment = "GOOD"
    elif mse_loss < 0.05:
        mse_assessment = "FAIR"
    else:
        mse_assessment = "POOR"
    
    report += f"  • MSE Assessment: {mse_assessment} (MSE: {mse_loss:.6f})\n"
    
    # Accuracy assessment
    if exact_match > 0.1:
        report += f"  • Exact Match Rate: HIGH ({exact_match:.1%}) - Many molecules perfectly reconstructed\n"
    elif exact_match > 0.01:
        report += f"  • Exact Match Rate: MODERATE ({exact_match:.1%}) - Some molecules perfectly reconstructed\n"
    else:
        report += f"  • Exact Match Rate: LOW ({exact_match:.1%}) - Few perfect reconstructions\n"
    
    if bit_accuracy > 0.95:
        report += f"  • Bit-wise Accuracy: EXCELLENT ({bit_accuracy:.1%}) - Very high bit-level reconstruction\n"
    elif bit_accuracy > 0.90:
        report += f"  • Bit-wise Accuracy: GOOD ({bit_accuracy:.1%}) - Good bit-level reconstruction\n"
    elif bit_accuracy > 0.80:
        report += f"  • Bit-wise Accuracy: FAIR ({bit_accuracy:.1%}) - Fair bit-level reconstruction\n"
    else:
        report += f"  • Bit-wise Accuracy: POOR ({bit_accuracy:.1%}) - Poor bit-level reconstruction\n"
    
    print(report)
    
    if save_to_file:
        with open(save_to_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {save_to_file}")


# Example usage
if __name__ == "__main__":
    """
    Example usage of the reconstruction metrics calculator
    """

    MODEL_PATH = f"model_checkpoints/Molecular_VAE_Representation/best_model.pth"
    TEST_CSV = "data/processed_data/reference_compounds_data.csv"
    SMILES_COLUMN = "Smiles_unify"
    
    print(f"Simple Reconstruction Quality Metrics Calculator for {MODEL_PATH}")
    print("=" * 50)
    
    try:
        # Calculate reconstruction metrics
        results = calculate_reconstruction_metrics(
            model_path=MODEL_PATH,
            test_csv=TEST_CSV,
            smiles_col=SMILES_COLUMN,
            verbose=True
        )
        
        # Print detailed report
        print_reconstruction_report(results)
        
        # Access specific metrics
        print("\nAccessing specific metrics:")
        print(f"MSE Loss: {results['reconstruction_metrics']['mse_loss']:.6f}")
        print(f"BCE Loss: {results['reconstruction_metrics']['bce_loss']:.6f}")
        print(f"Bit Accuracy: {results['reconstruction_metrics']['bit_accuracy']:.1%}")
        print(f"Exact Match: {results['reconstruction_metrics']['exact_match_accuracy']:.1%}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to update the MODEL_PATH and TEST_CSV variables with your actual file paths.")
