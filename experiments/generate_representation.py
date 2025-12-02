"""
Utility script for generating molecular embeddings via the trained VAE.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

from inference.vae_inference import MolecularVAEInference
import pandas as pd
import gc

def main():
    
    # Example configuration
    model_path = "model_checkpoints/Molecular_VAE_Representation/best_model.pth"  # Update this path
    
    # Initialize inference engine
    inference_engine = MolecularVAEInference(model_path,batch_size=100000)
    

    
    # Load training dataset
    try:
        cmp_library = pd.read_csv('data/processed_data/candidate_compounds_library_data.csv')
        reference_df = pd.read_csv('data/processed_data/reference_compounds_data.csv')
        cmp_smiles_list = cmp_library['Smiles_unify'].tolist()
        reference_smiles_list = reference_df['Smiles_unify'].tolist()
        
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        return
    
    
    # Get embeddings for small dataset
    cmp_embeddings = inference_engine.get_embeddings(cmp_smiles_list)
    reference_embeddings = inference_engine.get_embeddings(reference_smiles_list)

    print(f"Generated candidate embeddings shape: {cmp_embeddings.shape}")
    print(f"Generated reference embeddings shape: {reference_embeddings.shape}")
    
    inference_engine.save_embeddings(
        cmp_embeddings, 
        "data/processed_data/cmp_embeddings.npz", 
        smiles_list=cmp_smiles_list,
        valid_indices=list(range(len(cmp_embeddings))),
        compress=True
    )
    
    inference_engine.save_embeddings(
        reference_embeddings, 
        "data/processed_data/reference_embeddings.npz", 
        smiles_list=reference_smiles_list,
        valid_indices=list(range(len(reference_embeddings))),
        compress=True
    )
    

if __name__ == "__main__":
    main()
