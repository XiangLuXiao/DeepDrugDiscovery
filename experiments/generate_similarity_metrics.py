#!/usr/bin/env python3
"""
Compute similarity metrics between reference and candidate embeddings using
the trained attention-based molecular model.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

import pandas as pd
import numpy as np
import os
from models.molecular_attention import molecular_attention

def main():
    """Compute similarity matrix between reference and candidate compounds."""
    
    # Create output directory
    os.makedirs('data/similarity_metrics', exist_ok=True)
    
    # Load compound data
    try:
        reference_df = pd.read_csv('./data/processed_data/reference_compounds_data.csv')
        compound_df = pd.read_csv('./data/processed_data/candidate_compounds_library_data.csv')
        print(f"Loaded reference compounds: {len(reference_df)}")
        print(f"Loaded candidate compounds: {len(compound_df)}")
        
    except FileNotFoundError:
        print("Error: The dataset files were not found.")
        return
    
    # Load pre-computed embeddings
    try:
        cmp_embeddings = np.load("data/processed_data/cmp_embeddings.npz")
        query_FP_list = cmp_embeddings['embeddings']
        
        reference_embeddings = np.load("data/processed_data/reference_embeddings.npz")
        tar_FP_list = reference_embeddings['embeddings']
        
        print(f"Query embeddings shape: {query_FP_list.shape}")
        print(f"Target embeddings shape: {tar_FP_list.shape}")
        
    except FileNotFoundError:
        print("Error: The embedding files were not found.")
        print("Please generate embeddings first using the VAE inference script.")
        return
    
    print("=== Computing Similarity Matrix ===")
    # Compute similarity matrix using molecular attention
    sim_mat = molecular_attention(tar_FP_list, query_FP_list, '0')
    
    print(f"Similarity matrix shape: {sim_mat.shape}")
    print(f"Similarity range: [{sim_mat.min():.4f}, {sim_mat.max():.4f}]")
    
    # Save similarity matrix
    np.save('data/similarity_metrics/sim_mat.npy', sim_mat)
    print("Similarity matrix saved to: data/similarity_metrics/sim_mat.npy")
    
    print("Done!")

if __name__ == "__main__":
    main()
