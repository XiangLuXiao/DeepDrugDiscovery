#!/usr/bin/env python3
"""
Ligand-based screening analysis for each ablation-specific similarity matrix.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

import pandas as pd
import numpy as np
import os

def main():
    """Analyze similarity matrix to find high-similarity candidate compounds."""
    
    # Load compound data
    try:
        reference_df = pd.read_csv('./data/processed_data/reference_compounds_data.csv')
        compound_df = pd.read_csv('./data/processed_data/candidate_compounds_library_data.csv')
        print(f"Loaded reference compounds: {len(reference_df)}")
        print(f"Loaded candidate compounds: {len(compound_df)}")
        
    except FileNotFoundError:
        print("Error: The dataset files were not found.")
        return
    
    for ablation_name in ['ablation_gru_add', 'ablation_gru_concat', 'ablation_remove_features']:
        # Load similarity matrix
        try:
            sim_mat = np.load(f'data/similarity_metrics/sim_mat_{ablation_name}.npy')
            print(f"Loaded similarity matrix shape: {sim_mat.shape}")
            
        except FileNotFoundError:
            print("Error: Similarity matrix file not found.")
            print("Please run the similarity computation script first.")
            return
        
        print("=== Analyzing Similarity Matrix ===")
        
        # Initialize dictionaries to store results
        mTOR_library_idx_simi_500_dict = dict()
        mTOR_library_score_simi_500_dict = dict()
        mTOR_library_idx_high_simi_dict = dict()
        max_similarity_value = list()
        
        # Process each reference compound
        for i in range(sim_mat.shape[0]):
            high_score_index_list = []
            tmp_mat = sim_mat[i]
            
            # Get top 500 most similar compounds
            index_list = np.argsort(tmp_mat)[-1-500:]
            score_list = sim_mat[i][index_list]
            
            # Store top 500 results
            mTOR_library_score_simi_500_dict[i] = score_list
            mTOR_library_idx_simi_500_dict[i] = index_list
            
            # Find compounds with similarity >= 0.65 and < 1.0 (exclude perfect matches)
            for idx in index_list:
                if (sim_mat[i][idx] >= 0.5) & (sim_mat[i][idx] < 1):
                    high_score_index_list.append(idx)
            
            mTOR_library_idx_high_simi_dict[i] = high_score_index_list
        
        print(f"Found high similarity matches for {len(mTOR_library_idx_high_simi_dict)} reference compounds")
        
        # Extract results into lists
        target_list = []
        score_list = []
        cmpd_name_list = []
        cand_idx_list = []
        smiles_list = []
        cmpd_id_list = []
        
        for key in mTOR_library_idx_high_simi_dict.keys():
            target_id = key
            cmpd_name = reference_df.loc[key, 'Compound name in studies']
            index_list = mTOR_library_idx_high_simi_dict[key]
            
            if len(index_list) > 0:
                for cand_idx in index_list:
                    score = sim_mat[key][cand_idx]
                    smiles = compound_df.loc[cand_idx, 'Smiles_unify']
                    cmpd_id = compound_df.loc[cand_idx]
                    
                    target_list.append(key)
                    cmpd_name_list.append(cmpd_name)
                    smiles_list.append(smiles)
                    score_list.append(score)
                    cand_idx_list.append(cand_idx)
                    cmpd_id_list.append(cmpd_id)
        
        print(f"Total high similarity matches found: {len(target_list)}")
        
        # Create results dataframe
        candidate_library_df = pd.DataFrame()
        candidate_library_df['Smiles'] = smiles_list
        candidate_library_df['target_idx'] = target_list
        candidate_library_df['target_cmpd_name'] = cmpd_name_list
        candidate_library_df['similarity_score'] = score_list
        
        print(f"Created dataframe with {len(candidate_library_df)} entries")
        
        # Remove duplicate SMILES, keeping first occurrence
        candidate_library_df = candidate_library_df.drop_duplicates('Smiles', keep='first')
        print(f"After removing duplicates: {len(candidate_library_df)} entries")
        
        # Save results
        candidate_library_df.to_csv(f'./data/processed_data/candidate_library_{ablation_name}.csv', index=False)
        print(f"Results saved to: ./data/processed_data/candidate_library_{ablation_name}.csv")
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Average similarity score: {candidate_library_df['similarity_score'].mean():.4f}")
        print(f"Min similarity score: {candidate_library_df['similarity_score'].min():.4f}")
        print(f"Max similarity score: {candidate_library_df['similarity_score'].max():.4f}")
        print(f"Number of unique target compounds: {candidate_library_df['target_idx'].nunique()}")
        print("length of candidate library:", len(candidate_library_df))
        print("Done!")

if __name__ == "__main__":
    main()
