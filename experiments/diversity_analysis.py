#!/usr/bin/env python3
"""
Comprehensive molecular diversity calculator for chemical compound analysis.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

import os
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors, Fragments, MACCSkeys, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# RDKit emits noisy Morgan fingerprint deprecation warnings; silence them
RDLogger.DisableLog('rdApp.warning')

DEFAULT_SMILES_COLUMNS = [
    'Smiles_unify',
    'Smiles_removesalt',
    'Smiles',
    'smiles',
    'SMILES'
]


def clean_smiles_values(smiles_iterable):
    """Return a numpy array with invalid entries removed."""
    cleaned = []
    for entry in smiles_iterable:
        if pd.isna(entry):
            continue
        entry_str = str(entry).strip()
        if not entry_str or entry_str.lower() == 'nan' or entry_str.upper() == 'SANITIZE_NONE':
            continue
        cleaned.append(entry_str)
    return np.array(cleaned, dtype=object)


def get_smiles_array(smiles_source, smiles_column=None, preferred_columns=None):
    """Extract SMILES strings from a DataFrame or iterable."""
    if isinstance(smiles_source, pd.DataFrame):
        columns_to_check = []
        if smiles_column:
            columns_to_check.append(smiles_column)
        columns_to_check.extend((preferred_columns or DEFAULT_SMILES_COLUMNS))
        seen = set()
        ordered_columns = []
        for col in columns_to_check:
            if col and col not in seen:
                seen.add(col)
                ordered_columns.append(col)
        for column in ordered_columns:
            if column in smiles_source.columns:
                cleaned = clean_smiles_values(smiles_source[column])
                if cleaned.size:
                    return cleaned, column
        raise ValueError(
            f"None of the preferred SMILES columns {ordered_columns} contain valid values."
        )
    cleaned = clean_smiles_values(smiles_source)
    if not cleaned.size:
        raise ValueError("No valid SMILES strings provided.")
    return cleaned, smiles_column

class ComprehensiveDiversityCalculator:
    """Calculate multiple types of molecular diversity"""
    
    def __init__(self):
        self.results = {}
    
    def fingerprint_diversity(self, smiles_list, fp_type='morgan'):
        """Fingerprint-based diversity (Morgan, etc.)"""
        fps = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    if fp_type == 'morgan':
                        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    else:
                        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(fp)
            except:
                continue
        
        if len(fps) < 2:
            return {'fingerprint_diversity': 0, 'avg_similarity': 1.0}
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        diversity = 1 - avg_similarity
        
        return {
            'fingerprint_diversity': diversity,
            'avg_similarity': avg_similarity,
            'fingerprint_type': fp_type
        }
    
    def ring_system_diversity(self, smiles_list):
        """Ring system diversity beyond scaffolds"""
        ring_features = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    ring_info = mol.GetRingInfo()
                    features = [
                        rdMolDescriptors.CalcNumRings(mol),
                        rdMolDescriptors.CalcNumAromaticRings(mol),
                        rdMolDescriptors.CalcNumSaturatedRings(mol),
                        rdMolDescriptors.CalcNumHeterocycles(mol),
                        len([r for r in ring_info.AtomRings() if len(r) == 3]),  # 3-rings
                        len([r for r in ring_info.AtomRings() if len(r) == 4]),  # 4-rings
                        len([r for r in ring_info.AtomRings() if len(r) == 5]),  # 5-rings
                        len([r for r in ring_info.AtomRings() if len(r) == 6]),  # 6-rings
                        len([r for r in ring_info.AtomRings() if len(r) >= 7])   # 7+ rings
                    ]
                    ring_features.append(features)
            except:
                continue
        
        if len(ring_features) < 2:
            return {'ring_system_diversity': 0}
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(ring_features)
        distances = pairwise_distances(features_scaled, metric='euclidean')
        
        n = len(ring_features)
        diversity = np.sum(distances) / (n * (n - 1))
        
        return {
            'ring_system_diversity': diversity,
            'n_molecules': n
        }
    
    def chemical_space_diversity(self, smiles_list, n_components=10):
        """PCA-based chemical space diversity"""
        # Get Morgan fingerprints
        fps = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    fps.append(list(fp))
            except:
                continue
        
        if len(fps) < n_components:
            return {'chemical_space_diversity': 0, 'pca_explained_variance': 0}
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(fps)-1))
        pca_coords = pca.fit_transform(fps)
        
        # Calculate diversity in PCA space
        distances = pairwise_distances(pca_coords, metric='euclidean')
        n = len(fps)
        diversity = np.sum(distances) / (n * (n - 1))
        
        return {
            'chemical_space_diversity': diversity,
            'pca_explained_variance': np.sum(pca.explained_variance_ratio_),
            'n_components': n_components,
            'n_molecules': n
        }
    
    def calculate_all_diversity_metrics(self, smiles_list):
        """Calculate all diversity metrics"""
        print(f"Calculating diversity for {len(smiles_list)} molecules...")
        
        self.results = {}
        
        
        # Fingerprint diversity (Morgan)
        print("- Morgan fingerprint diversity...")
        self.results['morgan_fp'] = self.fingerprint_diversity(smiles_list, 'morgan')
        
        # Ring system diversity
        print("- Ring system diversity...")
        self.results['ring_system'] = self.ring_system_diversity(smiles_list)
        
        # Chemical space diversity
        print("- Chemical space (PCA) diversity...")
        self.results['chemical_space'] = self.chemical_space_diversity(smiles_list)
        
        return self.results
    
    def get_diversity_summary(self):
        """Get summary of all diversity metrics"""
        if not self.results:
            return "No diversity analysis has been performed yet."
        
        summary = []
        summary.append("Molecular Diversity Analysis Summary")
        summary.append("=" * 50)
        
        # Extract key diversity values
        metrics = [
            ('Morgan FP Diversity', self.results.get('morgan_fp', {}).get('fingerprint_diversity', 0)),
            ('Ring System Diversity', self.results.get('ring_system', {}).get('ring_system_diversity', 0)),
            ('Chemical Space Diversity', self.results.get('chemical_space', {}).get('chemical_space_diversity', 0))
        ]
        
        for name, value in metrics:
            summary.append(f"{name:<25}: {value:.3f}")
        
        return "\n".join(summary)
    
    def plot_diversity_comparison(self, figsize=(12, 8)):
        """Plot comparison of different diversity metrics"""
        if not self.results:
            print("No results to plot. Run calculate_all_diversity_metrics first.")
            return
        
        # Extract diversity values
        metrics = []
        values = []
        
        diversity_map = {
            'Morgan FP': self.results.get('morgan_fp', {}).get('fingerprint_diversity', 0),
            'Ring Systems': self.results.get('ring_system', {}).get('ring_system_diversity', 0),
            'Chemical Space': self.results.get('chemical_space', {}).get('chemical_space_diversity', 0)
        }
        
        metrics = list(diversity_map.keys())
        values = list(diversity_map.values())
        
        # Create plot
        plt.figure(figsize=figsize)
        bars = plt.bar(metrics, values, color='steelblue', alpha=0.7)
        
        plt.xlabel('Diversity Metric')
        plt.ylabel('Diversity Score')
        plt.title('Molecular Diversity Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_diversity_comparison_with_error_bars(self, means, stds, figsize=(12, 8)):
        """Plot comparison of different diversity metrics with error bars"""
        metrics = list(means.keys())
        mean_values = list(means.values())
        std_values = [stds[metric] for metric in metrics]
        
        # Create plot
        plt.figure(figsize=figsize)
        bars = plt.bar(metrics, mean_values, yerr=std_values, color='steelblue', 
                      alpha=0.7, capsize=5, error_kw={'elinewidth': 2})
        
        plt.xlabel('Diversity Metric')
        plt.ylabel('Diversity Score')
        plt.title('Molecular Diversity Comparison (Mean ± Std, n=5 samples)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, mean_values, std_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.01,
                    f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filename='diversity_results.json'):
        """Export all results to JSON"""
        import json
        if self.results:
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Results exported to {filename}")

def calculate_statistical_diversity(smiles_data, sample_size, n_iterations=5, seed=42,
                                   smiles_column=None, preferred_columns=None):
    """
    Calculate diversity metrics with statistical sampling
    
    Args:
        smiles_data: DataFrame or array with SMILES data
        sample_size: Number of molecules to sample each iteration
        n_iterations: Number of random samples to take
        seed: Random seed for reproducibility
        smiles_column: Optional column to pull SMILES from when passing a DataFrame
        preferred_columns: Optional override for the default column preference order
    
    Returns:
        Dictionary with mean and std for each diversity metric
    """
    np.random.seed(seed)
    smiles_array, _ = get_smiles_array(smiles_data, smiles_column, preferred_columns)
    n_available = len(smiles_array)
    if sample_size >= n_available:
        print(
            f"Requested sample size ({sample_size}) exceeds or equals available molecules ({n_available}). "
            "Using the full set once without sampling."
        )
        sample_size = n_available
        n_iterations = 1
        full_dataset_only = True
    else:
        full_dataset_only = False
    
    # Store results from each iteration
    all_results = []
    
    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1}/{n_iterations} ---")
        
        # Random sample
        if full_dataset_only:
            sampled_smiles = smiles_array
        else:
            indices = np.random.choice(len(smiles_array), size=sample_size, replace=False)
            sampled_smiles = smiles_array[indices]
        
        # Calculate diversity for this sample
        calc = ComprehensiveDiversityCalculator()
        results = calc.calculate_all_diversity_metrics(sampled_smiles)
        all_results.append(results)
    
    # Calculate means and standard deviations
    diversity_metrics = {
        'Morgan FP': 'morgan_fp',
        'Ring Systems': 'ring_system',
        'Chemical Space': 'chemical_space'
    }
    
    means = {}
    stds = {}
    
    for metric_name, result_key in diversity_metrics.items():
        values = []
        for result in all_results:
            if result_key in result:
                if result_key.endswith('_fp'):
                    values.append(result[result_key]['fingerprint_diversity'])
                elif result_key == 'ring_system':
                    values.append(result[result_key]['ring_system_diversity'])
                elif result_key == 'chemical_space':
                    values.append(result[result_key]['chemical_space_diversity'])
        
        if values:
            means[metric_name] = np.mean(values)
            stds[metric_name] = np.std(values)
        else:
            means[metric_name] = 0
            stds[metric_name] = 0
    
    return means, stds, all_results

# Quick examples
if __name__ == "__main__":
    candidate_path = "data/processed_data/candidate_library.csv"
    reference_path = "data/processed_data/reference_compounds_data.csv"
    candidate_compounds_path = "data/processed_data/candidate_compounds_library_data.csv"
    
    # First analysis - candidate library (full dataset)
    print("="*60)
    print("ANALYSIS 1: Candidate Library (Full Dataset)")
    print("="*60)
    
    calc1 = ComprehensiveDiversityCalculator()
    df = pd.read_csv(candidate_path)
    candidate_smiles, candidate_column = get_smiles_array(df)
    print(f"Loaded {len(candidate_smiles)} molecules from {candidate_path} (column '{candidate_column}')")
    
    results1 = calc1.calculate_all_diversity_metrics(candidate_smiles)
    print("\n" + calc1.get_diversity_summary())
    calc1.plot_diversity_comparison()
    calc1.export_results("plot/diversity_results_candidate_library.json")
    
    # Second analysis - reference compounds (full dataset)
    print("\n" + "="*60)
    print("ANALYSIS 2: Reference Compounds (Full Dataset)")
    print("="*60)
    
    calc_ref = ComprehensiveDiversityCalculator()
    reference_df = pd.read_csv(reference_path)
    reference_smiles, reference_column = get_smiles_array(reference_df)
    print(
        f"Loaded {len(reference_smiles)} reference molecules from {reference_path} (column '{reference_column}')"
    )
    
    reference_results = calc_ref.calculate_all_diversity_metrics(reference_smiles)
    print("\n" + calc_ref.get_diversity_summary())
    calc_ref.plot_diversity_comparison()
    calc_ref.export_results("plot/diversity_results_reference_compounds.json")
    
    # Third analysis - candidate compounds library (5 random samples)
    print("\n" + "="*60)
    print("ANALYSIS 3: Candidate Compounds Library (5 Random Samples)")
    print("="*60)
    
    cmp = pd.read_csv(candidate_compounds_path)
    cmp_smiles, cmp_column = get_smiles_array(cmp)
    sample_size = len(candidate_smiles)  # Same size as the first dataset actually used in Analysis 1
    
    print(
        f"Sampling {sample_size} molecules from {len(cmp_smiles)} available compounds (column '{cmp_column}')"
    )
    print("Performing 5 independent random samples...")
    
    # Calculate statistical diversity
    means, stds, all_results = calculate_statistical_diversity(
        cmp_smiles, sample_size, n_iterations=5
    )
    
    # Print statistical summary
    print("\n" + "="*50)
    print("Statistical Summary (n=5 samples):")
    print("="*50)
    for metric_name in means.keys():
        print(f"{metric_name:<20}: {means[metric_name]:.3f} ± {stds[metric_name]:.3f}")
    
    # Plot with error bars
    calc_stats = ComprehensiveDiversityCalculator()
    calc_stats.plot_diversity_comparison_with_error_bars(means, stds)
    
    # Export statistical results
    import json
    statistical_results = {
        'means': means,
        'standard_deviations': stds,
        'n_iterations': 5,
        'sample_size': sample_size,
        'total_available': len(cmp_smiles),
        'smiles_column': cmp_column,
        'individual_results': all_results
    }
    
    with open('plot/diversity_results_candidate_compounds_library_statistical.json', 'w') as f:
        json.dump(statistical_results, f, indent=2)
    
    print(f"\nStatistical results exported to: plot/diversity_results_candidate_compounds_library_statistical.json")
    
    # Compare the two analyses
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<20} {'Candidate':<12} {'Reference':<12} {'Cand. Samples (Mean±Std)':<25}")
    print("-" * 60)
    
    # Get Library 1 values
    lib1_values = {
        'Morgan FP': results1.get('morgan_fp', {}).get('fingerprint_diversity', 0),
        'Ring Systems': results1.get('ring_system', {}).get('ring_system_diversity', 0),
        'Chemical Space': results1.get('chemical_space', {}).get('chemical_space_diversity', 0)
    }
    reference_values = {
        'Morgan FP': reference_results.get('morgan_fp', {}).get('fingerprint_diversity', 0),
        'Ring Systems': reference_results.get('ring_system', {}).get('ring_system_diversity', 0),
        'Chemical Space': reference_results.get('chemical_space', {}).get('chemical_space_diversity', 0)
    }
    
    for metric in ['Morgan FP', 'Ring Systems', 'Chemical Space']:
        lib1_val = lib1_values[metric]
        ref_val = reference_values[metric]
        lib2_mean = means[metric]
        lib2_std = stds[metric]
        print(f"{metric:<20} {lib1_val:<12.3f} {ref_val:<12.3f} {lib2_mean:.3f}±{lib2_std:.3f}")
