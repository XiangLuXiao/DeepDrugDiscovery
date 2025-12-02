"""
Compound library analysis helpers for computing key property statistics, diversity,
and PCA-based chemical space coverage.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

import os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# RDKit warns about deprecated fingerprint helpers; silence to keep logs clean
RDLogger.DisableLog('rdApp.warning')

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

DEFAULT_SMILES_COLUMNS = [
    'Smiles_unify',
    'Smiles_removesalt',
    'Smiles',
    'smiles',
    'SMILES'
]


def clean_smiles_values(smiles_iterable):
    """Return cleaned SMILES list by filtering invalid values."""
    cleaned = []
    for entry in smiles_iterable:
        if pd.isna(entry):
            continue
        entry_str = str(entry).strip()
        if not entry_str or entry_str.lower() == 'nan' or entry_str.upper() == 'SANITIZE_NONE':
            continue
        cleaned.append(entry_str)
    return cleaned


def get_smiles_array(smiles_source, smiles_column=None, preferred_columns=None):
    """Extract SMILES list from a DataFrame or iterable."""
    if isinstance(smiles_source, pd.DataFrame):
        columns_to_check = []
        if smiles_column:
            columns_to_check.append(smiles_column)
        columns_to_check.extend(preferred_columns or DEFAULT_SMILES_COLUMNS)
        seen = set()
        ordered_columns = []
        for col in columns_to_check:
            if col and col not in seen:
                seen.add(col)
                ordered_columns.append(col)
        for column in ordered_columns:
            if column in smiles_source.columns:
                cleaned = clean_smiles_values(smiles_source[column])
                if cleaned:
                    return cleaned, column
        raise ValueError(
            f"None of the preferred SMILES columns {ordered_columns} contain valid values."
        )
    cleaned = clean_smiles_values(smiles_source)
    if not cleaned:
        raise ValueError("No valid SMILES strings provided.")
    return cleaned, smiles_column



def calculate_properties_and_fingerprint(smiles):
    """
    Worker function for parallel processing.
    Calculates physicochemical properties and Morgan fingerprint for a single SMILES string.
    Returns None if the SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Calculate physicochemical properties
        properties = {
            'SMILES': smiles,
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol)
        }
        return properties
    except Exception:
        return None

def plot_physchem_distributions(df, plot_name, dataset_name="Dataset", bin_count=40):
    """Generates and saves distribution plots for physicochemical properties."""
    print(f"[{dataset_name}] Generating physicochemical property distribution plots...")
    properties_to_plot = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotatableBonds']
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    palette = sns.color_palette("viridis", len(properties_to_plot))

    for i, prop in enumerate(properties_to_plot):
        sns.histplot(
            df[prop],
            kde=True,
            ax=axes[i],
            bins=bin_count,
            color=palette[i],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.85,
        )
        axes[i].set_title(f'Distribution of {prop}', fontsize=13)
        axes[i].set_xlabel(prop)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, linewidth=0.4, alpha=0.6)

    sns.despine(fig=fig)
    fig.tight_layout()
    out_dir = os.path.dirname(plot_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(plot_name, format='svg', bbox_inches='tight')
    plt.close()

def analyze_compound_library(smiles_list, plot_name, dataset_name="Dataset", bin_count=40):
    """Run property analysis and plotting for a SMILES list."""
    if not smiles_list:
        print(f"[{dataset_name}] No SMILES strings provided. Skipping.")
        return

    smiles_list = list(smiles_list)
    num_cores = min(20, cpu_count(), max(1, len(smiles_list)))
    print(f"[{dataset_name}] Using {num_cores} cores for parallel processing...")

    with Pool(num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(calculate_properties_and_fingerprint, smiles_list),
                total=len(smiles_list),
                desc=f"Processing {dataset_name}"
            )
        )

    # Filter out None results from invalid SMILES
    valid_results = [res for res in results if res is not None]
    print(f"[{dataset_name}] Successfully processed {len(valid_results)} valid molecules.")

    if not valid_results:
        print(f"[{dataset_name}] No valid molecules could be processed. Skipping.")
        return

    df_analysis = pd.DataFrame(valid_results)

    plot_physchem_distributions(df_analysis, plot_name, dataset_name, bin_count)


if __name__ == '__main__':
    dataset_configs = [
        {
            'name': 'Candidate Compounds Library',
            'path': 'data/processed_data/candidate_compounds_library_data.csv',
            'plot': 'plot/compound_library_physchem_distribution.svg',
            'bins': 40
        },
        {
            'name': 'Reference Compounds',
            'path': 'data/processed_data/reference_compounds_data.csv',
            'plot': 'plot/reference_compound_physchem_distribution.svg',
            'bins': 20
        }
    ]

    for config in dataset_configs:
        dataset_name = config['name']
        csv_path = config['path']
        plot_path = config['plot']
        bin_count = config.get('bins', 40)
        smiles_column = config.get('smiles_column')

        if not os.path.exists(csv_path):
            print(f"[{dataset_name}] File not found: {csv_path}. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        try:
            smiles_list, used_column = get_smiles_array(df, smiles_column)
        except ValueError as exc:
            print(f"[{dataset_name}] {exc} Skipping dataset.")
            continue

        print(
            f"[{dataset_name}] Loaded {len(smiles_list)} molecules from {csv_path} "
            f"(column '{used_column}')."
        )

        analyze_compound_library(smiles_list, plot_path, dataset_name, bin_count)
