"""
Utility helpers for building ablation-ready molecular datasets that store only
fingerprint representations.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

from joblib import Parallel, delayed
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch.utils.data import Dataset
import os
import gc  # For garbage collection


def gen_mogan(m, radius=2, nBits=2048):
    try:
        MorganGenerator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits, includeChirality=True)
        fp = MorganGenerator.GetFingerprint(m)
        return np.array(fp, dtype=np.int64)
    except Exception as e:
        print(f"Error generating fingerprint: {e}")
        return None

def gen_morgan_feature(mol_list, num_jobs):
    features_map = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(gen_mogan)(mol) for mol in tqdm(mol_list)
    )
    return np.array(features_map)

def gen_mol(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except:
        return None

def gen_mol_feature(smi_list, num_jobs):
    features_map = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(gen_mol)(smi) for smi in tqdm(smi_list)
    )
    return features_map

def smi_to_smi(x):
    try:
        mol = Chem.MolFromSmiles(x)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        return Chem.MolToSmiles(mol)
    except:
        return np.nan

def smiles_check(df, smi_col, jobs):
    """
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
        jobs: Number of worker, int
    Returns:
        df: DataFrame with a new column named 'Smiles_check'
    """
    mols = gen_mol_feature(df[smi_col].values, num_jobs=jobs)
    check_result = []
    for mol in tqdm(mols):
        a = bool(mol)
        if a:
            b = str(Chem.SanitizeMol(mol))
            check_result.append(b)
        else:
            check_result.append(a)

    df['Smiles_check'] = check_result
    print(df['Smiles_check'].value_counts(dropna=False))
    return df

def remove_salt(df, smi_col):
    """
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
    Returns:
        df: DataFrame with a new column named 'Smiles_removesalt': a col contains SMILES without any salt/solvent
    """
    Smiles_rs = []
    for smiles in tqdm(df[smi_col].values):
        frags = smiles.split(".")
        frags = sorted(frags, key=lambda x: len(x), reverse=True)
        Smiles_rs.append(frags[0])
            
    df['Smiles_removesalt'] = Smiles_rs
    return df

def smiles_unify(df, smi_col, jobs):
    """
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
    Returns:
        df: DataFrame with a new column named 'Smiles_unify': a col contains unified SMILES
    """
    Smiles_unify = []
    mols = gen_mol_feature(df[smi_col].values, num_jobs=jobs)
    for m in tqdm(mols):
        s_u = Chem.MolToSmiles(m)
        Smiles_unify.append(s_u)
    
    df['Smiles_unify'] = Smiles_unify
    return df

class Molecule:
    def __init__(self, mol, label, cpu_core=48):
        self.mol = mol
        self.label = label
        self.fingerprints = gen_mogan(self.mol)

class MolDataSet(Dataset):
    def __init__(self, data_list):
        self.data_list = np.array(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return MolDataSet(self.data_list[key])
        return self.data_list[key]

def construct_molecularNN_dataset(mol_list, label_list):
    output = [Molecule(mol, label) for mol, label in tqdm(zip(mol_list, label_list), total=len(mol_list))]
    return MolDataSet(output)

def molecularNN_mol_collate_func(batch):
    result = {
        'fingerprints': torch.from_numpy(np.array([x.fingerprints for x in batch])).float(),
        'target': torch.from_numpy(np.array([x.label for x in batch])).float()
    }
    return result

def vae_mol_collate_func(batch):
    """Collate function for VAE training with fingerprints and None features."""
    fingerprints = torch.from_numpy(np.array([x.fingerprints for x in batch])).float()
    features = None  # No features, pass None to VAE
    targets = torch.from_numpy(np.array([x.label for x in batch])).float()  # label contains fingerprints
    
    return fingerprints, features, targets

def dataset_loader(dataset, collate_func, batch_size, train=True, num_workers=2):
    """
    Custom dataset loader that mimics DataLoader but uses custom collate function.
    
    Args:
        dataset: The dataset to load from
        collate_func: Custom collate function
        batch_size: Batch size for loading
        train: Whether this is for training (affects shuffling)
        num_workers: Number of worker processes
    
    Returns:
        DataLoader: Configured data loader
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate_func,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

class VAEMolecule:
    """Molecule class specifically for VAE training with fingerprint reconstruction only."""
    def __init__(self, mol, target_data):
        self.mol = mol
        self.label = target_data  # Contains fingerprints for reconstruction
        self.fingerprints = gen_mogan(self.mol)

def construct_vae_dataset(mol_list, target_list):
    """Construct VAE dataset with fingerprint reconstruction only."""
    output = [VAEMolecule(mol, target) for mol, target in 
             tqdm(zip(mol_list, target_list), total=len(mol_list), desc="Constructing VAE dataset")]
    return MolDataSet(output)

def clear_memory():
    """Force garbage collection to free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def prepare_vae_dataset(smiles_list, config, logger):
    """
    Prepare molecular fingerprint data for VAE training (fingerprints only).
    
    Args:
        smiles_list (list): List of SMILES strings
        config (dict): Configuration dictionary
        logger: Logger instance
        
    Returns:
        tuple: (train_loader, val_loader, None)
    """
    logger.info(f"Preparing VAE dataset for {len(smiles_list)} molecules (fingerprints only)...")
    
    # Memory optimization: Process in smaller chunks to avoid memory explosion
    chunk_size = min(500000, len(smiles_list))  # Process max 500000 molecules at once
    all_valid_mols = []
    all_fingerprints = []
    
    # Process in chunks to control memory usage
    for i in tqdm(range(0, len(smiles_list), chunk_size), desc="Processing molecule chunks"):
        chunk_smiles = smiles_list[i:i+chunk_size]
        
        # Generate molecules for this chunk
        logger.info(f"Processing chunk {i//chunk_size + 1}: {len(chunk_smiles)} molecules")
        chunk_mol_list = gen_mol_feature(chunk_smiles, num_jobs=config['hardware'].get('num_jobs', 4))
        
        # Filter valid molecules in this chunk
        chunk_valid_mols = [mol for mol in chunk_mol_list if mol is not None]
        
        if len(chunk_valid_mols) == 0:
            logger.warning(f"No valid molecules in chunk {i//chunk_size + 1}")
            continue
        
        # Generate fingerprints for this chunk
        chunk_fingerprints = gen_morgan_feature(chunk_valid_mols, num_jobs=config['hardware'].get('num_jobs', 4))
        
        # Filter None values in this chunk
        chunk_valid_data = [(mol, fp) for mol, fp in zip(chunk_valid_mols, chunk_fingerprints) 
                           if fp is not None]
        
        if len(chunk_valid_data) == 0:
            logger.warning(f"No valid fingerprints in chunk {i//chunk_size + 1}")
            continue
        
        # Unpack and accumulate
        chunk_mols, chunk_fps = zip(*chunk_valid_data)
        all_valid_mols.extend(chunk_mols)
        all_fingerprints.extend(chunk_fps)
        
        # Clear chunk data to free memory
        del chunk_mol_list, chunk_valid_mols, chunk_fingerprints, chunk_valid_data
        del chunk_mols, chunk_fps
        
        # Force garbage collection
        clear_memory()
        
        logger.info(f"Chunk {i//chunk_size + 1} processed. Total valid molecules: {len(all_valid_mols)}")
    
    if len(all_valid_mols) == 0:
        raise ValueError("No valid molecules could be processed!")
    
    # Convert to numpy arrays
    fingerprints = np.array(all_fingerprints)
    
    # Clear intermediate lists to save memory
    del all_fingerprints
    clear_memory()  # Force cleanup
    
    logger.info(f"Total valid molecules processed: {len(fingerprints)}")
    logger.info(f"Success rate: {len(fingerprints)/len(smiles_list)*100:.2f}%")
    
    # Target should be fingerprints for reconstruction
    target_data = fingerprints
    
    # Sequential dataset creation to reduce peak memory usage
    logger.info("Creating VAE dataset with sequential processing...")
    
    # Create molecule data tuples
    molecule_data = list(zip(all_valid_mols, target_data))
    
    # Clear intermediate arrays to save memory
    del all_valid_mols, fingerprints, target_data
    clear_memory()  # Force cleanup
    
    # Split data first, then create datasets
    train_size = int(config['training'].get('train_split', 0.8) * len(molecule_data))
    
    # Split the raw data
    train_data = molecule_data[:train_size]
    val_data = molecule_data[train_size:]
    
    # Clear original data
    del molecule_data
    clear_memory()
    
    # Create datasets sequentially to reduce peak memory
    logger.info("Creating training dataset...")
    train_molecules = [VAEMolecule(mol, target) for mol, target in tqdm(train_data, desc="Creating train molecules")]
    train_dataset = MolDataSet(train_molecules)
    
    # Clear train data before creating validation dataset
    del train_data, train_molecules
    clear_memory()  # Force cleanup
    
    logger.info("Creating validation dataset...")
    val_molecules = [VAEMolecule(mol, target) for mol, target in tqdm(val_data, desc="Creating val molecules")]
    val_dataset = MolDataSet(val_molecules)
    
    # Clear val data
    del val_data, val_molecules
    clear_memory()  # Force cleanup
    
    # Create data loaders using custom dataset_loader
    train_loader = dataset_loader(
        train_dataset, 
        vae_mol_collate_func, 
        config['training']['batch_size'], 
        train=True,
        num_workers=config['hardware'].get('num_workers', 2)
    )
    
    val_loader = dataset_loader(
        val_dataset, 
        vae_mol_collate_func, 
        config['training']['batch_size'], 
        train=False,
        num_workers=config['hardware'].get('num_workers', 2)
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Fingerprint size: 2048")
    logger.info(f"Target data size (fingerprints only): 2048")
    
    return train_loader, val_loader, None

def generate_moleceularNN_embeddings(mol_list, model, logger, batch_size=1024, device='cuda:0'):
    """Generates embeddings for a list of molecules (fingerprints only)."""
    all_embeddings = []
    for i in tqdm(range(0, len(mol_list), batch_size), desc="Processing Batches"):
        batch_mol = mol_list[i:i+batch_size]
        try:
            batch_fp = gen_morgan_feature(batch_mol, 48)
            batch_fp = torch.tensor(batch_fp, dtype=torch.float).to(device)
        except ValueError as e:
            logger.error(f'Error processing batch {batch_mol}: {e}')
            raise e
        
        with torch.no_grad():
            batch_embedding = model(batch_fp, output_embedding=True)
        all_embeddings.append(batch_embedding.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)

def generate_embeddings(smiles):
    """Generates fingerprint embeddings for a SMILES string."""
    try:
        mol = gen_mol(smiles)
        fp = gen_mogan(mol)
    except ValueError as e:
        print(f'Error processing generate embeddings:{e}')
        raise e
        
    return fp

def generate_embeddings_features(smi_list, num_jobs):
    """Generates fingerprint embeddings for a list of SMILES strings."""
    mols = []
    
    features_map = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(generate_embeddings)(smi) for smi in tqdm(smi_list)
    )
    for i, feats in enumerate(features_map):
        mols.append(feats)
    return mols
