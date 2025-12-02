"""
Dataset utilities for building molecular neural network inputs and descriptors.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

from joblib import Parallel, delayed
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors, QED
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
import joblib
import os
import gc  # For garbage collection


def count_hydrogens(molecule):
    """Calculate the total number of hydrogen atoms in the molecule."""
    return sum(1 for atom in Chem.AddHs(molecule).GetAtoms() if atom.GetAtomicNum() == 1)

def count_halogens(molecule):
    """Calculate the number of halogen atoms (F, Cl, Br, I) in the molecule."""
    halogen_atomic_numbers = {9, 17, 35, 53}
    return sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() in halogen_atomic_numbers)

def count_aromatic_bonds(molecule):
    """Calculate the total number of aromatic bonds in the molecule."""
    return sum(1 for bond in molecule.GetBonds() if bond.GetBondType().name == 'AROMATIC')

def count_total_atoms(molecule):
    """Calculate the total number of atoms in the molecule, including hydrogens."""
    return Chem.AddHs(molecule).GetNumAtoms()

def compute_csp3_carbon_count(molecule):
    """Calculate the number of sp3 hybridized carbons in the molecule."""
    sp3_fraction = Chem.rdMolDescriptors.CalcFractionCSP3(molecule)
    total_carbon_atoms = sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6)
    return total_carbon_atoms * sp3_fraction

def count_aromatic_nitrogen_containing_rings(molecule):
    """Calculate the number of aromatic rings containing nitrogen atoms in the molecule."""
    ring_info = molecule.GetRingInfo().BondRings()
    ring_count = 0
    for ring in ring_info:
        is_aromatic = all(molecule.GetBondWithIdx(bond_idx).GetBondType().name == 'AROMATIC' for bond_idx in ring)
        contains_nitrogen = any(molecule.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomicNum() == 7 or 
                                molecule.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomicNum() == 7 for bond_idx in ring)
        if is_aromatic and contains_nitrogen:
            ring_count += 1
    return ring_count

def count_unsaturated_nonaromatic_carbocyclic_rings(molecule):
    """Calculate the number of unsaturated, non-aromatic carbocyclic rings in the molecule."""
    ring_info = molecule.GetRingInfo().BondRings()
    ring_count = 0
    for ring in ring_info:
        has_unsaturation = any(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'SINGLE' for bond_idx in ring)
        is_nonaromatic = all(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'AROMATIC' for bond_idx in ring)
        is_carbocycle = all(molecule.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomicNum() == 6 and 
                            molecule.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomicNum() == 6 for bond_idx in ring)
        if has_unsaturation and is_nonaromatic and is_carbocycle:
            ring_count += 1
    return ring_count

def count_unsaturated_nonaromatic_nitrogen_rings(molecule):
    """Calculate the number of unsaturated, non-aromatic rings containing nitrogen atoms in the molecule."""
    ring_info = molecule.GetRingInfo().BondRings()
    ring_count = 0
    for ring in ring_info:
        has_unsaturation = any(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'SINGLE' for bond_idx in ring)
        is_nonaromatic = all(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'AROMATIC' for bond_idx in ring)
        contains_nitrogen = any(molecule.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomicNum() == 7 or 
                                molecule.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomicNum() == 7 for bond_idx in ring)
        if has_unsaturation and is_nonaromatic and contains_nitrogen:
            ring_count += 1
    return ring_count

def count_unsaturated_nonaromatic_heterocyclic_rings(molecule):
    """Calculate the number of unsaturated, non-aromatic rings containing heteroatoms in the molecule."""
    ring_info = molecule.GetRingInfo().BondRings()
    ring_count = 0
    for ring in ring_info:
        has_unsaturation = any(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'SINGLE' for bond_idx in ring)
        is_nonaromatic = all(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'AROMATIC' for bond_idx in ring)
        contains_heteroatom = any(molecule.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomicNum() not in {1, 6} or 
                                  molecule.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomicNum() not in {1, 6} for bond_idx in ring)
        if has_unsaturation and is_nonaromatic and contains_heteroatom:
            ring_count += 1
    return ring_count

def gen_qed_properties(m):
    """Generate QED properties for the given molecule."""
    qed_properties = QED.properties(m)
    return {
        'QED_ALOGP': qed_properties.ALOGP,
        'QED_MW': qed_properties.MW,
        'QED_ROTB': qed_properties.ROTB,
        'QED_HBA': qed_properties.HBA,
        'QED_HBD': qed_properties.HBD,
        'QED_PSA': qed_properties.PSA,
        'QED_AROM': qed_properties.AROM,
    }

descriptor_functions = {
    'HydrogenAtomCount': count_hydrogens,
    'HalogenAtomCount': count_halogens,
    'AromaticBondCount': count_aromatic_bonds,
    'TotalAtomCount': count_total_atoms,
    'Sp3CarbonCount': compute_csp3_carbon_count,
    'AromaticNitrogenRingCount': count_aromatic_nitrogen_containing_rings,
    'UnsaturatedNonaromaticCarbocyclicRingCount': count_unsaturated_nonaromatic_carbocyclic_rings,
    'UnsaturatedNonaromaticNitrogenRingCount': count_unsaturated_nonaromatic_nitrogen_rings,
    'UnsaturatedNonaromaticHeterocyclicRingCount': count_unsaturated_nonaromatic_heterocyclic_rings,
    'HeteroatomCount': rdMolDescriptors.CalcNumHeteroatoms,
    'RingCount': rdMolDescriptors.CalcNumRings,
    'AmideBondCount': rdMolDescriptors.CalcNumAmideBonds,
    'QED_ALOGP': lambda m: gen_qed_properties(m)['QED_ALOGP'],
    'QED_MW': lambda m: gen_qed_properties(m)['QED_MW'],
    'QED_ROTB': lambda m: gen_qed_properties(m)['QED_ROTB'],
    'QED_HBA': lambda m: gen_qed_properties(m)['QED_HBA'],
    'QED_HBD': lambda m: gen_qed_properties(m)['QED_HBD'],
    'QED_PSA': lambda m: gen_qed_properties(m)['QED_PSA'],
    'QED_PSA': lambda m: gen_qed_properties(m)['QED_PSA'],
    'QED_AROM': lambda m: gen_qed_properties(m)['QED_AROM'],
}

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

def get_molecular_descriptors(molecule):
    """Get a dictionary of molecular descriptors for the given molecule."""
    descriptor_values = OrderedDict()
    for descriptor_name, function in descriptor_functions.items():
        try:
            descriptor_values[descriptor_name] = function(molecule)
        except:
            descriptor_values[descriptor_name] = 0.0
    return list(descriptor_values.values())

def gen_descriptors_feature(mol_list, num_jobs):
    features_map = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(get_molecular_descriptors)(mol) for mol in tqdm(mol_list)
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
        self.features = get_molecular_descriptors(self.mol)

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
        'features': torch.from_numpy(np.array([x.features for x in batch])).float(),
        'target': torch.from_numpy(np.array([x.label for x in batch])).float()
    }
    return result

def vae_mol_collate_func(batch):
    """Collate function for VAE training with fingerprints, features, and fingerprint targets."""
    fingerprints = torch.from_numpy(np.array([x.fingerprints for x in batch])).float()
    features = torch.from_numpy(np.array([x.features for x in batch])).float()
    targets = torch.from_numpy(np.array([x.label for x in batch])).float()  # label contains fingerprints only
    
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
    """Molecule class specifically for VAE training with target reconstruction data."""
    def __init__(self, mol, target_data, processed_features=None):
        self.mol = mol
        self.label = target_data  # Contains concatenated fingerprints + features for reconstruction
        self.fingerprints = gen_mogan(self.mol)
        
        # If processed features are provided, use them; otherwise compute original features
        if processed_features is not None:
            self.features = processed_features
        else:
            self.features = get_molecular_descriptors(self.mol)

def construct_vae_dataset(mol_list, target_list, processed_features_list=None):
    """Construct VAE dataset with target reconstruction data and optional processed features."""
    if processed_features_list is not None:
        output = [VAEMolecule(mol, target, feat) for mol, target, feat in 
                 tqdm(zip(mol_list, target_list, processed_features_list), 
                     total=len(mol_list), desc="Constructing VAE dataset")]
    else:
        output = [VAEMolecule(mol, target) for mol, target in 
                 tqdm(zip(mol_list, target_list), total=len(mol_list), desc="Constructing VAE dataset")]
    return MolDataSet(output)

def clear_memory():
    """Force garbage collection to free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def apply_feature_encoding(features, discretizer, encoder):
    """
    Apply the same feature encoding (discretization + one-hot) to new features with robust unknown category handling.
    
    Args:
        features (np.array): Raw molecular descriptors
        discretizer: Fitted KBinsDiscretizer
        encoder: Fitted OneHotEncoder
        
    Returns:
        np.array: Encoded features
    """
    try:
        # Apply discretization
        binned_values = discretizer.transform(features)
        
        # Check for out-of-range values and clip them
        n_bins = discretizer.n_bins_[0] if hasattr(discretizer, 'n_bins_') else discretizer.n_bins
        
        # Clip values to be within the expected range [0, n_bins-1]
        if isinstance(n_bins, (list, np.ndarray)):
            # If different bins per feature
            for i, bins in enumerate(n_bins):
                binned_values[:, i] = np.clip(binned_values[:, i], 0, bins - 1)
        else:
            # If same bins for all features
            binned_values = np.clip(binned_values, 0, n_bins - 1)
        
        # Apply one-hot encoding
        encoded_features = encoder.transform(binned_values)
        
        return encoded_features
        
    except ValueError as e:
        if "unknown categories" in str(e):
            print(f"Warning: Unknown categories detected. Applying robust encoding...")
            
            # Get the number of bins from the discretizer
            if hasattr(discretizer, 'n_bins_'):
                n_bins = discretizer.n_bins_
            else:
                n_bins = discretizer.n_bins
            
            # Apply discretization and clip to valid range
            binned_values = discretizer.transform(features)
            
            # Ensure all values are within the expected categorical range
            if isinstance(n_bins, (list, np.ndarray)):
                for i, bins in enumerate(n_bins):
                    binned_values[:, i] = np.clip(binned_values[:, i], 0, bins - 1)
            else:
                binned_values = np.clip(binned_values, 0, n_bins - 1)
            
            # Try encoding again
            try:
                encoded_features = encoder.transform(binned_values)
                return encoded_features
            except ValueError:
                # If it still fails, create a safe encoder
                print("Creating safe encoding by handling categories manually...")
                
                # Get categories from the trained encoder
                categories = encoder.categories_
                safe_binned = np.zeros_like(binned_values)
                
                for i, (col_cats, col_data) in enumerate(zip(categories, binned_values.T)):
                    # Map unknown categories to the first known category
                    safe_col = np.where(np.isin(col_data, col_cats), col_data, col_cats[0])
                    safe_binned[:, i] = safe_col
                
                encoded_features = encoder.transform(safe_binned)
                return encoded_features
        else:
            raise e

def prepare_vae_dataset(smiles_list, config, logger):
    """
    Prepare molecular fingerprint and feature data for VAE training with memory optimization.
    
    Args:
        smiles_list (list): List of SMILES strings
        config (dict): Configuration dictionary
        logger: Logger instance
        
    Returns:
        tuple: (train_loader, val_loader, normalization_params)
    """
    logger.info(f"Preparing VAE dataset for {len(smiles_list)} molecules with memory optimization...")
    
    # Memory optimization: Process in smaller chunks to avoid memory explosion
    chunk_size = min(500000, len(smiles_list))  # Process max 500000 molecules at once
    all_valid_mols = []
    all_fingerprints = []
    all_features = []
    
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
        
        # Generate fingerprints and features for this chunk
        chunk_fingerprints = gen_morgan_feature(chunk_valid_mols, num_jobs=config['hardware'].get('num_jobs', 4))
        chunk_features = gen_descriptors_feature(chunk_valid_mols, num_jobs=config['hardware'].get('num_jobs', 4))
        
        # Filter None values in this chunk
        chunk_valid_data = [(mol, fp, feat) for mol, fp, feat in zip(chunk_valid_mols, chunk_fingerprints, chunk_features) 
                           if fp is not None and feat is not None]
        
        if len(chunk_valid_data) == 0:
            logger.warning(f"No valid fingerprints/features in chunk {i//chunk_size + 1}")
            continue
        
        # Unpack and accumulate
        chunk_mols, chunk_fps, chunk_feats = zip(*chunk_valid_data)
        all_valid_mols.extend(chunk_mols)
        all_fingerprints.extend(chunk_fps)
        all_features.extend(chunk_feats)
        
        # Clear chunk data to free memory
        del chunk_mol_list, chunk_valid_mols, chunk_fingerprints, chunk_features, chunk_valid_data
        del chunk_mols, chunk_fps, chunk_feats
        
        # Force garbage collection
        clear_memory()
        
        logger.info(f"Chunk {i//chunk_size + 1} processed. Total valid molecules: {len(all_valid_mols)}")
    
    if len(all_valid_mols) == 0:
        raise ValueError("No valid molecules could be processed!")
    
    # Convert to numpy arrays
    fingerprints = np.array(all_fingerprints)
    features = np.array(all_features)
    
    # Clear intermediate lists to save memory
    del all_fingerprints, all_features
    clear_memory()  # Force cleanup
    
    logger.info(f"Total valid molecules processed: {len(fingerprints)}")
    logger.info(f"Success rate: {len(fingerprints)/len(smiles_list)*100:.2f}%")
    
    # Process features using KBinsDiscretizer and OneHotEncoder
    normalization_params = None
    if config['model_params']['feature_size'] > 0:
        logger.info("Processing features with KBinsDiscretizer and OneHotEncoder...")
        
        # Get n_bins from config or use default
        n_bins = config.get('feature_processing', {}).get('n_bins', 10)
        encoding_strategy = config.get('feature_processing', {}).get('encoding_strategy', 'uniform')
        
        # Initialize discretizer
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=encoding_strategy)
        binned_values = discretizer.fit_transform(features)
        
        # Initialize and fit OneHotEncoder with unknown category handling
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(binned_values)
        
        # Clear intermediate data
        del binned_values
        clear_memory()
        
        # Update config with actual encoded feature size
        original_feature_size = features.shape[1]
        encoded_feature_size = encoded_features.shape[1]
        config['model_params']['original_feature_size'] = original_feature_size
        config['model_params']['encoded_feature_size'] = encoded_feature_size
        
        logger.info(f"Original features: {original_feature_size}, Encoded features: {encoded_feature_size}")
        
        # Create model save directory if it doesn't exist
        model_save_path = config.get('output_dir', 'model_weights')
        os.makedirs(model_save_path, exist_ok=True)
        
        # Save discretizer and encoder
        discretizer_path = os.path.join(model_save_path, 'discretizer.joblib')
        encoder_path = os.path.join(model_save_path, 'encoder.joblib')
        
        joblib.dump(discretizer, discretizer_path)
        joblib.dump(encoder, encoder_path)
        
        logger.info(f"Discretizer saved to: {discretizer_path}")
        logger.info(f"Encoder saved to: {encoder_path}")
        
        # Store normalization parameters
        normalization_params = {
            'discretizer': discretizer,
            'encoder': encoder,
            'discretizer_path': discretizer_path,
            'encoder_path': encoder_path,
            'n_bins': n_bins,
            'encoding_strategy': encoding_strategy,
            'original_feature_size': original_feature_size,
            'encoded_feature_size': encoded_feature_size
        }
        
        # Update features with encoded values
        features = encoded_features
        
        # Clear original features array
        del encoded_features
        clear_memory()  # Force cleanup after encoding
        
        logger.info(f"Features shape after encoding: {features.shape}")
    
    # Target should always be fingerprints only (not concatenated with features)
    target_data = fingerprints
    
    # Sequential dataset creation to reduce peak memory usage
    logger.info("Creating VAE dataset with sequential processing...")
    
    # Create molecule data tuples
    molecule_data = list(zip(all_valid_mols, target_data, features))
    
    # Shuffle to obtain a different train/val split on every run
    shuffled_indices = np.random.permutation(len(molecule_data))
    molecule_data = [molecule_data[idx] for idx in shuffled_indices]
    
    # Clear intermediate arrays to save memory
    del all_valid_mols, fingerprints, features, target_data
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
    train_molecules = [VAEMolecule(mol, target, feat) for mol, target, feat in tqdm(train_data, desc="Creating train molecules")]
    train_dataset = MolDataSet(train_molecules)
    
    # Clear train data before creating validation dataset
    del train_data, train_molecules
    clear_memory()  # Force cleanup
    
    logger.info("Creating validation dataset...")
    val_molecules = [VAEMolecule(mol, target, feat) for mol, target, feat in tqdm(val_data, desc="Creating val molecules")]
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
    logger.info(f"Feature size: {normalization_params.get('encoded_feature_size', 'N/A') if normalization_params else 'N/A'}")
    logger.info(f"Target data size (fingerprints only): 2048")
    
    return train_loader, val_loader, normalization_params

def generate_moleceularNN_embeddings(mol_list, model, logger, batch_size=1024, device='cuda:0'):
    """Generates embeddings for a list of SMILES strings."""
    all_embeddings = []
    for i in tqdm(range(0, len(mol_list), batch_size), desc="Processing Batches"):
        batch_mol = mol_list[i:i+batch_size]
        try:
            batch_fp = gen_morgan_feature(batch_mol, 48)
            batch_feature = gen_descriptors_feature(batch_mol, 48)
            batch_fp = torch.tensor(batch_fp, dtype=torch.float).to(device)
            batch_feature = torch.tensor(batch_feature, dtype=torch.float).to(device)
        except ValueError as e:
            logger.error(f'Error processing batch {batch_mol}: {e}')
            raise e
        
        with torch.no_grad():
            batch_embedding = model(batch_fp, output_embedding=True, feature=batch_feature)
        all_embeddings.append(batch_embedding.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)

def generate_embeddings(smiles):
    """Generates embeddings for a list of SMILES strings."""

    try:
        mol = gen_mol(smiles)
        fp = gen_mogan(mol)
    except ValueError as e:
        print(f'Error processing generate embeddings:{e}')
        raise e
        
    return fp

def generate_embeddings_features(smi_list,num_jobs):
    mols = []
    
    features_map = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(generate_embeddings)(smi) for smi in tqdm(smi_list)
    )
    for i, feats in enumerate(features_map):
        mols.append(feats)
    return mols
