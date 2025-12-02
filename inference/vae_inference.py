"""
Molecular VAE inference engine for generating embeddings from SMILES batches.

This module supports streaming/batch inference, large dataset handling, and flexible outputs.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

import torch
import numpy as np
import pandas as pd
import joblib
import os
import sys
import yaml
from tqdm import tqdm
import logging
from typing import List, Union, Optional, Tuple, Iterator
import warnings
warnings.filterwarnings('ignore')

from models.molecular_vae import MolecularVAE
from datasets.molecularNN_dataset import (
    gen_mol_feature, gen_morgan_feature, gen_descriptors_feature,
    apply_feature_encoding, clear_memory
)


class MolecularVAEInference:
    """
    Inference class for getting embeddings from trained Molecular VAE.
    
    This class handles the complete pipeline from SMILES strings to latent embeddings,
    including feature preprocessing, batch processing, and memory management for large datasets.
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, batch_size: int = 256):
        """
        Initialize the inference engine.
        
        Args:
            model_path (str): Path to the trained model checkpoint (.pth file)
            config_path (str, optional): Path to config file. If None, loads from checkpoint.
            batch_size (int): Batch size for model inference (default: 256)
            
        Note:
            The inference engine expects the following files in the same directory as model_path:
            - best_model.pth (or your model checkpoint)
            - discretizer.joblib (if features were encoded during training)
            - encoder.joblib (if features were encoded during training)
            
            Example directory structure:
            model_checkpoints/
            ├── best_model.pth
            ├── discretizer.joblib
            └── encoder.joblib
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        self.logger = self._setup_logging()
        self.checkpoint = self._load_checkpoint()
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.checkpoint.get('config', {})
        
        # Initialize model
        self.model = self._load_model()
        
        # Load feature normalization parameters
        self.discretizer = None
        self.encoder = None
        self._load_feature_normalization()
        
        self.logger.info(f"VAE Inference engine initialized on {self.device}")
        self.logger.info(f"Model fingerprint size: {self.config['model_params']['fingerprint_size']}")
        self.logger.info(f"Model feature size: {self.config['model_params']['feature_size']}")
        self.logger.info(f"Latent embedding size: {self.config['model_params']['latent_size']}")
        self.logger.info(f"Inference batch size: {self.batch_size}")

    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration for inference operations.
        
        Configures logging with timestamp, level, and message formatting
        for tracking inference progress and debugging.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_checkpoint(self) -> dict:
        """
        Load model checkpoint from disk with safety checks.
        
        Attempts to load checkpoint safely with weights_only=True first,
        falling back to weights_only=False for compatibility with older checkpoints.
        
        Returns:
            dict: Loaded checkpoint containing model state and configuration
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            Exception: If checkpoint cannot be loaded
        """
        self.logger.info(f"Loading checkpoint from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        try:
            # Try loading with weights_only=True first (safer)
            try:
                from sklearn.preprocessing._discretization import KBinsDiscretizer
                from sklearn.preprocessing._encoders import OneHotEncoder
                
                # Use safe_globals to allow sklearn objects
                with torch.serialization.safe_globals([KBinsDiscretizer, OneHotEncoder]):
                    checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.logger.info("Checkpoint loaded safely with weights_only=True")
                
            except Exception as safe_error:
                self.logger.warning(f"Safe loading failed: {safe_error}")
                self.logger.info("Falling back to weights_only=False (checkpoint from trusted source)")
                
                # Fall back to weights_only=False for compatibility
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.logger.info("Checkpoint loaded with weights_only=False")
                
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        self.logger.info(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        return checkpoint

    def _load_model(self) -> MolecularVAE:
        """
        Initialize and load the trained MolecularVAE model.
        
        Creates model instance from config parameters, loads trained weights,
        and sets to evaluation mode for inference.
        
        Returns:
            MolecularVAE: Loaded and configured model in evaluation mode
        """
        model = MolecularVAE(self.config['model_params']).to(self.device)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        
        self.logger.info("Model loaded and set to evaluation mode")
        return model

    def _ensure_numpy_joblib_compatibility(self):
        """Alias legacy numpy modules expected by old joblib artifacts."""
        try:
            import numpy.core as np_core
        except ImportError:
            return

        alias_map = {
            'numpy._core': np_core,
            'numpy._core.numeric': getattr(np_core, 'numeric', None),
            'numpy._core.multiarray': getattr(np_core, 'multiarray', None),
        }

        for alias, target in alias_map.items():
            if target is None:
                continue
            if alias not in sys.modules:
                sys.modules[alias] = target

    def _load_feature_normalization(self):
        """
        Load feature normalization parameters from model directory.
        
        Attempts to load discretizer and encoder from the same directory as the model checkpoint.
        These are required if the model was trained with feature encoding. Logs warnings if
        files are missing but feature processing is expected.
        """
        # Get the directory containing the model checkpoint
        model_dir = os.path.dirname(self.model_path)

        # Define paths for discretizer and encoder in the same directory
        discretizer_path = os.path.join(model_dir, 'discretizer.joblib')
        encoder_path = os.path.join(model_dir, 'encoder.joblib')

        # Ensure numpy compatibility for legacy joblib files before loading
        self._ensure_numpy_joblib_compatibility()

        # Load discretizer
        if os.path.exists(discretizer_path):
            try:
                self.discretizer = joblib.load(discretizer_path)
                self.logger.info(f"Discretizer loaded from {discretizer_path}")
            except Exception as e:
                self.logger.error(f"Failed to load discretizer from {discretizer_path}: {e}")
                self.discretizer = None
        else:
            self.logger.warning(f"Discretizer not found at {discretizer_path}")
            self.discretizer = None
        
        # Load encoder
        if os.path.exists(encoder_path):
            try:
                self.encoder = joblib.load(encoder_path)
                self.logger.info(f"Encoder loaded from {encoder_path}")
            except Exception as e:
                self.logger.error(f"Failed to load encoder from {encoder_path}: {e}")
                self.encoder = None
        else:
            self.logger.warning(f"Encoder not found at {encoder_path}")
            self.encoder = None
        
        # Check if feature processing is expected but normalization parameters are missing
        if self.config['model_params']['feature_size'] > 0:
            if self.discretizer is None or self.encoder is None:
                self.logger.warning("Feature processing is enabled but normalization parameters not found!")
                self.logger.warning("Features will be used without encoding (may cause errors)")
                self.logger.info(f"Expected files: {discretizer_path}, {encoder_path}")
            else:
                self.logger.info("Feature normalization parameters loaded successfully")
        else:
            self.logger.info("Model configured without additional features - no normalization needed")

    def process_smiles(self, smiles_list: List[str], 
                      batch_size: int = 100000) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Process SMILES strings into fingerprints and features.
        
        Converts SMILES strings to RDKit molecules, generates Morgan fingerprints and
        molecular descriptors, and applies feature encoding if available.
        
        Args:
            smiles_list (List[str]): List of SMILES strings to process
            batch_size (int): Batch size for processing large datasets (default: 100000)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[int]]: 
                - fingerprints: Morgan fingerprint arrays
                - features: Molecular descriptor arrays (encoded if normalization available)
                - valid_indices: Indices of successfully processed molecules
                
        Raises:
            ValueError: If no valid molecules could be processed
        """
        self.logger.info(f"Processing {len(smiles_list)} SMILES strings...")
        
        all_fingerprints = []
        all_features = []
        valid_indices = []
        
        # Process in batches to manage memory
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Processing SMILES"):
            batch_smiles = smiles_list[i:i+batch_size]
            
            # Convert SMILES to molecules
            batch_mols = gen_mol_feature(batch_smiles, 
                                       num_jobs=self.config['hardware'].get('num_jobs', 4))
            
            # Filter valid molecules
            batch_valid_mols = []
            batch_valid_indices = []
            
            for j, mol in enumerate(batch_mols):
                if mol is not None:
                    batch_valid_mols.append(mol)
                    batch_valid_indices.append(i + j)
            
            if len(batch_valid_mols) == 0:
                continue
            
            # Generate fingerprints and features
            batch_fingerprints = gen_morgan_feature(batch_valid_mols, 
                                                  num_jobs=self.config['hardware'].get('num_jobs', 4))
            batch_features = gen_descriptors_feature(batch_valid_mols, 
                                                   num_jobs=self.config['hardware'].get('num_jobs', 4))
            
            # Filter out None values
            for k, (fp, feat) in enumerate(zip(batch_fingerprints, batch_features)):
                if fp is not None and feat is not None:
                    all_fingerprints.append(fp)
                    all_features.append(feat)
                    valid_indices.append(batch_valid_indices[k])
            
            # Clear memory
            clear_memory()
        
        if len(all_fingerprints) == 0:
            raise ValueError("No valid molecules could be processed!")
        
        fingerprints = np.array(all_fingerprints)
        features = np.array(all_features)
        
        self.logger.info(f"Successfully processed {len(fingerprints)} molecules "
                        f"({len(fingerprints)/len(smiles_list)*100:.1f}% success rate)")
        
        # Apply feature encoding if available
        if self.config['model_params']['feature_size'] > 0 and features.shape[1] > 0:
            if self.discretizer is not None and self.encoder is not None:
                self.logger.info("Applying feature encoding...")
                features = apply_feature_encoding(features, self.discretizer, self.encoder)
                self.logger.info(f"Features encoded to shape: {features.shape}")
            else:
                self.logger.warning("No feature encoding applied - using raw features")
        
        return fingerprints, features, valid_indices

    def get_embeddings(self, smiles_list: List[str], 
                      return_valid_indices: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
        """
        Get latent embeddings for a list of SMILES strings.
        
        Processes SMILES strings through the complete pipeline: molecule generation,
        feature extraction, and neural network encoding to latent embeddings.
        
        Args:
            smiles_list (List[str]): List of SMILES strings to encode
            return_valid_indices (bool): Whether to return indices of successfully processed molecules
            
        Returns:
            np.ndarray or Tuple[np.ndarray, List[int]]: 
                - If return_valid_indices=False: Embeddings array of shape (n_valid_molecules, latent_size)
                - If return_valid_indices=True: (embeddings, valid_indices) tuple
        """
        # Process SMILES to get fingerprints and features
        fingerprints, features, valid_indices = self.process_smiles(smiles_list)
        
        # Get embeddings from model
        self.logger.info(f"Generating embeddings for {len(fingerprints)} molecules...")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(fingerprints), self.batch_size), desc="Generating embeddings"):
                # Prepare batch
                batch_fps = fingerprints[i:i+self.batch_size]
                batch_feats = features[i:i+self.batch_size]
                
                # Convert to tensors
                fp_tensor = torch.from_numpy(batch_fps).float().to(self.device)
                feat_tensor = torch.from_numpy(batch_feats).float().to(self.device)
                
                # Get embeddings
                if self.config['model_params']['feature_size'] > 0:
                    embeddings = self.model(fp_tensor, feat_tensor, output_embedding=True)
                else:
                    embeddings = self.model(fp_tensor, output_embedding=True)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        if return_valid_indices:
            return embeddings, valid_indices
        else:
            return embeddings

    def get_embeddings_dataframe(self, df: pd.DataFrame, 
                                smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Get embeddings for molecules in a DataFrame.
        
        Processes SMILES from a pandas DataFrame and returns a new DataFrame
        with embeddings added as additional columns. Only successfully processed
        molecules are included in the result.
        
        Args:
            df (pd.DataFrame): DataFrame containing SMILES strings
            smiles_column (str): Name of the column containing SMILES (default: 'smiles')
            
        Returns:
            pd.DataFrame: DataFrame with original columns plus embedding columns
                         (embedding_0, embedding_1, ..., embedding_n)
                         
        Raises:
            ValueError: If specified SMILES column is not found in DataFrame
        """
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in DataFrame")
        
        smiles_list = df[smiles_column].tolist()
        embeddings, valid_indices = self.get_embeddings(smiles_list, return_valid_indices=True)
        
        # Create result DataFrame with only valid rows
        result_df = df.iloc[valid_indices].copy().reset_index(drop=True)
        
        # Add embedding columns
        embedding_columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)
        
        # Combine original data with embeddings
        result_df = pd.concat([result_df, embedding_df], axis=1)
        
        self.logger.info(f"Created result DataFrame with {len(result_df)} rows and {len(embedding_columns)} embedding dimensions")
        
        return result_df

    def get_embeddings_streaming(self, smiles_list: List[str], 
                               output_path: str,
                               processing_chunk_size: int = 50000,
                               compress: bool = True) -> Tuple[int, int]:
        """
        Generate embeddings for large datasets with streaming processing to minimize memory usage.
        
        Processes large datasets in chunks to avoid memory overflow, automatically
        saves results to disk, and provides memory-efficient handling of datasets
        with millions of molecules.
        
        Args:
            smiles_list (List[str]): List of SMILES strings to process
            output_path (str): Output file path (will add .npz extension if missing)
            processing_chunk_size (int): Number of SMILES to process at once (default: 50000)
            compress (bool): Whether to compress the .npz file (default: True)
            
        Returns:
            Tuple[int, int]: (total_processed, total_valid) - counts of molecules processed
            
        Note:
            This method is designed for datasets too large to fit in memory. It processes
            data in chunks and immediately saves results, keeping memory usage constant
            regardless of dataset size.
        """
        self.logger.info(f"Starting streaming inference for {len(smiles_list)} molecules...")
        self.logger.info(f"Processing in chunks of {processing_chunk_size}, inference batch size: {self.batch_size}")
        
        total_processed = 0
        total_valid = 0
        all_embeddings = []
        all_valid_smiles = []
        all_valid_indices = []
        
        # Ensure .npz extension
        if not output_path.endswith('.npz'):
            output_path = output_path.rstrip('.') + '.npz'
        
        # Process in chunks
        for chunk_start in tqdm(range(0, len(smiles_list), processing_chunk_size), 
                               desc="Processing chunks"):
            chunk_end = min(chunk_start + processing_chunk_size, len(smiles_list))
            chunk_smiles = smiles_list[chunk_start:chunk_end]
            
            try:
                # Process chunk
                fingerprints, features, valid_indices = self.process_smiles(chunk_smiles)
                
                if len(fingerprints) == 0:
                    self.logger.warning(f"No valid molecules in chunk {chunk_start}-{chunk_end}")
                    continue
                
                # Get embeddings for chunk
                chunk_embeddings = []
                with torch.no_grad():
                    for i in range(0, len(fingerprints), self.batch_size):
                        batch_fps = fingerprints[i:i+self.batch_size]
                        batch_feats = features[i:i+self.batch_size]
                        
                        fp_tensor = torch.from_numpy(batch_fps).float().to(self.device)
                        feat_tensor = torch.from_numpy(batch_feats).float().to(self.device)
                        
                        if self.config['model_params']['feature_size'] > 0:
                            embeddings = self.model(fp_tensor, feat_tensor, output_embedding=True)
                        else:
                            embeddings = self.model(fp_tensor, output_embedding=True)
                        
                        chunk_embeddings.append(embeddings.cpu().numpy())
                
                if chunk_embeddings:
                    chunk_embeddings = np.concatenate(chunk_embeddings, axis=0)
                    
                    # Store results
                    all_embeddings.append(chunk_embeddings)
                    
                    # Store metadata
                    chunk_valid_smiles = [chunk_smiles[i] for i in valid_indices]
                    chunk_valid_indices = [chunk_start + i for i in valid_indices]
                    
                    all_valid_smiles.extend(chunk_valid_smiles)
                    all_valid_indices.extend(chunk_valid_indices)
                    
                    total_valid += len(chunk_embeddings)
                
                total_processed += len(chunk_smiles)
                
                # Clear memory
                del fingerprints, features, chunk_embeddings
                clear_memory()
                
                self.logger.info(f"Processed chunk {chunk_start}-{chunk_end}: "
                               f"{len(valid_indices)} valid molecules")
                
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                continue
        
        # Combine all embeddings
        if all_embeddings:
            self.logger.info("Combining all embeddings...")
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            
            # Save results
            self.save_embeddings(
                final_embeddings, 
                output_path,
                smiles_list=smiles_list,
                valid_indices=all_valid_indices,
                compress=compress
            )
            
            self.logger.info(f"Streaming inference completed: {total_valid}/{total_processed} molecules processed")
            self.logger.info(f"Final embeddings shape: {final_embeddings.shape}")
            
            # Clear final arrays
            del final_embeddings, all_embeddings
            clear_memory()
        
        else:
            self.logger.error("No valid embeddings generated!")
        
        return total_processed, total_valid

    def save_embeddings(self, embeddings: np.ndarray, 
                       output_path: str,
                       smiles_list: Optional[List[str]] = None,
                       valid_indices: Optional[List[int]] = None,
                       compress: bool = True):
        """
        Save embeddings to .npz file with metadata.
        
        Saves embeddings along with optional SMILES strings and original indices
        for full traceability of results.
        
        Args:
            embeddings (np.ndarray): Embeddings array to save
            output_path (str): Output file path (will add .npz extension if missing)
            smiles_list (List[str], optional): Original SMILES list for metadata
            valid_indices (List[int], optional): Valid molecule indices for metadata
            compress (bool): Whether to use compression (default: True)
            
        Note:
            Compressed files are typically 50-80% smaller but take slightly longer to save/load.
            Use compress=False for fastest I/O if disk space is not a concern.
        """
        # Ensure .npz extension
        if not output_path.endswith('.npz'):
            output_path = output_path.rstrip('.') + '.npz'
        
        # Prepare save dictionary
        save_dict = {'embeddings': embeddings}
        
        if smiles_list is not None and valid_indices is not None:
            valid_smiles = np.array([smiles_list[i] for i in valid_indices])
            save_dict['smiles'] = valid_smiles
            save_dict['original_indices'] = np.array(valid_indices)
        
        # Save with or without compression
        if compress:
            np.savez_compressed(output_path, **save_dict)
        else:
            np.savez(output_path, **save_dict)
        
        file_size = os.path.getsize(output_path) / (1024**2)  # MB
        compression_str = "compressed" if compress else "uncompressed"
        self.logger.info(f"Embeddings saved to {output_path} ({compression_str}, {file_size:.1f} MB)")

    @staticmethod
    def load_embeddings(file_path: str, return_metadata: bool = False) -> Union[np.ndarray, Tuple]:
        """
        Load embeddings from .npz file.
        
        Args:
            file_path (str): Path to .npz embeddings file
            return_metadata (bool): Whether to return SMILES and indices if available
            
        Returns:
            np.ndarray or Tuple: 
                - If return_metadata=False: Embeddings array
                - If return_metadata=True: (embeddings, smiles, indices) tuple
                
        Raises:
            ValueError: If file is not in .npz format
        """
        if not file_path.endswith('.npz'):
            raise ValueError("File must be .npz format")
        
        data = np.load(file_path)
        embeddings = data['embeddings']
        
        if return_metadata:
            smiles = data.get('smiles', None)
            indices = data.get('original_indices', None)
            return embeddings, smiles, indices
        return embeddings

    @staticmethod
    def load_embeddings_chunked(file_path: str, chunk_size: int = 10000) -> Iterator[np.ndarray]:
        """
        Load embeddings in chunks for processing very large .npz files.
        
        Useful for processing embeddings that don't fit in memory, allowing
        streaming operations on large embedding datasets.
        
        Args:
            file_path (str): Path to .npz embeddings file
            chunk_size (int): Number of embeddings to load at once (default: 10000)
            
        Yields:
            np.ndarray: Chunks of embeddings of size (chunk_size, embedding_dim)
            
        Raises:
            ValueError: If file is not in .npz format
        """
        if not file_path.endswith('.npz'):
            raise ValueError("File must be .npz format")
        
        # Load full array (npz doesn't support memory mapping)
        data = np.load(file_path)
        embeddings = data['embeddings']
        total_size = embeddings.shape[0]
        
        for i in range(0, total_size, chunk_size):
            end_idx = min(i + chunk_size, total_size)
            yield embeddings[i:end_idx]

    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """
        Get comprehensive information about a .npz embeddings file.
        
        Provides detailed metadata about embeddings files including size,
        shape, data types, and available metadata.
        
        Args:
            file_path (str): Path to .npz embeddings file
            
        Returns:
            dict: File information including:
                - file_path: Path to the file
                - file_size_mb: File size in megabytes
                - shape: Embeddings array shape
                - dtype: Data type of embeddings
                - n_molecules: Number of molecules
                - embedding_dim: Dimensionality of embeddings
                - has_smiles: Whether SMILES metadata is included
                - has_indices: Whether original indices are included
                
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not in .npz format
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.endswith('.npz'):
            raise ValueError("File must be .npz format")
        
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024**2)
        
        info = {
            'file_path': file_path,
            'file_size_mb': file_size_mb,
            'file_extension': '.npz'
        }
        
        try:
            data = np.load(file_path)
            if 'embeddings' in data:
                embeddings = data['embeddings']
                info['shape'] = embeddings.shape
                info['dtype'] = str(embeddings.dtype)
                info['n_molecules'] = embeddings.shape[0]
                info['embedding_dim'] = embeddings.shape[1]
                info['has_smiles'] = 'smiles' in data
                info['has_indices'] = 'original_indices' in data
        
        except Exception as e:
            info['error'] = str(e)
        
        return info
