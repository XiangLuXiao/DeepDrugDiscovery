"""
Utility helpers for applying activation functions and attention metrics to fingerprints.

Provides GPU-aware routines for large-scale pairwise attention computations and activation sweeps.

Copyright (c) 2025 Xianglu Xiao
Author: Xianglu Xiao
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Union, List


def apply_activation_to_fingerprint(fingerprint: Union[np.ndarray, List], 
                                  activation_function: str = 'relu') -> torch.Tensor:
    """
    Apply an activation function to molecular fingerprints.
    
    Converts molecular fingerprint data to PyTorch tensors and applies the specified
    activation function. This is commonly used for preprocessing fingerprints before
    neural network input or for normalizing fingerprint values.
    
    Args:
        fingerprint (Union[np.ndarray, List]): Input molecular fingerprint data.
                                             Can be a single fingerprint or batch of fingerprints.
        activation_function (str): Activation function to apply. 
                                 Supported options: 'relu', 'sigmoid', 'tanh'
                                 Default: 'relu'
    
    Returns:
        torch.Tensor: Activated fingerprint tensor with the same shape as input
        
    Raises:
        ValueError: If an unsupported activation function is specified
    """
    fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32)
    
    if activation_function == 'relu':
        return F.relu(fingerprint_tensor)
    elif activation_function == 'sigmoid':
        return torch.sigmoid(fingerprint_tensor)
    elif activation_function == 'tanh':
        return torch.tanh(fingerprint_tensor)
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}. "
                        f"Supported options: 'relu', 'sigmoid', 'tanh'")


def molecular_attention(fp_list1, fp_list2, cuda_index='cpu'):
    """
    Calculate pairwise molecular attention matrix between two sets of fingerprints.
    
    Computes attention scores between all pairs of molecules from two datasets using
    appropriate attention metrics. For binary fingerprints, uses Jaccard attention
    (intersection over union). For continuous fingerprints, uses Tanimoto attention.
    
    The function automatically detects fingerprint type and applies GPU acceleration
    when available for efficient processing of large molecular datasets.
    
    Args:
        fp_list1: Query fingerprints - first set of molecular fingerprints
                 Shape: (n_molecules_1, fingerprint_dim)
        fp_list2: Key fingerprints - second set of molecular fingerprints  
                 Shape: (n_molecules_2, fingerprint_dim)
        cuda_index: GPU device specification. 
                   Options: 'cpu', integer (GPU index), or 'cuda:X'
                   Default: 'cpu'
    
    Returns:
        numpy.ndarray: Attention matrix of shape (n_molecules_1, n_molecules_2)
                      where entry [i,j] represents attention score between molecule i from fp_list1
                      and molecule j from fp_list2. Values range from 0 (no attention) to 1 (maximum attention).
    
    Attention Metrics:
        - Binary fingerprints: Jaccard attention = |A ∩ B| / |A ∪ B|
        - Continuous fingerprints: Tanimoto attention = A·B / (|A|² + |B|² - A·B)
    """
    if cuda_index == 'cpu':
        device = 'cpu'
    else:
        device = f'cuda:{cuda_index}' if torch.cuda.is_available() else 'cpu'
    
    fp1 = np.asarray(fp_list1)
    fp2 = np.asarray(fp_list2)
    
    is_binary = np.all(np.isin(fp1, [0, 1])) and np.all(np.isin(fp2, [0, 1]))
    
    if is_binary:
        attention_matrix = np.zeros([len(fp1), len(fp2)], dtype=np.float32)
        
        fp1_tensor = torch.ByteTensor(fp1).to(device)
        fp2_tensor = torch.ByteTensor(fp2).to(device)
        
        for i in tqdm(range(len(fp1_tensor))):
            ref_fp = fp1_tensor[i]
            intersection = torch.sum(ref_fp & fp2_tensor, dim=-1, dtype=torch.float16)
            union = torch.sum(ref_fp | fp2_tensor, dim=-1, dtype=torch.float16)
            attention_matrix[i] = (intersection / union).cpu().numpy()
        
    else:
        fp1_tensor = torch.tensor(fp1, dtype=torch.float32, device=device)
        fp2_tensor = torch.tensor(fp2, dtype=torch.float32, device=device)
        
        dot = torch.matmul(fp1_tensor, fp2_tensor.T)
        norm1_sq = torch.sum(fp1_tensor**2, dim=1, keepdim=True)
        norm2_sq = torch.sum(fp2_tensor**2, dim=1, keepdim=True).T
        
        attention_matrix = (dot / (norm1_sq + norm2_sq - dot + 1e-10)).cpu().numpy()
    
    if 'cuda' in str(device):
        torch.cuda.empty_cache()
    
    return attention_matrix
