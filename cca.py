#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import matrix_rank, solve
from scipy.linalg import qr, svd


"""
This code is largely based on:
    - Winkler AM, Renaud O, Smith SM, Nichols TE. Permutation inference for canonical correlation analysis. 
      Neuroimage. 2020 Oct 15;220:117065. doi: 10.1016/j.neuroimage.2020.117065. Epub 2020 Jun 27. 
      PMID: 32603857; PMCID: PMC7573815.
    - Github repository: https://github.com/andersonwinkler/PermCCA/tree/master
"""


def cca(Y, X, R, S):
    """
    Performs Canonical Correlation Analysis (CCA) between two data matrices.
    
    Input:
        Y (numpy.ndarray): First data matrix with shape (n_samples, n_features1).
        X (numpy.ndarray): Second data matrix with shape (n_samples, n_features1).
        R (int): Scaling parameter for Y's coefficients.
        S (int): Scaling parameter for X's coefficients.
    
    Returns:
        tuple: A tuple containing:
        - A (np.ndarray): Coefficient matrix for Y.
        - B (np.ndarray): Coefficient matrix for X.
        - cc (np.ndarray): Diagonal matrix of canonical correlations.
        - U (np.ndarray): Canonical variates for Y.
        - V (np.ndarray): Canonical variates for X.
    """
    
    N = Y.shape[0] 
     
    Qy, Ry, iY = qr(Y, mode='economic', pivoting=True) 
    Qx, Rx, iX = qr(X, mode='economic', pivoting=True) 
    K = min(matrix_rank(Y), matrix_rank(X)) 
    U, D, Vt = svd(Qy.T @ Qx, full_matrices=False) 
    L = U[:, :K] 
    M = Vt.T[:, :K] 
     
    cc = np.clip(np.diag(D[:K]), 0, 1) 
    cc = np.diag(cc)
    
    A = solve(Ry, L[:, :K]) * np.sqrt(N - R) 
    B = solve(Rx, M[:, :K]) * np.sqrt(N - S) 
     
    A_copy = A.copy() 
    B_copy = B.copy() 
    A[iY, :] = A_copy 
    B[iX, :] = B_copy 
     
    U = Y @ A 
    V = X @ B 
 
    return A, B, cc, U, V


def cca_pvalues(ccs_perm):
    """
    Calculates p-values for Canonical Correlation Analysis using Wilk's lambda statistic to assess statistical significance.

    Parameters:
        ccs_perm (list of np.ndarray): List of canonical correlation arrays from permutations,
                                   where the first element is from the observed data.
    
    Returns:
        numpy.ndarray: p-values for each canonical correlation.
    """
    
    cc1 = ccs_perm[0]
    nP = len(ccs_perm)
    
    lW1 = -np.flip(np.cumsum(np.flip(np.log(1 - cc1**2))))
    
    cnt = np.zeros_like(cc1)
    for cc in ccs_perm:
        lW_perm = -np.flip(np.cumsum(np.flip(np.log(1 - cc**2))))
        cnt += lW_perm >= lW1
        
    pvalues = cnt / nP
    pvalues_fwer = np.maximum.accumulate(pvalues)
    
    return pvalues, pvalues_fwer

