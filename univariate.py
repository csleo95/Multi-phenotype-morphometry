#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from scipy.linalg import pinv


def corr(Y, X, axis=0):
    """
    Computes Pearson's correlation coefficient(s) between arrays Y and X along specified axis.
    
    Input:
        Y (numpy.ndarray): First input array.
        X (numpy.ndarray): Second input array (must have the same size as Y along the specified axis).
        axis (int, optional): Axis along which to compute the correlation (0 or 1). Default is 0.
    
    Returns:
        numpy.ndarray: Correlation coefficient(s) between Y and X.
    """
    
    assert Y.shape[axis] == X.shape[axis]

    n = Y.shape[axis]

    Y = Y - np.mean(Y, axis=axis)
    X = X - np.mean(X, axis=axis)

    Y = Y / np.std(Y, axis=axis)
    X = X / np.std(X, axis=axis)

    if axis == 0:
        r = (Y.T @ X)/n
    elif axis == 1:
        r = (Y @ X.T)/n
    else:
        r = np.nan

    return r


def linreg(C, M, Y):
    """
    Perform linear regression and compute statistics for a given contrast.
    
    Input:
        C (numpy.ndarray): Contrast vector used to test a specific linear combination of regression coefficients.
        M (numpy.ndarray): Design matrix containing predictor variables (include intercept if needed).
        Y (numpy.ndarray): Response variable(s).
    
    Returns:
        tuple: A tuple containing:
        - Contrast estimate (C @ b)
        - t-statistic for the contrast
        - Two-tailed p-value for the t-statistic
        - Cohen's d effect size for the contrast
    """
    
    N = Y.shape[0]
    
    b = np.linalg.lstsq(M, Y, rcond=None)[0]
    
    res = Y - M @ b

    varres = np.sum(res**2, axis=0) / (N - np.linalg.matrix_rank(M))

    t_stat = (C @ b) / np.sqrt((C @ np.linalg.inv(M.T @ M) @ C) * varres)
    
    df = N - np.linalg.matrix_rank(M)
    pvalue = 2 * stats.t.sf(np.abs(t_stat), df=df)
    
    cohens_d = (C @ b) / np.sqrt(varres)
    
    return C @ b, t_stat, pvalue, cohens_d


def freedman_lane(Y, Z, idy=None, p=None, Rz=None, Hz=None):
    """
    Permute response Y using the Freedman-Lane procedure.

    Input:
        Y (numpy.ndarray): Response variable.
        Z (numpy.ndarray): Nuisance variables.
        idy (array_like, optional): A permutation of indices for Y. 
            If None, one is generated using seed `p`.
        p (int, optional): Random seed (default: 1).
        Rz (numpy.ndarray, optional): Precomputed residual-forming matrix (I − Hz).
        Hz (numpy.ndarray, optional): Precomputed hat matrix for Z.

    Returns:
        numpy.ndarray: Permuted response variable.
    """
    
    N = Y.shape[0]
    
    if idy is None:
        if p is None:
            p=1
        np.random.seed(p)
        N = Y.shape[0]
        idy = np.random.permutation(N)
    
    if Hz is None or Rz is None:
        Hz = Z @ pinv(Z)
        Rz = np.eye(N) - Hz

    Y_shuf = (Rz[idy, :] + Hz) @ Y
    return Y_shuf


def combine_pvalues(pvalues):
    """
    Combine p-values using Fisher's method.
    
    Parameters:
        pvalues (numpy.ndarray): Array of p-values to combine.
    
    Returns:
        numpy.ndarray: Fisher's combined test statistic.
    """
    
    fisher_stat = -2 * np.sum(np.log(pvalues), axis=0) 
    
    return fisher_stat