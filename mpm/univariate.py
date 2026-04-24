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
    if Y.shape[axis] != X.shape[axis]:
        raise ValueError("Y and X must have the same size along the observation axis")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    n = Y.shape[axis]

    Y = Y - np.mean(Y, axis=axis, keepdims=True)
    X = X - np.mean(X, axis=axis, keepdims=True)

    Y = Y / np.std(Y, axis=axis, keepdims=True, ddof=0)
    X = X / np.std(X, axis=axis, keepdims=True, ddof=0)

    if axis == 0:
        r = (Y.T @ X)/n
    elif axis == 1:
        r = (Y @ X.T)/n

    return r


def linreg(C, M, Y, alpha=0.05):
    """
    Perform ordinary least squares regression and compute statistics for a single contrast.
    
    Input:
        C (numpy.ndarray): Contrast vector used to test a linear combination of coefficients. Supported shapes are (p,) or (p, 1).
        M (numpy.ndarray): Design matrix containing predictor variables (include intercept if needed).
        Y (numpy.ndarray): Response variable(s).
        alpha (float): Optional; significance level used to compute the two-sided confidence interval.
    
    Returns:
        tuple: A tuple containing:
        - estimate (numpy.ndarray or float): Contrast estimate
        - t_stat (numpy.ndarray or float): t-statistic for the contrast
        - pvalue (numpy.ndarray or float): Two-tailed p-value for the t-statistic
        - cohens_d (numpy.ndarray or float): Cohen's d effect size for the contrast
        - df (int): Residual degrees of freedom
        - ci_lower (numpy.ndarray or float): Lower bound of the two-sided confidence interval
        - ci_upper (numpy.ndarray or float): Upper bound of the two-sided confidence interval
    """
    N = Y.shape[0]
    
    b = np.linalg.lstsq(M, Y, rcond=None)[0]
    res = Y - M @ b
    
    df = N - np.linalg.matrix_rank(M)
    estimate = C.T @ b
    varres = np.sum(res**2, axis=0) / df
    se = np.sqrt((C.T @ np.linalg.inv(M.T @ M) @ C) * varres)
    
    cohens_d = estimate / np.sqrt(varres)
    
    t_stat = estimate / se
    pvalue = 2 * stats.t.sf(np.abs(t_stat), df=df)
    
    tcrit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = estimate - tcrit * se
    ci_upper = estimate + tcrit * se
    
    return estimate, t_stat, pvalue, cohens_d, df, ci_lower, ci_upper


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


def combine_pvalues(pvalues, axis=0):
    """
    Combine p-values using Fisher's method.
    
    Parameters:
        pvalues (numpy.ndarray): Array of p-values to combine.
    
    Returns:
        numpy.ndarray: Fisher's combined test statistic.
    """
    fisher_stat = -2 * np.sum(np.log(pvalues), axis=axis) 
    
    return fisher_stat