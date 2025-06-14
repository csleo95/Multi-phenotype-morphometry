#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d


"""
This code is a translation of the original code from the repository:
https://github.com/andersonwinkler/PALM
"""


def palm_competitive(X, ord='ascend', mod=False):
    """
    Original function: https://github.com/andersonwinkler/PALM/blob/f20d0be2387530175faedefcbb93f422ab7f92dd/palm_competitive.m#L4
    
    Sort a set of values and return their competition ranks, i.e., 1224,
    or the modified competition ranks, i.e., 1334. This makes a difference
    only when there are ties in the data. The function returns the ranks
    in their original order as well as sorted.
    
    Input:
        X (np.ndarray): 2D numpy array with the original data. The function operates
                        on columns. To operate on rows or other dimensions, transpose
                        or permute the array's higher dimensions.
        ord (str): Sort as 'ascend' (default) or 'descend'.
        mod (bool): If True, returns the modified competition ranks, i.e., 1334.
                    This is correct for p-values and cdf. Otherwise, returns
                    standard competition ranks.

    Returns:
        tuple: A tuple containing:
            - unsrtR (np.ndarray): Competition ranks in the original order.
            - S (np.ndarray): Sorted values, just as in 'sort'.
            - srtR (np.ndarray): Competition ranks sorted as in S.
    """
    
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError('X must be a 2D array.')
    nR, nC = X.shape
    unsrtR = np.zeros_like(X, dtype=np.float32)

    if mod:
        if ord.lower() == 'ascend':
            ord = 'descend'
        elif ord.lower() == 'descend':
            ord = 'ascend'
        else:
            raise ValueError('ord must be "ascend" or "descend"')

    S = np.zeros_like(X, dtype=X.dtype)
    srtR = np.zeros_like(X, dtype=np.int32)

    for c in range(nC):  
        x = X[:, c].copy()

        infpos = np.isinf(x) & (x > 0)
        infneg = np.isinf(x) & (x < 0)
        finite_x = x[~infpos & ~infneg]

        if np.all(infpos | infneg):
            raise ValueError(
                'Data cannot be sorted. Maximum statistic is +Inf or -Inf '
                'for all permutations. Ensure that the input data, design, '
                'and contrasts are meaningful.')

        if np.any(infpos):
            x[infpos] = np.max(finite_x) + 1
        if np.any(infneg):
            x[infneg] = np.min(finite_x) - 1

        if ord.lower() == 'ascend':
            idx = np.argsort(x)
        elif ord.lower() == 'descend':
            idx = np.argsort(-x)
        else:
            raise ValueError('ord must be "ascend" or "descend"')

        S[:, c] = x[idx]
        rev_idx = np.argsort(idx)  

        srtR_col = np.arange(1, nR + 1)  
        dd = np.diff(S[:, c])

        if np.any(np.isnan(dd)):
            raise ValueError(
                'Data cannot be sorted. Check for NaNs that might be present, '
                'or precision issues that may cause over/underflow.')

        ties = np.concatenate(([False], dd == 0))

        for i in range(1, nR):
            if ties[i]:
                srtR_col[i] = srtR_col[i - 1]

        srtR[:, c] = srtR_col
        unsrtR[:, c] = srtR_col[rev_idx]

        if np.any(infpos):
            S[infpos[idx], c] = np.inf
        if np.any(infneg):
            S[infneg[idx], c] = -np.inf

    if mod:
        unsrtR = nR - unsrtR + 1

        S = np.flipud(S)
        srtR = np.flipud(nR - srtR + 1)

    return unsrtR, S, srtR


def palm_pareto(G, Gdist, rev, Pthr, G1out):
    """
    Original function: https://github.com/andersonwinkler/PALM/blob/f20d0be2387530175faedefcbb93f422ab7f92dd/palm_pareto.m
    
    Compute the p-values for a set of statistics G, taking
    as reference a set of observed values for G, from which
    the empirical cumulative distribution function (CDF) is
    generated. If the p-values are below Pthr, these are
    refined further using a tail approximation from the
    Generalized Pareto Distribution (GPD).
    
    Input:
        G (np.ndarray): Array of statistics to be converted to p-values.
        Gdist (np.ndarray): Array of observed values for the same statistic from which the empirical CDF is built.
                            It does not need to be sorted.
        rev (bool): If True, indicates that smaller values in G and Gdist are more significant.
        Pthr (float): P-value threshold below which the p-values will be refined using the GPD tail.
        G1out (bool): Boolean indicating whether to remove the first element of Gdist from the null distribution.

     Returns:
        tuple: A tuple containing:
            - P (np.ndarray): Array of computed p-values.
            - apar (float or None): Scale parameter of the GPD (None if not estimated).
            - kpar (float or None): Shape parameter of the GPD (None if not estimated).
            - upar (float or None): Location parameter of the GPD (None if not estimated).
    """
    
    apar = None
    kpar = None
    upar = None

    if G1out:
        Gdist = Gdist[1:]

    P = palm_datapval(G, Gdist, rev)
    Pidx = P < Pthr  

    if np.any(Pidx):
        nP = Gdist.shape[0]

        if rev:
            _, Gdist_sorted, Gcdf = palm_competitive(Gdist[:,np.newaxis], ord='descend', mod=True)
        else:
            _, Gdist_sorted, Gcdf = palm_competitive(Gdist[:,np.newaxis], ord='ascend', mod=True)
        
        Gcdf = Gcdf / nP
 
        Q = np.arange(751, 1000, 10) / 1000
        nQ = Q.size
        q = 0
        Ptail = None
  
        while (Ptail is None or np.any(np.isnan(Ptail))) and q < nQ - 2: # -2 is because of 0-based indexing
            qidx = Gcdf >= Q[q]
            Gtail = Gdist_sorted[qidx]
            qi = np.where(qidx)[0][0]

            if qi == 0:
                upar = Gdist_sorted[qi] - np.mean(Gdist_sorted[qi:qi + 2])
            else:
                upar = np.mean(Gdist_sorted[qi - 1:qi + 1])

            if rev:
                ytail = upar - Gtail
                y = upar - G[(G < upar) & Pidx]
            else:
                ytail = Gtail - upar
                y = G[(G > upar) & Pidx] - upar

            x = np.mean(ytail)
            s2 = np.var(ytail, ddof=1)
            apar = x * (x**2 / s2 + 1) / 2
            kpar = (x**2 / s2 - 1) / 2
            
            A2pval = andersondarling(np.round(gpdpvals(ytail, apar, kpar), decimals=8), kpar) #this yields exact translation from MATLAB

            if A2pval > 0.05:
                cte = len(Gtail) / nP
                Ptail = cte * gpdpvals(y, apar, kpar)
            else:
                q += 1
            
        if Ptail is not None and not np.isnan(Ptail).all():
            if rev:
                P_indices = (G < upar) & Pidx
                P[P_indices] = Ptail
            else:
                P_indices = (G > upar) & Pidx
                P[P_indices] = Ptail

    return P, apar, kpar, upar


def palm_datapval(G, Gdist, rev):
    """
    Original function: https://github.com/andersonwinkler/PALM/blob/f20d0be2387530175faedefcbb93f422ab7f92dd/palm_datapval.m
    Compute permutation p-values from a null distribution.

    Input:
        G (np.ndarray): Array of observed statistics.
        Gdist (np.ndarray): Array representing the null distribution of the statistic.
        rev (bool): If True, smaller values in G and Gdist are considered more significant.

    Returns:
        np.ndarray: Array of permutation p-values computed for each element in G.
    """

    if rev:
        P = np.mean(Gdist[:, np.newaxis] <= G[np.newaxis, :], axis=0)
    else:
        P = np.mean(Gdist[:, np.newaxis] >= G[np.newaxis, :], axis=0)
    return P


def gpdpvals(x, a, k):
    """
    Original function: https://github.com/andersonwinkler/PALM/blob/f20d0be2387530175faedefcbb93f422ab7f92dd/palm_pareto.m
    Compute the p-values for a GPD with parameters a (scale) and k (shape).

    Parameters:
        x (array-like): Data points at which to evaluate the GPD.
        a (float): Scale parameter of the GPD.
        k (float): Shape parameter of the GPD.

    Returns:
        np.ndarray: An array of p-values computed for each element in x.
    """
    eps = np.finfo(float).eps
    x = np.array(x)
    if np.abs(k) < eps:
        p = np.exp(-x / a)
    else:
        p = np.maximum(1 - k * x / a, 0) ** (1 / k)
    if k > 0:
        p[x > a / k] = 0
    return p


def andersondarling(z, k):
    """
    Original function: https://github.com/andersonwinkler/PALM/blob/f20d0be2387530175faedefcbb93f422ab7f92dd/palm_pareto.m
    
    Compute the Anderson-Darling statistic and return an approximated p-value based on the tables provided in:
    * Choulakian V, Stephens M A. Goodness-of-Fit Tests for the Generalized Pareto Distribution. Technometrics.
    2001;43(4):478-484.

    Parameters:
        z (np.ndarray): 1D array of cumulative probability values used to compute the Anderson-Darling statistic.
        k (float): Shape parameter for the reference tables (values below 0.5 are set to 0.5).

    Returns:
        float: The interpolated p-value based on the computed Anderson-Darling statistic.
    """

    ktable = np.array([0.9, 0.5, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5])
    ptable = np.array([0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    A2table = np.array([
        [0.3390, 0.4710, 0.6410, 0.7710, 0.9050, 1.0860, 1.2260, 1.5590],
        [0.3560, 0.4990, 0.6850, 0.8300, 0.9780, 1.1800, 1.3360, 1.7070],
        [0.3760, 0.5340, 0.7410, 0.9030, 1.0690, 1.2960, 1.4710, 1.8930],
        [0.3860, 0.5500, 0.7660, 0.9350, 1.1100, 1.3480, 1.5320, 1.9660],
        [0.3970, 0.5690, 0.7960, 0.9740, 1.1580, 1.4090, 1.6030, 2.0640],
        [0.4100, 0.5910, 0.8310, 1.0200, 1.2150, 1.4810, 1.6870, 2.1760],
        [0.4260, 0.6170, 0.8730, 1.0740, 1.2830, 1.5670, 1.7880, 2.3140],
        [0.4450, 0.6490, 0.9240, 1.1400, 1.3650, 1.6720, 1.9090, 2.4750],
        [0.4680, 0.6880, 0.9850, 1.2210, 1.4650, 1.7990, 2.0580, 2.6740],
        [0.4960, 0.7350, 1.0610, 1.3210, 1.5900, 1.9580, 2.2430, 2.9220],
    ])

    k = max(0.5, k)

    z = np.flipud(z)

    n = len(z)
    j = np.arange(1, n + 1)

    term1 = np.log(np.maximum(z, np.finfo(float).eps))
    term2 = np.log(1 - z[::-1])
    A2 = -n - (1 / n) * np.dot(2 * j - 1, (term1 + term2).T)

    i1 = np.zeros(len(ptable))
    for idx in range(len(ptable)):
        interp_func = interp1d(ktable, A2table[:, idx], kind='linear', fill_value='extrapolate')
        i1[idx] = interp_func(k)

    interp_func_pval = interp1d(i1, ptable, kind='linear', fill_value='extrapolate', assume_sorted=False)
    i2 = interp_func_pval(A2)
    A2pval = max(0, min(i2, 1))

    return A2pval