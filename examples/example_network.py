#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from mpm.univariate import linreg, freedman_lane, combine_pvalues
from mpm.pareto import palm_pareto

np.random.seed(1)                                                                     # fix random seed for reproducibility

# ── generate synthetic data ──
N = 90                                                                                # sample size
n_regions = 12                                                                        # number of regions
phenos = ["thickness", "area", "meancurv"]                                            # morphometric phenotypes
regions = [f"region_{i}" for i in range(n_regions)]                                   # region labels
columns = pd.MultiIndex.from_product([phenos, regions], names=["morpho", "region"])   # MultiIndex columns: morpho x region

Y = pd.DataFrame(np.random.randn(N, len(columns)), columns=columns)                   # harmonized neuroimaging data matrix
X = pd.DataFrame({"diagnosis": np.r_[np.zeros(N // 2), np.ones(N // 2)]})             # predictor of interest
Z = pd.DataFrame({                                                                    # nuisance covariates
    "age":   np.random.normal(30, 8, N),
    "sex":   np.random.binomial(1, 0.5, N),
    "euler": np.random.normal(-50, 15, N),
    "icv":   np.random.normal(1500, 120, N),
})

network_labels = pd.Series(                                                           # synthetic network labels for each region
    ["default"] * 4 + ["frontoparietal"] * 4 + ["somatomotor"] * 4,
    index=regions
)
networks = network_labels.unique()

# ── prepare Freedman-Lane permutation requirements ──
nP = 500                                                                              # number of permutations

Zc = StandardScaler(with_mean=True, with_std=False).fit_transform(Z.values)           # mean-center nuisance covariates
Z0 = np.column_stack([np.ones((N, 1)), Zc])                                           # nuisance design matrix with intercept
Hz = Z0 @ pinv(Z0)                                                                    # hat matrix for nuisance model
Rz = np.eye(N) - Hz                                                                   # residual-forming matrix for nuisance model

# ── compute network statistics across permutations ──
all_f = np.empty((nP, len(networks) * len(phenos)))                                   # Fisher statistics for each network x phenotype cell

for p in range(nP):
    if p == 0:                                                                        # first iteration is the observed data
        Yp = Y.values
    else:
        Yp = freedman_lane(Y.values, Zc, p=p, Rz=Rz, Hz=Hz)

    M = np.hstack((np.ones((N, 1)), X.values, Z.values))                              # design matrix with intercept
    C = np.zeros(M.shape[1]); C[1] = 1                                                # contrast vector for the diagnosis effect

    _, _, pvals, _, _, _, _ = linreg(C, M, Yp)                                                 # p-values for diagnosis effects
    pvals_2d = pd.DataFrame(
        pvals.reshape(len(phenos), len(regions)).T,
        index=regions,
        columns=phenos
    )

    k = 0
    for network in networks:
        idx = network_labels[network_labels == network].index
        for morpho in phenos:
            all_f[p, k] = combine_pvalues(pvals_2d.loc[idx, morpho].values)           # combine p-values within network
            k += 1

# ── compute p-values using GPD ──
pvals = np.empty(all_f.shape[1])

for j in range(all_f.shape[1]):
    P, _, _, _ = palm_pareto(
        G=np.array([all_f[0, j]]),
        Gdist=all_f[:, j],
        rev=False,
        Pthr=0.1,
        G1out=True
    )
    pvals[j] = P[0]

# ── FDR correction ──
_, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")

# ── pack results ──
pvalues_mat = pvals.reshape(len(networks), len(phenos))
pvalues_fdr_mat = pvals_fdr.reshape(len(networks), len(phenos))

res = {
    "pvalues":     pd.DataFrame(pvalues_mat, index=networks, columns=phenos),
    "pvalues_fdr": pd.DataFrame(pvalues_fdr_mat, index=networks, columns=phenos),
}
