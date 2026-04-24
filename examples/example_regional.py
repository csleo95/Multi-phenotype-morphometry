#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from mpm.univariate import linreg

np.random.seed(1)

# ── generate synthetic data ──
N = 90                                                                                # sample size
n_regions = 9                                                                         # number of regions
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

# ── linear regression across regions ──
M = np.hstack((np.ones((N, 1)), X.values, Z.values))                                  # design matrix with intercept
C = np.zeros(M.shape[1])                                                              # contrast vector for the diagnosis effect
C[1] = 1

betas, tstats, pvals, cohens, df, ci_lower, ci_upper = linreg(C, M, Y.values)

# ── FDR correction ──
_, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")

# ── pack results ──
stats = {
    'cohens':      cohens,
    'betas':       betas,
    'ci_lower':    ci_lower,
    'ci_upper':    ci_upper,
    'tstats':      tstats,
    'pvalues':     pvals,
    'pvalues_fdr': pvals_fdr
}

res = {}
for pheno in phenos:
    idx = [columns.get_loc((pheno, r)) for r in regions]
    df_pheno = pd.DataFrame(
        {stat_name: arr[idx] for stat_name, arr in stats.items()},
        index=regions,
    )
    df_pheno['df'] = df         
    res[pheno] = df_pheno