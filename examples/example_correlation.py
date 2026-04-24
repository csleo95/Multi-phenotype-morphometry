#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from mpm.univariate import linreg, corr, freedman_lane
from mpm.pareto import palm_pareto

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

# ── prepare Freedman-Lane permutation requirements ──
nP = 500                                                                              # number of permutations

Zc = StandardScaler(with_mean=True, with_std=False).fit_transform(Z.values)           # mean-center nuisance covariates 
Z0 = np.column_stack([np.ones((N, 1)), Zc])                                           # nuisance design matrix with intercept
Hz = Z0 @ pinv(Z0)                                                                    # hat matrix for nuisance model
Rz = np.eye(N) - Hz                                                                   # residual-forming matrix for nuisance model

# ── compute correlations across permutations ──
iu = np.triu_indices(len(phenos), k=1)                                                # upper-triangle indices
all_c = np.empty((nP, len(iu[0])))                                                    

for p in range(nP):                                                                   
    if p == 0:                                                                        # first iteration is the observed data
        Yp = Y.values                                                                  
    else:
        Yp = freedman_lane(Y.values, Zc, p=p, Rz=Rz, Hz=Hz)                           
    
    M = np.hstack((np.ones((N, 1)), X.values, Z.values))                              # design matrix with intercept
    C = np.zeros(M.shape[1]); C[1] = 1                                                # contrast vector for the diagnosis effect
    
    _, _, _, cohens, _, _, _ = linreg(C, M, Yp)                                                # Cohen's d for diagnosis effect 
    cohens_2d = cohens.reshape(len(phenos), len(regions)).T                           
    
    c = corr(cohens_2d, cohens_2d, axis=0)                                            
    all_c[p] = c[iu]                                                                  # store only unique off-diagonal elements

obs_c = all_c[0]                                                                      # observed correlations 

# ── compute p-values with using GPD ──
n_corrs = all_c.shape[1]                                                             
pvals = np.empty(n_corrs)                                                             

for j in range(n_corrs):                                                             
    P, _, _, _ = palm_pareto(                                                         
        G=np.array([np.abs(all_c[0, j])]),                                            
        Gdist=np.abs(all_c[:, j]),                                                    
        rev=False,                                                                     
        Pthr=0.1,                                                                      
        G1out=False                                                                    
    )
    pvals[j] = P[0] 

# ── FDR correction ──
_, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")                                                                                              

# ── reconstruct full symmetric matrices ──
obs_c_mat = np.full((len(phenos), len(phenos)), np.nan)                               
obs_c_mat[iu] = obs_c                                                                 
obs_c_mat[(iu[1], iu[0])] = obs_c                                                     
np.fill_diagonal(obs_c_mat, 1.0)                                                      

pvalues_mat = np.full((len(phenos), len(phenos)), np.nan)                            
pvalues_mat[iu] = pvals                                                              
pvalues_mat[(iu[1], iu[0])] = pvals                                                  

pvalues_fdr_mat = np.full((len(phenos), len(phenos)), np.nan)                         
pvalues_fdr_mat[iu] = pvals_fdr                                                       
pvalues_fdr_mat[(iu[1], iu[0])] = pvals_fdr 

# ── pack results ──      
res = {
    "corr_c":  pd.DataFrame(obs_c_mat, index=phenos, columns=phenos),                                                              
    "p_c":     pd.DataFrame(pvalues_mat, index=phenos, columns=phenos),                                                                
    "p_c_fdr": pd.DataFrame(pvalues_fdr_mat, index=phenos, columns=phenos)                                                        
}