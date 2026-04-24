#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from mpm.cca import cca, cca_pvalues
from mpm.univariate import corr, freedman_lane, linreg
from mpm.pareto import palm_pareto

np.random.seed(1)                                                                     # fix random seed for reproducibility

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

gene_gradients = ["C1", "C2", "C3"]                                                   # transcriptomic gradients
n_genes = 11                                                                          # number of genes
genes = [f"gene_{i}" for i in range(n_genes)]                                         # gene labels

T = pd.DataFrame(np.random.randn(n_regions, len(gene_gradients)),                     # transcriptomic gradient matrix
                 index=regions, columns=gene_gradients)

G = pd.DataFrame(np.random.randn(n_regions, n_genes),                                 # full gene expression matrix
                 index=regions, columns=genes)

# ── prepare Freedman-Lane permutation requirements ──
nP = 500                                                                              # number of permutations

Zc = StandardScaler(with_mean=True, with_std=False).fit_transform(Z.values)           # mean-center nuisance covariates
Z0 = np.column_stack([np.ones((N, 1)), Zc])                                           # nuisance design matrix with intercept
Hz = Z0 @ pinv(Z0)                                                                    # hat matrix for nuisance model
Rz = np.eye(N) - Hz                                                                   # residual-forming matrix for nuisance model

# ── compute CCA results across permutations ──
n_modes = min(len(phenos), len(gene_gradients))                                       # number of canonical modes
modes = [f"mode_{i}" for i in range(n_modes)]                                         # mode labels

all_cc = np.empty((nP, n_modes))                                                      # canonical correlations across permutations
all_l_pheno = np.empty((nP, n_modes, len(phenos)))                                    # phenotype loadings across permutations
all_l_gene = np.empty((nP, n_modes, len(genes)))                                      # gene loadings across permutations

for p in range(nP):
    if p == 0:                                                                        # first iteration is the observed data
        Yp = Y.values
    else:
        Yp = freedman_lane(Y.values, Zc, p=p, Rz=Rz, Hz=Hz)

    M = np.hstack((np.ones((N, 1)), X.values, Z.values))                              # design matrix with intercept
    C = np.zeros(M.shape[1]); C[1] = 1                                                # contrast vector for the diagnosis effect

    _, _, _, cohens, _, _, _ = linreg(C, M, Yp)                                                # Cohen's d for diagnosis effect
    cohens_2d = cohens.reshape(len(phenos), len(regions)).T                           # regions x phenotypes

    Cc = StandardScaler(with_mean=True, with_std=False).fit_transform(cohens_2d)      # mean-center imaging phenotypes
    Tc = StandardScaler(with_mean=True, with_std=False).fit_transform(T.values)       # mean-center transcriptomic gradients

    A, B, cc, U, V = cca(Cc, Tc, 0, 0)                                                # canonical weights, correlations and scores
    cc = np.asarray(cc)
    if cc.ndim == 2:
        cc = np.diag(cc)
    cc = np.ravel(cc)

    l_pheno = corr(V, cohens_2d)                                                      # phenotype loadings
    l_gene = corr(U, G.values)                                                        # gene loadings

    all_cc[p] = cc
    all_l_pheno[p] = l_pheno
    all_l_gene[p] = l_gene

l_pheno_obs = all_l_pheno[0]
l_gene_obs = all_l_gene[0]

# ── compute p-values for canonical correlations ──
p_r_unc, p_r_fwer = cca_pvalues([all_cc[p] for p in range(nP)])

# ── compute p-values for phenotype loadings using GPD ──
p_l_pheno = np.empty((n_modes, len(phenos)))

for i in range(n_modes):
    for j in range(len(phenos)):
        P, _, _, _ = palm_pareto(
            G=np.array([np.abs(all_l_pheno[0, i, j])]),
            Gdist=np.abs(all_l_pheno[:, i, j]),
            rev=False,
            Pthr=0.1,
            G1out=True
        )
        p_l_pheno[i, j] = P[0]

# ── compute p-values for gene loadings using GPD ──
p_l_gene = np.empty((n_modes, len(genes)))

for i in range(n_modes):
    for j in range(len(genes)):
        P, _, _, _ = palm_pareto(
            G=np.array([np.abs(all_l_gene[0, i, j])]),
            Gdist=np.abs(all_l_gene[:, i, j]),
            rev=False,
            Pthr=0.1,
            G1out=True
        )
        p_l_gene[i, j] = P[0]

# ── apply FDR correction to loading p-values ──
_, p_l_pheno_fdr, _, _ = multipletests(p_l_pheno.ravel(), method="fdr_bh")
_, p_l_gene_fdr, _, _ = multipletests(p_l_gene.ravel(), method="fdr_bh")

p_l_pheno_fdr = p_l_pheno_fdr.reshape(p_l_pheno.shape)
p_l_gene_fdr = p_l_gene_fdr.reshape(p_l_gene.shape)

# ── pack results ──
res = {
    "cca":           pd.DataFrame({"cc": all_cc[0],
                                   "pval": p_r_unc,
                                   "pval_fwer": p_r_fwer}, index=modes),
    "l_pheno":       pd.DataFrame(l_pheno_obs, index=modes, columns=phenos),
    "l_gene":        pd.DataFrame(l_gene_obs, index=modes, columns=genes),
    "p_l_pheno":     pd.DataFrame(p_l_pheno, index=modes, columns=phenos),
    "p_l_gene":      pd.DataFrame(p_l_gene, index=modes, columns=genes),
    "p_l_pheno_fdr": pd.DataFrame(p_l_pheno_fdr, index=modes, columns=phenos),
    "p_l_gene_fdr":  pd.DataFrame(p_l_gene_fdr, index=modes, columns=genes),
}
