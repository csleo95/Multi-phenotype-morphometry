#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.multitest import multipletests
from mpm.machine_learning import run_ml
from mpm.pareto import palm_pareto

np.random.seed(1)                                                                     

# ── generate synthetic data ──
N = 90                                                                                         # sample size
n_regions = 10                                                                                 # number of regions
phenos = ["thickness", "area", "meancurv"]                                                                         # morphometric phenotypes
regions = [f"region_{i}" for i in range(n_regions)]                                            # region labels
columns = pd.MultiIndex.from_product([phenos, regions], names=["morpho", "region"])            # MultiIndex columns: morpho x region

X = pd.DataFrame(np.random.randn(N, len(columns)), columns=columns)                            # neuroimaging predictors
Y = pd.DataFrame({"severity": np.random.randn(N)})                                             # clinical target
Z = pd.DataFrame({                                                                             # nuisance covariates
    "age":   np.random.normal(30, 8, N),
    "sex":   np.random.binomial(1, 0.5, N),
    "euler": np.random.normal(-50, 15, N),
    "icv":   np.random.normal(1500, 120, N),
})
samples = pd.Series(np.repeat(["site_1", "site_2", "site_3"], N // 3), name="sample")          # site labels

# ── create nested cross-validation folds ──
n_splits = 3
n_repeats_outer = 2
n_repeats_inner = 1

blocks = samples.astype(str).values                                                            # stratify by site
outer_folds = []
inner_folds = []

skf = RepeatedStratifiedKFold(n_splits=n_splits, 
                              n_repeats=n_repeats_outer, 
                              random_state=1)                                                  # get inner and outer folds
for train, test in skf.split(np.zeros((blocks.shape[0], 1)), 
                             blocks):
    outer_folds.append([train, test])
        
    inner_folds.append([])
    i = len(inner_folds)
    skf_inner = RepeatedStratifiedKFold(n_splits=n_splits, 
                                        n_repeats=n_repeats_inner, 
                                        random_state=1)
    for train_inner, test_inner in skf_inner.split(np.zeros((blocks.shape[0], 1))[train], 
                                                   blocks[train]):
        inner_folds[i - 1].append([train_inner, test_inner])

# ── compute prediction performance across permutations for cross-region models ──
nP = 50                                                                                        # total number of permutations
all_scores_morpho = np.empty((len(phenos), nP, len(outer_folds)))                              # phenotype x permutation x outer-fold

for i, pheno in enumerate(phenos):
    X_pheno = X.loc[:, X.columns.get_level_values("morpho") == pheno]                          # one phenotype across regions

    for p in range(nP):
        all_scores_morpho[i, p] = run_ml(
            p=p,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            Y=Y,
            X=X_pheno,
            Z=Z,
            samples=samples,
            preserve_cols=["age", "sex"],
            seed=1,
            select_features=False,
            percentile=None,
            n_jobs=1
        )

obs_scores_morpho = all_scores_morpho[:, 0, :].mean(axis=1)                                     # observed mean performance for each phenotype

# ── compute p-values using GPD ──
pvals_morpho = np.empty(len(phenos))

for i in range(len(phenos)):
    P, _, _, _ = palm_pareto(
        G=np.array([obs_scores_morpho[i]]),
        Gdist=all_scores_morpho[i].mean(axis=1),
        rev=False,
        Pthr=0.1,
        G1out=True
    )
    pvals_morpho[i] = P[0]

# ── FDR correction ──
_, pvals_fdr_morpho, _, _ = multipletests(pvals_morpho, method="fdr_bh")

# ── pack results ──
res = pd.DataFrame({
    "score": obs_scores_morpho,
    "pvalue": pvals_morpho,
    "pvalue_fdr": pvals_fdr_morpho,
}, index=phenos)
