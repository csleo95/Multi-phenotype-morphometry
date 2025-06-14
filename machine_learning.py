#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import SelectPercentile, f_regression, f_classif
from joblib import Parallel, delayed
from neuroCombat import neuroCombat, neuroCombatFromTraining
from univariate import corr, freedman_lane


def is_categorical(column):
    """
    Determines if a column contains only binary (0 and 1) values.
    
    Input:
        column (array-like): A column of data.
    
    Returns:
        bool: True if the column contains only 0s and 1s, False otherwise.
    """
    
    unique_values = np.unique(column)
    return set(unique_values).issubset({0, 1})


def harmonize(Y, X, Z, samples, preserve_cols):
    """
    Harmonize neuroimaging data using neuroCombat.
    
    Input:
        Y (pandas.DataFrame): Primary variables of interest.
        X (pandas.DataFrame): Neuroimaging data to be harmonized.
        Z (pandas.DataFrame): Covariate data.
        samples (pandas.Series): Sample/batch identifiers.
        preserve_cols (list): Columns from Z whose effect(s) should be preserved during harmonization.
    
    Returns:
        tuple: A tuple containing:
        - X_harmonized (pd.DataFrame): Harmonized neuroimaging data.
        - estimates (dict): Harmonization estimates from neuroCombat.
    """
    
    pheno = pd.concat([Y,Z[preserve_cols],samples], axis=1)

    cat_columns = [col for col in pheno.columns if pheno[col].dropna().isin([0, 1]).all() and col != 'sample']
    num_columns = [col for col in pheno.columns if not pheno[col].dropna().isin([0, 1]).all() and col != 'sample']

    harmonization = neuroCombat(X.T, pheno, 'sample', cat_columns, num_columns)
    
    X_harmonized = pd.DataFrame(harmonization['data']).T
    X_harmonized.index = X.index 
    X_harmonized.columns = pd.MultiIndex.from_tuples(X.columns)

    estimates = harmonization['estimates']
    
    return X_harmonized, estimates


def apply_harmonization(Y, X, Z, samples, estimates, preserve_cols):
    """
    Apply harmonization to neuroimaging data using neuroCombatFromTraining.
    
    Input:
        Y (pandas.DataFrame): Primary variables of interest.
        X (pandas.DataFrame): Neuroimaging data to be harmonized.
        Z (pandas.DataFrame): Covariate data.
        samples (pandas.Series): Sample/batch identifiers.
        estimates (dict): Previously estimated batch effect parameters.
        preserve_cols (list): Columns from Z whose effect(s) should be preserved during harmonization.
    
    Returns:
        pandas.DataFrame: Harmonized neuroimaging data.
    """
    
    pheno = pd.concat([Y,Z[preserve_cols],samples], axis=1)

    harmonization = neuroCombatFromTraining(X.T, pheno['sample'], estimates)
    
    X_harmonized = pd.DataFrame(harmonization['data']).T
    X_harmonized.index = X.index 
    X_harmonized.columns = pd.MultiIndex.from_tuples(X.columns)

    return X_harmonized


def prep_predictors(fold, Y, X, Z, samples, preserve_cols, select_features, percentile):
    """
    Prepares predictor variables through feature selection, harmonization, and covariate regression.
    
    Input:
        fold (tuple): Tuple containing train indices and test indices.
        Y (pandas.DataFrame): Response variable.
        X (pandas.DataFrame): Neuroimaging variables.
        Z (pandas.DataFrame): Nuisance covariate data.
        samples (pandas.Series): Sample/batch identifiers.
        preserve_cols (list):  Columns from Z whose effect(s) should be preserved during harmonization.
        select_features (bool): Whether to perform feature selection.
        percentile (int): Percentile of top features to select if select_features is True.
    
    Returns:
        numpy.ndarray: Processed predictors after feature selection, harmonization, 
                      covariate regression, and standardization
    """
    
    idxtrain = fold[0]
    idxtest = fold[1]
    
    if select_features:
        if is_categorical(Y):
            score_func = f_classif
        else:
            score_func = f_regression
            
        selector = SelectPercentile(score_func=score_func, percentile=percentile)
        selector.fit(X.iloc[idxtrain,:], Y.iloc[idxtrain,:].values.ravel())
        X_selected = selector.transform(X)
        selected_mask = selector.get_support()
        selected_columns = X.columns[selected_mask]
        X_selected = pd.DataFrame(X_selected, columns=selected_columns, index=X.index)
    
    else:
        X_selected = X.copy()
    
    Xharm = X_selected.values.copy()

    Xharm[idxtrain,:], estimates = harmonize(Y.iloc[idxtrain,:], 
                                             X_selected.iloc[idxtrain,:], 
                                             Z.iloc[idxtrain,:], 
                                             samples.iloc[idxtrain],
                                             preserve_cols)
        
    Xharm[idxtest,:] = apply_harmonization(Y.iloc[idxtest,:], 
                                           X_selected.iloc[idxtest,:], 
                                           Z.iloc[idxtest,:], 
                                           samples.iloc[idxtest], 
                                           estimates, 
                                           preserve_cols)
    
    N = Xharm[idxtrain,:].shape[0]
    intercept = np.ones((N, 1))
    M = np.hstack((intercept, Z.iloc[idxtrain,:].values))

    predictors = np.zeros_like(Xharm)

    coeffs = np.linalg.lstsq(M, Xharm[idxtrain,:], rcond=None)[0]
    intercept = coeffs[0]
    b = coeffs[1:]
    
    predictors[idxtrain,:] = Xharm[idxtrain,:] - (Z.iloc[idxtrain,:].values @ b + intercept)
    predictors[idxtest,:] = Xharm[idxtest,:] - (Z.iloc[idxtest,:].values @ b + intercept)

    scaler_predictors = StandardScaler().fit(predictors[idxtrain,:])
    predictors[idxtrain,:] = scaler_predictors.transform(predictors[idxtrain,:])
    predictors[idxtest,:] = scaler_predictors.transform(predictors[idxtest,:])
    
    return predictors


def fit(alpha, fold, predictors, Y, seed):
    """
    Fit a predictive model (classification or regression) and compute a performance metric.
    
    Input:
        alpha (float): Regularization parameter (C for LogisticRegression, alpha for Ridge).
        fold (tuple): Tuple containing train indices and test indices.
        predictors (numpy.ndarray): Processed predictor variables.
        Y (pandas.DataFrame): Response variable.
        seed (int): Random seed for reproducibility.
    
    Returns:
        float: Performance metric:
           - For classification: Matthews correlation coefficient.
           - For regression: Correlation between actual and predicted responses.
    """
    
    idxtrain = fold[0]
    idxtest = fold[1]
    
    if is_categorical(Y):
        model = LogisticRegression(penalty='l2', 
                                   solver='lbfgs', 
                                   C=alpha, 
                                   max_iter=1000, 
                                   class_weight='balanced',
                                   random_state=seed)
        model.fit(predictors[idxtrain,:], Y.iloc[idxtrain,:].values.ravel())
        Y_pred = model.predict(predictors[idxtest,:])
        r = matthews_corrcoef(Y.iloc[idxtest, :].values.ravel(), Y_pred)
        return r
    
    else:
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(predictors[idxtrain,:], Y.iloc[idxtrain,:].values.ravel())
        Y_pred = model.predict(predictors[idxtest,:])
        r = corr(Y.iloc[idxtest,:].values.ravel(), Y_pred)
        return r
    
    
def compute_score(outer_fold, inner_folds, Y, X, Z, samples, preserve_cols, seed, select_features, percentile, n_jobs):
    """
    Performs nested cross-validation to tune model hyperparameters and evaluate performance.
    
    Parameters:
        outer_fold (tuple): Contains indices for training and test sets for final evaluation.
        inner_folds (list of tuples): List of tuples containing indices for inner cross-validation folds.
        Y (pandas.DataFrame): Response variable.
        X (pandas.DataFrame): Neuroimaging variables.
        Z (pandas.DataFrame): Nuisance covariate data.
        samples (pandas.Series): Sample/batch identifiers.
        preserve_cols (list):  Columns from Z whose effect(s) should be preserved during harmonization.
        seed (int): Random seed for reproducibility.
        select_features (bool): Whether to perform feature selection.
        percentile (int): Percentile of top features to select if select_features is True.
        n_jobs (int): Number of parallel jobs to run. Use –1 to utilize all available cores.
    
    Returns:
        float: Performance score from the outer fold using the best hyperparameter.
    """
    idxtrain = outer_fold[0]

    predictors = Parallel(n_jobs=n_jobs, verbose=True)(
        delayed(prep_predictors)(
            fold,
            Y.iloc[idxtrain, :],
            X.iloc[idxtrain, :],
            Z.iloc[idxtrain, :],
            samples.iloc[idxtrain],
            preserve_cols,
            select_features,
            percentile
        )
        for fold in inner_folds
    )

    alphas = [1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04]

    def _alpha_score(alpha, predictors):
        scores = Parallel(n_jobs=1)(
            delayed(fit)(
                alpha,
                inner_folds[i],
                predictors[i],
                Y.iloc[idxtrain, :],
                seed
            )
            for i in range(len(inner_folds))
        )
        return np.mean(scores)

    all_rs = Parallel(n_jobs=n_jobs, verbose=True)(
        delayed(_alpha_score)(alpha, predictors)
        for alpha in alphas
    )

    best_index = int(np.argmax(all_rs))
    best_alpha = alphas[best_index]

    pred_outer = prep_predictors(
        outer_fold,
        Y,
        X,
        Z,
        samples,
        preserve_cols,
        select_features,
        percentile
    )
    r = fit(best_alpha, outer_fold, pred_outer, Y, seed)
    
    return r


def run_ml(p, outer_folds, inner_folds, Y, X, Z, samples, preserve_cols, seed, select_features, percentile, n_jobs=-1):
    """
    Runs machine learning analysis with optional permutation testing.
    
    If p=0, uses original data; if p>0, creates permuted data for permutation testing
    while preserving relationships with covariates for continuous outcomes.
    
    Parameters:
        p (int): Permutation number (0 for no permutation, >0 for permutation test).
        outer_folds (list): List of tuples containing indices for outer cross-validation.
        inner_folds (list): List of lists of tuples containing indices for inner cross-validation.
        Y (pandas.DataFrame): Response variable.
        X (pandas.DataFrame): Neuroimaging variables.
        Z (pandas.DataFrame): Nuisance covariate data.
        samples (pandas.Series): Sample/batch identifiers.
        preserve_cols (list):  Columns from Z whose effect(s) should be preserved during harmonization.
        seed (int): Random seed for reproducibility.
        select_features (bool): Whether to perform feature selection.
        percentile (int): Percentile of top features to select if select_features is True.
    
    Returns:
        list: Performance scores from the outer folds.
    """
    
    if p == 0:
        Yshuf = Y.copy()
    else:
        np.random.seed(p)
        N = Y.shape[0]
        idy = np.random.permutation(N)
    
        if is_categorical(Y):
            P = np.eye(N)[idy]
            Yshuf = P @ Y.values
        else:
            Z_centered = Z.values - np.mean(Z.values, axis=0)
            Hz = Z_centered @ np.linalg.pinv(Z_centered)
            Rz = np.eye(N) - Hz
            Yshuf = freedman_lane(Y.values, Z_centered, idy, Rz=Rz, Hz=Hz)
        
        Yshuf = pd.DataFrame(Yshuf, columns=Y.columns, index=Y.index)

    rs = []
    for outer_fold, inner_fold in zip(outer_folds, inner_folds):
        r = compute_score(outer_fold, inner_fold, Yshuf, X, Z, samples, preserve_cols, seed, select_features, percentile, n_jobs)
        rs.append(r)
    
    return rs



