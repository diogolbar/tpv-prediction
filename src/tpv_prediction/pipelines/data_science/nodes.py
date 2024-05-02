"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.5
"""
import logging
from typing import List, Dict

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier

from sklearn.metrics import f1_score

from skopt import forest_minimize


def optimize_model(X_train: pd.DataFrame, X_val:pd.DataFrame, y_train:pd.Series, y_val:pd.Series, parameters: Dict
) -> List:
    """
    Optimizes hyperparameters for a LightGBM model using Bayesian optimization.

    Args:
        X_train (pd.DataFrame): The input DataFrame containing the features for training.
        X_val (pd.DataFrame): The input DataFrame containing the features for validation.
        y_train (pd.Series): The input Series containing the target variable for training.
        y_val (pd.Series): The input Series containing the target variable for validation.
        parameters (Dict): A dictionary containing the parameters used for optimization.

    Returns:
        List: A list containing the best hyperparameters found by the optimization process.
    """
    def tune_lgbm(params):
    
        lr = params[0]
        max_depth = params[1]
        min_child_samples = params[2]
        subsample = params[3]
        colsample_bytree = params[4]
        n_estimators = params[5]
        reg_alpha = params[6]
        reg_lambda = params[7]
        gamma = params[8]
        
        
        model = LGBMClassifier(learning_rate=lr, num_leaves=2 ** max_depth, max_depth=max_depth, 
                            min_child_samples=min_child_samples, subsample=subsample,
                            colsample_bytree=colsample_bytree, bagging_freq=1,n_estimators=n_estimators, 
                            reg_alpha =reg_alpha, reg_lambda=reg_lambda, gamma=gamma,
                            random_state=0,class_weight="balanced", objective='multiclass',verbose=-1)
        
        model.fit(X_train, y_train.astype(int))
        
        y_pred = model.predict(X_val)
        
        f1_weighted = f1_score(y_val.astype(int), y_pred, average='weighted')

        return -f1_weighted

    space = [   (1e-3, 1e-1, 'log-uniform'), # lr
                (1, 10), # max_depth
                (1, 20), # min_child_samples
                (0.05, 1.), # subsample
                (0.05, 1.), # colsample_bytree
                (100,1000),# n_estimator
                (1e-6,1, 'log-uniform'), #reg_alpha
                (1e-6,1, 'log-uniform'),#reg_lambda
                (0,5)] #gamma
    

    res = forest_minimize(tune_lgbm, space, random_state=42, n_random_starts=parameters['random_starts'], n_calls=parameters['calls'], verbose=0, base_estimator=parameters['estimator'])
    best_params = res.x

    return best_params


def train_model(X_train: pd.DataFrame, y_train: pd.Series, hyperparameters: List) -> LGBMClassifier:
    """
    Trains a LightGBM model on the input training data.

    Args:
        X_train (pd.DataFrame): The input DataFrame containing the features for training.
        y_train (pd.Series): The input Series containing the target variable for training.
        hyperparameters (List): A list of hyperparameters for the LightGBM model.

    Returns:
        LGBMClassifier: The trained LightGBM classifier.
    """
    
    y_train = y_train.astype('category')

    best_params = hyperparameters

    lr = best_params[0]
    max_depth = best_params[1]
    min_child_samples = best_params[2]
    subsample = best_params[3]
    colsample_bytree = best_params[4]
    n_estimators = best_params[5]
    reg_alpha = best_params[6]
    reg_lambda = best_params[7]
    gamma = best_params[8]

    classifier = LGBMClassifier(learning_rate=lr, num_leaves=2 ** max_depth, max_depth=max_depth, 
                            min_child_samples=min_child_samples, subsample=subsample,
                            colsample_bytree=colsample_bytree, bagging_freq=1,n_estimators=n_estimators, 
                            reg_alpha =reg_alpha, reg_lambda=reg_lambda, gamma=gamma,
                            random_state=0,class_weight="balanced", objective='multiclass',verbose=-1)

    classifier.fit(X_train, y_train)

    return classifier