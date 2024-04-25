"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.5
"""
import logging
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from skopt import forest_minimize


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:

    X = data.drop(columns = 'target_class')
    y = data['target_class']

    stratified = parameters['stratify']

    if stratified == True:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=parameters["val_frac"], random_state=42,stratify=y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=parameters["val_frac"], random_state=42)
    
    return  X_train, X_val, y_train, y_val


def optimize_model(X_train: pd.DataFrame, X_val:pd.DataFrame, y_train:pd.Series, y_val:pd.Series, parameters:Dict
) -> List:


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
                (0,5)]#gammal]
    

    res = forest_minimize(tune_lgbm, space, random_state=42, n_random_starts=5, n_calls=10, verbose=0)
    best_params = res.x

    return best_params