"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.5
"""
import logging
from typing import Dict, Tuple, List

import pandas as pd

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    Splits the input DataFrame `data` into training and validation sets.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to be split.
        parameters (Dict): A dictionary containing the parameters used for splitting.

    Returns:
        Tuple: A tuple containing X_train, X_val, y_train, and y_val.
    """
    
    X = data.drop(columns = 'target_class')
    y = data['target_class']

    stratified = parameters['stratify']

    if stratified == True:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=parameters["val_frac"], random_state=42,stratify=y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=parameters["val_frac"], random_state=42)
    
    return  X_train, X_val, y_train, y_val


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



def evaluate_model(
    classifier: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluates the trained model on the test data and returns evaluation metrics.

    Args:
        classifier (LGBMClassifier): The trained LightGBM classifier.
        X_test (pd.DataFrame): The input DataFrame containing the features for testing.
        y_test (pd.Series): The input Series containing the target variable for testing.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics (F1 score, precision, recall, ROC AUC).
    """
    
    y_pred = classifier.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, classifier.predict_proba(X_test), multi_class='ovr')
    
    logger = logging.getLogger(__name__)
    logger.info("Model has a F1 score of %.3f, precision of %.3f, recall of %.3f, and ROC AUC of %.3f on test data.", f1, precision, recall, roc_auc)
    return {"f1_score": f1, "precision": precision, "recall": recall, "roc_auc": roc_auc}
