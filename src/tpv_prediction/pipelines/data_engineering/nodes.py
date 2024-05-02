"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.5
"""
from typing import Dict, Tuple

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(data: pd.DataFrame, parameters: Dict):
    """
    Preprocesses the input DataFrame `data` according to the
    specified `parameters`.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to
        be processed.
        parameters (Dict): A dictionary containing the parameters
        used for data processing.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    data["municipio"] = data["municipio"].str.replace(" ", "_")
    data["bairro"] = data["bairro"].str.replace(" ", "_")
    data["bairro"] = data["bairro"].fillna("")
    data["municipio+bairro"] = data["municipio"] + "_" + data["bairro"]

    data["tier_por_porte"] = (data["tier"] + 1) / (data["porte"] + 1)

    data = data.drop(columns=["municipio", "bairro"])

    le = LabelEncoder()

    for col in ["uf", "municipio+bairro"]:
        data[col] = le.fit_transform(data[col])

    conditions = [
        (data[parameters["target"]] <= 100),
        (data[parameters["target"]] <= 1000),
        (data[parameters["target"]] <= 10000),
        (data[parameters["target"]] <= 100000),
        (data[parameters["target"]] > 100000),
    ]

    values = [0, 1, 2, 3, 4]

    # Criar a coluna target_class com base nas condições e valores

    data["target_class"] = np.select(conditions, values)
    data = data.drop(columns=parameters["target"])
    data[parameters["categorical_list"]] = data[parameters[
        "categorical_list"]].astype(
        "category"
    )

    data["target_class"] = data["target_class"].astype("category")

    return data, le


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    Splits the input DataFrame `data` into training and validation sets.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to
        be split.
        parameters (Dict): A dictionary containing the parameters used
        for splitting.

    Returns:
        Tuple: A tuple containing X_train, X_val, y_train, and y_val.
    """

    X = data.drop(columns="target_class")
    y = data["target_class"]

    stratified = parameters["stratify"]

    if stratified is True:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=parameters["val_frac"], random_state=42,
            stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=parameters["val_frac"], random_state=42
        )

    return X_train, X_val, y_train, y_val
