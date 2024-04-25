"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.5
"""

import logging
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def preprocess_data(data:pd.DataFrame, parameters:Dict) -> pd.DataFrame:
    
    data['municipio']=data['municipio'].str.replace(' ','_')
    data['bairro']=data['bairro'].str.replace(' ','_')
    data['bairro'] = data['bairro'].fillna('')
    data['municipio+bairro'] = data['municipio']+'_'+data['bairro']

    data['tier_por_porte'] = (data['tier']+1)/(data['porte']+1)
    
    data = data.drop(columns = ['municipio','bairro'])
    
    le = LabelEncoder()

    for col in ['uf','municipio+bairro']:
        data[col] = le.fit_transform(data[col])

  

    conditions = [
    (data[parameters['target']] <= 100),
    (data[parameters['target']] <= 1000),
    (data[parameters['target']] <= 10000),
    (data[parameters['target']] <= 100000),
    (data[parameters['target']] > 100000)
    ]

    values = [0, 1, 2, 3, 4]

    # Criar a coluna target_class com base nas condições e valores
    data['target_class'] = np.select(conditions, values)

    data = data.drop(columns = parameters['target'] )
    return data