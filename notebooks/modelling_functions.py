import matplotlib.pyplot as plt

import pandas as pd

import shap

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import LearningCurveDisplay
from sklearn.model_selection import train_test_split

from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self,X):
        return self.model.predict(X)
    
    def predict_proba(self,X,):
        return self.model.predict_proba(X)

    def check_predict(self, X, y, multi = False, multi_class = 'ovr'):
        if hasattr(self.model, 'predict_proba'):  # Classification model
            p = self.model.predict_proba(X)
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            if multi == False:
                roc_auc = roc_auc_score(y, p, average='weighted')
            else:
                roc_auc = roc_auc_score(y, p, multi_class=multi_class, average='weighted')
            
            return {'Accuracy' : accuracy, 'Precision' : precision, 'Recall' : recall, 'F1 Score' : f1,'ROC-AUC' : roc_auc}
        else:  # Regression model
            p = self.model.predict(X)
            r2 = r2_score(y, p)
            MAE = mean_absolute_error(y, p)
            RMSE = mean_squared_error(y, p, squared=False)
            return {'r2_score' : r2,'RMSE' : RMSE, 'MAE' : MAE, 'MAE/Mean(%)' : round(100*MAE/y.mean(),2), 'RMSE/STD (%)' : round(100*RMSE/y.std(),2)}

    def interpret(self, X):
        
        try:
            explainer = shap.TreeExplainer(self.model)
        except:
            print("Model type not supported for interpretation")
            return
        
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X)
        plt.show()

    def learning_check(self, X, y):
        train_sizes, train_scores, test_scores = learning_curve(self.model, X, y, verbose=0)
        display = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores, score_name="Score")
        display.plot()
        plt.show()



def split_data(df, target_column, train_frac = 0.7, val_frac = 0.5, stratified=False):
    train = df.sample(frac=train_frac, random_state=42)
    test = df.drop(train.index)
    
    X = train.drop(columns=target_column)
    y = train[target_column]
    
    if stratified == True:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=val_frac, random_state=42,stratify=y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=val_frac, random_state=42)
    X_test = test.drop(columns=target_column)
    y_test = test[target_column]
    
    return X, X_train, X_val, X_test, y, y_train, y_val, y_test

def validation(X,y,model, name,multi=False):
    return pd.DataFrame(model.check_predict(X,y,multi), index=[name])

def compare_results(val,test):
    return pd.concat([val,test])