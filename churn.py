#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:38:31 2019

@author: nithin
@cmd   : python churn Churn_Modelling.csv RowNumber,CustomerId,Surname Geography,Gender Exited 200 RandomForestClassifier
"""

try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable
    
    
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn



def getinputs():
    # to get inputs from command arguments
    _dspath = sys.argv[1] if len(sys.argv) > 1 else 'Churn_Modelling.csv'
    _dropColumns = sys.argv[2].split(',') if len(sys.argv) > 2 else ['RowNumber', 'CustomerId', 'Surname']
    _encodeColumns = sys.argv[3].split(',') if len(sys.argv) > 3 else ['Geography', 'Gender']
    _outputColumns = sys.argv[4].split(',') if len(sys.argv) > 4 else ['Exited']
    _n_estimators = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    _algo = sys.argv[6] if len(sys.argv) > 6 else 'XGBClassifier' # DecisionTreeClassifier RandomForestClassifier XGBClassifier
    log_param("_dspath", _dspath)
    log_param("_dropColumns", _dropColumns)
    log_param("_encodeColumns", _encodeColumns)
    log_param("_outputColumns", _outputColumns)
    log_param("_n_estimators", _n_estimators)
    log_param("_algo", _algo)
    return _dspath, _dropColumns, _encodeColumns, _outputColumns, _n_estimators, _algo

def dropcolumns(customer_data, _dropColumns):
    # to drop columns from dataset
    customer_data.drop(_dropColumns, axis=1, inplace=True)

def encodecolumns(customer_data, _encodeColumns):
    # to encode string valued columns to numbers
    encoded = []
    for col in _encodeColumns:
        encoded.append(pd.get_dummies(customer_data[col]).iloc[:,1:])
    return encoded

def getXy(customer_data, _columns):
    # to separate inputs and outputs from the dataset
    X = customer_data.drop(_columns, axis=1)
    y = customer_data[_columns]
    return X,y

def buildclassifier(algo, n_estimators):
    # to build algorithm
    classifier = None
    if algo == 'RandomForestClassifier':
        classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    elif algo == 'DecisionTreeClassifier':
        classifier = tree.DecisionTreeClassifier()
    elif algo == 'XGBClassifier':
        params = {'learning_rate': 0.01, 'n_estimators': n_estimators, 'seed': 0, 'subsample': 1, 'colsample_bytree': 1,
                  'objective': 'binary:logistic', 'max_depth': 8, 'random_state' : 42}
        classifier = xgb.XGBClassifier(**params)
    return classifier

def eval_metrics(actual, pred):
    # to evaluate metrics
    print('----Classification Report---')
    print(classification_report(y_test,predictions))
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    print("Accuracy: %.1f%%" % (accuracy * 100.0))
    # log accuracy metric
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    
def plotcm(y_test, y_pred):
    # plotting the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    plt.savefig("cm.png", dpi = 200, bbox_inches = "tight")
    mlflow.log_artifact("cm.png")
    
def savemodel(classifier):
    # save the trained model
    mlflow.sklearn.log_model(classifier, "model")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    _dspath,_dropColumns,_encodeColumns,_outputColumns,_n_estimators,_algo = getinputs()
    customer_data = pd.read_csv(_dspath)
    dropcolumns(customer_data, _dropColumns)
    Geography, Gender = encodecolumns(customer_data, _encodeColumns)
    dropcolumns(customer_data, _encodeColumns)
    customer_data = pd.concat([customer_data,Geography,Gender], axis=1)
    X,y =  getXy(customer_data,_outputColumns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    classifier = buildclassifier(_algo,_n_estimators)
    classifier.fit(X_train, y_train.values.ravel())
    predictions = classifier.predict(X_test)
    eval_metrics(y_test, predictions)
    plotcm(y_test, predictions)
    savemodel(classifier)
#        run_id = mlflow.active_run().info.run_uuid
#        print("Run with id %s finished" % run_id)