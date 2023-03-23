import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import (get_scorer, make_scorer, f1_score, roc_auc_score, accuracy_score,
                             recall_score,
                             confusion_matrix, ConfusionMatrixDisplay)

import matplotlib.pyplot as plt

#
# definisco le funzioni che identificano le categorie di colonne
#
def cat_cols_selector(df, target_name):
    # the input is the dataframe
    
    # cols with less than THR values are considered categoricals
    THR = 10
    
    nunique = df.nunique()
    types = df.dtypes
    
    col_list = []
    
    for col in df.columns:
        if ((types[col] == 'object') or (nunique[col] < THR)):
            # print(col)
            if col != target_name:
                col_list.append(col)
    
    return col_list

def num_cols_selector(df, target_name):
    THR = 10
    
    types = df.dtypes
    nunique = df.nunique()
    
    col_list = []
    
    for col in df.columns:
        if (types[col] != 'object') and (nunique[col] >= THR): 
            # print(col)
            if col != target_name:
                col_list.append(col)
    
    return col_list

def plot_cm(model, x_test, y_test):
    y_pred_labels = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot();
    
def evaluate_metrics(model, X, y):
    test_pred = model.predict(X)
    test_probas = model.predict_proba(X)

    print('Validation set result:')

    roc_auc = round(roc_auc_score(y, test_probas[:,1]), 4)
    acc = round(accuracy_score(y, test_pred), 4)
    recall = round(recall_score(y, test_pred, pos_label="Yes"), 4)

    # this is the Object that will be saved in the Model Catalog
    metrics = {
        "accuracy" : acc,
        "roc_auc" : roc_auc,
        "recall" : recall
    }

    print(str(metrics))
    
    return metrics