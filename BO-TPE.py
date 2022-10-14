import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn import datasets
import scipy.stats as stats

d = datasets.load_digits()
X = d.data
y = d.target

#SVM
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
def objective(params):
    params = {
        'C': abs(float(params['C'])), 
        "kernel":str(params['kernel'])
    }
    clf = SVC(gamma='scale', **params)
    score = cross_val_score(clf, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=3), n_jobs=-1).mean()

    return {'loss':-score, 'status': STATUS_OK }

space = {
    'C': hp.normal('C', 0.1, 50),
    "kernel":hp.choice('kernel',['linear','poly','rbf','sigmoid'])
}

n_iter_search=100

Ttotal = 0
Stotal = 0
Sdata = []
Tdata = []

for STEP in range(25):
    trials = Trials()
    t1 = time.process_time()
    best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=n_iter_search)
    t2 = time.process_time()
    T = t2 - t1

    Ttotal += T
    Stotal += min(trials.losses())

    Tdata.append(T)
    Sdata.append(min(trials.losses()))

    print(STEP)

Ttotal /= 25
Stotal /= 25
print("Avg S:", Stotal)
print("Avg T:", Ttotal)
print("Sdata:", Sdata)
print("Tdata:", Tdata)
