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

from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

from utils import get_dataset, save

def run_RS(X, y, save_suffix, n_iter_search):
    rf_params = {
        'C': stats.uniform(0.0000001, 100),
        "gamma": stats.uniform(0, 20)
    }

    Ttotal = 0
    Stotal = 0
    Sdata = []
    Tdata = []
    Pdata = []

    for STEP in range(25):
        t1 = time.time()
        clf = SVC()
        Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='accuracy',n_jobs=1)
        Random.fit(X, y)
        t2 = time.time()
        T = t2 - t1

        Ttotal += T
        Stotal += Random.best_score_

        Tdata.append(T)
        Sdata.append(Random.best_score_)
        Pdata.append([Random.best_estimator_.C, Random.best_estimator_.gamma])

        print(STEP)

        save("RS_" + str(n_iter_search) + "_" + save_suffix, STEP, Random.best_score_, [Random.best_estimator_.C, Random.best_estimator_.gamma], T)

    Ttotal /= 25
    Stotal /= 25
    print("Avg S:", Stotal)
    print("Avg T:", Ttotal)
    print("Sdata:", Sdata)
    print("Tdata:", Tdata)
    print("P:", Pdata)

datasets = ["heart", "haberman", "breast"]
iter_values = [100, 1000, 10000]

for dataset in datasets:
    X, y = get_dataset(dataset)
    for n_iter in iter_values:
        run_RS(X, y, dataset, n_iter)