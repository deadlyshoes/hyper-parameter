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
import math

#SVM
import time
from sklearn.model_selection import GridSearchCV

from utils import save, get_dataset

def run_GS(X, y, dataset_name, n_iter_search):
    C_qt = {'100': 20, '1000': 100, '10000': 400}

    rf_params = {
        "C": np.logspace(-5, 2, C_qt[str(n_iter_search)]),
        "gamma": 2 * np.logspace(-10, 1, n_iter_search // C_qt[str(n_iter_search)])
    }

    Ttotal = 0
    Stotal = 0
    Sdata = []
    Tdata = []

    for STEP in range(25):
        t1 = time.process_time()
        clf = SVC()
        grid = GridSearchCV(clf, rf_params, cv=3, scoring='accuracy', n_jobs=1)
        grid.fit(X, y)
        t2 = time.process_time()
        T = t2 - t1

        Stotal += grid.best_score_
        Ttotal += T

        Sdata.append(grid.best_score_)
        Tdata.append(T)

        print(STEP)

        save("GS_" + str(n_iter_search) + "_" + dataset_name, STEP, grid.best_score_, T)

    Ttotal /= 25
    Stotal /= 25

    print("Avg S", Stotal)
    print("Svg T:", Ttotal)
    print("Sdata:", Sdata)
    print("Tdata:", Tdata)

datasets = ["heart", "haberman", "breast"]
iter_values = [100, 1000, 10000]

for dataset in datasets:
    X, y = get_dataset(dataset)
    for n_iter in iter_values:
        run_GS(X, y, dataset, n_iter)