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

from utils import get_dataset, save

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold

def run_BOTPE(X, y, save_suffix, n_iter_search):
    def objective(params):
        params = {
            'C': abs(float(params['C'])), 
            "gamma":float(params['gamma'])
        }
        clf = SVC(**params)
        score = cross_val_score(clf, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=3), n_jobs=1).mean()

        return {'loss':-score, 'status': STATUS_OK }

    space = {
        'C': hp.uniform('C', 0.0000001, 100),
        "gamma": hp.uniform('gamma', 0, 20),
    }

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

        save("BOTPE_" + str(n_iter_search) + "_" + save_suffix, STEP, min(trials.losses()), T)

    Ttotal /= 25
    Stotal /= 25
    print("Avg S:", Stotal)
    print("Avg T:", Ttotal)
    print("Sdata:", Sdata)
    print("Tdata:", Tdata)

datasets = ["heart", "haberman", "breast"]
iter_values = [100, 1000, 10000]

for dataset in datasets:
    X, y = get_dataset(dataset)
    for n_iter in iter_values:
        run_BOTPE(X, y, dataset, n_iter)