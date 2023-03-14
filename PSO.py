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

import optunity
import optunity.metrics

def run_PSO(X, y, save_suffix, n_iter_search):
    data=X
    labels=y.tolist()

    search = {
        'C': [0.0000001,100],
        'gamma':[0,20]
             }
    @optunity.cross_validated(x=data, y=labels, num_folds=3)
    def performance(x_train, y_train, x_test, y_test,C=None,gamma=None):
        # fit the model
        model = SVC(C=float(C),
                    gamma=float(gamma)
                                      )
        #predictions = model.predict(x_test)
        scores=np.mean(cross_val_score(model, X, y, cv=3, n_jobs=1,
                                        scoring="accuracy"))
        #return optunity.metrics.roc_auc(y_test, predictions, positive=True)
        return scores#optunity.metrics.accuracy(y_test, predictions)

    Ttotal = 0
    Stotal = 0
    Sdata = []
    Tdata = []

    for STEP in range(25):
        print("on step:", STEP)
        t1 = time.process_time()
        optimal_configuration, info, _ = optunity.maximize(performance,
                                                      solver_name='particle swarm',
                                                      num_evals=n_iter_search,
                                                       **search
                                                      )
        t2 = time.process_time()
        T = t2 - t1

        Ttotal += T
        Stotal += info.optimum

        Tdata.append(T)
        Sdata.append(info.optimum)

        save("PSO_" + str(n_iter_search) + "_" + save_suffix, STEP, info.optimum, T)

    Ttotal /= 25
    Stotal /= 25
    print("Avg S:", Stotal)
    print("Avg T:", Ttotal)
    print("Sdata:", Sdata)
    print("Tdata:", Tdata)

X, y = get_dataset("haberman")

run_PSO(X, y, "haberman", 100)
run_PSO(X, y, "haberman", 1000)
run_PSO(X, y, "haberman", 10000)

X, y = get_dataset("heart")

run_PSO(X, y, "heart", 100)
run_PSO(X, y, "heart", 1000)
run_PSO(X, y, "heart", 10000)

X, y = get_dataset("breast")

run_PSO(X, y, "breast", 100)
run_PSO(X, y, "breast", 1000)
run_PSO(X, y, "breast", 10000)