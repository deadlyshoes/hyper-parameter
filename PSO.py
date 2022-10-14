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


import optunity
import optunity.metrics

data=X
labels=y.tolist()

search = {
    'C': (0.1,50),
    'kernel':[0,4]
         }
@optunity.cross_validated(x=data, y=labels, num_folds=3)
def performance(x_train, y_train, x_test, y_test,C=None,kernel=None):
    # fit the model
    if kernel<1:
        ke='linear'
    elif kernel<2:
        ke='poly'
    elif kernel<3:
        ke='rbf'
    else:
        ke='sigmoid'
    model = SVC(C=float(C),
                kernel=ke
                                  )
    #predictions = model.predict(x_test)
    scores=np.mean(cross_val_score(model, X, y, cv=3, n_jobs=-1,
                                    scoring="accuracy"))
    #return optunity.metrics.roc_auc(y_test, predictions, positive=True)
    return scores#optunity.metrics.accuracy(y_test, predictions)

n_iter_search=100

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

Ttotal /= 25
Stotal /= 25
print("Avg S:", Stotal)
print("Avg T:", Ttotal)
print("Sdata:", Sdata)
print("Tdata:", Tdata)
