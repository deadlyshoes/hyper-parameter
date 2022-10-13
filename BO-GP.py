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
from skopt import Optimizer
from skopt import BayesSearchCV 
from skopt.space import Real, Categorical, Integer
rf_params = {
    'C': Real(0.1,50),
    "kernel":['linear','poly','rbf','sigmoid']
}

n_iter_search=100

Ttotal = 0
Stotal = 0
Sdata = []
Tdata = []

for STEP in range(25):
    t1 = time.process_time()
	clf = SVC(gamma='scale')
	Bayes = BayesSearchCV(clf, rf_params,cv=3,n_iter=n_iter_search, n_jobs=-1,scoring='accuracy')
	Bayes.fit(X, y)
    t2 = time.process_time()
    T = t2 - t1

    Ttotal += T
    Stotal += Bayes.best_score_

    Tdata.append(T)
    Sdata.append(Bayes.best_score_)

    print(STEP)

Ttotal /= 25
Stotal /= 25
print("Avg S:", Stotal)
print("Avg T:", Ttotal)
print("Sdata:", Sdata)
print("Tdata:", Tdata)
