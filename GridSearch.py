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

#SVM
import time
from sklearn.model_selection import GridSearchCV

C_points = []
for i in range(1, 101):
    C_points.append(i / 2)

rf_params = {
    'C': C_points,
    "kernel":['linear','poly','rbf','sigmoid']
}

Ttotal = 0
Stotal = 0
Sdata = []
Tdata = []

for _ in range(25):
    t1 = time.process_time()
    clf = SVC(gamma='scale')
    grid = GridSearchCV(clf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    t2 = time.process_time()
    T = t2 - t1

    Stotal += grid.best_score_
    Ttotal += T

    Sdata.append(grid.best_score_)
    Tdata.append(T)

Ttotal /= 25
Stotal /= 25

print("Avg S", Stotal)
print("Svg T:", Ttotal)
print("Sdata:", Sdata)
print("Tdata:", Tdata)
