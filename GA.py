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


from evolutionary_search import EvolutionaryAlgorithmSearchCV
rf_params = {
    'C': np.random.uniform(0.1,50,1000),
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
	ga1 = EvolutionaryAlgorithmSearchCV(estimator=clf,
                                   params=rf_params,
                                   scoring="accuracy",
                                   cv=3,
                                   verbose=0,
                                   population_size=10,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=14,
                                   n_jobs=-1)
ga1.fit(X, y)
    t2 = time.process_time()
    T = t2 - t1

    Ttotal += T
    Stotal += ga1.best_score_

    Tdata.append(T)
    Sdata.append(ga1.best_score_)

    print(STEP)

Ttotal /= 25
Stotal /= 25
print("Avg S:", Stotal)
print("Avg T:", Ttotal)
print("Sdata:", Sdata)
print("Tdata:", Tdata)
