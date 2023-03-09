import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import json

def interval_string_to_average_int(s):
    interval = [float(x) for x in s.split('-')]
    return (interval[0] + interval[1]) / 2.0

def get_dataset(dataset_name):
    X = None
    y = None

    if dataset_name == "heart":
        X_ls = []
        y_ls = []
        with open("datasets/heart.dat", "r") as statlog_dat_file:
            lines = statlog_dat_file.readlines()
            for line in lines:
                row_data = [float(x) for x in line.split()]
                X_ls.append(row_data[:-1])
                y_ls.append(row_data[-1])
        X = np.array(X_ls)
        transformer = MinMaxScaler()
        transformer.fit(X)
        X = transformer.transform(X)
        y = np.array(y_ls) - 1
    elif dataset_name == "breast":
        X_ls = []
        y_ls = []
        with open("datasets/breast-cancer.dat", "r") as breast_dat_file:
            for line in breast_dat_file.readlines():
                row_data = [x for x in line.split(',')]
                print(row_data)
                # 1. Class: no-recurrence-events, recurrence-events
                x = []
                if row_data[0] == "no-recurrence-events":
                    x.append(0)
                else:
                    x.append(1)
                # 2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
                x.append(interval_string_to_average_int(row_data[1]))
                # 3. menopause: lt40, ge40, premeno.
                if row_data[2] == "lt40":
                    x.append(0)
                elif row_data[2] == "ge40":
                    x.append(1)
                else:
                    x.append(2)
                # 4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59.
                x.append(interval_string_to_average_int(row_data[3]))
                # 5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39.
                x.append(interval_string_to_average_int(row_data[4]))
                # 6. node-caps: yes, no.
                if row_data[5] == "no":
                    x.append(0)
                else:
                    x.append(1)
                # 7. deg-malig: 1, 2, 3.
                x.append(float(row_data[6]))
                # 8. breast: left, right.
                if row_data[7] == "left":
                    x.append(0)
                else:
                    x.append(1)
                # 9. breast-quad: left-up, left-low, right-up, right-low, central.
                if row_data[8] == "left_up":
                    x.append(0)
                elif row_data[8] == "left_low":
                    x.append(1)
                elif row_data[8] == "right_up":
                    x.append(2)
                elif row_data[8] == "right_low":
                    x.append(3)
                else:
                    x.append(4)
                # 10. irradiat: yes, no.
                if row_data[9][0] == 'n':
                    y_ls.append(0)
                else:
                    y_ls.append(1)
                X_ls.append(x)
        X = np.array(X_ls)
        transformer = MinMaxScaler()
        transformer.fit(X)
        X = transformer.transform(X)
        y = np.array(y_ls)
    elif dataset_name == "haberman":
        X_ls = []
        y_ls = []
        with open("datasets/haberman.dat", "r") as iris_dat_file:
            for line in iris_dat_file.readlines():
                row_data = [x for x in line.split(',')]
                X_ls.append([float(x) for x in row_data[:-1]])
                if row_data[-1] == "1\n":
                    y_ls.append(0)
                else:
                    y_ls.append(1)
        X = np.array(X_ls)
        transformer = MinMaxScaler()
        transformer.fit(X)
        X = transformer.transform(X)
        y = np.array(y_ls)
    else:
        print("Unknown dataset")

    return X, y

def save(save_dir, i, S, T):
    Path("results/" + save_dir).mkdir(parents=True, exist_ok=True)
    with open("results/" + save_dir + "/S" + str(i), "w") as S_file:
        S_file.write(json.dumps(S))
    with open("results/" + save_dir + "/T" + str(i), "w") as T_file:
        T_file.write(json.dumps(T))