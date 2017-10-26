import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import time
import csv


def evaluate(X, Y, kf):
    linReg = LinearRegression()
    svmReg = svm.SVR()
    decTreeReg = tree.DecisionTreeRegressor()
    sgdReg = SGDRegressor()
    algos = np.array([linReg, svmReg, decTreeReg, sgdReg])
    aString = ["Linear Regression", "SVM Regression", "Decision Tree Regression", "SGD Regression"]
    all_results = []
    for i in range(0, algos.size):
        algo_result = []
        algo_result.append(aString[i])
        start = time.clock()
        print("ALGORITHM:", str(algos[i]))
        abs_error = cross_val_score(algos[i], X, Y, cv = kf, scoring ='neg_mean_absolute_error')
        mean_score = abs_error.mean()
        mean_score = -1 * mean_score
        print("mean absolute error:", mean_score)
        algo_result.append(mean_score)

        sq_error = cross_val_score(algos[i], X, Y, cv=kf, scoring='neg_mean_squared_error')
        mean_sqerror = -1 * sq_error.mean()
        mean_sqerror = np.sqrt(mean_sqerror)
        print("mean squared error: ", mean_sqerror)
        algo_result.append(mean_sqerror)

        med_abs_error = cross_val_score(algos[i], X, Y, cv=kf, scoring='neg_median_absolute_error')
        med_abs_error = -1 * med_abs_error.mean()
        print("median absolute error: ", med_abs_error)
        algo_result.append(med_abs_error)

        r2 = cross_val_score(algos[i], X, Y, cv=kf, scoring='r2')
        r2 = r2.mean()
        print("r2: ",r2)
        algo_result.append(r2)

        expl_var = cross_val_score(algos[i], X, Y, cv=kf, scoring='explained_variance')
        expl_var = expl_var.mean()
        print("explained var: ",expl_var)
        algo_result.append(expl_var)
        t = time.clock() - start
        print("Time:", t)
        algo_result.append(t)
        all_results.append(algo_result)
    return all_results;


def printResultsToCsv(all_results):
    with open("results.csv", 'w') as csvfile:
        resultswriter = csv.writer(csvfile, delimiter=",")
        for row in all_results:
            resultswriter.writerow(row)
    return;

all_results = []
first_row = ["", "mean absolute error", "mean squared error", "median absolute error", "r2", "explained variance", "runtime"]
all_results.append(first_row)

dataset1 = pd.read_csv("winequality-red.csv", delimiter=";")
dataset2 = pd.read_csv("winequality-white.csv", delimiter=";")

X1 = np.array(dataset1)
X1 = X1[:,:-1]
Y1 = np.array(dataset1)
Y1 = Y1[:,-1]
X1 = X1.astype('int')
Y1 = Y1.astype('int')

X2 = np.array(dataset2)
X2 = X2[:,:-1]
Y2 = np.array(dataset2)
Y2 = Y2[:,-1]
X2 = X2.astype('int')
Y2 = Y2.astype('int')

kf = KFold(n_splits=10)

print("EVALUATING DATASET1")
results1 = evaluate(X1,Y1,kf)
all_results.append(results1)
printResultsToCsv(all_results)

print("EVALUATING DATASET2")
all_results = []
all_results.append(first_row)
results2 = evaluate(X2,Y2,kf)
all_results.append(results2)

printResultsToCsv(all_results)
