import numpy as np
import pandas as pd
from sklearn import metrics, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]

filename = "The SUM without noise.csv"
sum_nonoise = pd.read_csv(filename, delimiter=";").drop(["Instance"],axis = 1)

sum_nonoise.loc[sum_nonoise["Target Class"] == "Very Large Number", "Target Class"] = 5
sum_nonoise.loc[sum_nonoise["Target Class"] == "Large Number", "Target Class"] = 4
sum_nonoise.loc[sum_nonoise["Target Class"] == "Medium Number", "Target Class"] = 3
sum_nonoise.loc[sum_nonoise["Target Class"] == "Small Number", "Target Class"] = 2
sum_nonoise.loc[sum_nonoise["Target Class"] == "Very Small Number", "Target Class"] = 1

# predict y value
# x - independent values

for size_index in range(len(sizes)):
    X = np.array(sum_nonoise[:sizes[size_index]])
    X = X[:,:-1]

    Y = np.array(sum_nonoise[:sizes[size_index]])
    Y = Y[:,-2]

    X = X.astype('int')
    Y = Y.astype('int')

    kf = KFold(n_splits=10)

    lin_reg = LinearRegression()

    abs_error = cross_val_score(lin_reg, X, Y, cv = kf, scoring ='neg_mean_absolute_error')
    mean_score = abs_error.mean()
    print("mean absolute error", sizes[size_index])
    print(-1 * mean_score)

    sq_error = cross_val_score(lin_reg, X, Y, cv=kf, scoring='neg_mean_squared_error')
    mean_sqerror = -1 * sq_error.mean()
    print("mean squared error", sizes[size_index])
    print(np.sqrt(mean_sqerror))

    # Reset y value for Logistic Regression and Classification
    Y = np.array(sum_nonoise[:sizes[size_index]])
    Y = Y[:, -1]
    Y = Y.astype('int')

    log_reg = LogisticRegression()
    accuracy = cross_val_score(log_reg, X, Y, cv=kf, scoring='accuracy')
    # accuracy = metrics.accuracy_score(Y, predicted)
    print("accuracy", sizes[size_index])
    print(accuracy)

    knn = neighbors.KNeighborsClassifier()
    k_nn = cross_val_score(knn, X, Y, cv=kf, scoring='accuracy')
    print("k nearest neighbours", sizes[size_index])
    print(k_nn)

