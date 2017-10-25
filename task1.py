import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]

filename = "The SUM dataset, without noise.csv"
sum_nonoise = pd.read_csv(filename, delimiter=";").drop(["Instance"],axis = 1)

sum_nonoise.loc[sum_nonoise["Target Class"] == "Very Large Number", "Target Class"] = 5
sum_nonoise.loc[sum_nonoise["Target Class"] == "Large Number", "Target Class"] = 4
sum_nonoise.loc[sum_nonoise["Target Class"] == "Medium Number", "Target Class"] = 3
sum_nonoise.loc[sum_nonoise["Target Class"] == "Small Number", "Target Class"] = 2
sum_nonoise.loc[sum_nonoise["Target Class"] == "Very Small Number", "Target Class"] = 1

# predict y value, x -> independent values
#

for size_index in range(len(sizes)):
    X = np.array(sum_nonoise[:sizes[size_index]])
    X = X[:,:-1]

    Y = np.array(sum_nonoise[:sizes[size_index]])
    Y = Y[:,-2]

    X = X.astype('int')
    Y = Y.astype('int')

    regr = LinearRegression()
    kf = KFold(n_splits=10)
    abs_error = cross_val_score(regr, X, Y, cv = kf, scoring = 'neg_mean_absolute_error')
    mean_score = abs_error.mean()
    print("Mean Absolute Error: ", -1 * mean_score)
    sq_error= cross_val_score(regr, X, Y, cv=kf, scoring='neg_mean_squared_error')
    mean_sqerror = -1 * sq_error.mean()
    print("rmse: ", np.sqrt(mean_sqerror))

