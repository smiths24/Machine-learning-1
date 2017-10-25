import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


dataset1 = pd.read_csv("kc_house_data.csv").drop(["date"],axis = 1)

dataset2 = pd.read_csv("train.csv").drop(["pickup_datetime"], axis = 1)
dataset2 = dataset2.drop(["dropoff_datetime"],axis = 1)
dataset2 = dataset2.drop(["id"], axis = 1)
dataset2 = dataset2.drop(["store_and_fwd_flag"], axis = 1)


X1 = np.array(dataset1)
X1 = X1[:,1:]
Y1 = np.array(dataset1)
Y1 = Y1[:,0]
#X1 = X1.astype('long')
#Y1 = Y1.astype('long')

X2 = np.array(dataset2)
X2 = X2[:,1:]
Y2 = np.array(dataset2)
Y2 = Y2[:,0]
#X2 = X2.astype('long')
#Y2 = Y2.astype('long')

kf = KFold(n_splits=10)

lin_reg = LinearRegression()
log_reg = LogisticRegression()
ridgeReg = Ridge()
ensembleReg = GradientBoostingRegressor()
sgdReg = SGDRegressor()
algos = np.array([lin_reg,log_reg,ridgeReg,ensembleReg,sgdReg])

for i in range(0,algos.size-1):
    print("DATASET:",i)
    abs_error = cross_val_score(algos[i], X1, Y1, cv = kf, scoring ='neg_mean_absolute_error')
    mean_score = abs_error.mean()
    print("DATASET 1 mean absolute error:")
    print(-1 * mean_score)

    abs_error = cross_val_score(algos[i], X2, Y2, cv = kf, scoring ='neg_mean_absolute_error')
    mean_score = abs_error.mean()
    print("DATASET2 mean absolute error:")
    print(-1 * mean_score)

    sq_error = cross_val_score(algos[i], X1, Y1, cv=kf, scoring='neg_mean_squared_error')
    mean_sqerror = -1 * sq_error.mean()
    print("DATASET2 mean squared error")
    print(np.sqrt(mean_sqerror))

    sq_error = cross_val_score(algos[i], X2, Y2, cv=kf, scoring='neg_mean_squared_error')
    mean_sqerror = -1 * sq_error.mean()
    print("DATASET1 mean squared error")
    print(np.sqrt(mean_sqerror))

    accuracy = cross_val_score(algos[i], X1, Y1, cv=kf, scoring='accuracy')
    print("DATASET1 accuracy: ",accuracy)
    accuracy = cross_val_score(algos[i], X1, Y1, cv=kf, scoring='accuracy')
    print("DATASET2 accuracy: ",accuracy)

    r2 = cross_val_score(algos[i], X1, Y1, cv=kf, scoring='r2')
    print("DATASET1 r2: ",r2)
    r2 = cross_val_score(algos[i], X1, Y1, cv=kf, scoring='r2')
    print("DATASET2 r2: ",r2)

    expl_var = cross_val_score(algos[i], X1, Y1, cv=kf, scoring='explained_variance')
    print("DATASET1 explained var: ",expl_var)
    expl_var = cross_val_score(algos[i], X1, Y1, cv=kf, scoring='explained_variance')
    print("DATASET2 explained var: ",expl_var)





