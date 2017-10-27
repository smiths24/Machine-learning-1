import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Normalise results list
def normalise_results(result_list):
    for item in range(len(result_list)):
        norm = np.linalg.norm(result_list)
        n = result_list[item] / norm
        result_list[item] = n
    return result_list


def evaluate(X, Y, kf, ds,set):
    lin_reg = LinearRegression()

    # Linear Regression
    lin_reg = LinearRegression()
    # - Mean Absolute Error
    lin_abs_error = cross_val_score(lin_reg, X, Y, cv=kf, scoring='neg_mean_absolute_error')
    lin_abs_error_norm = normalise_results(lin_abs_error)  # normalise results list
    mean_lin_abs_error = -1 * lin_abs_error_norm.mean()  # Multiply by -1 and find mean
    print(sizes[size_index], "lin mean absolute error: ", mean_lin_abs_error)

    # - Mean Squared Error
    lin_sq_error = cross_val_score(lin_reg, X, Y, cv=kf, scoring='neg_mean_squared_error')
    lin_sq_error_norm = normalise_results(lin_sq_error)  # normalise results list
    mean_lin_sq_error = -1 * lin_sq_error_norm.mean()  # Multiply by -1 and find mean
    print(sizes[size_index], "lin mean squared error: ", np.sqrt(mean_lin_sq_error))

    # Ridge Regression
    ridge_reg = Ridge()
    # ridge_reg.normalize()
    # - Mean Absolute Error
    ridge_abs_error = cross_val_score(ridge_reg, X, Y, cv=kf, scoring='neg_mean_absolute_error')
    ridge_abs_error_norm = normalise_results(ridge_abs_error)  # normalise results list
    mean_ridge_abs_error = -1 * ridge_abs_error_norm.mean()  # Multiply by -1 and find mean
    print(sizes[size_index], "ridge mean absolute error: ", mean_ridge_abs_error)

    # - Mean Squared Error
    ridge_sq_error = cross_val_score(ridge_reg, X, Y, cv=kf, scoring='neg_mean_squared_error')
    ridge_sq_error_norm = normalise_results(ridge_sq_error)  # normalise results list
    mean_ridge_sq_error = -1 * ridge_sq_error_norm.mean()  # Multiply by -1 and find mean
    print(sizes[size_index], "ridge mean squared error: ", np.sqrt(mean_ridge_sq_error))
    if (ds == "The SUM dataset, with noise.csv") or (ds == "The SUM dataset, without noise.csv"):
        if(ds == "The SUM dataset, with noise.csv"):
            Y = np.array(set[:sizes[size_index]])
            Y = Y[:, -1]
            Y = Y.astype('int')
        else:
            Y = np.array(set[:sizes[size_index]])
            Y = Y[:, -1]
            Y = Y.astype('int')

    if (ds == "kc_housing" and sizes[size_index] < 10000) or (ds == "year_predictions" and (sizes[size_index] < 10000)) or (ds == "The SUM dataset, with noise.csv" or ds == "The SUM dataset, without noise.csv"):
        # Logistic Regression
        log_reg = LogisticRegression()
        # - Accuracy
        accuracy = cross_val_score(log_reg, X, Y, cv=kf, scoring='accuracy')
        mean_accuracy = accuracy.mean()
        print(sizes[size_index], "accuracy: ", mean_accuracy)
        # - Average Precision
        score = make_scorer(precision_score, average="weighted")
        avg_precision = cross_val_score(log_reg, X, Y, cv=kf, scoring=score)
        mean_avg_precision = avg_precision.mean()
        print(sizes[size_index], "average precision: ", mean_avg_precision)

    # Other Classification - K-nearest Neighbours
    knn = neighbors.KNeighborsClassifier()
    # - Accuracy
    k_nn = cross_val_score(knn, X, Y, cv=kf, scoring='accuracy')
    mean_knn = k_nn.mean()
    print(sizes[size_index], "knn accuracy: ", mean_knn)
    # - Average Precision
    score = make_scorer(precision_score, average="weighted")
    knn_avg_precision = cross_val_score(knn, X, Y, cv=kf, scoring=score)
    mean_knn_avg_precision = knn_avg_precision.mean()
    print(sizes[size_index], "knn average precision: ", mean_knn_avg_precision)
    print("\n")

sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]

filename = "YearPredictionMSD.csv"
year_pred = pd.read_csv(filename)

filename = "The SUM dataset, with noise.csv"
sum_with_noise = pd.read_csv(filename, delimiter=";").drop(["Instance"], axis = 1)

sum_with_noise.loc[sum_with_noise["Noisy Target Class"] == "Very Large Number", "Noisy Target Class"] = 5
sum_with_noise.loc[sum_with_noise["Noisy Target Class"] == "Large Number", "Noisy Target Class"] = 4
sum_with_noise.loc[sum_with_noise["Noisy Target Class"] == "Medium Number", "Noisy Target Class"] = 3
sum_with_noise.loc[sum_with_noise["Noisy Target Class"] == "Small Number", "Noisy Target Class"] = 2
sum_with_noise.loc[sum_with_noise["Noisy Target Class"] == "Very Small Number", "Noisy Target Class"] = 1

filename = "The SUM dataset, without noise.csv"
sum_no_noise = pd.read_csv(filename, delimiter=";").drop(["Instance"], axis = 1)

sum_no_noise.loc[sum_no_noise["Target Class"] == "Very Large Number", "Target Class"] = 5
sum_no_noise.loc[sum_no_noise["Target Class"] == "Large Number", "Target Class"] = 4
sum_no_noise.loc[sum_no_noise["Target Class"] == "Medium Number", "Target Class"] = 3
sum_no_noise.loc[sum_no_noise["Target Class"] == "Small Number", "Target Class"] = 2
sum_no_noise.loc[sum_no_noise["Target Class"] == "Very Small Number", "Target Class"] = 1

filename = "kc_house_data.csv"
kc_house = pd.read_csv(filename, delimiter=",").drop(["id"],axis = 1)

# Iterate through the different chunk sizes and apply each algorithm and metric
for size_index in range(len(sizes)):

    X_pred = np.array(year_pred[:sizes[size_index]])
    X_pred = X_pred[:,1:]

    Y_pred = np.array(year_pred[:sizes[size_index]])
    Y_pred = Y_pred[:,0]

    X_pred = X_pred.astype('int')
    Y_pred = Y_pred.astype('int')

    kf = KFold(n_splits=10, random_state=0)
    print("Year Predictions")
    evaluate(X_pred,Y_pred, kf,"year_predictions", year_pred)

    X_w_noise = np.array(sum_with_noise[:sizes[size_index]])
    X_w_noise = X_w_noise[:, :-1]

    Y_w_noise = np.array(sum_with_noise[:sizes[size_index]])
    Y_w_noise = Y_w_noise[:, -2]

    X_w_noise = X_w_noise.astype('int')
    Y_w_noise = Y_w_noise.astype('int')
    print("SUM with noise")
    evaluate(X_w_noise,Y_w_noise,kf,"The SUM dataset, with noise.csv", sum_with_noise)

    X_no_noise = np.array(sum_no_noise[:sizes[size_index]])
    X_no_noise = X_no_noise[:, :-1]

    Y_no_noise = np.array(sum_no_noise[:sizes[size_index]])
    Y_no_noise = Y_no_noise[:, -2]

    X_no_noise = X_no_noise.astype('int')
    Y_no_noise = Y_no_noise.astype('int')
    print("SUM no noise")
    evaluate(X_no_noise,Y_no_noise,kf,"no_noise", sum_no_noise)

    X_housing = np.array(kc_house[:sizes[size_index]])
    X_housing = X_housing[:, 2:6]

    Y_housing = np.array(kc_house[:sizes[size_index]])
    Y_housing = Y_housing[:, 1]

    X_housing = X_housing.astype('int')
    Y_housing = Y_housing.astype('int')
    print("Housing")
    evaluate(X_housing,Y_housing,kf,"kc_housing", kc_house)





