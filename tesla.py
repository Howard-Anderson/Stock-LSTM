"""
        Tesla Stock Price Dataset:

    Authors: Howard Anderson, Gopinath TK.

    Filename: tesla.py.

    Date: 07/02/2023.

    Depends on: 
        [ Libraries/ Modules ]:
            numpy, pandas, matplotlib, scikit-learn.
        [ Files ]:
            tesla.csv
"""

# Importing Libraries / Modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Loading the DataSet as a DataFrame.
tesla = pd.read_csv("TSLA.csv")

# Independent Variables - X, Dependent Variable - Y.
X = tesla[["Open","High","Low","Volume"]]
Y = tesla["Close"]

# Spliting the DataSet.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.34, random_state = 34)

# Instantiating and Training the Models.
# RandomForestRegressor.
rfr = RandomForestRegressor(n_estimators = 0x7D).fit(x_train, y_train)

# LinearRegression.
linear_regression = LinearRegression().fit(x_train, y_train)

# KNeighborsRegressor.
knn_regressor = KNeighborsRegressor(n_neighbors = 10).fit(x_train, y_train)

# SVR (Suppor Vector Machine Regressor).
""" Note: Perform drops when the Sample size get more than 10000. """
svr_regressor = SVR().fit(x_train, y_train) 

# LinearSVR (Linear Support Vector Machine Regressor).
linear_svr = LinearSVR(epsilon = 0.01).fit(x_train, y_train)

# LassoRegression.
lasso_regression = Lasso().fit(x_train, y_train)

# RidgeRegression.
ridge_regression = Ridge().fit(x_train, y_train)

# DecisionTreeRegressor.
dtr = DecisionTreeRegressor(max_depth = 7).fit(x_train, y_train)

# StochasticGradientDescentRegressor.
sgdr = SGDRegressor().fit(x_train, y_train)

# Predictions from the Models.
rfr_pred = rfr.predict(x_test)
linear_regression_pred = linear_regression.predict(x_test)
knn_regressor_pred = knn_regressor.predict(x_test)
svr_regerssor_pred = svr_regressor.predict(x_test)
linear_svr_pred = linear_svr.predict(x_test)
lasso_regression_pred = lasso_regression.predict(x_test)
ridge_regression_pred = ridge_regression.predict(x_test)
dtr_pred = dtr.predict(x_test)
sgdr_pred = sgdr.predict(x_test)

# Accuracy of the Models.
print(f"\nAccuracy of RandomForestRegressor: {r2_score(y_test, rfr_pred)}")
print(f"\nAccuracy of LinearRegression: {r2_score(y_test, linear_regression_pred)}")
print(f"\nAccuracy of KNeighborsRegressor: {r2_score(y_test, knn_regressor_pred)}")
print(f"\nAccuracy of SVR [Support Vector Regression]: {r2_score(y_test, svr_regerssor_pred)}")
print(f"\nAccuracy of LinearSVR [Linear Support Vector Regression]: {r2_score(y_test, linear_svr_pred)} ")
print(f"\nAccuracy of LassoRegression: {r2_score(y_test, lasso_regression_pred)}")
print(f"\nAccuracy of RidgeRegression: {r2_score(y_test, ridge_regression_pred)}")
print(f"\nAccuracy of DecisionTreeRegressor: {r2_score(y_test, dtr_pred)}")
print(f"\nAccuracy of StochasticGradientDescentRegressor: {r2_score(y_test, sgdr_pred)}")

# Writing the Predicted Data into a File.
results = {	
			"Open" : x_test["Open"],
			"High" : x_test["High"],
			"Low" : x_test["Low"],
			"Volume" : x_test["Volume"],
			"Actual Values": y_test,
			"Random Forest Regressor" : rfr_pred, 
			"Linear Regression" : linear_regression_pred, 
			"KNeighborsRegressor" : knn_regressor_pred,
			"Support Vector Regressor" : svr_regerssor_pred,
			"Linear Support Vector Regressor" : linear_svr_pred,
			"Lasso Regression" : lasso_regression_pred,
			"Ridge Regression" : ridge_regression_pred,
			"Decision Tree Regressor" : dtr_pred,
			"Stochastic Gradient Descent Regressor" : sgdr_pred
			}
results = pd.DataFrame(results)
results.to_csv("tesla_pred.csv")

