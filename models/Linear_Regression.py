import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def Lasso_implementation(X_train, y_train, X_test, y_test, alpha=1):
    """Implement the Lasso model based on the split training and test set returning model and RMSE.

        Parameters:
            X_train (Pandas DataFrame): Features of training part of the dataset
            y_train (Pandas Series): Labels of training part of the dataset
            X_test (Pandas DataFrame): Features of test part of the dataset
            y_test (Pandas Series): Labels of test part of the dataset

        Returns:
            model: Lasso model implementation using sklearn
            RMSE: Root Mean Squared Error using sklearn metrics
    """
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = MSE ** 0.5

    return model, RMSE

def Ridge_implementation(X_train, y_train, X_test, y_test, alpha=1):
    """Implement the Lasso model based on the split training and test set returning model and RMSE.

        Parameters:
            X_train (Pandas DataFrame): Features of training part of the dataset
            y_train (Pandas Series): Labels of training part of the dataset
            X_test (Pandas DataFrame): Features of test part of the dataset
            y_test (Pandas Series): Labels of test part of the dataset

        Returns:
            model: Lasso model implementation using sklearn
            RMSE: Root Mean Squared Error using sklearn metrics
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = MSE ** 0.5

    return model, RMSE
