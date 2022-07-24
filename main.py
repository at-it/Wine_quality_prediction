import pandas as pd
import matplotlib.pyplot as plt
from dataset.dataset import split_dataset_for_regression_prediction
from dataset.dataset import initialize_dataset
from explanatory_analysis import explanatory_analysis
from models import Linear_Regression

li = []

df = initialize_dataset()
X, y, X_train, y_train, X_test, y_test = split_dataset_for_regression_prediction(df, 'quality', 0.2)

explanatory_analysis.run(X)
model, RMSE = Linear_Regression.Ridge_implementation(X_train, y_train, X_test, y_test, alpha=0.5)
li.append(RMSE)
model, RMSE = Linear_Regression.Lasso_implementation(X_train, y_train, X_test, y_test, alpha=0.5)
li.append(RMSE)
print(li)



