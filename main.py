import pandas as pd
import matplotlib.pyplot as plt
from dataset.dataset import split_dataset_for_regression_prediction
from dataset.dataset import initialize_dataset
from explanatory_analysis import explanatory_analysis
from models import Linear_Regression
from evaluation.evaluation import gather_all_models, implement_and_evaluate

df = initialize_dataset()
X, y, X_train, y_train, X_test, y_test = split_dataset_for_regression_prediction(
    df, 'quality', 0.2)
explanatory_analysis.run(X)
all_models = gather_all_models(X_train, y_train, X_test, y_test)
implement_and_evaluate(all_models)
