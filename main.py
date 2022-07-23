import pandas as pd
import matplotlib.pyplot as plt
from dataset.dataset import split_dataset_for_wine_quality_prediction
from dataset.dataset import initialize_dataset
from explanatory_analysis import explanatory_analysis


df = initialize_dataset()
X, y, X_train, y_train, X_test, y_test = split_dataset_for_wine_quality_prediction(df)

explanatory_analysis.run(X)




