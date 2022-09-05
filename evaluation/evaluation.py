from models import Linear_Regression

def implement_and_evaluate(models: dict):
    for i in models.keys():
        print(models[i])

def gather_all_models(X_train, y_train, X_test, y_test, alpha=0.5):
    
    models = {}
    model, RMSE = Linear_Regression.Lasso_implementation(X_train, y_train, X_test, y_test, alpha=0.5)
    models['Lasso'] = {'model': model, 'RMSE':RMSE}
    model, RMSE = Linear_Regression.Ridge_implementation(X_train, y_train, X_test, y_test, alpha=0.5)
    models['Ridge'] = {'model': model, 'RMSE':RMSE}
    
    return models