from dfx_experiments.utils import *
from dfx_experiments.experiments.get_datasets import *
import shap


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = create_company_bankruptcy_prediction_data()
    print("finish load data company bankruptcy prediction")

    model = tabular_dnn(len(list(X_train.columns)), len(set(y_train)))
    model_name = 'DNN'

    X_train_, y_train_ = set_deep_model_data(X_train, y_train)

    size = 100
    print(f"train DeepExplainer on {model_name}")
    explainer = shap.DeepExplainer(model, X_train_[:size])
    shap_values = explainer.shap_values(X_train_.iloc[:size].values, check_additivity=False)
    shap.summary_plot(shap_values, X_train_[:size])

    size = 500
    print(f"train DeepExplainer on {model_name}")
    explainer = shap.DeepExplainer(model, X_train_[:size])
    shap_values = explainer.shap_values(X_train_.iloc[:size].values, check_additivity=False)
    shap.summary_plot(shap_values, X_train_[:size])

