# --- Imports
from dfx_experiments.experiments.get_datasets import create_wine_data
from dfx.doe_xai import DoeXai
from dfx.plotter import Plotter

# --- Other imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
from numpy import *
seed = 42


if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_wine_data()

    # --- Model Training
    model = RandomForestClassifier(n_estimators=500, random_state=random.seed(seed))
    model.fit(X_train, y_train)
    print("Fitting of Random Forest Classifier finished")

    xgb_predict = model.predict(X_test)
    xgb_score = accuracy_score(y_test, xgb_predict)
    print(f'test score : {xgb_score}')
    print("=" * 80)
    print(classification_report(y_test, model.predict(X_test)))

    # --- SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(X_train.columns),
                design_file_name=os.path.join(os.getcwd(), '../../..', 'resources/res_V_design.csv'))
    # features: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
    # 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
    cont = dx.find_feature_contribution(user_list=[['nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue'],
                                                   ['alcohol', 'malic_acid', 'ash'], ['hue', 'proline']])

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)