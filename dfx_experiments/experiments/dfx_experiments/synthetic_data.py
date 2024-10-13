from dfx.doe_xai import DoeXai
from dfx.plotter import Plotter

import os
import shap
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

seed = 42

original_feature_contributions_dict = {"x1": [], "x2": [], "x3": [], "x4": [], "x5": []}


def set_target(row):
    v = 0
    if row['x1'] > 0.65:
        v = 1
        original_feature_contributions_dict["x1"].append(1)
    elif row['x2'] > 0.7:
        v = 1
        original_feature_contributions_dict["x2"].append(1)
    elif row['x3'] > 0.75:
        v = 1
        original_feature_contributions_dict["x3"].append(1)
    return v


# --- Create Synthetic Data
size = 1000
df = pd.DataFrame(
    {"x1": np.random.sample(size),
     "x2": np.random.sample(size),
     "x3": np.random.sample(size),
     "x4": np.random.sample(size),
     "x5": np.random.sample(size),
     })

y = df.apply(lambda row: set_target(row), axis=1)
original_feature_contributions = {k: np.round(np.sum(v)/ size, 3) for k, v in original_feature_contributions_dict.items()}
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=seed)

# --- Model Training
model = LogisticRegression(random_state=np.random.seed(seed))
model.fit(X_train, y_train)

test_score = model.score(X_test, y_test)
print(test_score)

# --- SHAP
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

# --- DOE
dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(df.columns))
cont = dx.find_feature_contribution(only_orig_features=True)
print("original:", original_feature_contributions)
print("DFX:", {k: np.round(v, 3) for k, v in cont.items()})
print("SHAP:",  {f"x{i+1}": np.round(v, 3) for i, v in enumerate(np.abs(shap_values).mean(axis=0))})


# --- Plot
p = Plotter(X_train)
p.plot_doe_feature_contribution(cont)