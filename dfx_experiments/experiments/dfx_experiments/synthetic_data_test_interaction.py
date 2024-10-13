# --- Imports
from dfx.doe_xai import DoeXai
from dfx.plotter import Plotter

# --- Other imports
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
from numpy import *
import numpy as np
seed = 42


# --- helper function
def set_target(row):
    t = np.random.sample(1)
    if (row['x1'] >= 0.55) and (row['x2'] >= 0.55) and (row['x3'] >= 0.55):
        return 1
    elif t<= 0.01:
        return 1
    else:
        return 0


# --- Create Synthetic Data
df = pd.DataFrame({"x1": np.random.sample(1000),
       "x2": np.random.sample(1000),
       "x3": np.random.sample(1000),
       "x4": np.random.randint(10, size=1000).tolist()})
y = df.apply(lambda row: set_target(row), axis=1)
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=seed)

# --- Model Training
model = LogisticRegression(random_state=random.seed(seed))
model.fit(X_train, y_train)

test_score = model.score(X_test, y_test)
print(test_score)

# --- SHAP
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

# --- DOE
dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(df.columns))
cont = dx.find_feature_contribution(user_list=[["x1", "x2", "x3"], ["x1", "x2"], ["x1", "x3"], ["x3", "x2"]])
print(cont)

# --- Plot
p = Plotter(X_train)
p.plot_doe_feature_contribution(cont)