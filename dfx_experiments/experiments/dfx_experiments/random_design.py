# --- Imports
from dfx_experiments.experiments.get_datasets import create_wine_data
from dfx.doe_xai import DoeXai
from dfx.plotter import Plotter

# --- Other imports
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
seed = 42


def replace_zeds_randomly(zeds_df):
    zeds_df_c = zeds_df.copy()
    base = 50
    df = pd.DataFrame(np.random.randint(0, 100, size=zeds_df_c.shape), columns=list(zeds_df_c.columns))
    for row_index, row in enumerate(df.itertuples()):
        for col_index, v in enumerate(row[1:]):
            if v >= base:
                zeds_df_c[col_index][row_index] = zeds_df_c[col_index][row_index] * (-1)
    return zeds_df_c


if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_wine_data()

    # --- Model Training
    model = RandomForestClassifier(n_estimators=500, random_state=seed)
    model.fit(X_train, y_train)
    print("Fitting of Gradient Boosting Classifier finished")

    # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=list(X_train.columns))
    # replace zeds randomly to see the effect of a random design
    dx.zeds_df = replace_zeds_randomly(dx.zeds_df)
    cont = dx.find_feature_contribution(only_orig_features=True)
    print(cont)

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)