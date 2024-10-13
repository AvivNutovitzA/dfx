# Thesis Code

all example data stored in git lfs, if you would like to download the data and test the examples connect to git lfs

all dfx data and experiments can be found under dfx_experiments folder.
for very basic dfx use, given your x_train, y_train, here is a quick example:

```

from dfx.doe_xai import DoeXai
from dfx_experiments.utils import *

model = RandomForestClassifier(n_estimators=50, random_state=seed)
dx = DoeXai(x_data=new_x_train, y_data=y_train, model=model)

cont = dx.find_feature_contribution(only_orig_features=True)
dfx_importance_as_df, dfx_indices = dfx_contribution_to_df(cont) 

```