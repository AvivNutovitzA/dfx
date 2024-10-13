from dfx_experiments.utils import *
from dfx.doe_xai import DoeXai
from dfx_experiments.experiments.get_datasets import *

seed = 42
np.random.seed(seed)
random.seed(seed)


def get_data_by_name(name):
    f = f'create_{name}_data'
    return eval(f+'()')


data_set_name = 'company_bankruptcy_prediction'
X_train, y_train, X_test, y_test = get_data_by_name(data_set_name)
new_X_train, new_X_test = reduce_multicollinearity(X_train, X_test, data_set_name)

model = RandomForestClassifier(n_estimators=50, random_state=seed)
model, score = train_model_get_score_by_model_name(model, 'rf', new_X_train, y_train, new_X_test, y_test)
dx = DoeXai(x_data=new_X_train, y_data=y_train, model=model)
isinstance_zero_contributions = dx.explain_instance(instance_index=0)
isinstance_one_contributions = dx.explain_instance(instance_index=1)
r_df = pd.concat([dfx_contribution_to_df(isinstance_zero_contributions)[0],
                  dfx_contribution_to_df(isinstance_one_contributions)[0]], axis=1)

