from dfx_experiments.utils import *
from dfx.doe_xai import DoeXai
from dfx_experiments.experiments.get_datasets import *


seed = 42
np.random.seed(seed)
random.seed(seed)


def get_data_by_name(name):
    f = f'create_{name}_data'
    return eval(f+'()')


data_sets = ['stroke', 'fetal_health', 'bank_churners', 'hr_employee_attrition', 'cervical_cancer', 'mobile_price',
             'churn_modelling', 'company_bankruptcy_prediction', 'airline_passenger_satisfaction',
             'banking_marketing_targets']


data_sets1 = ['hr_employee_attrition', 'cervical_cancer', 'mobile_price', 'company_bankruptcy_prediction']

for ds_name in data_sets:
    x_train, y_train, x_test, y_test = get_data_by_name(ds_name)
    new_x_train, new_x_test = reduce_multicollinearity(x_train, x_test, ds_name)

    model = RandomForestClassifier(n_estimators=50, random_state=seed)
    model, score = train_model_get_score_by_model_name(model, 'rf', new_x_train, y_train, new_x_test, y_test)
    dx = DoeXai(x_data=new_x_train, y_data=y_train, model=model)

    all_interactions = create_all_feature_interactions_from_list(new_x_train.columns, 2)
    contributions = dx.find_feature_contribution(user_list=all_interactions)
    dfx_importance_as_df, _ = dfx_contribution_to_df(contributions)
    dx.output_process_files(f"find_interactions_on_ds_{ds_name}")
    dfx_importance_as_df.to_csv(f"find_interactions_feature_contributions_on_ds_{ds_name}.csv")

    contribution_sign = dx.find_feature_contribution_sign(user_list=all_interactions)
    dfx_importance_with_sign_as_df, _ = dfx_contribution_to_df(contribution_sign)
    dx.output_process_files(f"find_interactions_on_ds_{ds_name}")
    dfx_importance_with_sign_as_df.to_csv(f"find_interactions_feature_contributions_sign_on_ds_{ds_name}.csv")

    contributions_ = dx.find_feature_contribution(only_orig_features=True)
    dfx_importance_as_df, _ = dfx_contribution_to_df(contributions_)
    dx.output_process_files(f"only_orig_features_on_ds_{ds_name}")
    dfx_importance_as_df.to_csv(f"only_orig_features_contributions_on_ds_{ds_name}.csv")

    contribution_sign = dx.find_feature_contribution_sign(only_orig_features=True)
    dfx_importance_sign_as_df, _ = dfx_contribution_to_df(contribution_sign)
    dx.output_process_files(f"only_orig_features_sign_on_ds_{ds_name}")
    dfx_importance_sign_as_df.to_csv(f"only_orig_features_contributions_sign_on_ds_{ds_name}.csv")




