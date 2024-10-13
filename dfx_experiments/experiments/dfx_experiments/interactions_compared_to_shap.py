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

# test triplets
l1 = ['Total debt/Total net worth', 'Cash Flow to Equity', 'Allocation rate per person',
      'Long-term Liability to Current Assets', 'Degree of Financial Leverage (DFL)', 'Tax rate (A)',
      'Total assets to GNP price', 'Cash Reinvestment %', 'Average Collection Days', 'Inventory/Current Liability']
tested_interactions = create_all_feature_interactions_from_list(l1, 3)
contributions = dx.find_feature_contribution(user_list=tested_interactions)
dfx_importance_as_df, _ = dfx_contribution_to_df(contributions)


# form important triplets test pairs
l2 = ['Tax rate (A)', 'Cash Reinvestment %', 'Inventory/Current Liability',
      'Total assets to GNP price', 'Long-term Liability to Current Assets',
      'Cash Flow to Equity', 'Average Collection Days', 'Cash Flow to Equity',
      'Allocation rate per person']
tested_interactions1 = create_all_feature_interactions_from_list(l2, 2)
contributions1 = dx.find_feature_contribution(user_list=tested_interactions1)
dfx_importance_as_df1, _ = dfx_contribution_to_df(contributions1)


# create interaction for shap
interaction = (new_X_train['Cash Reinvestment %'] - new_X_train['Cash Reinvestment %'].mean()) * \
              (new_X_train['Long-term Liability to Current Assets'] - new_X_train['Long-term Liability to Current Assets'].mean())
interaction = (new_X_train['Cash Reinvestment %']) * (new_X_train['Long-term Liability to Current Assets'])
new_X_train['Cash Reinvestment %_Long-term Liability to Current Assets'] = interaction.values
interaction1 = (new_X_train['Cash Reinvestment %']) * (new_X_train['Long-term Liability to Current Assets'])
interaction2 = (new_X_train['Cash Reinvestment %'] - new_X_train['Cash Reinvestment %'].mean()) * \
               (new_X_train['Average Collection Day'] - new_X_train['Average Collection Day'].mean())
interaction3 = (new_X_train['Cash Reinvestment %']) * (new_X_train['Average Collection Day'])

test_interaction = (new_X_test['Cash Reinvestment %']) * (new_X_test['Long-term Liability to Current Assets'])
test_interaction = (new_X_test['Cash Reinvestment %'] - new_X_test['Cash Reinvestment %'].mean()) * \
                   (new_X_test['Long-term Liability to Current Assets'] - new_X_test['Long-term Liability to Current Assets'].mean())
new_X_test['Cash Reinvestment %_Long-term Liability to Current Assets'] = test_interaction.values
model1 = RandomForestClassifier(n_estimators=50, random_state=seed)
model1, _ = train_model_get_score_by_model_name(model1, 'rf', new_X_train, y_train, new_X_test, y_test)
shap_values_df, _ = create_shap_values('RF', model1, new_X_train)