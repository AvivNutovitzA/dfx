from dfx_experiments.utils import *
from dfx_experiments.experiments.get_datasets import *
from dfx.doe_xai import DoeXai

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

tf.random.set_seed(seed)


def get_data_by_name(name):
    f = f'create_{name}_data'
    return eval(f+'()')


all_data_set_names = ['stroke', 'fetal_health', 'bank_churners', 'hr_employee_attrition',
                      'cervical_cancer', 'mobile_price', 'churn_modelling',
                      'company_bankruptcy_prediction', 'airline_passenger_satisfaction',
                      'banking_marketing_targets']

list_of_models_names = ['LR', 'DTC', 'RF', 'DNN', 'CNN']

shap_results_df = pd.DataFrame(columns=list_of_models_names, index=all_data_set_names)
mutual_information_results_df = pd.DataFrame(columns=list_of_models_names, index=all_data_set_names)
permutation_results_df = pd.DataFrame(columns=list_of_models_names, index=all_data_set_names)
model_importance_results_df = pd.DataFrame(columns=list_of_models_names, index=all_data_set_names)
random_results_df = pd.DataFrame(columns=list_of_models_names, index=all_data_set_names)

for data_set_name in all_data_set_names:

    set_results_folder(data_set_name)
    X_train, y_train, X_test, y_test = get_data_by_name(data_set_name)
    new_X_train, new_X_test = reduce_multicollinearity(X_train, X_test, data_set_name, output=True)

    # create mutual info feature importance
    mutual_info_df = mutual_info_to_df(new_X_train, y_train)
    mutual_info_df.to_csv('/'.join([datasets_folder, data_set_name, f'model_mutual_info_as_df.csv']), index=False)

    list_of_models = [LogisticRegression(random_state=random.seed(seed)),
                      DecisionTreeClassifier(random_state=random.seed(seed)),
                      RandomForestClassifier(n_estimators=50, random_state=random.seed(seed)),
                      tabular_dnn(len(list(new_X_train.columns)), len(set(y_train))),
                      tabular_cnn(len(list(new_X_train.columns)), len(set(y_train)))]

    model_scores = []
    top_5_features_match_proportion_scores = []
    top_10_features_match_proportion_scores = []
    for model, model_name in zip(list_of_models, list_of_models_names):

        # train model
        model, score = train_model_get_score_by_model_name(model, model_name, new_X_train, y_train, new_X_test, y_test)
        model_scores.append(score)

        # create shap values
        shap_values_as_df, shap_indices = create_shap_values(model_name, model, new_X_train)
        shap_values_as_df.to_csv('/'.join([datasets_folder, data_set_name, f'{model_name}_model_shap_importance_as_df.csv']), index=False)

        # dfx
        is_keras_nn_model = model_name in ['DNN, CNN']
        dx = DoeXai(x_data=new_X_train, y_data=y_train, model=model, is_keras_nn_model=is_keras_nn_model)
        cont = dx.find_feature_contribution(only_orig_features=True)
        dfx_importance_as_df, dfx_indices = dfx_contribution_to_df(cont)
        dfx_importance_as_df.to_csv('/'.join([datasets_folder, data_set_name, f'{model_name}_model_dfx_importance_as_df.csv']), index=False)

        # model feature importance
        if model_name in ['LR', 'SVM']:
            model_feature_importance = np.abs(model.coef_)
        elif model_name in ['DTC', 'RF']:
            model_feature_importance = np.abs(model.feature_importances_)
        else:
            model_feature_importance = pd.Series([-1] * len(new_X_train.columns))
        model_feature_importance_df = model_feature_importance_to_df(model_feature_importance, new_X_train.columns)
        model_feature_importance_df.to_csv('/'.join([datasets_folder, data_set_name, f'{model_name}_model_feature_importance_as_df.csv']), index=False)

        # create permutation_importance
        if model_name not in ['DNN', 'CNN']:
            permutation_importance_df = permutation_importance_to_df(model, new_X_train, y_train)
        else:
            permutation_importance_df = permutation_importance_to_df(model, new_X_train, y_train, True)
        permutation_importance_df.to_csv('/'.join([datasets_folder, data_set_name, f'{model_name}_model_permutation_importance_as_df.csv']), index=False)

        # create random importance
        random_importance_df = random_importance_to_df(new_X_train.columns)
        random_importance_df.to_csv('/'.join([datasets_folder, data_set_name, f'{model_name}_model_random_as_df.csv']), index=False)

        # pearson test shap
        t1 = set_data_for_statistical_tests(dfx_importance_as_df.sort_values('feature_name'))
        t2 = set_data_for_statistical_tests(shap_values_as_df.sort_values('feature_name'))
        pr_stat, _ = stats.pearsonr(t1, t2)

        # update shap _results
        shap_results_df.at[data_set_name, model_name] = np.round(pr_stat, 3)

        # pearson test mutual_information
        t2 = set_data_for_statistical_tests(mutual_info_df.sort_values('feature_name'))
        pr_stat, _ = stats.pearsonr(t1, t2)

        # update mutual_information _results
        mutual_information_results_df.at[data_set_name, model_name] = np.round(pr_stat, 3)

        # pearson test mutual_information
        t2 = set_data_for_statistical_tests(model_feature_importance_df.sort_values('feature_name'))
        pr_stat, _ = stats.pearsonr(t1, t2)

        # update mutual_information _results
        model_importance_results_df.at[data_set_name, model_name] = np.round(pr_stat, 3)

        # pearson test permutation importance
        t2 = set_data_for_statistical_tests(permutation_importance_df.sort_values('feature_name'))
        pr_stat, _ = stats.pearsonr(t1, t2)

        # update permutation importance _results
        permutation_results_df.at[data_set_name, model_name] = np.round(pr_stat, 3)

        # pearson test random importance
        t2 = set_data_for_statistical_tests(random_importance_df.sort_values('feature_name'))
        pr_stat, _ = stats.pearsonr(t1, t2)

        # update random importance _results
        random_results_df.at[data_set_name, model_name] = np.round(pr_stat, 3)

        # select feature based on shap
        if model_name != 'DNN' and model_name != 'CNN':
            dfx_predictions, dfx_features_in_group = get_scores_by_adding_selected_features(new_X_train, y_train, new_X_test, y_test, dfx_indices, data_set_name,
                                                   model_name, 'dfx', max_features=15)
            shap_predictions, shap_features_in_group = get_scores_by_adding_selected_features(new_X_train, y_train, new_X_test, y_test, shap_indices, data_set_name,
                                                   model_name, max_features=15)

            # one data set contains 9 features only
            top_5_features_match_proportion_scores.append(len(list(set(dfx_features_in_group[4]) & set(shap_features_in_group[4])))/5)
            if len(dfx_features_in_group) >= 10:
                top_10_features_match_proportion_scores.append(len(list(set(dfx_features_in_group[9]) & set(shap_features_in_group[9])))/10)
            else:
                top_10_features_match_proportion_scores.append(len(list(set(dfx_features_in_group[8]) & set(shap_features_in_group[8])))/ 9)

    shap_results_df.at[data_set_name, 'number_of_features'] = int(new_X_train.shape[1])
    shap_results_df.at[data_set_name, 'top_5_features_match_proportion_max'] = np.round(max(top_5_features_match_proportion_scores), 2)
    shap_results_df.at[data_set_name, 'top_5_features_match_proportion_min'] = np.round(min(top_5_features_match_proportion_scores), 2)
    shap_results_df.at[data_set_name, 'top_10_features_match_proportion_max'] = np.round(max(top_10_features_match_proportion_scores), 2)
    shap_results_df.at[data_set_name, 'top_10_features_match_proportion_min'] = np.round(min(top_10_features_match_proportion_scores), 2)

    model_importance_results_df.at[data_set_name, 'number_of_features'] = int(new_X_train.shape[1])
    permutation_results_df.at[data_set_name, 'number_of_features'] = int(new_X_train.shape[1])
    mutual_information_results_df.at[data_set_name, 'number_of_features'] = int(new_X_train.shape[1])
    random_results_df.at[data_set_name, 'number_of_features'] = int(new_X_train.shape[1])

# write _results
shap_results_df.to_csv('/'.join([results_folder, 'shap_comparison_results_df.csv']))
model_importance_results_df.to_csv('/'.join([results_folder, 'model_feature_importance_comparison_results_df.csv']))
mutual_information_results_df.to_csv('/'.join([results_folder, 'model_mutual_info_comparison_results_df.csv']))
permutation_results_df.to_csv('/'.join([results_folder, 'permutation_importance_comparison_results_df.csv']))
random_results_df.to_csv('/'.join([results_folder, 'random_comparison_results_df.csv']))
