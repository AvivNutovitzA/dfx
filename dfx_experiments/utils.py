import os
import gc
import shap
from numpy import random
# import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
from scipy import stats
from itertools import combinations

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
# from sklearn.exceptions import ConvergenceWarning
from statsmodels.stats.outliers_influence import variance_inflation_factor

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Dense, Input, Flatten, MaxPooling1D, Dropout, Conv1D, BatchNormalization, Reshape


tf.compat.v1.disable_v2_behavior()
seed = 42
random.seed(seed)
tf.random.set_seed(seed)


results_folder = '/'.join([os.path.dirname(os.path.abspath(__file__)), 'results'])
datasets_folder = '/'.join([results_folder, 'datasets'])
# models_folder = '/'.join([results_folder, 'models'])
plots_folder = '/'.join([results_folder, 'plots'])

models_and_names = {'LR': LogisticRegression(random_state=seed),
                    'DTC': DecisionTreeClassifier(random_state=seed),
                    'RF': RandomForestClassifier(n_estimators=50, random_state=seed)
                    }


def load_data(file_name, size=-1):
    example_path = 'data'
    df = pd.read_csv(os.path.join(os.getcwd(), '../..', f'{example_path}/{file_name}_data.csv'))
    if size > -1:
        df = df.sample(size, random_state=seed)

    if file_name == 'wine':
        y = df['y']
        df = df.drop(columns=['y'])
        return df, y

    elif file_name == 'fake_job_posting':
        df.fillna(" ", inplace=True)
        df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + \
                     df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] \
                     + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function']
        return df['text'], df['fraudulent']

    elif file_name == 'hotel_bookings':
        X = df.drop(["is_canceled"], axis=1)
        y = df["is_canceled"]
        return X, y

    elif file_name == 'hr_employee_attrition':
        target_map = {'Yes': 1, 'No': 0}
        # Use the pandas apply method to numerically encode our attrition target variable
        y = df["Attrition"].apply(lambda x: target_map[x])
        X = df.drop(["Attrition"], axis=1)
        return X, y

    elif file_name == 'nomao':
        X = df.drop(["__TARGET__"], axis=1)
        y = df["__TARGET__"]
        return X, y

    elif file_name == 'placement_full_class':
        X = df[['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p',
                'specialisation', 'mba_p']]
        y = df['status']
        return X, y

    elif file_name == 'rain_weather_aus':
        df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'RISK_MM', 'Date'], axis=1)
        df = df.dropna(how='any')
        X = df.loc[:, df.columns != 'RainTomorrow']
        y = df['RainTomorrow']
        return X, y

    elif file_name == 'cervical_cancer':
        df = df.replace('?', np.nan)
        df = df.rename(columns={'Biopsy': 'Cancer'})
        df = df.apply(pd.to_numeric)
        df = df.fillna(df.mean().to_dict())
        X = df.drop('Cancer', axis=1)
        y = df['Cancer']
        return X, y

    elif file_name == 'glass':
        features = df.columns[:-1].tolist()
        X = df[features]
        y = df['Type']
        return X, y

    elif file_name == 'mobile_price':
        y = df.price_range
        X = df.drop(["price_range"], axis=1)
        return X, y

    elif file_name == 'clinvar_conflicting':
        toBeConsidered = ['CHROM', 'POS', 'REF', 'ALT', 'AF_ESP', 'AF_EXAC', 'AF_TGP',
                          'CLNDISDB', 'CLNDN', 'CLNHGVS', 'CLNVC', 'MC', 'ORIGIN', 'CLASS',
                          'Allele', 'Consequence', 'IMPACT', 'SYMBOL', 'Feature_type',
                          'Feature', 'BIOTYPE', 'STRAND', 'CADD_PHRED', 'CADD_RAW']
        df2 = df[toBeConsidered]
        df2 = df2.dropna()
        cutdowns = []
        for i in df2.columns.values:
            if df2[i].nunique() < 1000:
                cutdowns.append(i)
        df_final = df2[cutdowns]
        df_final['CHROM'] = df_final['CHROM'].astype(str)
        from sklearn.feature_extraction import FeatureHasher
        fh = FeatureHasher(n_features=5, input_type='string')
        hashed1 = fh.fit_transform(df_final['REF'])
        hashed1 = hashed1.toarray()
        hashedFeatures1 = pd.DataFrame(hashed1)
        nameList = {}
        for i in hashedFeatures1.columns.values:
            nameList[i] = "REF" + str(i + 1)
        hashedFeatures1.rename(columns=nameList, inplace=True)
        hashed2 = fh.fit_transform(df_final['ALT'])
        hashed2 = hashed2.toarray()
        hashedFeatures2 = pd.DataFrame(hashed2)
        nameList2 = {}
        for i in hashedFeatures2.columns.values:
            nameList2[i] = "ALT" + str(i + 1)
        hashedFeatures2.rename(columns=nameList2, inplace=True)
        binaryFeature1 = pd.get_dummies(df_final['CLNVC'])
        df_final = df_final.drop(columns=['MC'], axis=1)
        hashed0 = fh.fit_transform(df_final['CHROM'])
        hashed0 = hashed0.toarray()
        hashedFeatures0 = pd.DataFrame(hashed0)
        nameList0 = {}
        for i in hashedFeatures0.columns.values:
            nameList0[i] = "CHROM" + str(i + 1)
        hashedFeatures0.rename(columns=nameList0, inplace=True)
        hashed3 = fh.fit_transform(df_final['Allele'])
        hashed3 = hashed3.toarray()
        hashedFeatures3 = pd.DataFrame(hashed3)
        nameList3 = {}
        for i in hashedFeatures3.columns.values:
            nameList3[i] = "Allele" + str(i + 1)
        hashedFeatures3.rename(columns=nameList3, inplace=True)
        hashed4 = fh.fit_transform(df_final['Consequence'])
        hashed4 = hashed4.toarray()
        hashedFeatures4 = pd.DataFrame(hashed4)
        nameList4 = {}
        for i in hashedFeatures4.columns.values:
            nameList4[i] = "Consequence" + str(i + 1)
        hashedFeatures4.rename(columns=nameList4, inplace=True)
        binaryFeature3 = pd.get_dummies(df_final['IMPACT'])
        df_final = df_final.drop(columns=['Feature_type'], axis=1)
        binaryFeature4 = pd.get_dummies(df_final['BIOTYPE'], drop_first=True)
        binaryFeature5 = pd.get_dummies(df_final['STRAND'], drop_first=True)
        df3 = pd.concat(
            [binaryFeature1, binaryFeature3, binaryFeature4, binaryFeature5, hashedFeatures1, hashedFeatures2,
             hashedFeatures3, hashedFeatures4, hashedFeatures0, df_final['CLASS']], axis=1)
        df3 = df3.dropna()
        df3.rename(columns={1: "one", 16: "sixteen"}, inplace=True)
        y = df3['CLASS']
        X = df3.drop(columns=['CLASS'], axis=1)
        return X, y

    elif file_name == 'heart_failure_clinical':
        y = df['DEATH_EVENT']
        X = df.drop('DEATH_EVENT', axis=1)
        return X, y

    elif file_name == 'churn_modelling':
        df['EstimatedSalary'] = df['EstimatedSalary'].astype(int)
        df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Geography'], inplace=True)
        le = preprocessing.LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        return X, y
    elif file_name == 'hr_leaving':
        y = df['left']
        X = df.drop('left', axis=1)
        return X, y
    elif file_name == 'bank_churners':
        df = pd.get_dummies(df, drop_first=True)
        norm = MinMaxScaler().fit(df)
        data_norm_arr = norm.transform(df)
        X = pd.DataFrame(data=data_norm_arr,
                         columns=['CLIENTNUM', 'Customer_Age', 'Dependent_count', 'Months_on_book',
                                  'Total_Relationship_Count', 'Months_Inactive_12_mon',
                                  'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                                  'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                                  'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                                  'Attrition_Flag_Existing Customer', 'Gender_M',
                                  'Education_Level_Doctorate', 'Education_Level_Graduate',
                                  'Education_Level_High School', 'Education_Level_Post-Graduate',
                                  'Education_Level_Uneducated', 'Education_Level_Unknown',
                                  'Marital_Status_Married', 'Marital_Status_Single',
                                  'Marital_Status_Unknown', 'Income_Category_$40K - $60K',
                                  'Income_Category_$60K - $80K', 'Income_Category_$80K - $120K',
                                  'Income_Category_Less than $40K', 'Income_Category_Unknown',
                                  'Card_Category_Gold', 'Card_Category_Platinum', 'Card_Category_Silver'])
        X = df.drop("Attrition_Flag_Existing Customer", axis=1)
        y = df["Attrition_Flag_Existing Customer"]
        return X, y
    elif file_name == 'fetal_health':
        X = df.drop(["fetal_health"], axis=1)
        y = df["fetal_health"]
        return X, y
    elif file_name == 'stroke':
        df.drop("id", axis=1, inplace=True)
        for column in ['bmi']:
            df[column].fillna(df[column].mode()[0], inplace=True)
        for label, content in df.items():
            if pd.api.types.is_string_dtype(content):
                df[label] = content.astype("category").cat.as_ordered()
        for label, content in df.items():
            if not pd.api.types.is_numeric_dtype(content):
                df[label] = pd.Categorical(content).codes + 1
        X = df.drop("stroke", axis=1)
        y = df["stroke"]
        return X, y
    elif file_name == 'company_bankruptcy_prediction':
        df.columns = [str(col).strip() for col in list(df.columns)]
        X = df.drop(["Bankrupt?"], axis=1)
        y = df['Bankrupt?']
        return X, y
    elif file_name == 'airline_passenger_satisfaction':
        df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
        X = df.drop(["satisfaction"], axis=1)
        y = df['satisfaction']
        return X, y
    elif file_name == 'banking_marketing_targets':
        X = df.drop(["y"], axis=1)
        target_map = {'yes': 1, 'no': 0}
        y = df['y'].apply(lambda x: target_map[x])
        return X, y
    else:
        raise ValueError(f"file name can be one of the following: wine, fake_job_posting, hotel_bookings, "
                         f"hr_employee_attrition, nomao, placement_full_class, rain_weather_aus, cervical_cancer, "
                         f"glass or mobile_price. "
                         f"file_name that passed is {type(file_name)}")


def random_importance_to_df(feature_names):
    return pd.DataFrame({'feature_name': feature_names,
                          'random_feature_importance': [random.random() for _ in range(len(feature_names))]})


def dfx_contribution_to_df(contribution):
    dfx_df = pd.DataFrame.from_dict(contribution, orient='index')
    dfx_df = dfx_df.reset_index()
    dfx_df.columns = ['feature_name', 'dfx_feature_importance']
    dfx_df = dfx_df.sort_values('dfx_feature_importance', ascending=False)
    indices = list(reversed(np.argsort(np.array([np.abs(values) for key, values in contribution.items()]))))
    return dfx_df, indices


def shap_values_to_df(shap_values, feature_names):
    shap_sum = np.abs(shap_values).mean(axis=0)
    if len(shap_sum.shape) > 1:
        shap_sum = shap_sum.mean(axis=0)
    importance_df = pd.DataFrame([feature_names, shap_sum.tolist()]).T
    importance_df.columns = ['feature_name', 'shap_feature_importance']
    importance_df = importance_df.sort_values('shap_feature_importance', ascending=False)
    indices = list(reversed(np.argsort(shap_sum.tolist())))
    return importance_df, indices


def model_feature_importance_to_df(model_feature_importance, feature_names):
    if len(model_feature_importance.shape) > 1:
        model_feature_importance = np.abs(model_feature_importance).mean(axis=0)
    tmp = pd.DataFrame({feature_name: [feature_importance] for feature_name, feature_importance in
                        zip(feature_names, model_feature_importance)}).T
    tmp = tmp.reset_index()
    tmp.columns = ['feature_name', 'model_feature_importance']
    tmp = tmp.sort_values('model_feature_importance', ascending=False)
    return tmp


def permutation_importance_to_df(model, X, y, is_deep_model=False):
    if not is_deep_model:
        results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
        feature_importances = results.importances_mean
    else:
        feature_importances = [-1] * len(X.columns)
    tmp = pd.DataFrame({feature_name: [feature_importance] for feature_name, feature_importance in
                         zip(X.columns, feature_importances)}).T
    tmp = tmp.reset_index()
    tmp.columns = ['feature_name', 'permutation_feature_importance']
    tmp = tmp.sort_values('permutation_feature_importance', ascending=False)
    return tmp


def f_score_pvalue_to_df(X_train, y_train):
    fs, pvalues = f_classif(X_train, y_train)
    f_pvalue_df = pd.DataFrame({'feature_name': X_train.columns, 'f_score_pvalue': pvalues})
    f_pvalue_df['f_score_pvalue'] = f_pvalue_df['f_score_pvalue'].fillna(-1)
    f_pvalue_df = f_pvalue_df.sort_values('f_score_pvalue', ascending=True)
    return f_pvalue_df


def mutual_info_to_df(X_train, y_train):
    mutual_info = mutual_info_classif(X_train, y_train)
    mutual_info_df = pd.DataFrame({'feature_name': X_train.columns, 'mutual_info_score': mutual_info})
    mutual_info_df['mutual_info_score'] = mutual_info_df['mutual_info_score'].fillna(-1)
    mutual_info_df = mutual_info_df.sort_values('mutual_info_score', ascending=False)
    return mutual_info_df


def set_data_for_statistical_tests(df):
    try:
        # df
        df = df.set_index('feature_name')
        col = df.columns[0]
        if min(df[col]) < 0:
            df += min(df[col]) * -1
        if max(df[col]) > 0:
            df[col] = df.values / max(df.values)
        df = pd.Series(df.values.flatten(), dtype=float)
    except:
        # series
        if min(df) < 0:
            df += min(df) * -1
        if max(df) > 0:
            df = df.values / max(df.values)
            df = pd.Series(df, dtype=float)
    return df


def create_col_mean_from_dfs(dfs, col):
    return pd.concat([df[col] for df in dfs], axis=1).mean(axis=1)


def run_4_tests(t1, t2, col1, col2):
    t_stat, t_pvalue = stats.ttest_ind(t1, t2)
    r_stat, r_pvalue = stats.pearsonr(t1, t2)
    s_stat, s_pvalue = stats.spearmanr(t1, t2)
    k_stat, k_pvalue = stats.kendalltau(t1, t2)
    return pd.DataFrame({f'{col1}_vs_{col2}_stats': [t_stat, r_stat, s_stat, k_stat],
                         f'{col1}_vs_{col2}_pvalue': [t_pvalue, r_pvalue, s_pvalue, k_pvalue]},
                        index=['ttest', 'pearson', 'spearman', 'kendalltau'])


def run_4_tests_on_list_of_dfs(dfs, first_col, second_col):
    t1 = set_data_for_statistical_tests(create_col_mean_from_dfs(dfs, first_col))
    t2 = set_data_for_statistical_tests(create_col_mean_from_dfs(dfs, second_col))
    return run_4_tests(t1, t2, first_col, second_col)


def create_one_metric_df_per_data_set(dfs, list_of_models_names):
    dfx_dfs = []
    permutation_feature_importance_dfs = []
    model_feature_importance_dfs = []
    shap_feature_importance_dfs = []
    random_feature_importance_dfs = []
    for model_name in list_of_models_names:
        t_dfs = dfs[model_name]
        dfx_dfs.append(set_data_for_statistical_tests(create_col_mean_from_dfs(t_dfs, 'dfx_feature_importance')))
        permutation_feature_importance_dfs.append(
            set_data_for_statistical_tests(create_col_mean_from_dfs(t_dfs, 'permutation_feature_importance')))
        model_feature_importance_dfs.append(
            set_data_for_statistical_tests(create_col_mean_from_dfs(t_dfs, 'model_feature_importance')))
        shap_feature_importance_dfs.append(
            set_data_for_statistical_tests(create_col_mean_from_dfs(t_dfs, 'shap_feature_importance')))
        random_feature_importance_dfs.append(
            set_data_for_statistical_tests(create_col_mean_from_dfs(t_dfs, 'random_feature_importance')))

    all_data = pd.DataFrame({'dfx_feature_importance_mean': list(pd.concat(dfx_dfs, axis=1).mean(axis=1)),
                             'permutation_feature_importance_mean': list(pd.concat(permutation_feature_importance_dfs, axis=1).mean(axis=1)),
                             'model_feature_importance_mean': list(pd.concat(model_feature_importance_dfs, axis=1).mean(axis=1)),
                             'shap_feature_importance_mean': list(pd.concat(shap_feature_importance_dfs, axis=1).mean(axis=1)),
                             'random_feature_importance_mean': list(pd.concat(random_feature_importance_dfs, axis=1).mean(axis=1))})
    return all_data


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def tabular_dnn(n_of_input_columns, n_of_output_classes, feature_selection_dropout=0.2,
                first_dense=256, second_dense=256, third_dense=256, dense_dropout=0.5,
                activation_type=gelu):

    inputs = Input(shape=(n_of_input_columns,))
    inputs_normalization = BatchNormalization()(inputs)
    inputs_feature_selection = Dropout(feature_selection_dropout)(inputs_normalization)

    x = Dense(first_dense, activation=activation_type)(inputs_feature_selection)
    x = Dropout(dense_dropout)(x)
    x = Dense(second_dense, activation=activation_type)(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(third_dense, activation=activation_type)(x)
    x = Dropout(dense_dropout)(x)
    output = Dense(n_of_output_classes, activation="sigmoid")(x)
    model = Model(inputs, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'accuracy'])
    return model


def tabular_cnn(n_of_input_columns, n_of_output_classes):
    inputs = Input(shape=(n_of_input_columns,))
    inputs_normalization = BatchNormalization()(inputs)
    inputs_feature_selection = Dropout(0.3)(inputs_normalization)
    x = Reshape((n_of_input_columns, 1))(inputs_feature_selection)
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=5, padding='same')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(n_of_output_classes, activation='softmax')(x)
    model = Model(inputs, output)
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=1000,
        decay_rate=0.8)
    optimizer = SGD(learning_rate=lr_schedule)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    return model


def clean_deep_shap_values(shap_values, input_shape):
    if isinstance(shap_values, list):
        return np.mean(shap_values, axis=0).reshape(input_shape)
    else:
        return shap_values.reshape(input_shape)


def set_deep_model_data(x_train, y_train):
    return x_train, to_categorical(y_train, len(set(y_train)))


def _add_selected_features_iteratively_train_and_predict(x_train, y_train, X_test, order_features, model_name=None, max_features=25):

    used_features = []
    features_in_group = []
    predictions = []
    for o in order_features[0: max_features]:
        used_features.append(x_train.columns[o])
        models_dict = {'LR': LogisticRegression(random_state=random.seed(seed)),
                       'DTC': DecisionTreeClassifier(random_state=random.seed(seed)),
                       'RF': RandomForestClassifier(n_estimators=50, random_state=random.seed(seed)),
                       'DNN': tabular_dnn(len(used_features), len(set(y_train))),
                       'CNN': tabular_cnn(len(used_features), len(set(y_train)))}
        model = models_dict[model_name]
        if model_name == 'CNN' or model_name == 'DNN':
            # CNN or DNN
            y_train_ = to_categorical(y_train, len(set(y_train)))
            model.fit(x_train[used_features], y_train_, epochs=30, batch_size=int(x_train.shape[0]/10))
            case_predictions = pd.DataFrame(model.predict(X_test[used_features])).astype(float)
        else:
            # other model types
            model.fit(x_train[used_features], y_train)
            case_predictions = pd.DataFrame(model.predict_proba(X_test[used_features]))
        predictions.append(case_predictions)
        features_in_group.append(used_features.copy())
        del model
        gc.collect()
    return predictions, features_in_group


def get_scores_by_adding_selected_features(
        x_train,
        y_train,
        x_test,
        y_test,
        order_features,
        data_set_name,
        model_name=None,
        method='shap',
        max_features=25,
        plot=True):
    plot_name = f'{plots_folder}/{method}/{data_set_name}/scores_by_adding_selected_features_on_model_{model_name}.png'
    title_name = f'{method} data={data_set_name} model={model_name}'
    predictions, features_in_group = _add_selected_features_iteratively_train_and_predict(
        x_train, y_train, x_test, order_features, model_name, max_features)
    scores = pd.DataFrame()
    scores['accuracy'] = pd.Series([np.round(accuracy_score(y_test, p.idxmax(axis=1)), 3) for p in predictions]).cummax()
    scores['mutual_info'] = pd.Series([np.round(
        mutual_info_classif(x_train[features], y_train, n_neighbors=len(features), random_state=seed).sum(), 3)
        for features in features_in_group]).cummax()
    scores.index.name = 'number of features'

    if plot:
        fig = px.line(scores, title=title_name)
        fig.update_traces(mode="markers+lines")
        fig.write_image(plot_name)
    return predictions, features_in_group


def set_results_folder(data_set_name):
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    if not os.path.exists(datasets_folder):
        os.mkdir(datasets_folder)

    if not os.path.exists(plots_folder):
        os.mkdir(plots_folder)

    if not os.path.exists(f'{datasets_folder}/{data_set_name}/'):
        os.mkdir(f'{datasets_folder}/{data_set_name}')

    if not os.path.exists(f'{plots_folder}/dfx'):
        os.mkdir(f'{plots_folder}/dfx')

    if not os.path.exists(f'{plots_folder}/shap'):
        os.mkdir(f'{plots_folder}/shap')

    if not os.path.exists(f'{plots_folder}/dfx/{data_set_name}'):
        os.mkdir(f'{plots_folder}/dfx/{data_set_name}')

    if not os.path.exists(f'{plots_folder}/shap/{data_set_name}'):
        os.mkdir(f'{plots_folder}/shap/{data_set_name}')


def reduce_multicollinearity(X_train, X_test, data_set_name, output:bool = False):
    vif = pd.DataFrame()
    vif["features"] = X_train.columns
    vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    if output:
        vif.to_csv(f'{datasets_folder}/{data_set_name}/VIF_of_features.csv', index=False)

    # remove Multicollinearity features from X
    new_columns = vif[vif['VIF'] < 10]['features']
    new_X_train = X_train[new_columns]
    new_X_test = X_test[new_columns]
    return new_X_train, new_X_test


def train_model_get_score_by_model_name(model, model_name, new_x_train, y_train, new_x_test, y_test):
    if model_name == 'CNN' or model_name == 'DNN':
        print()
        x_train_, y_train_ = set_deep_model_data(new_x_train, y_train)
        print(f'train deep model {model_name}')
        model.fit(x_train_, y_train_, epochs=30, batch_size=int(x_train_.shape[0] / 10))
        # evaluate the model
        x_train_, y_train_ = set_deep_model_data(new_x_test, y_test)
        scores = model.evaluate(x_train_, y_train_, verbose=0)
        print(f"finish fit model {model_name}")
        return model, scores[1]
    else:
        model.fit(new_x_train, y_train)
        score = model.score(new_x_test, y_test)
        print(f"finish fit model {model_name}")
        return model, score


def create_shap_values(model_name, model, x_train):
    shap_values = None
    if model_name in ['LR']:
        print()
        print(f"train LinearExplainer on {model_name}")
        explainer = shap.LinearExplainer(model, x_train)
        shap_values = explainer.shap_values(x_train)

    elif model_name in ['DTC', 'RF']:
        # size = min(X_train.shape[0], 2000)
        print()
        print(f"train TreeExplainer on {model_name}")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_train)

    elif model_name in ['DNN', 'CNN']:
        size = min(x_train.shape[0], 2500)
        print()
        print(f"train DeepExplainer on {model_name}")
        explainer = shap.DeepExplainer(model, x_train[:size])
        shap_values = explainer.shap_values(x_train.iloc[:size].values, check_additivity=False)
        shap_values = clean_deep_shap_values(shap_values, x_train.iloc[:size].shape)

    shap_values_as_df, shap_indices = shap_values_to_df(shap_values, list(x_train.columns))
    return shap_values_as_df, shap_indices


def create_all_feature_interactions_from_df(x_train, levels=3):
    return create_all_feature_interactions_from_list(x_train.columns, levels)


def create_all_feature_interactions_from_list(l, levels=3):
    return [list(l) for l in list(combinations(l, levels))]
