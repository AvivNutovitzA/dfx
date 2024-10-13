# --- Imports
from dfx_experiments.utils import load_data

# --- Other imports
import string
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
seed = 42


def create_fake_job_posting_data_and_tv(size=-1):
    # thanks to - https://www.kaggle.com/madz2000/text-classification-using-keras-nb-97-accuracy
    # for the prepossess and model training

    # --- helper functions
    def get_simple_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_words(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stop:
                pos = pos_tag([i.strip()])
                word = lemmatizer.lemmatize(i.strip(), get_simple_pos(pos[0][1]))
                final_text.append(word.lower())
        return " ".join(final_text)

    X, y = load_data('fake_job_posting', size=size)

    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    lemmatizer = WordNetLemmatizer()
    X = X.apply(lemmatize_words)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    tv = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3), max_features=100)
    tv_train_reviews = tv.fit_transform(X_train)
    tv_test_reviews = tv.transform(X_test)

    train_clean_data = pd.DataFrame(tv_train_reviews.toarray(), columns=tv.get_feature_names())
    test_clean_data = pd.DataFrame(tv_test_reviews.toarray(), columns=tv.get_feature_names())
    return train_clean_data, y_train, test_clean_data, y_test


def create_hotel_booking_data(size=-1):
    # thanks to - https://www.kaggle.com/marcuswingen/eda-of-bookings-and-ml-to-predict-cancelations
    # for the prepossess and model training

    # --- helper functions
    def get_clean_enc_cat_features(enc_cat_f, cat_f):
        rv = []
        for enc_cat in enc_cat_f:
            idx = enc_cat.split('_')[0].split('x')[1]
            feature_value = enc_cat.split('_')[1]
            rv.append((cat_f[int(idx)] + '_' + feature_value))
        return rv

    X, y = load_data('hotel_bookings', size=size)

    num_features = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
                    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
                    "babies", "is_repeated_guest", "previous_cancellations",
                    "previous_bookings_not_canceled", "agent", "company",
                    "required_car_parking_spaces", "total_of_special_requests", "adr"]

    cat_features = ["hotel", "arrival_date_month", "meal", "market_segment",
                    "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"]
    features = num_features + cat_features
    X = X[features]
    # Separate features and predicted value
    features = num_features + cat_features
    num_transformer = SimpleImputer(strategy="constant")
    # Preprocessing for categorical features:
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))])
    # Bundle preprocessing for numerical and categorical features:
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                                   ("cat", cat_transformer, cat_features)], remainder='drop')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_train = preprocessor.fit_transform(X_train)
    enc_cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names()
    clean_enc_cat_features = get_clean_enc_cat_features(enc_cat_features, cat_features)
    labels = np.concatenate([num_features, clean_enc_cat_features])
    X_train = pd.DataFrame(X_train, columns=labels)
    X_test = pd.DataFrame(preprocessor.transform(X_test), columns=labels)

    return X_train, y_train, X_test, y_test


def create_wine_data(size=-1):
    X, y = load_data('wine', size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_hr_employee_attrition_data(size=-1):
    # thanks to -  https://www.kaggle.com/arthurtok/employee-attrition-via-ensemble-tree-based-methods
    # for the prepossess and model training

    X, y = load_data('hr_employee_attrition', size)

    categorical = []
    for col, value in X.iteritems():
        if value.dtype == 'object':
            categorical.append(col)

    # Store the numerical columns in a list numerical
    numerical = X.columns.difference(categorical)

    x_cat = X[categorical]
    x_cat = pd.get_dummies(x_cat)
    x_num = X[numerical]
    x_final = pd.concat([x_num, x_cat], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(x_final, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_nomao_data(size=-1):
    X, y = load_data('nomao', size)
    clean_y = np.array([int(yy) - 1 for yy in y])

    X_train, X_test, y_train, y_test = train_test_split(X, clean_y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_placement_full_class_data(size=-1):
    # thanks to - https://www.kaggle.com/vinayshaw/will-you-get-a-job-or-not-eda-prediction
    # for the prepossess and model training

    X, y = load_data('placement_full_class', size)

    X["gender"] = X.gender.map({"M": 0, "F": 1})
    X["ssc_b"] = X.ssc_b.map({"Others": 0, "Central": 1})
    X["hsc_b"] = X.hsc_b.map({"Others": 0, "Central": 1})
    X["hsc_s"] = X.hsc_s.map({"Commerce": 0, "Science": 1, "Arts": 2})
    X["degree_t"] = X.degree_t.map({"Comm&Mgmt": 0, "Sci&Tech": 1, "Others": 2})
    X["workex"] = X.workex.map({"No": 0, "Yes": 1})
    X["specialisation"] = X.specialisation.map({"Mkt&HR": 0, "Mkt&Fin": 1})

    y = y.map({"Not Placed": 0, "Placed": 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_rain_weather_aus_data(size=-1):
    # thanks to - https://www.kaggle.com/aninditapani/will-it-rain-tomorrow
    # for the prepossess and model training

    X, y = load_data('rain_weather_aus', size)

    X['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    y = y.map({"No": 0, "Yes": 1})

    # See unique values and convert them to int using pd.getDummies()
    categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']

    # transform the categorical columns
    X = pd.get_dummies(X, columns=categorical_columns)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_cervical_cancer_data(size=-1):
    X, y = load_data('cervical_cancer', size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_glass_data(size=-1):
    X, y = load_data('glass', size)
    # mapping y {1,2,3} -> {0, 1, 2}, {5, 6 , 7} -> {3, 4, 5}
    y = y.apply(lambda x: x-1 if x < 5 else x-2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_mobile_price_data(size=-1):
    X, y = load_data('mobile_price', size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_clinvar_conflicting_data(size=-1):
    # thanks to - https://www.kaggle.com/alanabd/genetic-variant-classifications-ile-ml
    # for the prepossess and model training
    X, y = load_data('clinvar_conflicting', size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_heart_failure_clinical_data(size=-1):
    # thanks to - https://www.kaggle.com/bhanuprakash06/genetic-conflict-classification
    # for the prepossess and model training
    X, y = load_data('heart_failure_clinical', size)
    scaler = StandardScaler()
    columns = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return pd.DataFrame(X_train, columns=columns), y_train, pd.DataFrame(X_test, columns=columns), y_test


def create_churn_modelling_data(size=-1):
    # thanks to - https://www.kaggle.com/sankha1998/churn-prediction-deep-learning-perceptron
    # for the prepossess and model training
    X, y = load_data('churn_modelling', size)
    columns = list(X.columns)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return pd.DataFrame(X_train, columns=columns), y_train, pd.DataFrame(X_test, columns=columns), y_test


def create_hr_leaving_data(size=-1):
    # thanks to - https://www.kaggle.com/kanishkkavdia/hr-leaving-logistic-regression
    # for the prepossess and model training
    X, y = load_data('hr_leaving', size)
    X["Department"] = LabelEncoder().fit_transform(X["Department"])
    X["salary"] = LabelEncoder().fit_transform(X["salary"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_bank_churners_data(size=-1):
    # thanks to - https://www.kaggle.com/kanishkkavdia/hr-leaving-logistic-regression
    # for the prepossess and model training
    X, y = load_data('bank_churners', size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_fetal_health_data(size=-1):
    # thanks to - https://www.kaggle.com/karnikakapoor/fetal-health-classification
    # for the prepossess and model training
    X, y = load_data('fetal_health', size)
    col_names = list(X.columns)
    X_df = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X_df, columns=col_names)
    y = y-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_stroke_data(size=-1):
    # thanks to - https://www.kaggle.com/karnikakapoor/fetal-health-classification
    # for the prepossess and model training
    X, y = load_data('stroke', size)
    col_names = list(X.columns)
    X_df = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X_df, columns=col_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_company_bankruptcy_prediction_data(size=-1):
    X, y = load_data('company_bankruptcy_prediction', size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_airline_passenger_satisfaction_data(size=-1):
    # thanks to -https://www.kaggle.com/fedniko/fedorov-airline-satisfaction
    # for the prepossess and model training
    X, y = load_data('airline_passenger_satisfaction', size)
    X = X.fillna(0)

    def one_hot_encode(db, feature):
        one_hot = pd.get_dummies(db[feature], prefix=feature)
        db = db.drop(feature, axis=1)
        return db.join(one_hot)

    def transform(x):
        db = x.copy()
        db.pop('id')
        if 'satisfaction' in x:
            db.pop('satisfaction')
        db = one_hot_encode(db, 'Gender')
        db = one_hot_encode(db, 'Customer Type')
        db = one_hot_encode(db, 'Type of Travel')
        db = one_hot_encode(db, 'Class')
        return db

    X = transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test


def create_banking_marketing_targets_data(size=7000):
    # thanks to -https://www.kaggle.com/fedniko/fedorov-airline-satisfaction
    # for the prepossess and model training
    X, y = load_data('banking_marketing_targets', size)
    X = X.fillna(0)
    ohe_cols = list(X.select_dtypes(include='object').columns.values)
    X = pd.get_dummies(X, prefix=ohe_cols, columns=ohe_cols, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, y_train, X_test, y_test
