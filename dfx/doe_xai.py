import warnings
import itertools
from typing import Union, Optional, List, Any, Tuple, Dict
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from dfx.design_creator import DesignCreator
from dfx.data_modifier import DataModifier
from dfx.predictor import Predictor
from dfx_experiments.utils import *


class DoeXai:

    def __init__(self,
                 x_data: Union[pd.DataFrame, csr_matrix, np.ndarray],
                 y_data: Union[pd.DataFrame, pd.Series, np.ndarray],
                 model: Any,
                 is_keras_nn_model: bool = False,
                 verbose: int = 0,
                 feature_names: Optional[List[str]] = None,
                 design_file_name: Optional[str] = None,
                 ):
        """

        :param x_data: original data used for explanation, data from the training
        :param y_data: original target used for explanation, target from the training
        :param model: any ML model that we would like to explain, the model must support "predict" method if it is
         a keras neural net model or "predict_proba" if it is a skleran model
        :param feature_names: Optional, in case that X is a matrix and not a dataframe DFX support insert the list of
        feature names we would like to explain
        :param design_file_name: Optional, DFX uses the base_36.csv design in the resources folder
        :param verbose: show stages of not
        :param is_keras_nn_model: for CNN and DNN models of keras set the value to be true
        """
        # condition on x_data
        if isinstance(x_data, pd.DataFrame):
            self.x_original_data = x_data.values
            self.feature_names = list(x_data.columns)
        elif isinstance(x_data, csr_matrix):
            self.x_original_data = x_data.toarray()
            if feature_names:
                self.feature_names = feature_names
            else:
                raise Exception("Must pass feature_names if x_data is csr_matrix")
        elif isinstance(x_data, np.ndarray):
            self.x_original_data = x_data
            if feature_names:
                self.feature_names = feature_names
            else:
                raise Exception("Must pass feature_names if x_data is np.ndarray")
        else:
            raise ValueError(f"x_data can by pandas DataFrame or numpy ndarray or scipy.sparse csr_matrix ONLY, "
                             f"but passed {type(x_data)}")
        # condition on y_data
        if isinstance(y_data, pd.DataFrame):
            self.y_original_data = y_data.values
        elif isinstance(y_data, np.ndarray):
            self.y_original_data = y_data
        elif isinstance(y_data, pd.Series):
            self.y_original_data = y_data.reset_index(drop=True)
        else:
            raise ValueError(f"y_data can by pandas DataFrame or Series or numpy ndarray ONLY, but passed {type(y_data)}")
        self.model = model
        if design_file_name:
            self.dc = DesignCreator(feature_matrix=None, file_name=design_file_name)
        else:
            self.dc = DesignCreator(feature_matrix=self.x_original_data)
        self.verbose = verbose
        if is_keras_nn_model:
            predict_method = getattr(model, "predict", None)
            if not callable(predict_method):
                raise ValueError("the used model has no method predict")
        else:
            predict_proba_method = getattr(model, "predict_proba", None)
            if not callable(predict_proba_method):
                raise ValueError("the used model has no method predict_proba")
        self.is_keras_nn_model = is_keras_nn_model

        reference_values = [row.mean() for row in self.x_original_data.T]
        lists_of_designs, list_of_all_positions_per_design = self.dc.get_lists_of_design_from_df_for_tabluar_data(
            reference_values)

        dm = DataModifier(self.x_original_data, lists_of_designs, list_of_all_positions_per_design, len(reference_values))
        self.zeds_df, data_modified_list = dm.set_tabular_data_for_prediction()

        p = Predictor(data_modified_list, self.y_original_data, self.model, self.is_keras_nn_model)
        self.all_predictions_all_targets, self.all_predictions_df = p.create_tabular_gs_df()

    def _contribution_base(self, user_list: List[str] = None, only_orig_features: bool = False) -> Tuple[Any, List[str]]:
        y = self.all_predictions_df.mean(axis=1)
        orig_x, x = self._get_x_for_feature_contribution(user_list, only_orig_features)
        x_ = self.find_interactions(orig_x, x, y, only_orig_features)
        return self._fit_linear_approximation(x_, y)

    def find_feature_contribution(self, user_list: List[str] = None, only_orig_features: bool = False) \
            -> Dict[str, float]:
        m, selected_features_x = self._contribution_base(user_list, only_orig_features)
        return self._create_contribution(m, selected_features_x)

    def find_feature_contribution_sign(self, user_list: List[str] = None, only_orig_features: bool = False) \
            -> Dict[str, int]:
        m, selected_features_x = self._contribution_base(user_list, only_orig_features)
        return self._create_contributions_sign(m, selected_features_x)

    @staticmethod
    def _create_contribution(m: Any, selected_features_x: List[str]) -> Dict[str, float]:
        contributions = {}
        for index, col in enumerate(selected_features_x):
            contributions[col] = np.abs(m.coef_[index])
        return contributions

    @staticmethod
    def _create_contributions_sign(m: Any, selected_features_x: List[str]) -> Dict[str, int]:
        contributions_sign = {}
        for index, col in enumerate(selected_features_x):
            contributions_sign[col] = 1 if m.coef_[index] >= 0 else -1
        return contributions_sign

    def _fit_linear_approximation(self, x: pd.DataFrame, y: pd.Series, run_fffs: bool = False) -> Tuple[Any, List[str]]:
        selected_features_x = list(x.columns)
        if run_fffs:
            feature_selector = SFS(
                LinearRegression(normalize=True),
                k_features=max(int(np.sqrt(x.shape[1])), self.zeds_df.shape[1]),
                forward=True,
                verbose=2,
                cv=5,
                n_jobs=-1,
                scoring='r2')

            features = feature_selector.fit(x, y)
            selected_columns = list(features.k_feature_names_)
            selected_columns.extend([list(x.columns)[i] for i in list(self.zeds_df.columns.astype(int))])
            selected_features_x = pd.DataFrame(x)[set(selected_columns)]
            m = self.get_best_linear_model(selected_features_x, y)
            m.fit(selected_features_x, y)
        else:
            m = self.get_best_linear_model(x, y)
        m.fit(x, y)
        return m, selected_features_x

    def _get_x_for_feature_contribution(self, user_list: List[str] = None, only_orig_features: bool = False) -> \
            Tuple[pd.DataFrame, pd.DataFrame]:
        x = self.zeds_df.copy()
        try:
            x.columns = self.feature_names
        except:
            pass
        orig_x = x.copy(deep=True)

        if only_orig_features:
            return orig_x, x

        if user_list:
            for new_feature in user_list:
                feature_name = str(new_feature[0])
                feature_value = x[new_feature[0]]
                for index, elements in enumerate(new_feature):
                    if index > 0:
                        feature_name += '_|_' + str(new_feature[index])
                        feature_value = feature_value * x[new_feature[index]]
                x[feature_name] = feature_value
        else:
            list_of_columns_pairs = list(itertools.combinations(x.columns, 2))
            for pair in list_of_columns_pairs:
                new_feature = str(pair[0]) + '_|_' + str(pair[1])
                x[new_feature] = x[pair[0]] * x[pair[1]]

        x.columns = x.columns.astype(str)
        return orig_x, x

    def output_process_files(self, output_files_prefix: str):
        # self.df_output.to_csv(f'{output_files_prefix}_feature_contributions.csv', index=False)
        self.zeds_df.to_csv(f'{output_files_prefix}_zeds_df.csv', index=False)
        self.all_predictions_df.to_csv(f'{output_files_prefix}_gs_df.csv', index=False)

    def explain_instance(self, instance_index: int, user_list=None, only_orig_features=True):
        """
        the function set a local explanation at a specific index of the training data after the response vector part
        was completed
        :param instance_index:
        :param user_list:
        :param only_orig_features:
        :return:
        """
        y = self.all_predictions_df.iloc[:, instance_index]
        orig_x, x = self._get_x_for_feature_contribution(user_list, only_orig_features)
        m, selected_features_x = self._fit_linear_approximation(x, y)
        return self._create_contribution(m, selected_features_x)

    @staticmethod
    def get_best_linear_model(x: pd.DataFrame, y: pd.Series) -> Any:
        model = ElasticNet(max_iter=1000, random_state=42)
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
        # define grid
        grid = dict()
        grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0]
        grid['l1_ratio'] = [(i+1)/100 for i in range(0, 10)]
        # define search
        search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, verbose=0)
        # perform the search
        with warnings.catch_warnings():
            results = search.fit(x, y)
        if max(results.best_estimator_.coef_) == 0:
            return LinearRegression()
        else:
            return results.best_estimator_

    @staticmethod
    def find_interactions(orig_x: pd.DataFrame, x: pd.DataFrame, y: pd.Series, only_orig_features: bool,
                          coef_filter: float = 0.85) -> pd.DataFrame:
        if only_orig_features:
            return orig_x
        selected_features = []
        pos = orig_x.shape[1]
        for col in x.iloc[:, pos:].columns:
            t = pd.concat([orig_x.copy(), x[[col]]], axis=1)
            m = Ridge(alpha=0.5, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.fit(t, y)
                if np.abs(m.coef_[pos]) >= np.quantile(np.abs(m.coef_), coef_filter):
                    selected_features.append(col)
        return pd.concat([orig_x, x[selected_features]], axis=1) if selected_features else orig_x

