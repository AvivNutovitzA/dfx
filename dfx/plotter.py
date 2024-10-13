import pandas as pd
import numpy as np
import matplotlib.style
import matplotlib.pyplot as plt
matplotlib.style.use('classic')


class Plotter():
    def __init__(self, x_train, plot_top=20):
        # condition on x_data
        if isinstance(x_train, pd.DataFrame):
            self.x_data = x_train.values
            self.feature_names = list(x_train.columns)
        else:
            raise ValueError(f"x_train can by pandas DataFrame ONLY, "f"but passed {type(x_train)}")
        assert len(self.feature_names) == x_train.shape[1]
        self.X_train = x_train
        self.plot_top = plot_top

    def _set_number_of_features(self, contributions=None):
        if contributions is not None:
            if contributions.shape[0] >= self.plot_top:
                return self.plot_top
            else:
                return contributions.shape[0]
        else:
            if self.plot_top >= self.X_train.shape[1]:
                return self.X_train.shape[1]
            else:
                return self.plot_top

    # feature importance
    def plot_model_coef(self, model):
        plt.figure()
        number_of_features = self._set_number_of_features()
        indices = np.argsort(model.coef_[0])[0: number_of_features]
        features_to_show = self.feature_names
        plt.title("Model Feature importance/coefficient")
        plt.barh(range(number_of_features), model.coef_[0][indices], color="b", align="center")
        plt.yticks(range(number_of_features), np.array(features_to_show)[indices.astype(int)])
        plt.ylim([-1, number_of_features])
        plt.show()

    def plot_model_importance(self, model):
        plt.figure()
        number_of_features = self._set_number_of_features()
        indices = np.argsort(model.feature_importances_)[0: number_of_features]
        features_to_show = self.feature_names
        plt.title("Model Feature importance/coefficient")
        plt.barh(range(number_of_features), model.feature_importances_[indices], color="b", align="center")
        plt.yticks(range(number_of_features), np.array(features_to_show)[indices.astype(int)])
        plt.ylim([-1, number_of_features])
        plt.show()

    # doe contribution
    def plot_doe_feature_contribution(self, class_feature_contributions,  color='b'):
        plt.figure()
        contributions = np.array([np.abs(values) for key, values in class_feature_contributions.items()])
        number_of_features = self._set_number_of_features(contributions)
        indices = np.argsort(contributions)[0: number_of_features]
        features_to_show = list(class_feature_contributions.keys())
        plt.title(f"DFX Feature Contribution")
        plt.barh(range(number_of_features), contributions[indices], color=color, align="center")
        plt.yticks(range(number_of_features), np.array(features_to_show)[indices.astype(int)])
        plt.ylim([-1, number_of_features])
        plt.show()
