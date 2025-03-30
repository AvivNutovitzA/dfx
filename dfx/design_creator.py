import os
import random
import numpy as np
import pandas as pd


random.seed = 42


def get_base():
    try:
        return pd.read_csv(os.path.join(os.getcwd(), '../../..', 'dfx/resources/pb36.csv'))  # header=None
    except:
        return pd.read_csv(os.path.join(os.getcwd(), 'dfx/resources/pb36.csv'))  # header=None


class DesignCreator:
    # base_36_file = 'resources/base_36.csv'

    def __init__(self, feature_matrix, file_name=None):
        if file_name:
            self.design_df = pd.read_csv(file_name)
            self.design_df.columns = self.design_df.columns.astype(int)
        else:
            self.create_design_from_feature_matrix(feature_matrix)
        self.base_n = -1

    def get_lists_of_design_from_df_for_tabluar_data(self, feature_means):
        lists_of_designs = []
        list_of_all_positions_per_design = []
        for row in self.design_df.iterrows():
            list_of_design = []
            list_of_positions_per_design = []
            for position in list(row[1].index):
                # position_val = int(str(position).split('V')[1])-1
                if row[1][position] == -1:
                    list_of_design.append(feature_means[position])
                    list_of_positions_per_design.append(position)
            lists_of_designs.append(list_of_design)
            list_of_all_positions_per_design.append(list_of_positions_per_design)
        return lists_of_designs, list_of_all_positions_per_design

    def get_lists_of_design_from_df_for_images(self):
        lists_of_designs = []
        for row in self.design_df.iterrows():
            list_of_design = []
            for position in list(row[1].index):
                # position_val = int(str(position).split('V')[1])-1
                if row[1][position] == -1:
                    list_of_design.append((position + 1) * 3)
                    list_of_design.append((position + 1) * 3 - 1)
                    list_of_design.append((position + 1) * 3 - 2)
            lists_of_designs.append(list_of_design)
        return lists_of_designs

    @staticmethod
    def clean_row(row):
        return [1 if char == '+' else -1 for char in [char for char in row]]

    def _create_base_design(self):
        # self.design_df = pd.DataFrame.from_records(base[0].apply(lambda x: self.clean_row(x)))
        self.design_df = get_base()
        self.base_n = self.design_df.shape[1]

    def create_design_from_feature_matrix(self, feature_matrix):
        self.n_features = feature_matrix.shape[1]
        self._create_base_design()
        self._doubling(feature_matrix)
        self._select_matrix_columns(feature_matrix)

    def _doubling(self, feature_matrix):
        while self.design_df.shape[1] < feature_matrix.shape[1] + 1:
            self.design_df = self._doubling_step()

    def _doubling_step(self):
        m1 = np.c_[self.design_df.values, -self.design_df.values]
        m2 = np.c_[self.design_df.values, self.design_df.values]
        res = np.r_[m1, m2]
        del m1, m2
        return pd.DataFrame(res)

    def _select_matrix_columns(self, feature_matrix):
        np.random.seed(42)
        self.design_df = self.design_df[[i for i in list(self.design_df.columns) if int(i) > 0]]
        if self.n_features < self.base_n and self.base_n > -1:
            selected_columns = [col for col in self.design_df.columns if int(col) < self.n_features + 1]
        else:
            selected_columns = np.random.choice(
                list(self.design_df.columns), size=feature_matrix.shape[1], replace=False)
        self.design_df = self.design_df[selected_columns]
        self.design_df.columns = list(range(0, self.design_df.shape[1]))
        selected_columns = [c1 for c1, p1 in
                            zip(self.design_df.columns, list(np.random.uniform(size=len(self.design_df.columns)))) if
                            p1 >= 0.5]
        self.design_df[selected_columns] = self.design_df[selected_columns] * -1

    # -- for image data not in use now --
    # -----------------------------------
    def generate_image_design_block(self):
        arr_64 = list(self.design_df.columns)
        arr_1024 = []
        arr_1024_column_names = []
        index = 0
        index_64 = 0
        index_1024 = 0
        n = 8
        m = 4
        for l in range(0, n):
            for k in range(0, m):
                index_64 = index
                for j in range(0, n):
                    for i in range(0, m):
                        v = arr_64[index_64]
                        index_1024 += 1
                        arr_1024.append(pd.DataFrame(self.design_df[v]))
                        arr_1024_column_names.append('V{}'.format(index_1024))
                    index_64 += 1
            index += n
        df_1 = pd.concat(arr_1024, axis=1)
        df_1.columns = arr_1024_column_names
        return df_1
