from typing import Tuple, List

import numpy as np
import pandas as pd


class DataModifier:
    def __init__(self, orig_data: pd.DataFrame, list_of_designs: List[pd.DataFrame],
                 list_of_all_positions_per_design: List[pd.DataFrame], number_of_features: int):
        self.orig_data = orig_data
        self.list_of_designs = list_of_designs
        self.list_of_all_positions_per_design = list_of_all_positions_per_design
        self.number_of_features = number_of_features

    def set_tabular_data_for_prediction(self) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        zeds = []
        data_modified_list = []
        for designs, positions_per_design in zip(self.list_of_designs, self.list_of_all_positions_per_design):
            train_df_copy = pd.DataFrame(self.orig_data.copy())
            zed = [1] * self.number_of_features
            for selected_index, reference_value in zip(positions_per_design, designs):
                train_df_copy[selected_index] = reference_value
                zed[selected_index] = -1
            zeds.append(zed)
            data_modified_list.append(train_df_copy)
        return pd.DataFrame.from_records(zeds), data_modified_list

    # -- for image data not in use now --
    # -----------------------------------
    def set_image_data_for_prediction(self, image: np.array) -> Tuple[pd.DataFrame, List[np.array]]:
        # working on a single image one at a time
        zeds = []
        data_modified_list = []
        for design, positions_per_design in zip(self.list_of_designs):
            image_copy = image.copy()
            zed = [1] * len(design)
            # set the design on the new image
            new_image = self.set_all_reference_values_per_image(image_copy, design)
            for z in design:
                zed[z - 1] = 0
            zeds.append(zed)
            data_modified_list.append(new_image)
        return pd.DataFrame.from_records(zeds), data_modified_list

    @staticmethod
    def find_image_blocks(image: np.array, window_r: int = 8, window_c: int = 8) -> np.array:
        blocks = []
        for index_r, r in enumerate(range(0, image.shape[0] - window_r + 1, window_r)):
            for index_c, c in enumerate(range(0, image.shape[1] - window_c + 1, window_c)):
                blocks.append(image[r:r + window_r, c:c + window_c])
        return np.array(blocks)

    @staticmethod
    def find_block_means(blocks: np.array) -> np.array:
        return [int(np.mean(block)) for block in blocks]

    @staticmethod
    def find_block_for_reference_position(image: np.array, reference_position: int, window_r: int = 8,
                                          window_c: int = 8) -> np.array:
        block_indexes = []
        for val_r in (range(image.shape[0])):
            for val_c in (range(image.shape[1])):
                for val_d in range(image.shape[2]):
                    block_indexes.append((val_c // window_c) + (val_r // window_r) * 4)
        return block_indexes[reference_position]

    @staticmethod
    def image_set_reference_values_by_positions(im: np.array, reference_value: int, reference_position: int) \
            -> np.array:
        image_copy_1 = im.copy().ravel()
        image_copy_1[reference_position] = reference_value
        return image_copy_1.reshape(im.shape)

    def set_all_reference_values_per_image(self, image: np.array, reference_positions: List[int], block: bool = True) \
            -> np.array:
        if not block:
            image_mean = int(np.mean(image))
            for reference_position in reference_positions:
                image = self.image_set_reference_values_by_positions(image, image_mean, reference_position - 1)
        else:
            block_means = self.find_block_means(self.find_image_blocks(image))
            block_reference_positions = [self.find_block_for_reference_position(image, reference_position - 1) for
                                        reference_position in reference_positions]
            for reference_position, block_reference_position in zip(reference_positions, block_reference_positions):
                image = self.image_set_reference_values_by_positions(image, block_means[block_reference_position - 1],
                                                                     reference_position - 1)
            return image
