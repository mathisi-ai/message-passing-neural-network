import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import torch as to
from pandas import pandas as pd

from message_passing_nn.fixtures.filenames import *
from message_passing_nn.utils.logger import get_logger


class Saver:
    def __init__(self, model_directory: str, results_directory: str) -> None:
        self.model_directory = model_directory
        self.results_directory = results_directory

    def save_model(self, epoch: int, configuration_id: str, model: to.nn.Module) -> None:
        current_folder = os.path.join(self.model_directory, configuration_id)
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
        path_and_filename = os.path.join(current_folder, "_".join([EPOCH, str(epoch), MODEL_STATE_DICTIONARY]))
        to.save(model.state_dict(), path_and_filename)
        get_logger().info("Saved model checkpoint in " + path_and_filename)

    def save_results(self, results: Dict, configuration_id: str = '') -> None:
        current_folder = os.path.join(self.results_directory, configuration_id)
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
        results_dataframe = self._construct_dataframe_from_nested_dictionary(results)
        path_and_filename = os.path.join(current_folder,
                                         "_".join([datetime.now().strftime("%d-%b-%YT%H_%M"), RESULTS_CSV]))
        results_dataframe.to_csv(path_and_filename)
        get_logger().info("Saved results in " + path_and_filename)

    def save_distance_maps(self, distance_maps: List[Tuple], configuration_id: str = ''):
        current_folder = os.path.join(self.results_directory, configuration_id)
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
        path_and_filename = os.path.join(current_folder,
                                         "_".join([datetime.now().strftime("%d-%b-%YT%H_%M"), DISTANCE_MAPS]))
        with open(path_and_filename, 'wb') as file:
            pickle.dump(distance_maps, file)
        get_logger().info("Saved inference outputs in " + path_and_filename)

    @staticmethod
    def _construct_dataframe_from_nested_dictionary(results: Dict) -> pd.DataFrame:
        results_dataframe = pd.DataFrame.from_dict({(i, j): results[i][j]
                                                    for i in results.keys()
                                                    for j in results[i].keys()},
                                                   orient='index')
        return results_dataframe
