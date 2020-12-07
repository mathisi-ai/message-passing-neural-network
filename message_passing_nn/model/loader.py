from typing import Dict

import torch as to
from torch import nn

from message_passing_nn.utils.model_selector import load_model


class Loader:
    def __init__(self, model: str) -> None:
        self.model = load_model(model)

    def load_model(self, data_dimensions: dict, path_to_model: str) -> nn.Module:
        model_parameters = self._get_model_parameters_from_path(path_to_model)
        self.model = self.model(time_steps=int(model_parameters['time_steps']),
                                number_of_nodes=data_dimensions["number_of_nodes"],
                                number_of_node_features=data_dimensions["number_of_node_features"],
                                fully_connected_layer_input_size=data_dimensions["fully_connected_layer_input_size"],
                                fully_connected_layer_output_size=data_dimensions["fully_connected_layer_output_size"])
        self.model.load_state_dict(to.load(path_to_model))
        self.model.eval()
        return self.model

    @staticmethod
    def _get_model_parameters_from_path(path_to_model: str) -> Dict:
        model_configuration = path_to_model.split("/")[-2].split("__")
        model_parameters = {}
        for model_parameter in model_configuration:
            key, value = model_parameter.split("&")[0], model_parameter.split("&")[1]
            model_parameters.update({key: value})
        return model_parameters
