from typing import Dict, Any, Tuple

import numpy as np
import torch as to
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from message_passing_nn.fixtures.amino_acid_maps import map_amino_acid_codes, amino_acid_to_index
from message_passing_nn.data.preprocessor import Preprocessor
from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.loss_function_selector import LossFunctionSelector
from message_passing_nn.utils.loss_with_penalty import LossWithPenalty
from message_passing_nn.utils.model_selector import ModelSelector
from message_passing_nn.utils.optimizer_selector import OptimizerSelector
from message_passing_nn.utils.postgres_connector import PostgresConnector


class Trainer:
    def __init__(self, preprocessor: Preprocessor, device: str, postgres_connector: PostgresConnector = None) -> None:
        self.preprocessor = preprocessor
        self.device = device
        self.postgres_connector = postgres_connector
        self.model = None
        self.loss_function = None
        self.optimizer = None

    def instantiate_attributes(self, data_dimensions: Tuple, configuration_dictionary: Dict) -> None:
        node_features_size, labels_size = data_dimensions
        number_of_nodes = node_features_size[0]
        number_of_node_features = node_features_size[1]
        fully_connected_layer_output_size = labels_size[0]
        self.model = ModelSelector.load_model(configuration_dictionary['model'])
        self.model = self.model(time_steps=configuration_dictionary['time_steps'],
                                number_of_nodes=number_of_nodes,
                                number_of_node_features=number_of_node_features,
                                fully_connected_layer_input_size=number_of_nodes * number_of_node_features,
                                fully_connected_layer_output_size=fully_connected_layer_output_size,
                                device=self.device)
        get_logger().info('Loaded the ' + configuration_dictionary['model'] +
                          ' model. Model weights size: ' + self.model.get_model_size() + ' MB')
        self.model.to(self.device)
        penalty = True if self.postgres_connector else False
        self.loss_function = self._instantiate_the_loss_function(configuration_dictionary, penalty=penalty)
        get_logger().info('Loss function: ' + configuration_dictionary['loss_function'])
        self.optimizer = self._instantiate_the_optimizer(
            OptimizerSelector.load_optimizer(configuration_dictionary['optimizer']))
        get_logger().info('Optimizer: ' + configuration_dictionary['optimizer'])

    def do_train(self, training_data: DataLoader, epoch: int) -> float:
        training_loss = np.average(list(map(self._do_train_batch, training_data)))
        get_logger().info('[Iteration %d] training loss: %.6f' % (epoch, training_loss))
        return training_loss

    def _do_train_batch(self, training_data: DataLoader) -> float:
        node_features, all_neighbors, labels, _ = training_data
        node_features, all_neighbors, labels = (node_features.to(self.device),
                                                all_neighbors.to(self.device),
                                                labels.to(self.device))
        current_batch_size = self._get_current_batch_size(labels)
        self.optimizer.zero_grad()
        outputs = self.model(node_features, all_neighbors, batch_size=current_batch_size)
        loss = self.loss_function.calculate_loss(node_features, outputs, labels)
        self._do_backpropagate(loss)
        return loss.item()

    def do_evaluate(self, evaluation_data: DataLoader, epoch: int = None) -> float:
        with to.no_grad():
            evaluation_loss = []
            if len(evaluation_data):
                for node_features, all_neighbors, labels_validation, _ in evaluation_data:
                    node_features, all_neighbors, labels_validation = (node_features.to(self.device),
                                                                       all_neighbors.to(self.device),
                                                                       labels_validation.to(self.device))
                    current_batch_size = self._get_current_batch_size(labels_validation)
                    outputs = self.model(node_features, all_neighbors, current_batch_size)
                    loss = self.loss_function.calculate_loss(node_features, outputs, labels_validation)
                    evaluation_loss.append(float(loss))
                evaluation_loss = np.average(evaluation_loss)
                if epoch is not None:
                    get_logger().info('[Iteration %d] validation loss: %.6f' % (epoch, evaluation_loss))
                else:
                    get_logger().info('Test loss: %.6f' % evaluation_loss)
            else:
                get_logger().warning('No evaluation data found!')
        return evaluation_loss

    def _do_backpropagate(self, loss: to.Tensor) -> None:
        loss.backward()
        self.optimizer.step()

    def _instantiate_the_loss_function(self, configuration_dictionary: dict, penalty: bool = False):
        if penalty:
            penalty_dictionary = {}
            self.postgres_connector.open_connection()
            penalty = self.postgres_connector.execute_query(fields=["residue", "penalty"],
                                                            use_case='penalty')
            self.postgres_connector.close_connection()
            for residue, penalty in penalty:
                residue_index = amino_acid_to_index[map_amino_acid_codes[residue]]
                penalty_dictionary.update({residue_index: np.array(penalty)})
            return LossWithPenalty(LossFunctionSelector.load_loss_function(configuration_dictionary['loss_function'])(),
                                   penalty=penalty_dictionary,
                                   scaling_factor=configuration_dictionary['scaling_factor'],
                                   penalty_decimals=configuration_dictionary['penalty_decimals'],
                                   batch_size=configuration_dictionary['batch_size'])
        else:
            return LossFunctionSelector.load_loss_function(configuration_dictionary['loss_function'])

    def _instantiate_the_optimizer(self, optimizer: Any) -> Optimizer:
        model_parameters = list(self.model.parameters())
        try:
            optimizer = optimizer(model_parameters, lr=0.001, momentum=0.9)
        except:
            optimizer = optimizer(model_parameters, lr=0.001)
        return optimizer

    @staticmethod
    def _get_current_batch_size(features: to.Tensor) -> int:
        return len(features)
