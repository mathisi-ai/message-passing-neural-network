from typing import Dict, Tuple

import numpy as np
import torch as to
from torch.utils.data.dataloader import DataLoader

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.loss_function.loss_function_selector import load_loss_function
from message_passing_nn.utils.loss_function.loss_function_wrapper import LossFunctionWrapper
from message_passing_nn.utils.model_selector import load_model
from message_passing_nn.utils.optimizer_selector import load_optimizer
from message_passing_nn.utils.postgres_connector import PostgresConnector


class Trainer:
    def __init__(self, preprocessor: DataPreprocessor, device: str,
                 postgres_connector: PostgresConnector = None) -> None:
        self.preprocessor = preprocessor
        self.device = device
        self.postgres_connector = postgres_connector
        self.model = None
        self.loss_function = None
        self.optimizer = None

    def build(self, data_dimensions: dict, configuration_dictionary: Dict) -> None:
        self._instantiate_the_model(configuration_dictionary, data_dimensions)
        self._instantiate_the_loss_function(configuration_dictionary)
        self._instantiate_the_optimizer(configuration_dictionary)

    def do_train_step(self, training_data: DataLoader, epoch: int) -> float:
        training_loss = []
        for data in training_data:
            node_features, all_neighbors, labels = self._send_to_device(data)
            self.optimizer.zero_grad()
            outputs = self.model(node_features, all_neighbors, batch_size=labels.shape[0])
            loss = self.loss_function(outputs, labels, node_features)
            training_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
        get_logger().info('[Iteration %d] training loss: %.6f' % (epoch, np.average(training_loss)))
        return np.average(training_loss)

    def do_evaluate_step(self, evaluation_data: DataLoader, epoch: int = None) -> float:
        with to.no_grad():
            evaluation_loss = []
            if len(evaluation_data):
                for data in evaluation_data:
                    node_features, all_neighbors, labels = self._send_to_device(data)
                    outputs = self.model(node_features, all_neighbors, batch_size=labels.shape[0])
                    loss = self.loss_function(outputs, labels, node_features)
                    evaluation_loss.append(loss.item())
                evaluation_loss = np.average(evaluation_loss)
                if epoch is not None:
                    get_logger().info('[Iteration %d] validation loss: %.6f' % (epoch, evaluation_loss))
                else:
                    get_logger().info('Test loss: %.6f' % evaluation_loss)
            else:
                get_logger().warning('No evaluation data found!')
        return evaluation_loss

    def _instantiate_the_model(self, configuration_dictionary: dict, data_dimensions: dict) -> None:
        self.model = load_model(configuration_dictionary['model'])
        self.model = self.model(time_steps=configuration_dictionary['time_steps'],
                                number_of_nodes=data_dimensions["number_of_nodes"],
                                number_of_node_features=data_dimensions["number_of_node_features"],
                                fully_connected_layer_input_size=data_dimensions["fully_connected_layer_input_size"],
                                fully_connected_layer_output_size=data_dimensions["fully_connected_layer_output_size"],
                                device=self.device)
        self.model.to(self.device)
        get_logger().info('Loaded the {} model on {}. Model size: {} MB'.format(configuration_dictionary['model'],
                                                                                self.device,
                                                                                self.model.get_model_size()))

    def _instantiate_the_loss_function(self, configuration: dict) -> None:
        loss_function, penalty = load_loss_function(configuration, self.postgres_connector)
        self.loss_function = LossFunctionWrapper(loss_function, penalty)
        get_logger().info('Loss function: {}'.format(configuration['loss_function']))

    def _instantiate_the_optimizer(self, configuration_dictionary: dict) -> None:
        optimizer = load_optimizer(configuration_dictionary['optimizer'])
        model_parameters = list(self.model.parameters())
        try:
            self.optimizer = optimizer(model_parameters, lr=0.001, momentum=0.9)
        except:
            self.optimizer = optimizer(model_parameters, lr=0.001)
        get_logger().info('Optimizer: {}'.format(configuration_dictionary['optimizer']))

    def _send_to_device(self, data: Tuple[to.Tensor, to.Tensor, to.Tensor, str]) \
            -> Tuple[to.Tensor, to.Tensor, to.Tensor]:
        node_features, all_neighbors, labels = (data[0].to(self.device),
                                                data[1].to(self.device),
                                                data[2].to(self.device))
        return node_features, all_neighbors, labels
