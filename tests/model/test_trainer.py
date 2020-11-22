from unittest import TestCase

import torch as to
from message_passing_nn.infrastructure.graph_dataset import GraphDataset

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model.trainer import Trainer
from tests.fixtures.postgres_variables import TEST_DATASET
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, NEIGHBORS_SERIALIZED, \
    FEATURES_SERIALIZED, LABELS_SERIALIZED
from message_passing_nn.utils.postgres_connector import PostgresConnector


class TestTrainer(TestCase):
    def setUp(self) -> None:
        time_steps = 1
        loss_function = "MSE"
        optimizer = "SGD"
        model = "RNN"
        device = "cpu"
        self.configuration_dictionary = {"model": model,
                                         "loss_function": loss_function,
                                         "optimizer": optimizer,
                                         "time_steps": time_steps}
        self.postgres_connector = PostgresConnector()
        data_preprocessor = DataPreprocessor()
        self.model_trainer = Trainer(data_preprocessor, device)

    def test_instantiate_attributes(self):
        # Given
        number_of_nodes = BASE_GRAPH.size()[0]
        number_of_node_features = BASE_GRAPH_NODE_FEATURES.size()[1]
        data_dimensions = (BASE_GRAPH_NODE_FEATURES.size(), BASE_GRAPH.view(-1).size())

        # When
        self.model_trainer.instantiate_attributes(data_dimensions, self.configuration_dictionary)

        # Then
        self.assertTrue(self.model_trainer.model.number_of_nodes == number_of_nodes)
        self.assertTrue(
            self.model_trainer.model.number_of_node_features == number_of_node_features)
        self.assertTrue(self.model_trainer.optimizer.param_groups)

    def test_do_train(self):
        # Given
        data_dimensions = ((4, 4), (7, 1))
        self.model_trainer.instantiate_attributes(data_dimensions,
                                                  self.configuration_dictionary)
        self._insert_test_data(1)
        dataset = GraphDataset(self.postgres_connector)
        training_data, _, _ = DataPreprocessor().train_validation_test_split(dataset, 1, 0.0, 0.0)

        # When
        training_loss = self.model_trainer.do_train(training_data=training_data, epoch=1)

        # Then
        self.assertTrue(training_loss > 0.0)
        self._truncate_table()

    def test_do_evaluate(self):
        # Given
        data_dimensions = ((4, 4), (7, 1))
        self.model_trainer.instantiate_attributes(data_dimensions,
                                                  self.configuration_dictionary)
        self._insert_test_data(1)
        dataset = GraphDataset(self.postgres_connector)
        training_data, _, _ = DataPreprocessor().train_validation_test_split(dataset, 1, 0.0, 0.0)

        # When
        validation_loss = self.model_trainer.do_evaluate(evaluation_data=training_data, epoch=1)

        # Then
        self.assertTrue(validation_loss > 0.0)
        self._truncate_table()

    def _insert_test_data(self, dataset_size):
        self.postgres_connector.open_connection()
        features = FEATURES_SERIALIZED
        neighbors = NEIGHBORS_SERIALIZED
        labels = LABELS_SERIALIZED
        for index in range(dataset_size):
            self.postgres_connector.execute_insert_dataset(str(index), features, neighbors, labels)
        self.postgres_connector.close_connection()

    def _truncate_table(self) -> None:
        self.postgres_connector.open_connection()
        self.postgres_connector.truncate_table(TEST_DATASET)
        self.postgres_connector.close_connection()
