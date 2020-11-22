import os
from unittest import TestCase

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.file_system_repository import FileSystemRepository
from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.model.trainer import Trainer
from message_passing_nn.use_case.grid_search import GridSearch
from message_passing_nn.utils.postgres_connector import PostgresConnector
from message_passing_nn.utils.saver import Saver
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, FEATURES_SERIALIZED, \
    NEIGHBORS_SERIALIZED, LABELS_SERIALIZED
from tests.fixtures.postgres_variables import *


class TestTraining(TestCase):
    def setUp(self) -> None:
        self.features = BASE_GRAPH_NODE_FEATURES
        self.adjacency_matrix = BASE_GRAPH
        self.labels = BASE_GRAPH.view(-1)
        self.dataset_name = 'training-test-data'
        self.tests_data_directory = os.path.join('tests', 'test_data')
        tests_model_directory = os.path.join('tests', 'model_checkpoints')
        tests_results_directory = os.path.join('tests', 'grid_search_results')
        device = "cpu"
        self.data_path = os.path.join("./", self.tests_data_directory, self.dataset_name)
        self.repository = FileSystemRepository(self.tests_data_directory, self.dataset_name)
        self.data_preprocessor = DataPreprocessor()
        self.data_preprocessor.enable_test_mode()
        self.postgres_connector = PostgresConnector()
        self.postgres_connector.open_connection()
        self.model_trainer = Trainer(self.data_preprocessor, device)
        self.saver = Saver(tests_model_directory, tests_results_directory)

    def test_start_for_multiple_batches_of_the_same_size(self):
        # Given
        dataset_size = 6
        grid_search_dictionary = {
            "model": ["RNN"],
            "epochs": [10],
            "batch_size": [3],
            "validation_split": [0.2],
            "test_split": [0.1],
            "loss_function": ["MSE"],
            "optimizer": ["SGD"],
            "time_steps": [1],
            "validation_period": [5]
        }
        self._insert_test_data(dataset_size)
        dataset = GraphDataset(self.postgres_connector)

        grid_search = GridSearch(dataset,
                                 self.data_preprocessor,
                                 self.model_trainer,
                                 grid_search_dictionary,
                                 self.saver)

        # When
        losses = grid_search.start()
        configuration_id = list(losses["training_loss"].keys())[0]

        # Then
        self.assertTrue(losses["training_loss"][configuration_id][grid_search_dictionary["epochs"][0]] > 0.0)
        self.assertTrue(
            losses["validation_loss"][configuration_id][grid_search_dictionary["validation_period"][0]] > 0.0)
        self.assertTrue(losses["test_loss"][configuration_id]["final_epoch"] > 0.0)

        # Tear down
        self._truncate_table()

    def test_start_for_multiple_batches_of_differing_size(self):
        # Given
        dataset_size = 5
        grid_search_dictionary = {
            "model": ["RNN"],
            "epochs": [10],
            "batch_size": [3],
            "validation_split": [0.2],
            "test_split": [0.1],
            "loss_function": ["MSE"],
            "optimizer": ["SGD"],
            "time_steps": [1],
            "validation_period": [5]
        }
        self._insert_test_data(dataset_size)
        dataset = GraphDataset(self.postgres_connector)

        grid_search = GridSearch(dataset,
                                 self.data_preprocessor,
                                 self.model_trainer,
                                 grid_search_dictionary,
                                 self.saver)

        # When
        losses = grid_search.start()
        configuration_id = list(losses["training_loss"].keys())[0]

        # Then
        self.assertTrue(losses["training_loss"][configuration_id][grid_search_dictionary["epochs"][0]] > 0.0)
        self.assertTrue(
            losses["validation_loss"][configuration_id][grid_search_dictionary["validation_period"][0]] > 0.0)
        self.assertTrue(losses["test_loss"][configuration_id]["final_epoch"] > 0.0)

        # Tear down
        self._truncate_table()

    def test_start_a_grid_search(self):
        # Given
        dataset_size = 6
        grid_search_dictionary = {
            "model": ["RNN"],
            "epochs": [10, 15],
            "batch_size": [3, 4],
            "validation_split": [0.2],
            "test_split": [0.1],
            "loss_function": ["MSE"],
            "optimizer": ["SGD"],
            "time_steps": [1],
            "validation_period": [5]
        }
        self._insert_test_data(dataset_size)
        dataset = GraphDataset(self.postgres_connector)

        grid_search = GridSearch(dataset,
                                 self.data_preprocessor,
                                 self.model_trainer,
                                 grid_search_dictionary,
                                 self.saver)

        # When
        losses = grid_search.start()
        configuration_id = list(losses["training_loss"].keys())[0]

        # Then
        self.assertTrue(losses["training_loss"][configuration_id][grid_search_dictionary["epochs"][0]] > 0.0)
        self.assertTrue(
            losses["validation_loss"][configuration_id][grid_search_dictionary["validation_period"][0]] > 0.0)
        self.assertTrue(losses["test_loss"][configuration_id]["final_epoch"] > 0.0)

        # Tear down
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
