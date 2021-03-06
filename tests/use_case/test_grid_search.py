import os
from typing import List, Tuple
from unittest import TestCase

import itertools

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.file_system_repository import FileSystemRepository
from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.model.trainer import Trainer
from message_passing_nn.use_case.grid_search import GridSearch
from message_passing_nn.utils.postgres_connector import PostgresConnector
from message_passing_nn.utils.saver import Saver
from tests.fixtures.matrices_and_vectors import *
from tests.fixtures.environment_variables import *
from tests.fixtures.residue_list import map_amino_acid_codes


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
        self.postgres_connector = PostgresConnector()
        self.model_trainer = Trainer(self.data_preprocessor, device, self.postgres_connector)
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
            "loss_function": ["MSEPenalty"],
            "optimizer": ["Adagrad"],
            "time_steps": [1],
            "validation_period": [5],
            "scaling_factor": [0.1],
            "penalty_decimals": [0]
        }
        self._insert_test_data(dataset_size)
        dataset = GraphDataset(self.postgres_connector)
        grid_search_configurations = self._get_all_grid_search_configurations(grid_search_dictionary)

        grid_search = GridSearch(dataset,
                                 self.data_preprocessor,
                                 self.model_trainer,
                                 grid_search_configurations,
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
            "optimizer": ["Adagrad"],
            "time_steps": [1],
            "validation_period": [5],
            "scaling_factor": [0.1],
            "penalty_decimals": [0]
        }
        self._insert_test_data(dataset_size)
        dataset = GraphDataset(self.postgres_connector)
        grid_search_configurations = self._get_all_grid_search_configurations(grid_search_dictionary)

        grid_search = GridSearch(dataset,
                                 self.data_preprocessor,
                                 self.model_trainer,
                                 grid_search_configurations,
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
            "optimizer": ["Adagrad"],
            "time_steps": [1],
            "validation_period": [5],
            "scaling_factor": [0.1],
            "penalty_decimals": [0]
        }
        self._insert_test_data(dataset_size)
        dataset = GraphDataset(self.postgres_connector)
        grid_search_configurations = self._get_all_grid_search_configurations(grid_search_dictionary)

        grid_search = GridSearch(dataset,
                                 self.data_preprocessor,
                                 self.model_trainer,
                                 grid_search_configurations,
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
        dataset_values = """(pdb_code varchar primary key, features float[][], neighbors float[][], labels float[])"""
        penalty_values = """(residue varchar primary key, matrix float[][], penalty float[][])"""
        self.postgres_connector.create_table(TEST_DATASET, dataset_values)
        self.postgres_connector.create_table(TEST_PENALTY, penalty_values)
        features = FEATURES_SERIALIZED
        neighbors = NEIGHBORS_SERIALIZED
        labels = LABELS_SERIALIZED
        penalty = PENALTY_SERIALIZED
        for index in range(dataset_size):
            self.postgres_connector.execute_insert_dataset(str(index), features, neighbors, labels)
        for residue in map_amino_acid_codes:
            self.postgres_connector.execute_insert_penalty(residue, penalty)
        self.postgres_connector.close_connection()

    def _truncate_table(self) -> None:
        self.postgres_connector.open_connection()
        self.postgres_connector.truncate_table(TEST_DATASET)
        self.postgres_connector.close_connection()

    @staticmethod
    def _get_all_grid_search_configurations(grid_search_parameters) -> List[Tuple[Tuple]]:
        all_grid_search_configurations = []
        for key in grid_search_parameters.keys():
            all_grid_search_configurations.append([(key, value) for value in grid_search_parameters[key]])
        return list(itertools.product(*all_grid_search_configurations))
