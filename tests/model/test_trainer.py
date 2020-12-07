from unittest import TestCase

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.model.trainer import Trainer
from message_passing_nn.utils.postgres_connector import PostgresConnector
from tests.fixtures.matrices_and_vectors import *
from tests.fixtures.environment_variables import *
from tests.fixtures.residue_list import map_amino_acid_codes


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
        data_dimensions = {"number_of_nodes": 4,
                           "number_of_node_features": 4,
                           "fully_connected_layer_input_size": 16,
                           "fully_connected_layer_output_size": 8}

        # When
        self.model_trainer.build(data_dimensions, self.configuration_dictionary)

        # Then
        self.assertTrue(self.model_trainer.model.number_of_nodes == data_dimensions["number_of_nodes"])
        self.assertTrue(self.model_trainer.model.number_of_node_features == data_dimensions["number_of_node_features"])
        self.assertTrue(self.model_trainer.optimizer.param_groups)

    def test_do_train(self):
        # Given
        data_dimensions = {"number_of_nodes": 4,
                           "number_of_node_features": 4,
                           "fully_connected_layer_input_size": 16,
                           "fully_connected_layer_output_size": 8}
        self.model_trainer.build(data_dimensions, self.configuration_dictionary)
        self._insert_test_data(1)
        dataset = GraphDataset(self.postgres_connector)
        training_data, _, _ = DataPreprocessor().train_validation_test_split(dataset, 1, 0.0, 0.0)

        # When
        training_loss = self.model_trainer.do_train_step(training_data=training_data, epoch=1)

        # Then
        self.assertTrue(training_loss > 0.0)
        self._truncate_table()

    def test_do_evaluate(self):
        # Given
        data_dimensions = {"number_of_nodes": 4,
                           "number_of_node_features": 4,
                           "fully_connected_layer_input_size": 16,
                           "fully_connected_layer_output_size": 8}
        self.model_trainer.build(data_dimensions, self.configuration_dictionary)
        self._insert_test_data(1)
        dataset = GraphDataset(self.postgres_connector)
        training_data, _, _ = DataPreprocessor().train_validation_test_split(dataset, 1, 0.0, 0.0)

        # When
        validation_loss = self.model_trainer.do_evaluate_step(evaluation_data=training_data, epoch=1)

        # Then
        self.assertTrue(validation_loss > 0.0)
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
