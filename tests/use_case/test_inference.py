import os
from datetime import datetime
from unittest import TestCase

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.model import Loader, Inferencer
from message_passing_nn.use_case import Inference
from message_passing_nn.utils.postgres_connector import PostgresConnector
from message_passing_nn.utils.saver import Saver
from tests.fixtures.environment_variables import *
from tests.fixtures.matrices_and_vectors import *
from tests.fixtures.residue_list import map_amino_acid_codes


class TestInference(TestCase):
    def test_start(self):
        # Given
        tests_model_directory = os.path.join("tests", "test_data",
                                             "model-checkpoints-test",
                                             "configuration&id__model&RNN__epochs&10__loss_function&MSE__optimizer"
                                             "&Adagrad__batch_size&100__validation_split&0.2__test_split"
                                             "&0.1__time_steps&1__validation_period&5",
                                             "Epoch_5_model_state_dictionary.pth")
        tests_results_directory = os.path.join('tests', 'results_inference')
        device = "cpu"
        data_preprocessor = DataPreprocessor()
        loader = Loader("RNN")
        inferencer = Inferencer(data_preprocessor, device)
        saver = Saver(tests_model_directory, tests_results_directory)
        self.postgres_connector = PostgresConnector()
        self._insert_test_data(dataset_size=1)
        dataset = GraphDataset(self.postgres_connector)
        inference = Inference(dataset, data_preprocessor, loader, inferencer, saver)

        # When
        inference.start()

        # Then
        filename_expected = datetime.now().strftime("%d-%b-%YT%H_%M") + "_distance_maps.pickle"
        self.assertTrue(os.path.isfile(os.path.join(tests_results_directory, filename_expected)))

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
