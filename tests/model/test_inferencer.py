from unittest import TestCase

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.model import Inferencer
from message_passing_nn.utils.model_selector import load_model
from message_passing_nn.utils.postgres_connector import PostgresConnector
from tests.fixtures.matrices_and_vectors import *
from tests.fixtures.environment_variables import *
from tests.fixtures.residue_list import map_amino_acid_codes


class TestInferencer(TestCase):
    def test_do_inference(self):
        # Given
        data_preprocessor = DataPreprocessor()
        device = "cpu"
        inferencer = Inferencer(data_preprocessor, device)
        data_dimensions = {"number_of_nodes": 4,
                           "number_of_node_features": 4,
                           "fully_connected_layer_input_size": 16,
                           "fully_connected_layer_output_size": 8}
        model = load_model("RNN")
        model = model(time_steps=1,
                      number_of_nodes=data_dimensions["number_of_nodes"],
                      number_of_node_features=data_dimensions["number_of_node_features"],
                      fully_connected_layer_input_size=data_dimensions["fully_connected_layer_input_size"],
                      fully_connected_layer_output_size=data_dimensions["fully_connected_layer_output_size"])
        self.postgres_connector = PostgresConnector()
        self._insert_test_data(1)
        dataset = GraphDataset(self.postgres_connector)
        inference_data, _, _ = DataPreprocessor().train_validation_test_split(dataset, 1, 0.0, 0.0)
        output_label_pairs_expected = [to.tensor((0, 1, 0, 2, 1, 2, 0, 0)), to.tensor((0, 1, 0, 2, 1, 2, 0, 0))]

        # When
        output_label_pairs = inferencer.do_inference(model, inference_data)

        # Then
        self.assertEqual(output_label_pairs[0][0].squeeze().size(), output_label_pairs_expected[0].size())
        self.assertEqual(output_label_pairs[0][1].squeeze().size(), output_label_pairs_expected[1].size())

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
