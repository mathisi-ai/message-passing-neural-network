from unittest import TestCase

import torch as to

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.model import Inferencer
from message_passing_nn.utils.model_selector import ModelSelector
from message_passing_nn.utils.postgres_connector import PostgresConnector
from tests.fixtures.matrices_and_vectors import FEATURES_SERIALIZED, NEIGHBORS_SERIALIZED, LABELS_SERIALIZED
from tests.fixtures.postgres_variables import TEST_DATASET


class TestInferencer(TestCase):
    def test_do_inference(self):
        # Given
        data_preprocessor = DataPreprocessor()
        device = "cpu"
        inferencer = Inferencer(data_preprocessor, device)
        data_dimensions = ((4, 4), (7, 1))
        model = ModelSelector.load_model("RNN")
        model = model(time_steps=1,
                      number_of_nodes=data_dimensions[0][0],
                      number_of_node_features=data_dimensions[0][1],
                      fully_connected_layer_input_size=data_dimensions[0][0] * data_dimensions[0][1],
                      fully_connected_layer_output_size=data_dimensions[1][0])
        self.postgres_connector = PostgresConnector()
        self._insert_test_data(1)
        dataset = GraphDataset(self.postgres_connector)
        inference_data, _, _ = DataPreprocessor().train_validation_test_split(dataset, 1, 0.0, 0.0)
        output_label_pairs_expected = [to.tensor((0, 1, 0, 2, 1, 2, 0)), to.tensor((0, 1, 0, 2, 1, 2, 0))]

        # When
        output_label_pairs = inferencer.do_inference(model, inference_data)

        # Then
        self.assertEqual(output_label_pairs[0][0].squeeze().size(), output_label_pairs_expected[0].size())
        self.assertEqual(output_label_pairs[0][1].squeeze().size(), output_label_pairs_expected[1].size())

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
