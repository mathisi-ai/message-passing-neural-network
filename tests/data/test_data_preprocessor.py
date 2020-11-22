from unittest import TestCase

import torch as to
from message_passing_nn.infrastructure.graph_dataset import GraphDataset

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from tests.fixtures.postgres_variables import TEST_DATASET
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, FEATURES_SERIALIZED, \
    NEIGHBORS_SERIALIZED, LABELS_SERIALIZED
from message_passing_nn.utils.postgres_connector import PostgresConnector


class TestGraphPreprocessor(TestCase):
    def setUp(self) -> None:
        self.data_preprocessor = DataPreprocessor()
        self.postgres_connector = PostgresConnector()

    def test_train_validation_test_split(self):
        # Given
        self._insert_test_data(dataset_size=10)
        dataset = GraphDataset(self.postgres_connector)
        train_validation_test_split_expected = [7, 2, 1]

        # When
        train_validation_test_split = self.data_preprocessor.train_validation_test_split(dataset,
                                                                                         batch_size=1,
                                                                                         validation_split=0.2,
                                                                                         test_split=0.1)
        train_validation_test_split = [len(dataset) for dataset in train_validation_test_split]

        # Then
        self.assertEqual(train_validation_test_split_expected, train_validation_test_split)
        self._truncate_table()

    def test_extract_data_dimensions(self):
        # Given
        self._insert_test_data(dataset_size=1)
        dataset = GraphDataset(self.postgres_connector)
        data_dimensions_expected = (to.Size([4, 4]), to.Size([7]))

        # When
        data_dimensions = self.data_preprocessor.extract_data_dimensions(dataset)

        # Then
        self.assertEqual(data_dimensions_expected, data_dimensions)
        self._truncate_table()

    def test_flatten_when_sizes_match(self):
        # Given
        dataset_length = 2
        labels = BASE_GRAPH.view(-1)
        tensors = to.cat((labels, labels))
        tensors_flattened_expected = tensors.view(-1)

        # When
        tensors_flattened = self.data_preprocessor.flatten(tensors, desired_size=dataset_length * len(labels))

        # Then
        self.assertTrue(to.allclose(tensors_flattened_expected, tensors_flattened))

    def test_flatten_when_sizes_do_not_match(self):
        # Given
        dataset_length = 3
        labels = BASE_GRAPH.view(-1)
        tensors = to.cat((labels, labels))
        tensors_flattened_expected = to.cat((tensors.view(-1), to.zeros_like(labels)))

        # When
        tensors_flattened = self.data_preprocessor.flatten(tensors, desired_size=dataset_length * len(labels))

        # Then
        self.assertTrue(to.allclose(tensors_flattened_expected, tensors_flattened))

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

