import os
from typing import List
from unittest import TestCase
import shutil
from datetime import datetime
import torch as to
from message_passing_nn.infrastructure.graph_dataset import GraphDataset

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model import Loader, Inferencer
from message_passing_nn.infrastructure.file_system_repository import FileSystemRepository
from message_passing_nn.usecase import Inference
from message_passing_nn.utils.saver import Saver


class TestInference(TestCase):
    def test_start(self):
        # Given
        dataset_size = 1
        features = to.ones(4, 2)
        adjacency_matrix = to.ones(4, 4)
        labels = to.ones(16)
        dataset_name = 'inference-test-data'
        tests_data_directory = os.path.join('tests', 'test_data')
        tests_model_directory = os.path.join("tests", "test_data", "model-checkpoints-test/configuration&id__model&"
                                             "RNN__epochs&10__loss_function&MSE__optimizer&Adagrad__batch_size&"
                                             "100__validation_split&0.2__test_split&0.1__time_steps&"
                                             "1__validation_period&5", "Epoch_5_model_state_dictionary.pth")
        tests_results_directory = os.path.join('tests', 'results_inference')
        device = "cpu"
        repository = FileSystemRepository(tests_data_directory, dataset_name)
        data_path = os.path.join(tests_data_directory, dataset_name)
        data_preprocessor = DataPreprocessor()
        data_preprocessor.enable_test_mode()
        loader = Loader("RNN")
        inferencer = Inferencer(data_preprocessor, device)
        saver = Saver(tests_model_directory, tests_results_directory)
        dataset = GraphDataset(data_path, test_mode=True)
        inference = Inference(dataset,
                              data_preprocessor,
                              loader,
                              inferencer,
                              saver)

        adjacency_matrix_filenames, features_filenames, labels_filenames = self._save_test_data(adjacency_matrix,
                                                                                                dataset_size,
                                                                                                features,
                                                                                                labels,
                                                                                                repository)

        # When
        inference.start()

        # Then
        filename_expected = datetime.now().strftime("%d-%b-%YT%H_%M") + "_distance_maps.pickle"
        self.assertTrue(os.path.isfile(tests_results_directory + "/" + filename_expected))

        # Tear down
        self._remove_files(dataset_size,
                           features_filenames,
                           adjacency_matrix_filenames,
                           labels_filenames,
                           tests_data_directory,
                           dataset_name,
                           tests_results_directory)

    @staticmethod
    def _save_test_data(adjacency_matrix, dataset_size, features, labels, repository):
        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        adjacency_matrix_filenames = [str(i) + '_training_adjacency-matrix' '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            repository.save(features_filenames[i], features)
            repository.save(adjacency_matrix_filenames[i], adjacency_matrix)
            repository.save(labels_filenames[i], labels)
        return adjacency_matrix_filenames, features_filenames, labels_filenames

    @staticmethod
    def _remove_files(dataset_size: int,
                      features_filenames: List[str],
                      adjacency_matrix_filenames: List[str],
                      labels_filenames: List[str],
                      tests_data_directory: str,
                      dataset: str,
                      tests_results_directory: str) -> None:
        for i in range(dataset_size):
            os.remove(os.path.join(tests_data_directory, dataset, features_filenames[i]))
            os.remove(os.path.join(tests_data_directory, dataset, adjacency_matrix_filenames[i]))
            os.remove(os.path.join(tests_data_directory, dataset, labels_filenames[i]))
        shutil.rmtree(tests_results_directory)
