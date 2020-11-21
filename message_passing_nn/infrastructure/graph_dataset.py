import os
import pickle
from typing import List, Tuple

import torch as to
from torch.utils.data import Dataset
from tqdm import tqdm

from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.postgres_connector import PostgresConnector


class GraphDataset(Dataset):
    def __init__(self, data_directory: str, postgres_connector: PostgresConnector = None, test_mode: bool = False) -> None:
        self.data_directory = data_directory
        self.postgres_connector = postgres_connector
        self.test_mode = test_mode
        self.dataset = self._load_data() if self.data_directory else []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[to.Tensor, to.Tensor, to.Tensor, str]:
        return self.dataset[index][0], self.dataset[index][1], self.dataset[index][2], self.dataset[index][3]

    def _load_data(self) -> List[Tuple[to.Tensor, to.Tensor, to.Tensor, str]]:
        get_logger().info("Loading dataset")
        files_in_path = self._extract_name_prefixes_from_filenames()
        dataset = []
        size = 0
        disable_progress_bar = self.test_mode
        for filename_index in tqdm(range(len(files_in_path)), disable=disable_progress_bar):
            filename = files_in_path[filename_index]
            try:
                dataset.append(
                    (self._get_features(filename), self._get_all_neighbors(filename), self._get_labels(filename),
                     filename))
            except Exception as e:
                get_logger().info(e)
                get_logger().info("Skipped " + filename)
            size += self._get_size(dataset[-1])
        get_logger().info(
            "Loaded " + str(len(dataset)) + " files. Size: " + str(int(size * 0.000001)) + " MB")
        return dataset

    @staticmethod
    def _to_list(dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> List[Tuple[to.Tensor, to.Tensor]]:
        return [(dataset[index][0], dataset[index][1]) for index in range(len(dataset))]

    @staticmethod
    def _extract_labels(dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> List[to.Tensor]:
        return [dataset[index][2] for index in range(len(dataset))]

    def _get_labels(self, filename: str) -> to.Tensor:
        with open(os.path.join(self.data_directory, filename + 'labels.pickle'), 'rb') as labels_file:
            labels = pickle.load(labels_file).float()
        return labels

    def _get_features(self, filename: str) -> to.Tensor:
        with open(os.path.join(self.data_directory, filename + 'features.pickle'), 'rb') as features_file:
            features = pickle.load(features_file).float()
        return features

    def _get_all_neighbors(self, filename: str) -> to.Tensor:
        with open(os.path.join(self.data_directory, filename + 'adjacency-matrix.pickle'), 'rb') as adjacency_matrix_file:
            adjacency_matrix = pickle.load(adjacency_matrix_file).float()
            number_of_nodes = adjacency_matrix.shape[0]
            all_neighbors_list = []
            max_number_of_neighbors = -1
            for node_id in range(adjacency_matrix.shape[0]):
                neighbors = to.nonzero(adjacency_matrix[node_id], as_tuple=True)[0].tolist()
                if len(neighbors) > max_number_of_neighbors:
                    max_number_of_neighbors = len(neighbors)
                all_neighbors_list.append(neighbors)
            all_neighbors = self._get_minus_ones_tensor(number_of_nodes, max_number_of_neighbors)
            for node_id in range(number_of_nodes):
                all_neighbors[node_id, :len(all_neighbors_list[node_id])] = to.tensor(all_neighbors_list[node_id])
        return all_neighbors

    @staticmethod
    def _get_minus_ones_tensor(number_of_nodes: int, max_number_of_neighbors: int) -> to.Tensor:
        return to.zeros(number_of_nodes, max_number_of_neighbors) - to.ones(number_of_nodes, max_number_of_neighbors)

    @staticmethod
    def _get_size(data: Tuple[to.Tensor, to.Tensor, to.Tensor, str]) -> int:
        return int(data[0].element_size() * data[0].nelement() +
                   data[1].element_size() * data[1].nelement() +
                   data[2].element_size() * data[2].nelement())

    def _extract_name_prefixes_from_filenames(self) -> List[str]:
        return list(set([self._reconstruct_filename(file) for file in self._get_data_filenames()]))

    def _get_data_filenames(self) -> List[str]:
        return sorted([file for file in os.listdir(self.data_directory) if file.endswith(".pickle")])

    @staticmethod
    def _reconstruct_filename(file: str) -> str:
        return "_".join(file.split("_")[:-1]) + "_"

    def enable_test_mode(self) -> None:
        self.test_mode = True
