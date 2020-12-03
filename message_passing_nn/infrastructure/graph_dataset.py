from typing import List, Tuple

import torch as to
from torch.utils.data import Dataset

from message_passing_nn.fixtures.characters import BYTES_TO_MB
from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.postgres_connector import PostgresConnector


class GraphDataset(Dataset):
    def __init__(self, postgres_connector: PostgresConnector = None) -> None:
        self.postgres_connector = postgres_connector
        self.dataset = self._load_data() if self.postgres_connector else []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[to.Tensor, to.Tensor, to.Tensor, str]:
        features, neighbors, labels, pdb_code = [to.tensor(data)
                                                 if not isinstance(data, str) else data
                                                 for data in self.dataset[index]]
        return features, neighbors, labels, pdb_code

    def _load_data(self) -> List[Tuple[to.Tensor, to.Tensor, to.Tensor, str]]:
        get_logger().info("Loading dataset")
        self.postgres_connector.open_connection()
        dataset = self.postgres_connector.execute_query(fields=["features", "neighbors", "labels", "pdb_code"],
                                                        use_case='dataset')
        dataset_on_memory_in_mb = str(self._get_size_in_memory(dataset))
        get_logger().info("Loaded " + str(len(dataset)) + " entries. Size: " + dataset_on_memory_in_mb + " MB")
        self.postgres_connector.close_connection()
        return dataset

    @staticmethod
    def _to_list(dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> List[Tuple[to.Tensor, to.Tensor]]:
        return [(dataset[index][0], dataset[index][1]) for index in range(len(dataset))]

    @staticmethod
    def _get_size_in_memory(dataset: Tuple[to.Tensor, to.Tensor, to.Tensor, str]) -> int:
        example_features, example_neighbors, example_labels = [to.tensor(data) for data in dataset[0][:-1]]
        return int((example_features[0].element_size() * example_features[0].nelement() +
                    example_neighbors[1].element_size() * example_neighbors[1].nelement() +
                    example_labels[2].element_size() * example_labels[2].nelement() * BYTES_TO_MB)) * len(dataset)
