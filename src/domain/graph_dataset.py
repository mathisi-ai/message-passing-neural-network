from typing import List, Tuple, Any

from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, raw_dataset: List[Tuple[Any, Any]]) -> None:
        self.node_features = self.to_graph(raw_dataset)
        self.labels = self.extract_labels(raw_dataset)

    def __len__(self) -> int:
        return len(self.node_features)

    def __getitem__(self, index: int) -> Tuple:
        return self.node_features[index], self.labels[index]

    @staticmethod
    def to_graph(raw_dataset: List[Tuple[Any, Any]]) -> List[Tuple]:
        return [raw_dataset[index][0] for index in range(len(raw_dataset))]

    @staticmethod
    def extract_labels(raw_dataset: List[Tuple[Any, Any]]) -> List[Any]:
        return [raw_dataset[index][1].view(-1).float() for index in range(len(raw_dataset))]
