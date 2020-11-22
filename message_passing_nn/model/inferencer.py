from typing import List, Tuple

import torch as to
from torch import nn
from torch.utils.data import DataLoader

from message_passing_nn.data import DataPreprocessor


class Inferencer:
    def __init__(self, data_preprocessor: DataPreprocessor, device: str) -> None:
        self.preprocessor = data_preprocessor
        self.device = device

    def do_inference(self, model: nn.Module, inference_data: DataLoader) -> List[Tuple[to.Tensor, to.Tensor, str]]:
        outputs_labels_pairs = []
        with to.no_grad():
            for node_features, all_neighbors, labels, tag in inference_data:
                node_features, all_neighbors, labels = (node_features.to(self.device),
                                                        all_neighbors.to(self.device),
                                                        labels.to(self.device))
                outputs = model.forward(node_features, all_neighbors, batch_size=1)
                outputs_labels_pairs.append((outputs, labels, tag))
        return outputs_labels_pairs
