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
            for data in inference_data:
                node_features, all_neighbors, labels, pdb_code = self._send_to_device(data)
                outputs = model.forward(node_features, all_neighbors, batch_size=1)
                outputs_labels_pairs.append((outputs, labels, pdb_code))
        return outputs_labels_pairs

    def _send_to_device(self, data: Tuple[to.Tensor, to.Tensor, to.Tensor, str]) \
            -> Tuple[to.Tensor, to.Tensor, to.Tensor, str]:
        node_features, all_neighbors, labels, pdb_code = (data[0].to(self.device),
                                                          data[1].to(self.device),
                                                          data[2].to(self.device),
                                                          data[3])
        return node_features, all_neighbors, labels, pdb_code
