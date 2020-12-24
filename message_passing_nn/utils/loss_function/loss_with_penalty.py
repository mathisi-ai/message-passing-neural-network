import torch as to
from typing import Tuple


class LossWithPenalty:
    def __init__(self, loss_function, penalty: dict, scaling_factor: float, penalty_decimals: int, batch_size: int):
        self.loss_function = loss_function
        self.penalty = penalty
        self.scaling_factor = scaling_factor
        self.penalty_decimals = penalty_decimals
        self.batch_size = batch_size

    def forward(self, outputs: to.Tensor, labels: to.Tensor, features: to.Tensor) -> to.Tensor:
        batch_size = outputs.shape[0]
        number_of_nodes = features.shape[1]
        loss = to.zeros((batch_size, number_of_nodes))
        outputs = outputs.reshape(batch_size, number_of_nodes, 2)
        labels = labels.reshape(batch_size, number_of_nodes, 2)
        for batch in range(batch_size):
            for node_id in range(number_of_nodes):
                if self._residue_exists(features[batch, node_id]):
                    loss[batch, node_id] = self._get_node_error(outputs[batch, node_id], labels[batch, node_id])
                    if self.penalty:
                        loss[batch, node_id] += self._add_penalty(features[batch, node_id], outputs[batch, node_id])
        return to.mean(loss)

    @staticmethod
    def _get_node_error(outputs: to.Tensor, labels: to.Tensor):
        phi_squared_error = (outputs[0] - labels[0]) ** 2
        psi_squared_error = (outputs[1] - labels[1]) ** 2
        return to.sqrt(phi_squared_error + psi_squared_error)

    def _add_penalty(self, one_hot_residue: to.Tensor, node_outputs: to.Tensor) -> Tuple[to.Tensor, int]:
        penalty_index = self._get_residue_index(one_hot_residue)
        phi, psi = node_outputs[0].item(), node_outputs[1].item()
        phi_index, psi_index = self._get_index_of(phi), self._get_index_of(psi)
        node_penalty = self.scaling_factor * self.penalty[penalty_index][psi_index, phi_index]
        return node_penalty

    @staticmethod
    def _get_residue_index(one_hot_residue: to.Tensor) -> int:
        return ((one_hot_residue == 1).nonzero()).item()

    def _get_index_of(self, value: float) -> int:
        return int(value * 10 ** self.penalty_decimals)

    @staticmethod
    def _residue_exists(one_hot_residue: to.Tensor) -> bool:
        return to.sum(one_hot_residue) == 1
