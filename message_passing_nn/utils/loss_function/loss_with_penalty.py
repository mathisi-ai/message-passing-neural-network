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
        loss_step_1 = self.loss_function(outputs, labels)
        loss_step_2 = to.zeros_like(loss_step_1)
        if self.penalty:
            loss_step_2 = self._get_penalty_loss(features, labels, loss_step_2)
        return to.add(loss_step_1, loss_step_2)

    def _get_penalty_loss(self, features: to.Tensor, labels: to.Tensor, loss_step_2: to.Tensor) -> to.Tensor:
        batch_size = labels.shape[0]
        number_of_nodes = features.shape[1]
        for batch in range(batch_size):
            counter = 0
            batch_loss, counter = self._add_node_penalty(counter,
                                                         features[batch],
                                                         labels[batch].reshape(number_of_nodes, 2),
                                                         to.zeros_like(loss_step_2),
                                                         number_of_nodes)
            loss_step_2 += batch_loss / counter
        return loss_step_2

    def _add_node_penalty(self,
                          counter: int,
                          features: to.Tensor,
                          labels: to.Tensor,
                          batch_loss: to.Tensor,
                          number_of_nodes: int) -> Tuple[to.Tensor, int]:
        for node_id in range(number_of_nodes):
            one_hot_residue = features[node_id, :]
            node_labels = labels[node_id]
            node_penalty, counter = self._get_node_penalty(one_hot_residue, node_labels, counter)
            batch_loss += node_penalty
        return batch_loss, counter

    def _get_node_penalty(self, one_hot_residue: to.Tensor, node_labels: to.Tensor, counter: int) \
            -> Tuple[to.Tensor, int]:
        if self._residue_exists(one_hot_residue):
            node_penalty = self._calculate_node_penalty(node_labels, one_hot_residue)
            counter += 1
            return node_penalty, counter
        else:
            return to.tensor(0.0), counter

    def _calculate_node_penalty(self, node_labels: to.Tensor, one_hot_residue: to.Tensor) -> to.Tensor:
        penalty_index = self._get_residue_index(one_hot_residue)
        phi, psi = node_labels[0].item(), node_labels[1].item()
        phi_index, psi_index = self._get_index_of(phi), self._get_index_of(psi)
        return self.scaling_factor * self.penalty[penalty_index][psi_index, phi_index]

    @staticmethod
    def _get_residue_index(one_hot_residue: to.Tensor) -> int:
        return ((one_hot_residue == 1).nonzero()).item()

    def _get_index_of(self, value: float) -> int:
        return int(value * 10 ** self.penalty_decimals)

    @staticmethod
    def _residue_exists(one_hot_residue: to.Tensor) -> bool:
        return to.sum(one_hot_residue) == 1
