import torch as to
from torch.nn import MSELoss


class LossWithPenalty:
    def __init__(self, loss_function, penalty: dict, scaling_factor: float, penalty_decimals: int, batch_size: int):
        self.loss_function = loss_function
        self.penalty = penalty
        self.scaling_factor = scaling_factor
        self.penalty_decimals = penalty_decimals
        self.batch_size = batch_size

    def __call__(self, labels, outputs, features):
        loss_step_1 = self.loss_function(outputs, labels)
        loss_step_2 = to.zeros_like(loss_step_1)
        if self.penalty:
            counter = 0
            for batch in range(labels.shape[0]):
                number_of_nodes = features.shape[1]
                labels_reshaped = labels[batch].reshape(number_of_nodes, 2)
                for node_id in range(number_of_nodes):
                    loss_step_2, counter = self._add_penalty(features[batch], labels_reshaped, loss_step_2, node_id, counter)
            loss_step_2 += loss_step_2/counter
        return loss_step_1 + loss_step_2

    def _add_penalty(self, features, labels_reshaped, loss_step_2, node_id, counter):
        if to.sum(features[node_id, :]) == 1:
            penalty_index = ((features[node_id, :] == 1).nonzero()).item()
            if labels_reshaped[node_id][0] > 0.0 and labels_reshaped[node_id][1] > 0.0:
                index_phi, index_psi = (self._get_index_from(labels_reshaped[node_id][0]),
                                        self._get_index_from(labels_reshaped[node_id][1]))
                counter += 1
                return to.add(loss_step_2, self.scaling_factor * self.penalty[penalty_index][index_psi, index_phi]), counter
        else:
            return 0.0, counter

    def _get_index_from(self, value: float):
        return int(value * 10 ** self.penalty_decimals)
