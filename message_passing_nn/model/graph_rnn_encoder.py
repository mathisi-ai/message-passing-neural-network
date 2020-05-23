from typing import List

import torch as to
import torch.nn as nn

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model.node import Node


class GraphRNNEncoder(nn.Module):
    def __init__(self,
                 time_steps: int,
                 number_of_nodes: int,
                 number_of_node_features: int,
                 fully_connected_layer_input_size: int,
                 fully_connected_layer_output_size: int,
                 device: str) -> None:
        super(GraphRNNEncoder, self).__init__()
        node_features_tensor_shape = [number_of_node_features, number_of_node_features]
        nodes_tensor_shape = [number_of_nodes, number_of_nodes]

        self.time_steps = time_steps
        self.number_of_nodes = number_of_nodes
        self.number_of_node_features = number_of_node_features
        self.fully_connected_layer_input_size = fully_connected_layer_input_size
        self.fully_connected_layer_output_size = fully_connected_layer_output_size
        self.device = device

        self.w_graph_node_features = self._get_parameter(node_features_tensor_shape)
        self.w_graph_neighbor_messages = self._get_parameter(node_features_tensor_shape)
        self.u_graph_node_features = self._get_parameter(nodes_tensor_shape)
        self.u_graph_neighbor_messages = self._get_parameter(node_features_tensor_shape)
        self.linear = to.nn.Linear(self.fully_connected_layer_input_size, self.fully_connected_layer_output_size)
        self.sigmoid = to.nn.Sigmoid()

    @classmethod
    def of(cls,
           time_steps: int,
           number_of_nodes: int,
           number_of_node_features: int,
           fully_connected_layer_input_size: int,
           fully_connected_layer_output_size: int,
           device: str):
        return cls(time_steps,
                   number_of_nodes,
                   number_of_node_features,
                   fully_connected_layer_input_size,
                   fully_connected_layer_output_size,
                   device)

    def forward(self,
                node_features: to.Tensor,
                adjacency_matrix: to.Tensor,
                batch_size: int) -> to.Tensor:
        outputs = to.zeros(batch_size, self.fully_connected_layer_output_size, device=self.device)
        for batch in range(batch_size):
            outputs[batch] = self.sigmoid(
                self.linear(
                    DataPreprocessor.flatten(
                        self.encode(node_features[batch], adjacency_matrix[batch]),
                        self.fully_connected_layer_input_size)))
        return outputs

    def encode(self, node_features: to.Tensor, adjacency_matrix: to.Tensor) -> to.Tensor:
        messages = self._send_messages(node_features, adjacency_matrix)
        encodings = self._encode_nodes(node_features, messages)
        return encodings

    def _send_messages(self, node_features: to.Tensor, adjacency_matrix: to.Tensor) -> to.Tensor:
        messages = to.zeros((self.number_of_nodes, self.number_of_nodes, self.number_of_node_features),
                            device=self.device)
        for step in range(self.time_steps):
            messages = self._compose_messages(node_features, adjacency_matrix, messages)
        return messages

    def _encode_nodes(self, node_features: to.Tensor, messages: to.Tensor) -> to.Tensor:
        node_encoding_messages = to.zeros(self.number_of_nodes, self.number_of_node_features, device=self.device)
        node_encoding_features = self.u_graph_node_features.matmul(node_features)
        for node_id in range(self.number_of_nodes):
            node_encoding_messages[node_id] = self.u_graph_neighbor_messages.matmul(to.sum(messages[node_id], dim=0))
        return to.relu(node_encoding_features + node_encoding_messages)

    def _apply_recurrent_layer(self, node_features: to.Tensor, messages: to.Tensor, node_id: int) -> to.Tensor:
        node_encoding_features = self.u_graph_node_features.matmul(node_features[node_id])
        node_encoding_messages = self.u_graph_neighbor_messages.matmul(to.sum(messages[node_id], dim=0))
        return to.relu(node_encoding_features + node_encoding_messages)

    def _compose_messages(self,
                          node_features: to.Tensor,
                          adjacency_matrix: to.Tensor,
                          messages: to.Tensor) -> to.Tensor:
        new_messages = to.zeros(messages.shape, device=self.device)
        for node_id in range(self.number_of_nodes):
            all_neighbors = to.nonzero(adjacency_matrix[node_id], as_tuple=True)[0]
            for end_node_id in all_neighbors:
                messages_from_the_other_neighbors = to.zeros(node_features[node_id].shape[0], device=self.device)
                if len(all_neighbors) > 1:
                    end_node_index = (all_neighbors == end_node_id).nonzero()[0][0].item()
                    for neighbor in to.cat((all_neighbors[:end_node_index], all_neighbors[end_node_index + 1:])):
                        messages_from_the_other_neighbors += self.w_graph_neighbor_messages.matmul(
                            messages[neighbor, node_id])
                new_messages[node_id, end_node_id] = to.relu(
                    to.add(self.w_graph_node_features.matmul(node_features[node_id]),
                           messages_from_the_other_neighbors))
        return new_messages

    def _get_parameter(self, tensor_shape: List[int]) -> nn.Parameter:
        return nn.Parameter(nn.init.kaiming_normal_(to.zeros(tensor_shape, device=self.device)), requires_grad=True)
