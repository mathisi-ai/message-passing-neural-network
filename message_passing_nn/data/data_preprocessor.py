from typing import Tuple

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.utils.logger import get_logger


class DataPreprocessor:
    def __init__(self):
        pass

    def train_validation_test_split(self,
                                    dataset: GraphDataset,
                                    batch_size: int,
                                    validation_split: float = 0.2,
                                    test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        test_index, validation_index = self._get_validation_and_test_indexes(dataset, validation_split, test_split)

        training_data = self._get_train_split(batch_size, dataset, validation_index)

        validation_data = self._get_validation_split(batch_size, dataset, test_index, validation_index, validation_split)

        test_data = self._get_test_split(batch_size, dataset, test_index, test_split)

        get_logger().info("Train/validation/test split: {} batches of {}".format("/".join([str(len(training_data)),
                                                                                           str(len(validation_data)),
                                                                                           str(len(test_data))]),
                                                                                 str(batch_size)))
        return training_data, validation_data, test_data

    @staticmethod
    def get_dataloader(dataset: GraphDataset, batch_size: int = 1) -> DataLoader:
        return DataLoader(dataset, batch_size)

    @staticmethod
    def extract_data_dimensions(dataset: GraphDataset) -> dict:
        data_dimensions = {"number_of_nodes": int(dataset[0][0].shape[0]),
                           "number_of_node_features": int(dataset[0][0].shape[1]),
                           "fully_connected_layer_input_size": int(dataset[0][0].shape[0] * dataset[0][0].shape[1]),
                           "fully_connected_layer_output_size": int(dataset[0][2].shape[0])}
        return data_dimensions

    @staticmethod
    def _get_train_split(batch_size: int, dataset: GraphDataset, validation_index: int) -> DataLoader:
        train_sampler = SubsetRandomSampler(list(range(validation_index)))
        training_data = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        return training_data

    @staticmethod
    def _get_validation_split(batch_size: int,
                              dataset: GraphDataset,
                              test_index: int,
                              validation_index: int,
                              validation_split: float) -> DataLoader:
        if validation_split:
            validation_sampler = SubsetRandomSampler(list(range(validation_index, test_index)))
            validation_data = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)
        else:
            validation_data = DataLoader(GraphDataset())
        return validation_data

    @staticmethod
    def _get_test_split(batch_size: int, dataset: GraphDataset, test_index: int, test_split: float) -> DataLoader:
        if test_split:
            test_sampler = SubsetRandomSampler(list(range(test_index, len(dataset.dataset))))
            test_data = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        else:
            test_data = DataLoader(GraphDataset())
        return test_data

    @staticmethod
    def _get_validation_and_test_indexes(dataset: GraphDataset,
                                         validation_split: float,
                                         test_split: float) -> Tuple[int, int]:
        validation_index = int((1 - validation_split - test_split) * len(dataset))
        test_index = int((1 - test_split) * len(dataset))
        return test_index, validation_index
