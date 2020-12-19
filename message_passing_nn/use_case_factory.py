import os

from typing import List, Tuple

import itertools

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.model.inferencer import Inferencer
from message_passing_nn.model.loader import Loader
from message_passing_nn.model.trainer import Trainer
from message_passing_nn.use_case.grid_search import GridSearch
from message_passing_nn.use_case.inference import Inference
from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.postgres_connector import PostgresConnector
from message_passing_nn.utils.saver import Saver


class UseCaseFactory:
    def __init__(self) -> None:
        self.use_case = None
        self.model = None
        self.device = os.environ['DEVICE']
        self.model_directory = os.environ['MODEL_DIRECTORY']
        self.results_directory = os.environ['RESULTS_DIRECTORY']

    def build(self, use_case_name: str, grid_search_parameters: dict = None):
        if use_case_name.lower() == "grid-search":
            self.use_case = self._build_grid_search(grid_search_parameters)
        elif use_case_name.lower() == "inference":
            self.use_case = self._build_inference()
        else:
            raise RuntimeError("Invalid use case. Please use either grid-search or inference!")
        return self.use_case

    def start(self):
        try:
            self.use_case.start()
        except Exception:
            get_logger().exception("message")

    def _build_grid_search(self, grid_search_parameters) -> GridSearch:
        data_preprocessor = DataPreprocessor()
        postgres_connector = PostgresConnector()
        dataset = GraphDataset(postgres_connector)
        model_trainer = Trainer(data_preprocessor, self.device, postgres_connector)
        saver = Saver(self.model_directory, self.results_directory)
        grid_search_configurations = self._get_all_grid_search_configurations(grid_search_parameters)
        return GridSearch(dataset, data_preprocessor, model_trainer, grid_search_configurations, saver)

    def _build_inference(self) -> Inference:
        self.model = os.environ['MODEL']
        data_preprocessor = DataPreprocessor()
        model_loader = Loader(self.model)
        model_inferencer = Inferencer(data_preprocessor, self.device)
        postgres_connector = PostgresConnector()
        dataset = GraphDataset(postgres_connector)
        saver = Saver(self.model_directory, self.results_directory)
        return Inference(dataset, data_preprocessor, model_loader, model_inferencer, saver)

    @staticmethod
    def _get_all_grid_search_configurations(grid_search_parameters) -> List[Tuple[Tuple]]:
        all_grid_search_configurations = []
        for key in grid_search_parameters.keys():
            all_grid_search_configurations.append([(key, value) for value in grid_search_parameters[key]])
        return list(itertools.product(*all_grid_search_configurations))
