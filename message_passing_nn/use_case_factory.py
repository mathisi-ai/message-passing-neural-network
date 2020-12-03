import os

from message_passing_nn.infrastructure.graph_dataset import GraphDataset

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.file_system_repository import FileSystemRepository
from message_passing_nn.model.inferencer import Inferencer
from message_passing_nn.model.loader import Loader
from message_passing_nn.model.trainer import Trainer
from message_passing_nn.use_case.grid_search import GridSearch
from message_passing_nn.use_case.inference import Inference
from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.saver import Saver
from message_passing_nn.utils.postgres_connector import PostgresConnector


class UseCaseFactory:
    def __init__(self, parameters_path: str, test_mode: bool = False) -> None:
        self.parameters_path = parameters_path
        self.test_mode = test_mode
        self.use_case = None
        self.grid_search_dictionary = None
        self._init_environment_variables()

    def build(self, use_case_name: str):
        if use_case_name.lower() == "grid-search":
            self.use_case = self._create_grid_search()
        elif use_case_name.lower() == "inference":
            self.use_case = self._create_inference()
        else:
            raise RuntimeError("Invalid use case. Please use either grid-search or inference!")
        return self.use_case

    def start(self):
        try:
            self.use_case.start()
        except Exception:
            get_logger().exception("message")

    def _create_grid_search(self) -> GridSearch:
        data_preprocessor = DataPreprocessor()
        postgres_connector = PostgresConnector()
        dataset = GraphDataset(postgres_connector)
        model_trainer = Trainer(data_preprocessor, os.environ['DEVICE'], postgres_connector)
        saver = Saver(os.environ['MODEL_DIRECTORY'], os.environ['RESULTS_DIRECTORY'])
        return GridSearch(dataset, data_preprocessor, model_trainer, self.grid_search_dictionary, saver)

    @staticmethod
    def _create_inference() -> Inference:
        data_preprocessor = DataPreprocessor()
        model_loader = Loader(os.environ['MODEL'])
        model_inferencer = Inferencer(data_preprocessor, os.environ['DEVICE'])
        postgres_connector = PostgresConnector()
        dataset = GraphDataset(postgres_connector)
        saver = Saver(os.environ['MODEL_DIRECTORY'], os.environ['RESULTS_DIRECTORY'])
        return Inference(dataset, data_preprocessor, model_loader, model_inferencer, saver)

    def _init_environment_variables(self):
        self.grid_search_dictionary = {}
        parameters = FileSystemRepository.load_json(self.parameters_path)
        for key, value in parameters.items():
            if isinstance(value, str):
                os.environ[key] = value
            elif isinstance(value, list):
                self.grid_search_dictionary.update({key.lower(): value})
            else:
                raise RuntimeError("Incorrect parameter type. Please use only str and list types!")
