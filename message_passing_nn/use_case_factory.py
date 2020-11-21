import os

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.file_system_repository import FileSystemRepository
from message_passing_nn.model.inferencer import Inferencer
from message_passing_nn.model.loader import Loader
from message_passing_nn.model.trainer import Trainer
from message_passing_nn.usecase.grid_search import GridSearch
from message_passing_nn.usecase.inference import Inference
from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.saver import Saver


class UseCaseFactory:
    def __init__(self, parameters_path: str) -> None:
        self.parameters_path = parameters_path
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
        data_path = self._get_data_path()
        data_preprocessor = DataPreprocessor()
        trainer = Trainer(data_preprocessor, os.environ['DEVICE'])
        saver = Saver(os.environ['MODEL_DIRECTORY'], os.environ['RESULTS_DIRECTORY'])
        return GridSearch(data_path, data_preprocessor, trainer, self.grid_search_dictionary, saver)

    def _create_inference(self) -> Inference:
        data_path = self._get_data_path()
        data_preprocessor = DataPreprocessor()
        model_loader = Loader(os.environ['MODEL'])
        model_inferencer = Inferencer(data_preprocessor, os.environ['DEVICE'])
        saver = Saver(os.environ['MODEL_DIRECTORY'], os.environ['RESULTS_DIRECTORY'])
        return Inference(data_path, data_preprocessor, model_loader, model_inferencer, saver)

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

    @staticmethod
    def _get_data_path() -> str:
        return os.path.join("./", os.environ['DATA_DIRECTORY'], os.environ['DATASET_NAME'])
