import os
from typing import Type

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.file_system_repository import FileSystemRepository
from message_passing_nn.model.inferencer import Inferencer
from message_passing_nn.model.loader import Loader
from message_passing_nn.model.trainer import Trainer
from message_passing_nn.usecase import UseCase
from message_passing_nn.usecase.grid_search import GridSearch
from message_passing_nn.usecase.inference import Inference
from message_passing_nn.utils.grid_search_parameters_parser import GridSearchParametersParser
from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.saver import Saver


class MessagePassingNN:
    def __init__(self, use_case: UseCase) -> None:
        self.use_case = use_case

    def start(self):
        try:
            self.use_case.start()
        except Exception:
            get_logger().exception("message")


def create_grid_search(parameters_path: str) -> MessagePassingNN:
    _init_environment_variables(FileSystemRepository, parameters_path)
    grid_search_dictionary = GridSearchParametersParser().get_grid_search_dictionary()
    data_path = _get_data_path()
    data_preprocessor = DataPreprocessor()
    trainer = Trainer(data_preprocessor, os.environ['DEVICE'])
    saver = Saver(os.environ['MODEL_DIRECTORY'], os.environ['RESULTS_DIRECTORY'])
    grid_search = GridSearch(data_path,
                             data_preprocessor,
                             trainer,
                             grid_search_dictionary,
                             saver)
    return MessagePassingNN(grid_search)


def create_inference(parameters_path: str) -> MessagePassingNN:
    _init_environment_variables(FileSystemRepository, parameters_path)
    data_path = _get_data_path()
    data_preprocessor = DataPreprocessor()
    model_loader = Loader(os.environ['MODEL'])
    model_inferencer = Inferencer(data_preprocessor, os.environ['DEVICE'])
    saver = Saver(os.environ['MODEL_DIRECTORY'], os.environ['RESULTS_DIRECTORY'])
    inference = Inference(data_path, data_preprocessor, model_loader, model_inferencer, saver)
    return MessagePassingNN(inference)


def _init_environment_variables(file_system_repository: Type['FileSystemRepository'], parameters_path: str):
    parameters = file_system_repository.load_json(parameters_path)
    for key, value in parameters.items():
        os.environ[key] = value


def _get_data_path() -> str:
    return os.path.join("./", os.environ['DATA_DIRECTORY'], os.environ['DATASET_NAME'])
