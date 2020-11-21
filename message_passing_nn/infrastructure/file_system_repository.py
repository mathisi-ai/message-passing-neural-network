import json
import pickle

import torch as to


class FileSystemRepository:
    def __init__(self, data_directory: str, dataset: str) -> None:
        super().__init__()
        self.data_directory = data_directory + dataset + '/'
        self.test_mode = False

    @staticmethod
    def load_json(file_path: str) -> dict:
        with open(file_path, 'r') as f:
            parameters = json.load(f)
        return parameters

    def save(self, filename: str, data_to_save: to.Tensor) -> None:
        with open(self.data_directory + filename, 'wb') as file:
            pickle.dump(data_to_save, file)
