import json
import pickle

import os
import torch as to


class FileSystemRepository:
    def __init__(self, data_directory: str, dataset_name: str) -> None:
        super().__init__()
        self.data_directory = os.path.join(data_directory, dataset_name)
        self.test_mode = False

    @staticmethod
    def load_json(file_path: str) -> dict:
        with open(file_path, 'r') as f:
            parameters = json.load(f)
        return parameters

    def save(self, filename: str, data_to_save: to.Tensor) -> None:
        with open(os.path.join("./", self.data_directory, filename), 'wb') as file:
            pickle.dump(data_to_save, file)
