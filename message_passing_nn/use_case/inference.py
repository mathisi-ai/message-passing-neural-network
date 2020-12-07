from typing import Tuple

from torch.utils.data.dataloader import DataLoader

from message_passing_nn.data import DataPreprocessor
from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.model.inferencer import Inferencer
from message_passing_nn.model.loader import Loader
from message_passing_nn.use_case import UseCase
from message_passing_nn.utils import Saver
from message_passing_nn.utils.logger import get_logger


class Inference(UseCase):
    def __init__(self,
                 dataset: GraphDataset,
                 data_preprocessor: DataPreprocessor,
                 loader: Loader,
                 inferencer: Inferencer,
                 saver: Saver) -> None:
        self.dataset = dataset
        self.data_preprocessor = data_preprocessor
        self.loader = loader
        self.inferencer = inferencer
        self.saver = saver

    def start(self) -> None:
        get_logger().info('Starting Inference')
        inference_dataset, data_dimensions = self._prepare_dataset()
        model = self.loader.load_model(data_dimensions, self.saver.model_directory)
        outputs_labels_pairs = self.inferencer.do_inference(model, inference_dataset)
        self.saver.save_distance_maps(outputs_labels_pairs)
        get_logger().info('Finished Inference')

    def _prepare_dataset(self) -> Tuple[DataLoader, dict]:
        inference_dataset = self.data_preprocessor.get_dataloader(self.dataset)
        data_dimensions = self.data_preprocessor.extract_data_dimensions(self.dataset)
        return inference_dataset, data_dimensions
