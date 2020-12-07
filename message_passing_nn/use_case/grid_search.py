from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data.dataloader import DataLoader

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.model.trainer import Trainer
from message_passing_nn.use_case import UseCase
from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.saver import Saver


class GridSearch(UseCase):
    def __init__(self,
                 dataset: GraphDataset,
                 data_preprocessor: DataPreprocessor,
                 trainer: Trainer,
                 grid_search_configurations: List[Tuple[Tuple]],
                 saver: Saver) -> None:
        self.dataset = dataset
        self.data_preprocessor = data_preprocessor
        self.trainer = trainer
        self.grid_search_configurations = grid_search_configurations
        self.saver = saver
        self.results = {'training_loss': {},
                        'validation_loss': {},
                        'test_loss': {}}

    def start(self) -> Dict:
        get_logger().info('Starting Grid Search')
        configuration_id = ''
        for configuration in self.grid_search_configurations:
            configuration_dictionary, dataloaders = self._build_a_configuration(configuration)
            self._train_a_single_configuration(configuration_dictionary['configuration_id'],
                                               dataloaders,
                                               configuration_dictionary['epochs'],
                                               configuration_dictionary['validation_period'])
        self.saver.save_results(self.results, configuration_id)
        get_logger().info('Finished Training')
        return self.results

    def _train_a_single_configuration(self,
                                      configuration_id: str,
                                      dataloaders: Tuple[DataLoader, DataLoader, DataLoader],
                                      epochs: int,
                                      validation_period: int) -> None:
        get_logger().info('Starting training:'.format(configuration_id))
        training_data, validation_data, test_data = dataloaders
        validation_loss_max = np.inf
        for epoch in range(1, epochs + 1):
            training_loss = self.trainer.do_train_step(training_data, epoch)
            self.results['training_loss'][configuration_id].update({epoch: training_loss})
            if epoch % validation_period == 0:
                validation_loss = self.trainer.do_evaluate_step(validation_data, epoch)
                self._save_best_model(configuration_id, epoch, validation_loss, validation_loss_max)
                self.results['validation_loss'][configuration_id].update({epoch: validation_loss})
        test_loss = self.trainer.do_evaluate_step(test_data)
        self.results['test_loss'][configuration_id].update({'final_epoch': test_loss})
        get_logger().info('Finished training:'.format(configuration_id))

    def _save_best_model(self, configuration_id, epoch, validation_loss, validation_loss_max):
        if validation_loss < validation_loss_max:
            self.saver.save_model(epoch, configuration_id, self.trainer.model)

    def _build_a_configuration(self, configuration: Tuple[Tuple]) \
            -> Tuple[dict, Tuple[DataLoader, DataLoader, DataLoader]]:
        configuration_dictionary = self._get_configuration_dictionary(configuration)
        dataloaders, data_dimensions = self._prepare_dataset(configuration_dictionary)
        self.trainer.build(data_dimensions, configuration_dictionary)
        self._update_results_dict_with_configuration_id(configuration_dictionary)
        return configuration_dictionary, dataloaders

    @staticmethod
    def _get_configuration_dictionary(configuration: Tuple[Tuple]) -> dict:
        configuration_dictionary = dict(((key, value) for key, value in configuration))
        configuration_id = 'configuration&id'
        for key, value in configuration_dictionary.items():
            configuration_id += '__' + '&'.join([key, str(value)])
        configuration_dictionary.update({'configuration_id': configuration_id})
        return configuration_dictionary

    def _prepare_dataset(self, configuration_dictionary: Dict) \
            -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], dict]:
        dataloaders = self.data_preprocessor.train_validation_test_split(self.dataset,
                                                                         configuration_dictionary['batch_size'],
                                                                         configuration_dictionary['validation_split'],
                                                                         configuration_dictionary['test_split'])
        data_dimensions = self.data_preprocessor.extract_data_dimensions(self.dataset)
        return dataloaders, data_dimensions

    def _update_results_dict_with_configuration_id(self, configuration_dictionary: Dict) -> None:
        for key in self.results:
            self.results[key].update({configuration_dictionary['configuration_id']: {}})
