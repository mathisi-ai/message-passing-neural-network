import numpy as np
from torch.nn.modules.module import Module

from message_passing_nn.fixtures.amino_acid_maps import amino_acid_to_index, map_amino_acid_codes
from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.loss_function.loss_functions import loss_functions
from message_passing_nn.utils.loss_function.loss_with_penalty import LossWithPenalty
from message_passing_nn.utils.postgres_connector import PostgresConnector


def load_loss_function(configuration: dict, postgres_connector: PostgresConnector = None) -> Module or LossWithPenalty:
    if configuration["loss_function"] in loss_functions:
        if 'penalty' in configuration["loss_function"].lower():
            penalty_dictionary = build_penalty_loss(postgres_connector)
            return LossWithPenalty(loss_function=loss_functions[configuration["loss_function"]](),
                                   penalty=penalty_dictionary,
                                   scaling_factor=configuration['scaling_factor'],
                                   penalty_decimals=configuration['penalty_decimals'],
                                   batch_size=configuration['batch_size']), True
        return loss_functions[configuration["loss_function"]](), False
    else:
        get_logger().info("The {} is not available".format(configuration["loss_function"]))
        raise Exception


def build_penalty_loss(postgres_connector: PostgresConnector):
    penalty_dictionary = {}
    postgres_connector.open_connection()
    penalty = postgres_connector.execute_query(fields=["residue", "penalty"], use_case='penalty')
    postgres_connector.close_connection()
    for residue, penalty in penalty:
        residue_index = amino_acid_to_index[map_amino_acid_codes[residue]]
        penalty_dictionary.update({residue_index: np.array(penalty)})
    return penalty_dictionary
