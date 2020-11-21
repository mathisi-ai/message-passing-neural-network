import os
from typing import List, Dict

import numpy as np

from message_passing_nn.fixtures.characters import GRID_SEARCH_SEPARATION_CHARACTER
from message_passing_nn.utils.logger import get_logger


class GridSearchParametersParser:
    def __init__(self) -> None:
        pass

    def get_grid_search_dictionary(self) -> Dict:
        return {
            'model': self._parse_string_selections(os.environ["MODEL"]),
            'epochs': self._parse_integer_range(os.environ["EPOCHS"]),
            'loss_function': self._parse_string_selections(os.environ["LOSS_FUNCTION"]),
            'optimizer': self._parse_string_selections(os.environ["OPTIMIZER"]),
            'batch_size': self._parse_integer_range(os.environ["BATCH_SIZE"]),
            'validation_split': self._parse_float_range(os.environ["VALIDATION_SPLIT"]),
            'test_split': self._parse_float_range(os.environ["TEST_SPLIT"]),
            'time_steps': self._parse_integer_range(os.environ["TIME_STEPS"]),
            'validation_period': self._parse_integer_range(os.environ["VALIDATION_PERIOD"])
        }

    def _parse_integer_range(self, field: str) -> List[int]:
        integer_range = field.split(GRID_SEARCH_SEPARATION_CHARACTER)
        if len(integer_range) == 1:
            return [int(integer_range[0])]
        elif len(integer_range) == 3:
            min_range, max_range, number_of_values = integer_range
            integer_range = np.linspace(int(min_range), int(max_range), int(number_of_values))
            return [int(number) for number in integer_range]
        else:
            get_logger().info("Incorrect values for integer range. Please either provide "
                              "a single integer or three integers separated by & (min&max&values)")
            raise Exception

    def _parse_float_range(self, field: str) -> List[float]:
        float_range = field.split(GRID_SEARCH_SEPARATION_CHARACTER)
        if len(float_range) == 1:
            return [float(float_range[0])]
        elif len(float_range) == 3:
            min_range, max_range, number_of_values = float_range
            float_range = np.linspace(float(min_range), float(max_range), int(number_of_values))
            return [float(number) for number in float_range]
        else:
            get_logger().info("Incorrect values for float range. Please either provide "
                              "a single float or two floats and an integer separated by & (min&max&values)")
            raise Exception

    @staticmethod
    def _parse_string_selections(string_selection: str) -> List[str]:
        return string_selection.split(GRID_SEARCH_SEPARATION_CHARACTER)
