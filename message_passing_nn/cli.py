import logging
import os

import click

from message_passing_nn.infrastructure.file_system_repository import FileSystemRepository
from message_passing_nn.use_case_factory import UseCaseFactory
from message_passing_nn.utils.logger import setup_logging, get_logger


@click.group("message-passing-nn")
@click.option('--debug', default=False, help='Set the logs to debug level', show_default=True, is_flag=True)
def main(debug):
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)


@click.command('grid-search', help='Starts the grid search')
@click.argument("parameters_path", envvar='PARAMETERS_PATH')
def start_training(parameters_path: str) -> None:
    get_logger().info("Starting grid search")
    grid_search_parameters = init_environment_variables(parameters_path)
    use_case = UseCaseFactory()
    grid_search = use_case.build(use_case_name="grid-search", grid_search_parameters=grid_search_parameters)
    grid_search.start()


@click.command('inference', help='Starts the inference')
@click.argument("parameters_path", envvar='PARAMETERS_PATH')
def start_inference(parameters_path: str) -> None:
    get_logger().info("Starting inference")
    init_environment_variables(parameters_path)
    use_case = UseCaseFactory()
    inference = use_case.build(use_case_name="inference")
    inference.start()


def init_environment_variables(parameters_path):
    grid_search_dictionary = {}
    parameters = FileSystemRepository.load_json(parameters_path)
    for key, value in parameters.items():
        if isinstance(value, str):
            os.environ[key] = value
        elif isinstance(value, list):
            grid_search_dictionary.update({key.lower(): value})
        else:
            raise RuntimeError("Incorrect parameter type. Please use only str and list types!")
    return grid_search_dictionary


main.add_command(start_training)
main.add_command(start_inference)

if __name__ == '__main__':
    main()
