import logging

import click

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
    use_case = UseCaseFactory(parameters_path)
    grid_search = use_case.build("grid-search")
    grid_search.start()


@click.command('inference', help='Starts the inference')
@click.argument("parameters_path", envvar='PARAMETERS_PATH')
def start_inference(parameters_path: str) -> None:
    get_logger().info("Starting inference")
    use_case = UseCaseFactory(parameters_path)
    inference = use_case.build("inference")
    inference.start()


main.add_command(start_training)
main.add_command(start_inference)

if __name__ == '__main__':
    main()
