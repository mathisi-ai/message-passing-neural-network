import logging

import click

from message_passing_nn.create_message_passing_nn import create_grid_search, create_inference
from message_passing_nn.utils.logger import setup_logging, get_logger


@click.group("message-passing-nn")
@click.option('--debug', default=False, help='Set the logs to debug level', show_default=True, is_flag=True)
def main(debug):
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)


@click.command('grid-search', help='Starts the grid search')
@click.argument("parameters_path", envvar='PARAMETERS_PATH')
def start_training(parameters_path: str) -> None:
    message_passing_nn = create_grid_search(parameters_path)
    message_passing_nn.start()


@click.command('inference', help='Starts the inference')
@click.argument("parameters_path", envvar='PARAMETERS_PATH')
def start_inference(parameters_path: str) -> None:
    get_logger().info("Starting inference")
    message_passing_nn = create_inference(parameters_path)
    message_passing_nn.start()


main.add_command(start_training)
main.add_command(start_inference)

if __name__ == '__main__':
    main()
