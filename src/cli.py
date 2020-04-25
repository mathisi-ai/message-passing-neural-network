import logging

import click
import sys

from src.fixtures.loss_functions import loss_functions
from src.fixtures.optimizers import optimizers
from src.message_passing_nn import create


@click.command("start-training", help='Starts the training')
@click.option('--dataset', help='Select which dataset to use', required=True, type=str)
@click.option('--epochs', default=10, help='Set the number of epochs', show_default=True, type=int)
@click.option('--loss_function', default='MSE', help='Set the loss function', show_default=True,
              type=click.Choice(list(loss_functions.keys())))
@click.option('--optimizer', default='SGD', help='Set the optimizer', show_default=True,
              type=click.Choice(list(optimizers.keys())))
@click.option('--data_path', default='data/', help='Set the path of your data folder', required=True,
              type=str)
@click.option('--batch_size', default=1, help='Set the batch size', required=True, type=int)
def start_training(dataset: str, epochs: int, loss_function: str, optimizer: str, data_path: str, batch_size: int) -> None:
    get_logger().info("Starting training")
    message_passing_nn = create(dataset, epochs, loss_function, optimizer, data_path, batch_size)
    message_passing_nn.start()


@click.group("message-passing-nn")
@click.option('--debug', default=False, help='Set the logs to debug level', show_default=True, is_flag=True)
def main(debug):
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)


def setup_logging(log_level):
    get_logger().setLevel(log_level)

    logOutputFormatter = logging.Formatter(
        '%(asctime)s %(levelname)s - %(message)s [%(filename)s:%(lineno)s] [%(relativeCreated)d]')

    stdoutStreamHandler = logging.StreamHandler(sys.stdout)
    stdoutStreamHandler.setLevel(log_level)
    stdoutStreamHandler.setFormatter(logOutputFormatter)

    get_logger().addHandler(stdoutStreamHandler)

    stderrStreamHandler = logging.StreamHandler(sys.stdout)
    stderrStreamHandler.setLevel(logging.WARNING)
    stderrStreamHandler.setFormatter(logOutputFormatter)

    get_logger().addHandler(stderrStreamHandler)


def get_logger() -> logging.Logger:
    return logging.getLogger('message_passing_nn')


main.add_command(start_training)

if __name__ == '__main__':
    main()
