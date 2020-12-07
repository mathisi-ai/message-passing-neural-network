from torch.optim.optimizer import Optimizer

from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.optimizers import optimizers


def load_optimizer(optimizer_selection: str) -> Optimizer:
    if optimizer_selection in optimizers:
        return optimizers[optimizer_selection]
    else:
        get_logger().info("The {} is not available".format(optimizer_selection))
        raise Exception
