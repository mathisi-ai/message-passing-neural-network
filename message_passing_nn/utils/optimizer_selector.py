from torch.optim.optimizer import Optimizer

from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.optimizers import optimizers


class OptimizerSelector:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_optimizer(optimizer_selection: str) -> Optimizer:
        if optimizer_selection in optimizers:
            return optimizers[optimizer_selection]
        else:
            get_logger().info("The " + optimizer_selection + " is not available")
            raise Exception
