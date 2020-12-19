from torch import nn

from message_passing_nn.utils.logger import get_logger
from message_passing_nn.utils.models import models


def load_model(model_selection: str) -> nn.Module:
    if model_selection in models:
        return models[model_selection]
    else:
        get_logger().info("The " + model_selection + " model is not available")
        raise Exception
