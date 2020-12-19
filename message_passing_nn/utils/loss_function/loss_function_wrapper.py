class LossFunctionWrapper:
    def __init__(self, loss_function, penalty):
        self.loss_function = loss_function
        self.penalty = penalty

    def __call__(self, outputs, labels, features=None):
        if self.penalty:
            return self.loss_function.forward(outputs, labels, features)
        else:
            return self.loss_function(outputs, labels)
