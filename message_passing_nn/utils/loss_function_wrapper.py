class LossFunctionWrapper:
    def __init__(self, loss_function, penalty):
        self.loss_function = loss_function()
        self.penalty = penalty

    def __call__(self, labels, outputs, features=None):
        if self.penalty:
            self.loss_function(labels, outputs, features)
        else:
            self.loss_function(labels, outputs)
