import torch

class Optimizer(object):
    def step(self):
        raise NotImplementedError
            

class SGD(Optimizer):
    '''
    Gradient Descent (for an individual or batch of examples)
    '''
    def __init__(self, params, lr):
        super(SGD).__init__()

        self.params = params
        self.lr = lr
    
    def step(self):
        """ 
        w = w - eta * dL/dw
        """
        for (weight, grad) in self.params:
            weight.sub_(self.lr * grad)