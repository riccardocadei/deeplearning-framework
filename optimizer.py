import torch

class Optimizer(object):
    def step(self):
        raise NotImplementedError
            

class SGD(Optimizer):
    '''
    Implements stochastic gradient descent 
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