import torch

class Optimizer(object):
    def step(self):
        raise NotImplementedError
            

class BGD(Optimizer):
    '''
    Batch Gradient Descent
    '''
    def __init__(self, params, lr):
        super(BGD).__init__()

        self.params = params
        self.lr = lr
    
    def step(self):
        """
        Batch Gradient Descent step: 
        w = w - eta * [1/n*sum([dL/dw])]
        """
        for (weight, grad) in self.params:
            weight.sub_(self.lr * grad)