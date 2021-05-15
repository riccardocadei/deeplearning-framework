import torch

class Optimizer(object):
    def step(self):
        raise NotImplementedError
            
class SGD(Optimizer):
    '''
    Stochastic Gradient Descent
    '''
    def __init__(self, params, lr):
        super(SGD).__init__()

        self.params = params
        self.lr = lr
    
    def step(self):
        """
        Stochastic Gradient Descent step: 
        w = w - eta * [[dL/dw]]
        """
        for (weight, grad) in self.params:
            weight.sub_(self.lr * grad)

class BGD(Optimizer):
    '''
    Batch Gradient Descent
    '''
    def __init__(self, params, lr, batch_size):
        super(SGD).__init__()

        self.params = params
        self.lr = lr
        self.batch_size = batch_size
    
    def step(self):
        """
        Batch Gradient Descent step: 
        w = w - eta * [1/n*sum([dL/dw])]
        """
        for (weight, grad) in self.params:
            weight.sub_(self.lr * torch.mul(grad, 1/self.batch_size))