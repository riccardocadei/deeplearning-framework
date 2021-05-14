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