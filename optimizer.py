class Optimizer(object):
    def step(self):
        raise NotImplementedError
            
class SGD(Optimizer):
    '''
    Stochastic gradient descent
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


# Riccardo C: other first order optimizer could be consider (maybe something with a bit of inertia)