import torch

def linear(x, weight, bias=None):
    """
    f(x) = w*x + b
    """
    if bias is not None:
        return x.mm(weight).add(bias) 
    else:
        return x.mm(weight)

def relu(x):
    """
    ReLu(x) = max(0,x)
    """
    return torch.max(x, torch.zeros(x.size()))

def drelu(x):
    """
    dReLu(x) = 1 if (x>0), 0 otherwise
    """
    return (x > 0).float()

# Riccardo C: not sure if this implementation is efficient from a computational point of view
def tanh(x):
    """
    TanH(x) = (e^x - e^-x) / (e^x + e^-x)
    """
    return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())

def dtanh(tanh_x):
    """
    dTanH(x) = 1 - TanH(x)^2
    """
    return 1 - tanh_x**2

# Riccardo C: not sure if this implementation is efficient from a computational point of view
def sigmoid(x):
    """
    Sigmoid(x) = 1 / (1+e^-x)
    """
    return 1 / (1 + (-x).exp())

def dsigmoid(sigmoid_x):
    """
    dSigmoid(x) = Sigmoid(x) * (1-Sigmoid(x)) = Sigmoid(x) - Sigmoid(x)^2
    """
    return sigmoid_x - sigmoid_x**2

def mse(f_x, y):
    """
    MSE(f_x,y) = sum( (f_x-y)^2 )
    """
    return (f_x - y).pow(2).sum()

def dmse(f_x, y):
    """
    dMSE(f_x,y) = 2* (f_x-y) 
    """ 
    return 2 * (f_x - y)