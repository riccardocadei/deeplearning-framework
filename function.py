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
    dMSE(f_x,y) = sum(2* (f_x-y))
    """ 
    return torch.sum(2 * (f_x - y))

def cross_entropy(p, t):
    """
    Cross_entropy(p, t) = sum_n(- sum_k(t*log(p)))
    """
    l = - torch.sum(torch.mul(t, torch.log(p)), 1)
    return l.sum()

def dcross_entropy(p, t):
    """
    dCross_entropy(p,t) = sum_n(-sum_k(t/p))
    """
    l = - torch.sum(torch.mul(t, torch.pow(p,-1)), 1)
    return l.sum()

def mae(x, y):
    """
    MAE(x,y) = sum(abs(x-y))
    """
    return torch.abs(x-y).sum()

def dmae(x, y):
    """
    dMAE(x,y) = sum(sign(x))
    """
    return torch.sign(x).sum()