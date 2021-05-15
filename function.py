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
    dMSE(f_x,y) = 2* (f_x-y)
    """ 
    return 2 * (f_x - y)

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    return torch.exp(x) / torch.sum(torch.exp(x))

def cross_entropy(p, t):
    """
    Cross_entropy(p, t) = - sum(t*log(p))
    """
    #print('p', p)
    p = softmax(p)
    #print('ps',p)
    l = - torch.mul(t, torch.log(p)).sum(dim=1)
    #print('l', l)
    return l

def dcross_entropy(p, t):
    """
    dCross_entropy(p,t) = - t/p
    """
    p = softmax(p)
    return p-t

def mae(x, y):
    """
    MAE(x,y) = sum(abs(x-y))
    """
    return torch.abs(x-y).sum()

def dmae(x, y):
    """
    dMAE(x,y) = sign(x-y)
    """
    return torch.sign(x-y)