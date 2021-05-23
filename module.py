import torch
import math
import function


class Module(object):
    '''
    Abstract Module Class of a Neural Network Layer (module)
    '''
    def forward (self, *input):
        raise NotImplementedError

    def backward (self, *gradwrtoutput):
        raise NotImplementedError

    def param (self):
        return []


class Linear(Module):
    """
    Fully Connected Layer
    """
    def __init__(self, input_dim, output_dim, bias=True, init_option='normal'):
        super(Linear).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = torch.zeros(input_dim, output_dim)
        self.dweight = torch.zeros(input_dim, output_dim)

        if bias:
            self.bias = torch.zeros(1,output_dim)
            self.dbias = torch.zeros(1,output_dim)
        else:
            self.bias = None
            self.dbias = None

        self.init_parameters(init_option)

    def forward(self, input):
        """
        Forward step: w*x + b
        """
        self.input = input
        return function.linear(self.input, self.weight, self.bias)

    def backward(self, gradwrtoutput):
        """
        Backward step: compute the loss derived with respect to the input
        """
        self.gradwrtoutput = gradwrtoutput
        # [[dL/dW]] = [dL/ds] * x.t
        self.dweight.add_(self.input.t().mm(gradwrtoutput))
        # [dL/db] = [dL/ds]
        if self.dbias is not None: self.dbias.add_(gradwrtoutput)
        # [dL/dx] = W.t * [dL/ds]
        return gradwrtoutput.mm(self.weight.t())

    def param(self):
        """
        Collect in a list of couples the parameters (weight and bias) of the module and 
        the loss derivative with respect to them
        """
        if self.bias is None and self.dbias is None:
            return [(self.weight, self.dweight)]
        else:
            return [(self.weight, self.dweight), (self.bias, self.dbias)]


    def init_parameters(self, init_option):
        """
        Weights and biases initialization
        """
        if init_option=='none':
            pass
        elif init_option=='standard':
            #Standard normal initialization
            self.weight = self.weight.normal_(0, 1)
        elif init_option=='normal':
            #ResNet Xavier initialization
            n = self.output_dim
            self.weight = self.weight.normal_(0, math.sqrt(2. / n))
        elif init_option=='xavier':
            #Original Xavier initialization
            n_in = self.input_dim
            n_out = self.output_dim
            std = math.sqrt(2. / (n_in + n_out))
            self.weight = self.weight.normal_(0, std)
        else:
            raise TypeError('Weights inizialiazation not known')

class ReLU(Module):
    """
    Elementwise activation function: ReLU
    """
    def __init__(self):
        super(ReLU).__init__()

    def forward(self, input):
        """
        Forward step: ReLU(s)
        """
        self.input = input
        return function.relu(input)

    def backward(self, gradwrtoutput):
        """
        Backward step: compute the loss derived with respect 
        to the input
        """
        return gradwrtoutput * function.drelu(self.input)

class TanH(Module):
    """
    Elementwise activation function: TanH
    """
    def __init__(self):
        super(TanH).__init__()

    def forward(self, input):
        """
        Forward step: TanH(s)
        """
        self.input = input
        return function.tanh(input)

    def backward(self, gradwrtoutput):
        """
        Backward step: compute the loss derived with respect
        to the input
        """
        return gradwrtoutput * function.dtanh(self.input)

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid).__init__()

    def forward(self, input):
        """
        Forward step: Sigmoid(s)
        """
        self.input = input
        self.output = function.sigmoid(input)
        return self.output
    
    def backward(self, gradwrtoutput):
        """
        Backward step: compute the loss derived with respect
        to the input
        """
        return gradwrtoutput * function.dsigmoid(self.output)


class Sequential(Module):
    """
    General Class to create a Neural Network given a sequence
    of consecutive layers
    """ 
    def __init__(self, *args):
        super(Sequential).__init__()
        
        self.sequential_modules = []
        for module in args:
            self.sequential_modules.append(module)
    
    def __call__(self, input):
        return self.forward(input)
            
    def forward(self, input):
        """
        Forward step of the network
        """
        self.input = input
        output = input
        for module in self.sequential_modules:
            output = module.forward(output)
        self.output = output
        return self.output
    
    def backward(self, gradwrtoutput):
        """
        Backward step of the network
        """
        for module in reversed(self.sequential_modules):
            gradwrtoutput = module.backward(gradwrtoutput)

    def param(self):
        '''
        Collect in a list of couples the parameters of each module of the network, 
        together with the loss derivative with respect to them
        '''
        parameters = []
        for module in self.sequential_modules:
            parameters.extend(module.param())
        return parameters

    def zero_grad(self):
        '''
        Reset all the gradients of the network to 0
        '''
        for couple in self.param():
            weight, gradient = couple
            if (weight is None) or (gradient is None):
                continue
            else:
                gradient.zero_()


class Loss(Module):
    """
    General class to define the following loss
    functions:
    - 'MSE': Mean Squared Error (L2 Norm)
    - 'CrossEntropy': Cross-Entropy
    - 'MAE': Mean Absolute Error
    """
    def __init__(self, model, fun='MSE'):
        super(Loss).__init__()
        self.model = model
        self.function = fun

    def __call__(self, output, target):
        self.forward(output, target)
        return self

    def forward(self, output, target):
        """
        Forward step, depending on the chosen loss function
        """
        self.output = output
        self.target = target
        if self.function=='MSE':
            self.loss = function.mse(self.output, self.target)
        elif self.function=='CrossEntropy':
            self.loss = function.cross_entropy(self.output, self.target)
        elif self.function=='MAE':
            self.loss = function.mae(self.output, self.target)
        return self.loss

    def backward(self):
        """
        Backward step: compute the loss function derivative with respect to the output
        """
        if self.function=='MSE':
            dloss = function.dmse(self.output, self.target)
        elif self.function=='CrossEntropy':
            dloss = function.dcross_entropy(self.output, self.target)
        elif self.function=='MAE':
            dloss = function.dmae(self.output, self.target)
        
        self.model.backward(dloss)