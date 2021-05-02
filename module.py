import torch
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

    # Riccardo C: different weights initialization could be implemented (i.e. Xavier)
    def init_parameters(self, init_option):
        """
        Weights and biases initialization
        """
        if init_option=='none':
            pass
        elif init_option=='normal':
            # Riccardo C: feel free to change these values (are completely random)
            self.weight = self.weight.normal_(0, 1)
            # Riccardo C: I don't know why but starting with bias normal distributd the train 
            # doesn't connverge at all if you have any explanation, let me know!!!
            # self.bias = self.bias.normal_(0, 1)
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

# Riccardo C: other modules could be implemented, i.e. Batch Normalization
# class BatchNorm(Module)


class Sequential(Module):
    # Riccardo C: this description could be improved
    """
    General Class to create a Multi Layer Perceptron 
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
        Forwrard step of the network
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


# Riccardo C: maybe we can create a new loss.py file where we can create a new class Loss (defined on Module)
# and add there all the losses. I have already implemented MSE. We could consider also CrossEntropyLoss and/or MAE
# In alternative, other Loss Function could be added directly in this module using if/else statement

class MSE(Module):
    """
    Mean Squared Error (L2 Norm)
    """
    def __init__(self, model):
        super(MSE).__init__()
        self.model = model
    
    def __call__(self, output, target):
        self.forward(output, target)
        return self

    def forward(self, output, target):
        """
        Forward step: sum( (f(x)-y)^2 )
        """
        self.output = output
        self.target = target
        self.loss = function.mse(self.output, self.target)
        return self.loss

    def backward(self):
        """
        Backward step: compute the mean-squared error derivative with respect to the output
        """
        dloss = function.dmse(self.output, self.target)
        self.model.backward(dloss)

# Riccardo C: could be implemented
# class CrossEntropyLoss(Module)

# Riccardo C: could be implemented
# class MAE(Module)