import torch
from optimizer import *
from plotter import *
import math

def train(model, loss_function, train_input, train_target, nb_epochs, lr, optim='SGD', batch_size=1, show_plot=False):
    """
    Trains the model using the data provided as training set
    """
    n = train_input.size(0)
    
    if optim == 'SGD':
        #Stochastic Gradient Descent
        optimizer = SGD(model.param(), lr)
    elif optim == 'BGD':
        #Batch Gradient Descent
        optimizer = BGD(model.param(), lr, batch_size)
    elif optim == 'GD':
        #Gradient Descent
        batch_size = n
        optimizer = BGD(model.param(), lr, batch_size)
    
    losses = []
    val_losses = []
    
    # create validation set
    val_dim = math.floor(n*0.1)
    val_input = train_input[:val_dim]
    train_input = train_input[val_dim:]
    val_target = train_target[:val_dim]
    train_target = train_target[val_dim:]
    
    for epoch in range(1,nb_epochs+1):
        epoch_loss = 0
        if optim == 'SGD':
            for i in range(n-val_dim):
                # forward step
                output = model(train_input[i].view(1,-1)) 
                # compute the loss
                loss = loss_function(output, train_target[i]) 
                epoch_loss += loss.loss.item()
                # reset the grad
                model.zero_grad() 
                # backward
                loss.backward() 
                # sgd step
                optimizer.step()
        else:
            for input, target in zip(train_input.split(batch_size), train_target.split(batch_size)):
                # reset the grad
                model.zero_grad() 
                for i in range(batch_size):
                    # forward step
                    output = model(input[i].view(1,-1)) 
                    # compute the loss
                    loss = loss_function(output, target[i]) 
                    epoch_loss += loss.loss.item()  
                    # backward
                    loss.backward() 
                # bgd step
                optimizer.step()
        
        if epoch % 10 == 0:
            print('Epoch {}/{}, Loss: {}'.format(epoch, nb_epochs, epoch_loss/n))
            losses.append(epoch_loss/n)
            #evaluate loss on the validation set
            epoch_val_loss = 0
            for i in range(val_dim):
                output = model(val_input[i].view(1,-1))
                loss = loss_function(output, val_target[i])
                epoch_val_loss += loss.loss.item()
            val_losses.append(epoch_val_loss/val_dim)

    if show_plot:
        if loss_function.function == 'MSE':
            metric = 'MSE Loss'
        elif loss_function.function == 'CrossEntropy':
            metric = 'Cross-Entropy Loss'
        elif loss_function.function == 'MAE':
            metric = 'MAE Loss'
        plot_train_val(losses, val_losses, 10, al_param=False, metric=metric, save=True, model_name='')


def test(model, test_input, test_label):
    '''
    Test the model computing the error rate (in %)
    '''
    test_pred_label = model(test_input)
    _, test_class  = torch.max(test_label, 1)
    _, test_pred_class = torch.max(test_pred_label, 1)
    
    nb_errors = 0
    n = test_input.size(0)
    for k in range(test_input.size(0)):
        if test_class[k] != test_pred_class[k]:
            nb_errors += 1
    return nb_errors/n*100
