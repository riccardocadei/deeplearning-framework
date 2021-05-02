import torch
from optimizer import SGD

# Riccardo C: This function can be improved a lot:
# - Batch Gradient Descent could be integreted
# - Error rate should be evaluted during the traing either on the training set but also on a validation set
# - Loss/Error rate evolution should be be plotted (feel free to copy paste the function that I wrote for MiniProject1)
# - save the best model (when the validation loss is minimum)

def train(model, loss_function, train_input, train_target, nb_epochs, lr, plot=False):
    """
    Training
    """

    optimizer = SGD(model.param(), lr)

    losses = []
    for epoch in range(1,nb_epochs+1):

        epoch_loss = 0
        n = train_input.size(0)
        for i in range(n):
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
        if epoch % 10 == 0:
            print('Epoch {}/{}, Loss: {}'.format(epoch, nb_epochs, epoch_loss/n))
            losses.append(epoch_loss/n)

    #if plot:
        #plot the evolution of the loss

def test(model, test_input, test_label):
    '''
    Test the model computing the error rate 
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
