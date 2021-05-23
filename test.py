import torch
torch.set_grad_enabled(False)
import math
from dataset import generate_dataset_disk
from module import *
from training import train, test
from plotter import *

# Load the dataset
train_input, train_label = generate_dataset_disk()
test_input, test_label = generate_dataset_disk(plot=False)

model = Sequential(Linear(2, 25), ReLU(),
                   Linear(25,25), ReLU(),
                   Linear(25,25), ReLU(),
                   Linear(25,2), Sigmoid())
loss_function = Loss(model, fun='MSE')
batch_size = 25
nb_epochs = 300
lr = 1e-3

train(model,
     loss_function,
    train_input, 
    train_label, 
    nb_epochs, 
    lr, 
    batch_size=batch_size, 
    show_plot=True)

train_error = test(model, train_input, train_label)
test_error = test(model, test_input, test_label)

print('Train Error: {}%'.format(train_error))
print('Test Error: {}%'.format(test_error))
