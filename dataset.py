import torch
import math

import matplotlib.pyplot as plt

def generate_dataset_disk(n=1000, one_hot_encoding=True, plot=True):
    """
    Generate a dataset of n points in [0,1]^2 with label equal to 0
    if outside the disk of radius 1/sqrt(2*pi) and 1 inside
    """
    input = torch.empty(n,2).uniform_(0,1)
    center = torch.empty(1,2).fill_(1/2)
    label = ((input - center).pow(2).sum(1).sqrt() < 1/math.sqrt(2*math.pi)).long()
    if plot:
        plot_dataset_disk(input, label)
    if one_hot_encoding:
        mask = torch.eye(2) 
        label = mask[label]
    return input, label

# Riccardo C: This plot could be improved a bit and added in the report
def plot_dataset_disk(input, label):
    """
    Plot the dataset in 2D
    """
    colors = []
    for i in label:
        if (i==0):
            colors.append('m')
        else:
            colors.append('c')
    plt.scatter(input[:,0], input[:,1], color=colors)
    plt.xlabel("x")
    plt.ylabel("y")