import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def get_color(method):
    
    if 'Poisson' in method:
        color = 'xkcd:blue'
    elif 'Unrolled_ADMM' in method:
        color = 'xkcd:purple'
    elif 'ADMMNet' in method:
        color = 'xkcd:blue'
    elif 'Richard-Lucy' in method:
        color = 'xkcd:green' 
    elif 'Tikhonet' in method:
        color = 'xkcd:orange'
    elif method == 'ShapeNet':
        color = 'xkcd:pink'
    elif method == 'FPFS':
        color = 'xkcd:red'
    elif method == 'ngmix':
        color = 'xkcd:pink'
    elif method == 'No_Deconv':
        color = 'black'
    else:
        color = 'xkcd:brown'
        
    return color

def get_label(method):
    
    if 'Poisson' in method:
        label = 'Unrolled ADMM (Poisson)'
    elif 'Unrolled_ADMM' in method:
        label = 'Unrolled ADMM'
    elif 'Richard-Lucy' in method:
        label = 'Richardson-Lucy'
    elif method == 'Wiener':
        label = 'Wiener'
    elif 'Tikhonet' in method:
        label = 'Tikhonet'
    elif 'Identity' in method:
        label = 'Tikhonet (Identity)'
    elif method == 'ShapeNet':
        label = 'ShapeNet'
    elif method == 'FPFS':
        label = 'FPFS'
    elif method == 'ngmix':
        label = 'ngmix'
    elif method == 'No_Deconv':
        label = 'No Deconv'
    else:
        label = method
        
    return label

def plot_loss(train_loss, val_loss, epoch_min, model_save_path, model_name):
    n_epochs = len(train_loss)
    plt.figure(figsize=(12,7))
    plt.plot(range(1, n_epochs+1), train_loss, '-o', markersize=4, label='Train Loss')
    plt.plot(range(1, n_epochs+1), val_loss, '-o', markersize=4, label='Valid Loss')
    plt.plot([epoch_min+1], [val_loss[epoch_min]], 'ro', markersize=7, label='Best Epoch')
    plt.title(f'{model_name} Loss Curve', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.yscale("log")
    plt.legend(fontsize=15)
    file_name = f'./{model_save_path}/{model_name}_loss_curve.jpg'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()