import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def get_color(method):
    
    if 'Unrolled_ADMM' in method:
        color = 'xkcd:purple'
    elif 'WienerNet' in method:
        color = 'xkcd:green'
    elif 'ResUNet' in method:
        color = 'xkcd:red'
    elif method == 'No_Deconv':
        color = 'black'
    else:
        color = 'xkcd:brown'
        
    return color

def get_label(method):
    
    if 'Unrolled_ADMM' in method:
        label = 'Unrolled ADMM'
    elif method == 'No_Deconv':
        label = 'No Deconv'
    else:
        label = method
        
    return label

def plot_loss(train_loss, val_loss, epoch_min, model_save_path, model_name):
    n_epochs = len(train_loss)
    plt.figure(figsize=(12,7))
    plt.plot(range(1, n_epochs+1), train_loss, '-o', markersize=4, label='Training Loss')
    plt.plot(range(1, n_epochs+1), val_loss, '-o', markersize=4, label='Validation Loss')
    plt.plot([epoch_min+1], [val_loss[epoch_min]], 'ro', markersize=7, label='Best Epoch')
    plt.title(f'{model_name} Loss Curve', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.yscale("log")
    plt.legend(fontsize=15)
    file_name = f'./{model_save_path}/{model_name}_loss_curve.jpg'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()