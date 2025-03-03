import os

import ffmpeg
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from torch.fft import fft2, fftshift, ifft2, ifftn, ifftshift

from utils.dataio import *


def standardize(img):
    """Standardize the image to have zero mean and unit standard deviation.

    Args:
        img (`numpy.ndarray`): Input image.

    Returns:
        `numpy.ndarray`: Standardized image.
    """
    return (img - img.mean()) / img.std()

def normalize(img):
    """Normalize the image to `[0, 1]`.

    Args:
        img (`numpy.ndarray`): Input image.

    Returns:
        `numpy.ndarray`: Normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())


def PSF(theta, k, w, delay):
    tf = (torch.exp(-1j*k*(delay - w(theta))) + torch.exp(1j*k*(delay - w(theta+torch.pi)))) / 2
    psf = ifftshift(ifft2(fftshift(tf, dim=[-2,-1])), dim=[-2,-1]).abs()
    psf /= psf.sum(axis=(-2,-1)) # Normalization.
    return psf


def TF(theta, k, w, delay):
    tf = (torch.exp(-1j*k*(delay - w(theta))) + torch.exp(1j*k*(delay - w(theta+np.pi)))) / 2
    return tf


def condition_number(psf):
    """Calculate the condition number of a PSF.

    Args:
        tf (`numpy.ndarray`): PSF image.

    Returns:
        `float`: Condition number.
    """
    # H = fft2(psf)
    H = psf
    return H.abs().max() / H.abs().min()    


def visualize_deconv(results_dir, IP_rec, SOS, time, IP_max, IP_min, SOS_max, SOS_min, params):
    """Visualize the reconstructed IP and SOS for multi-channel deconvolution."""
    fig = plt.figure(figsize=(13,6))
    ax = plt.subplot(1,2,1)
    norm_IP = Normalize(vmax=IP_max, vmin=IP_min)
    plt.imshow(standardize(IP_rec), cmap='gray', norm=norm_IP)
    plt.title("Reconstructed initial pressure", fontsize=16)
    plt.text(400, 30, "t = {:.2f} s".format(time), color='white', fontsize=15)
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = plt.colorbar(cax=cax, norm=norm_IP)
    cb.set_ticks([IP_max, IP_min])
    cb.set_ticklabels(['Max', 'Min'], fontsize=13)
    
    ax = plt.subplot(1,2,2)
    norm_SOS  = Normalize(vmax=SOS_max, vmin=SOS_min)
    plt.imshow(SOS, cmap='magma', norm=norm_SOS)
    plt.title("Given speed of sound", fontsize=16)
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = plt.colorbar(cax=cax, norm=norm_SOS)
    cb.ax.tick_params(labelsize=12)
    cb.set_label("$m \cdot s^{-1}$", fontsize=12)
    plt.suptitle(f'Multi-channel deconvolution ({params})', fontsize=17)
    plt.savefig(os.path.join(results_dir, 'visualization.png'), bbox_inches='tight')
    plt.close()

def visualize_apact(results_dir, IP_rec, SOS_rec, time, IP_max, IP_min, SOS_max, SOS_min, params):
    """Visualize the reconstructed IP and SOS for APACT."""
    fig = plt.figure(figsize=(13,6))
    ax = plt.subplot(1,2,1)
    norm_IP = Normalize(vmax=IP_max, vmin=IP_min)
    plt.imshow(standardize(IP_rec), cmap='gray', norm=norm_IP)
    plt.title("Reconstructed initial pressure", fontsize=16)
    plt.text(381, 30, "t = {:.1f} s".format(time), color='white', fontsize=15)
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = plt.colorbar(cax=cax, norm=norm_IP)
    cb.set_ticks([IP_max, IP_min])
    cb.set_ticklabels(['Max', 'Min'], fontsize=13)
    
    ax = plt.subplot(1,2,2)
    norm_SOS  = Normalize(vmax=SOS_max, vmin=SOS_min)
    plt.imshow(SOS_rec, cmap='magma', norm=norm_SOS)
    plt.title("Reconstructed speed of sound", fontsize=16)
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = plt.colorbar(cax=cax, norm=norm_SOS)
    cb.ax.tick_params(labelsize=12)
    cb.set_label("$m \cdot s^{-1}$", fontsize=12)
    plt.suptitle(f'APACT ({params})', fontsize=17)
    plt.savefig(os.path.join(results_dir, 'visualization.png'), bbox_inches='tight')
    plt.close()


def visualize_nf_apact(results_dir, IP_rec, SOS_rec, loss_list, time, IP_max, IP_min, SOS_max, SOS_min, params):
    """Visualize the reconstructed IP and SOS and the training process for NF-APACT."""
    fig = plt.figure(figsize=(13,10.2))
    gs = gridspec.GridSpec(2, 2)
    ax = plt.subplot(gs[0:1,:])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    plt.plot(range(len(loss_list)), loss_list, '-o', markersize=4.5, linewidth=2, label='loss')
    plt.title("Loss Curve", fontsize=16)
    plt.title("t = {:.2f} s".format(time), loc='right', x=0.94, y=0.91, color='black', fontsize=15)
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.yscale('log')
    
    ax = plt.subplot(gs[1:2,0:1])
    norm_IP = Normalize(vmax=IP_max, vmin=IP_min)
    plt.imshow(standardize(IP_rec), cmap='gray', norm=norm_IP)
    plt.title("Reconstructed initial pressure", fontsize=16)
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = plt.colorbar(cax=cax, norm=norm_IP)
    cb.set_ticks([IP_max, IP_min])
    cb.set_ticklabels(['Max', 'Min'], fontsize=13)
    
    ax = plt.subplot(gs[1:2,1:2])
    norm_SOS  = Normalize(vmax=SOS_max, vmin=SOS_min)
    plt.imshow(SOS_rec, cmap='magma', norm=norm_SOS)
    plt.title("Reconstructed speed of sound", fontsize=16)
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = plt.colorbar(cax=cax, norm=norm_SOS)
    # cb.ax.set_yticks([1500, 1520, 1540, 1560, 1580, 1600])
    cb.ax.tick_params(labelsize=12)
    cb.set_label("$m \cdot s^{-1}$", fontsize=12)
    plt.suptitle(f'NF-APACT ({params})', fontsize=17)
    plt.savefig(os.path.join(results_dir, 'visualization.png'), bbox_inches='tight')
    plt.close()
    
    
def make_video(results_dir, loss_list, task_params, frame_rate=3):
    """Create video to show the convergence of NF-APACT."""
    video_dir = os.path.join(results_dir, 'video')
    os.makedirs(video_dir, exist_ok=True)
    for idx, loss in enumerate(loss_list):
        IP = load_mat(os.path.join(video_dir, f'IP_rec_{idx}.mat'))
        SOS = load_mat(os.path.join(video_dir, f'SOS_rec_{idx}.mat'))
        
        # Visualization
        fig = plt.figure(figsize=(13, 6.2))
        gs = GridSpec(6, 14)
        ax = plt.subplot(1,2,1)
        norm_IP = Normalize(vmax=task_params['IP_max'], vmin=task_params['IP_min'])
        plt.imshow(standardize(IP), cmap='gray', norm=norm_IP)
        plt.title("Reconstructed initial pressure", fontsize=16)
        plt.axis('off')
        cax = fig.add_axes([ax.get_position().x1+0.012, ax.get_position().y0, 0.02, ax.get_position().height])
        cb = plt.colorbar(cax=cax, norm=norm_IP)
        cb.set_ticks([task_params['IP_max'], task_params['IP_min']])
        cb.set_ticklabels(['Max', 'Min'], fontsize=13)
        
        ax = plt.subplot(1,2,2)
        norm_SOS  = Normalize(vmax=task_params['SOS_max'], vmin=task_params['SOS_min'])
        plt.imshow(SOS, cmap='magma', norm=norm_SOS)
        plt.title("Reconstructed speed of sound", fontsize=16)
        plt.axis('off')
        cax = fig.add_axes([ax.get_position().x1+0.012, ax.get_position().y0, 0.02, ax.get_position().height])
        cb = plt.colorbar(cax=cax, norm=norm_SOS)
        # cb.ax.set_yticks([1500, 1520, 1540, 1560, 1580, 1600])
        cb.ax.tick_params(labelsize=12)
        cb.set_label("$m \cdot s^{-1}$", fontsize=12)
        
        plt.suptitle("    Iteration: {}          Loss={:.4e}".format(idx, loss), fontsize=24, horizontalalignment='center')
        # plt.tight_layout()
        plt.savefig(os.path.join(video_dir, f'frame{str(idx).zfill(2)}.jpg'), bbox_inches='tight', dpi=256, transparent=True)
        plt.close()
        
    # Assemble video from frames.
    (
        ffmpeg
        .input(os.path.join(video_dir, 'frame*.jpg'), pattern_type='glob', framerate=frame_rate)
        .output(os.path.join(results_dir, 'video.mp4'), vcodec='h264', loglevel="quiet")
        .overwrite_output()
        .run()
    )
    

def make_video_icon(results_dir, task_params, frame_rate=4):
    """Create video of NF-APACT convergence for the icon on website."""
    video_dir = os.path.join(results_dir, 'video')
    os.makedirs(video_dir, exist_ok=True)
    log = load_log(os.path.join(results_dir, 'log.json'))
    for idx, loss in enumerate(log['loss']):
        IP = load_mat(os.path.join(video_dir, f'IP_rec_{idx}.mat'))
        SOS = load_mat(os.path.join(video_dir, f'SOS_rec_{idx}.mat'))
        
        # Visualization
        fig = plt.figure(figsize=(12.8, 6.4))
        gs = GridSpec(6, 14)
        ax = plt.subplot(1,2,1)
        norm_IP = Normalize(vmax=task_params['IP_max'], vmin=task_params['IP_min'])
        plt.imshow(standardize(IP), cmap='gray', norm=norm_IP)
        plt.axis('off')
        
        ax = plt.subplot(1,2,2)
        norm_SOS  = Normalize(vmax=task_params['SOS_max'], vmin=task_params['SOS_min'])
        plt.imshow(SOS, cmap='magma', norm=norm_SOS)
        plt.axis('off')
        plt.savefig(os.path.join(video_dir, f'frame_icon{str(idx).zfill(2)}.jpg'), bbox_inches='tight', dpi=256, transparent=True)
        plt.close()
        
    # Assemble video from frames.
    (
        ffmpeg
        .input(os.path.join(video_dir, 'frame_icon*.jpg'), pattern_type='glob', framerate=frame_rate)
        .output(os.path.join(results_dir, 'video_icon.mp4'), vcodec='h264', loglevel="quiet")
        .overwrite_output()
        .run()
    )
    