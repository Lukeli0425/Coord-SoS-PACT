import numpy as np
import torch
from torch import nn


def get_N_freq_nyquist(samples):
    nyquist_rate = 1 / (2 * (2 * 1 / samples))
    return int(np.floor(np.log(nyquist_rate, 2)))


class PosEncodingNeRF(nn.Module):
    def __init__(self, in_features, N_freq=4, sidelength=None, use_nyquist=False):
        super().__init__()

        self.in_features = in_features
        self.N_freq = N_freq
        self.out_dim = in_features + 2 * in_features * self.N_freq

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.N_freq):
            for j in range(self.in_features):
                c = coords[..., j]
                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], self.out_dim)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
            

class Sine(nn.Module):
    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega
        
    def forward(self, x):
        return torch.sin(self.omega * x)
    
       
class SIREN(nn.Module):
    def __init__(self, hidden_layers, in_features, hidden_features, out_features, activation_fn='sin', pos_encoding=False, N_freq=0):
        super().__init__()
        
        if activation_fn == 'sin':
            self.activation_fn = Sine()
        elif activation_fn == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation_fn == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation_fn == 'softplus':
            self.activation_fn = nn.Softplus()
        else:
            raise NotImplementedError('Activation function not supported.')
        
        if pos_encoding:
            self.pos_encoding = PosEncodingNeRF(in_features, N_freq=N_freq)
            in_features = self.pos_encoding.out_dim
        else:
            self.pos_encoding = False
        self.mlp = nn.Sequential(
            nn.Sequential(nn.Linear(in_features, hidden_features), self.activation_fn),
            *[nn.Sequential(nn.Linear(hidden_features, hidden_features), self.activation_fn) for _ in range(hidden_layers)],
            nn.Linear(hidden_features, out_features)
        )

        self.mlp.apply(sine_init)
        self.mlp[0].apply(first_layer_sine_init)
        
    def forward(self, x):
        coords = self.pos_encoding(x) if self.pos_encoding else x
        return self.mlp(coords), x

