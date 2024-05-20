import torch
from torch import nn
import numpy as np


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
    def __init__(self, num_hidden_layers, activation_fn, in_features, hidden_features, out_features):
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
            raise ValueError('Activation function not supported.')
            
        self.mlp = nn.Sequential(
            nn.Sequential(nn.Linear(in_features, hidden_features), self.activation_fn),
            *[nn.Sequential(nn.Linear(hidden_features, hidden_features), self.activation_fn) for _ in range(num_hidden_layers)],
            nn.Linear(hidden_features, out_features)
        )
        
        self.mlp.apply(sine_init)
        self.mlp[0].apply(first_layer_sine_init)
        
    def forward(self, x):
        return self.mlp(x), x


if __name__ == '__main__':
    model = SIREN(num_hidden_layers=3, activation_fn='sin', in_features=3, hidden_features=256, out_features=1)
    print(model.mlp[0].weight)
    # print(model)