# baseline implementation of PINNs
# paper: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
# link: https://www.sciencedirect.com/science/article/pii/S0021999118307125
# code: https://github.com/maziarraissi/PINNs

import torch
import torch.nn as nn

class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(PINNs, self).__init__()

        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        # Asegurarse de que la última capa tenga el doble de neuronas (para partes real e imaginaria)
        layers.append(nn.Linear(in_features=hidden_dim, out_features=2*out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x, t):
        src = torch.cat((x, t), dim=-1)
        output = self.linear(src)
        # Separar la salida en partes real e imaginaria
        real, imag = output[..., :out_dim], output[..., out_dim:]
        return torch.complex(real, imag)
