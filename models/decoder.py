from configs.config import config as cfg

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,in_features,out_features,hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            cfg["activation"],
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        return self.mlp(x)

class ScalerDecoder(nn.Module):
    def __init__(self,in_features,hidden_dim):
        super().__init__()
        self.decoder = MLP(in_features=in_features,out_features=1,hidden_dim=hidden_dim)

    def forward(self, h):
        return self.decoder(h)
