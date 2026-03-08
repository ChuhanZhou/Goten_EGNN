from unittest import case

from sympy.strategies.core import switch

from configs.config import config as cfg

import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from abc import ABC, abstractmethod

class MLP(nn.Module):
    def __init__(self,in_features,out_features,hidden_dim=None,pre_norm=False):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_features

        self.pre_norm = pre_norm
        self.ln = nn.LayerNorm(in_features)

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            cfg["activation"],
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        if self.pre_norm:
            x = self.ln(x)
        return self.mlp(x)

def get_decoder(label):
    match (label):
        case "mu":
            raise NotImplementedError("Don't know how to decode: {}".format(label))
        case "alpha":
            return ScalerDecoder(cfg["node_dim"])
        case "homo":
            return ScalerDecoder(cfg["node_dim"])
        case "lumo":
            return ScalerDecoder(cfg["node_dim"])
        case "gap":
            raise NotImplementedError("Don't know how to decode: {}".format(label))
        case "r2":
            raise NotImplementedError("Don't know how to decode: {}".format(label))
        case "zpve":
            return ScalerDecoder(cfg["node_dim"])
        case "u0":
            return ScalerDecoder(cfg["node_dim"])
        case "u":
            return ScalerDecoder(cfg["node_dim"])
        case "h":
            return ScalerDecoder(cfg["node_dim"])
        case "g":
            return ScalerDecoder(cfg["node_dim"])
        case "cv":
            return ScalerDecoder(cfg["node_dim"])
        case other:
            raise NotImplementedError("Don't know how to decode: {}".format(label))

class GraphDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, scaler, vector, batch_index):
        pass

class EquivariantDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim,out_label,mean=0,std=1):
        super().__init__()
        self.decoder = MLP(in_features=in_features, out_features=1, hidden_dim=hidden_dim)

class ScalerDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None):
        super().__init__()
        self.decoder = MLP(in_features=in_features,out_features=1,hidden_dim=hidden_dim)

    def forward(self, scaler, vector, batch_index):
        graph_scaler = scatter_sum(scaler, batch_index, dim=0)
        out = self.decoder(graph_scaler)
        return out

    def toInvariant(self, X):
        outs = []
        for X_l in X:
            l_norm = torch.norm(X_l, dim=1)
            outs.append(l_norm)
        return torch.cat(outs, dim=1)

