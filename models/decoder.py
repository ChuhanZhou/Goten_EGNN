from configs.config import config as cfg

import torch
import torch.nn as nn
from torch_scatter import scatter_sum,scatter_mean
from abc import ABC, abstractmethod

class MLP(nn.Module):
    def __init__(self,in_features,out_features,hidden_dim=None,pre_norm=False,act_fn=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_features

        if act_fn is None:
            act_fn = cfg["activation"]

        self.ln = None
        if pre_norm:
            self.ln = nn.LayerNorm(in_features)

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        if self.ln is not None:
            x = self.ln(x)
        return self.mlp(x)

class Gate(nn.Module): # gate
    def __init__(self,in_features,out_features,hidden_dim=None,pre_norm=False,act_fn=None):
        super().__init__()

        self.mlp = MLP(in_features,out_features,hidden_dim,pre_norm,act_fn)
        self.w = nn.Linear(in_features,out_features)

    def forward(self, s, v):
        return self.mlp(s.unsqueeze(1)) * self.w(v)

def get_decoder(label):
    match (label):
        case "mu": # dipole moment
            return DipoleMomentDecoder(cfg["node_dim"])
        case "alpha": # isotropic polarizability
            return IsotropicPolarizabilityDecoder(cfg["node_dim"])
        case "homo":
            return IntensiveScalerDecoder(cfg["node_dim"])
        case "lumo":
            return IntensiveScalerDecoder(cfg["node_dim"])
        case "gap": # gap(lumo-homo)
            raise NotImplementedError("Don't know how to decode: {}".format(label))
        case "r2": # electronic spatial extent
            return ElectronicSpatialExtentDecoder(cfg["node_dim"])
        case "zpve": # zero point vibrational energy
            return IntensiveScalerDecoder(cfg["node_dim"])
        case "u0": # internal energy at 0K
            return ExtensiveScalerDecoder(cfg["node_dim"])
        case "u": # internal energy at 298.15K
            return ExtensiveScalerDecoder(cfg["node_dim"])
        case "h": # enthalpy at 298.15K
            return ExtensiveScalerDecoder(cfg["node_dim"])
        case "g": # free energy at 298.15K
            return ExtensiveScalerDecoder(cfg["node_dim"])
        case "cv": # heat capacity at 298.15K
            return ExtensiveScalerDecoder(cfg["node_dim"])
        case other:
            raise NotImplementedError("Don't know how to decode: {}".format(label))

class GraphDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, mass_center_vec, scaler, vector, batch_index):
        pass

class IntensiveScalerDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.decoder = MLP(in_features=in_features,out_features=1,hidden_dim=hidden_dim,act_fn=nn.Softplus())

    def forward(self, mass_center_vec, scaler, vector, batch_index):
        graph_scaler = scatter_mean(scaler, batch_index, dim=0)
        out = self.decoder(graph_scaler)
        return out

class ExtensiveScalerDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.decoder = MLP(in_features=in_features,out_features=1,hidden_dim=hidden_dim,act_fn=nn.Softplus())

    def forward(self, mass_center_vec, scaler, vector, batch_index):
        node_scaler = self.decoder(scaler)
        out = scatter_sum(node_scaler, batch_index, dim=0)
        return out

class DipoleMomentDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None,out_features=1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.decoder_q = MLP(in_features=in_features, out_features=1, hidden_dim=hidden_dim)
        self.decoder_mu = Gate(in_features=in_features, out_features=1,hidden_dim=hidden_dim)
        self.out_features = out_features

    def forward(self, mass_center_vec, scaler, vector, batch_index):
        q_i = self.decoder_q(scaler)
        mu_i = self.decoder_mu(scaler,vector).squeeze()
        node_mu = mu_i + q_i * mass_center_vec
        graph_mu = scatter_sum(node_mu, batch_index, dim=0)
        out = graph_mu
        if self.out_features == 1:
            out = torch.norm(out, dim=-1, keepdim=True)
        return out

class IsotropicPolarizabilityDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.decoder_alpha = MLP(in_features=in_features,out_features=1,hidden_dim=hidden_dim)
        self.decoder_nu = Gate(in_features=in_features,out_features=1,hidden_dim=hidden_dim)

    def forward(self, mass_center_vec, scaler, vector, batch_index):
        I_3 = torch.eye(3,device=scaler.device)
        nu = self.decoder_nu(scaler, vector).squeeze()
        outer_1 = nu.unsqueeze(-1) * mass_center_vec.unsqueeze(-2)
        outer_2 = mass_center_vec.unsqueeze(-1) * nu.unsqueeze(-2)
        alpha = self.decoder_alpha(scaler).unsqueeze(-1) * I_3 + outer_1 + outer_2
        out = scatter_sum(alpha, batch_index, dim=0)
        out = out.diagonal(dim1=-2, dim2=-1).sum(-1) / 3
        out = out.unsqueeze(-1)
        return out

class ElectronicSpatialExtentDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.decoder_q = MLP(in_features=in_features, out_features=1, hidden_dim=hidden_dim)

    def forward(self, mass_center_vec, scaler, vector, batch_index):
        q_i = self.decoder_q(scaler)
        r2 = (mass_center_vec ** 2).sum(dim=-1, keepdim=True)
        out = scatter_sum(q_i * r2, batch_index, dim=0)
        return out

