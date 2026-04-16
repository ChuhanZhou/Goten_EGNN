from configs.config import config as cfg

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch.autograd import grad
from abc import ABC, abstractmethod
import math

class MLP(nn.Module):
    def __init__(self,in_features,out_features,hidden_dim=None,pre_norm=False,norm=False,act_fn=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_features

        if act_fn is None:
            act_fn = cfg["activation"]

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_features) if pre_norm else nn.Identity(),
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim) if norm else nn.Identity(),
            act_fn,
            #nn.Dropout(cfg["dropout"]),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        return self.mlp(x)

class Gate(nn.Module): # gate
    def __init__(self,in_features,out_features,hidden_dim=None,pre_norm=False,act_fn=None):
        super().__init__()

        self.mlp = MLP(in_features,out_features,hidden_dim,pre_norm,act_fn)
        self.w = nn.Linear(in_features,out_features,bias=False)

    def forward(self, s, v):
        return self.mlp(s.unsqueeze(1)) * self.w(v)

class GateV2(nn.Module): # gate
    def __init__(self,in_features,out_features,hidden_dim=None,pre_norm=False,act_fn=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, in_features+out_features,bias=False)
        self.mlp = MLP(in_features*2,out_features * 2,hidden_dim,pre_norm,act_fn)

    def forward(self, s, v):
        vec_v,vec_w = torch.split(self.w(v), [self.in_features,self.out_features], dim=-1)
        vec_v_norm = torch.norm(vec_v, dim=1)
        x = self.mlp(torch.cat([s, vec_v_norm], dim=-1))
        s,x = torch.split(x, [self.out_features, self.out_features], dim=-1)
        v = x.unsqueeze(-2) * vec_w
        return s, v

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self,x):
        return self.softplus(x) - self.shift

def get_decoder(label,mean=0,std=1):
    match (label):
        case "mu": # dipole moment
            return DipoleMomentDecoder(cfg["node_dim"],out_features=1,mean=mean,std=std)
        case "mu_3d":  # dipole moment
            return DipoleMomentDecoder(cfg["node_dim"],out_features=3,mean=mean,std=std)
        case "alpha": # isotropic polarizability
            return ExtensiveScalerDecoder(cfg["node_dim"])
        case "homo":
            return IntensiveScalerDecoder(cfg["node_dim"])
        case "lumo":
            return IntensiveScalerDecoder(cfg["node_dim"])
        case "gap": # gap(lumo-homo)
            raise NotImplementedError("Don't know how to decode: {}".format(label))
        case "r2": # electronic spatial extent
            return ElectronicSpatialExtentDecoder(cfg["node_dim"],mean=mean,std=std)
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
        case "e&f":
            return EnergyForceDecoder(cfg["node_dim"],mean=mean,std=std)
        case other:
            raise NotImplementedError("Don't know how to decode: {}".format(label))

class GraphDecoder(nn.Module):
    def __init__(self,mean=0,std=1):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    @abstractmethod
    def forward(self, pos, scaler, vector, batch_index):
        pass

    def standardize(self,v):
        return (v - self.mean) / self.std

    def destandardize(self,v):
        return v * self.std + self.mean

class IntensiveScalerDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.decoder = MLP(in_features=in_features,out_features=1,hidden_dim=hidden_dim)

    def forward(self, pos, scaler, vector, batch_index):
        graph_scaler = scatter(scaler, batch_index, dim=0, reduce="mean")
        out = self.decoder(graph_scaler)
        return out

class ExtensiveScalerDecoder(GraphDecoder):
    def __init__(self, in_features, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.decoder = MLP(in_features=in_features,out_features=1,hidden_dim=hidden_dim)

    def forward(self, pos, scaler, vector, batch_index):
        node_scaler = self.decoder(scaler)
        out = scatter(node_scaler, batch_index, dim=0, reduce="sum")
        return out

class ScalerDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.node_decoder = MLP(in_features=in_features, out_features=hidden_dim, hidden_dim=hidden_dim)
        self.graph_decoder = MLP(in_features=in_features, out_features=hidden_dim, hidden_dim=hidden_dim)
        #self.finial_decoder = MLP(in_features=hidden_dim * 2, out_features=1, hidden_dim=hidden_dim)
        self.finial_decoder = nn.Linear(in_features=hidden_dim*2, out_features=1)

        self.act_fn = cfg["activation"]

    def forward(self, pos, scaler, vector, batch_index):
        node_scaler = self.act_fn(self.node_decoder(scaler))
        node_scaler = scatter(node_scaler, batch_index, dim=0, reduce="sum")
        graph_scaler = scatter(scaler, batch_index, dim=0, reduce="mean")
        graph_scaler = self.act_fn(self.graph_decoder(graph_scaler))
        mix_scaler = torch.cat([graph_scaler, node_scaler], dim=-1)
        out = self.finial_decoder(mix_scaler)
        return out

class DipoleMomentDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None,out_features=1,mean=0,std=1):
        super().__init__(mean, std)
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.gate_0 = GateV2(in_features=in_features, out_features=hidden_dim)
        self.act_fn = cfg["activation"]
        self.gate_1 = GateV2(in_features=hidden_dim, out_features=1)
        self.out_features = out_features

    def forward(self, pos, scaler, vector, batch_index):
        q_i, mu_i = self.gate_0(scaler,vector)
        q_i = self.act_fn(q_i)
        q_i, mu_i = self.gate_1(q_i, mu_i)
        mu_i = mu_i.squeeze()

        q_i = self.destandardize(q_i)

        node_mu = mu_i + q_i * pos
        graph_mu = scatter(node_mu, batch_index, dim=0, reduce="sum")
        out = graph_mu
        if self.out_features == 1:
            out = torch.norm(out, dim=-1, keepdim=True)

        out = self.standardize(out)
        return out

class ElectronicSpatialExtentDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None,mean=0,std=1):
        super().__init__(mean, std)
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.decoder_q = MLP(in_features=in_features, out_features=1, hidden_dim=hidden_dim,act_fn=ShiftedSoftplus())

    def forward(self, pos, scaler, vector, batch_index):
        q_i = self.decoder_q(scaler)

        r2_i = torch.norm(pos,dim=1,keepdim=True) ** 2

        out = scatter(q_i * r2_i, batch_index, dim=0, reduce="sum")

        #out = self.standardize(out)

        return out

class EnergyForceDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None,mean=0,std=1):
        super().__init__(mean, std)
        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.decoder = MLP(in_features=in_features, out_features=1, hidden_dim=hidden_dim)

    def forward(self, pos, scaler, vector, batch_index):
        node_scaler = self.decoder(scaler)
        energy_out = scatter(node_scaler, batch_index, dim=0, reduce="sum")

        energy = energy_out

        force_out = -grad(
            outputs=energy,
            inputs=pos,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
            retain_graph=True,
        )[0]

        return [energy_out, force_out]
