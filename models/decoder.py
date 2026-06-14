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

def get_decoder(label,mean=0,std=1,need_guide=False):
    decoder = None
    match (label):
        case "mu": # dipole moment
            decoder = DipoleMomentDecoder(cfg["node_dim"],out_features=1,mean=mean,std=std)
        #case "mu_3d":  # dipole moment
        #    decoder = DipoleMomentDecoder(cfg["node_dim"],out_features=3,mean=mean,std=std)
        case "alpha": # isotropic polarizability
            decoder = ExtensiveScalarDecoder(cfg["node_dim"])
        case "homo":
            decoder = IntensiveScalarDecoder(cfg["node_dim"])
        case "lumo":
            decoder = IntensiveScalarDecoder(cfg["node_dim"])
        case "gap": # gap(lumo-homo)
            #return GapDecoder(cfg["node_dim"])
            raise NotImplementedError("Don't know how to decode: {}".format(label))
        case "r2": # electronic spatial extent
            decoder = ElectronicSpatialExtentDecoder(cfg["node_dim"],mean=mean,std=std)
        case "zpve": # zero point vibrational energy
            decoder = ExtensiveScalarDecoder(cfg["node_dim"])
        case "u0": # internal energy at 0K
            decoder = ExtensiveScalarDecoder(cfg["node_dim"])
        case "u": # internal energy at 298.15K
            decoder = ExtensiveScalarDecoder(cfg["node_dim"])
        case "h": # enthalpy at 298.15K
            decoder = ExtensiveScalarDecoder(cfg["node_dim"])
        case "g": # free energy at 298.15K
            decoder = ExtensiveScalarDecoder(cfg["node_dim"])
        case "cv": # heat capacity at 298.15K
            decoder = ExtensiveScalarDecoder(cfg["node_dim"])
        case "e&f":
            return EnergyForceDecoder(cfg["node_dim"],mean=mean,std=std)
        case "scalar_ext":
            decoder = ExtensiveScalarDecoder(cfg["node_dim"])
        case "scalar_int":
            decoder = IntensiveScalarDecoder(cfg["node_dim"])
        case "scalar_mix":
            decoder = ScalarDecoder(cfg["node_dim"])
        case other:
            raise NotImplementedError("Don't know how to decode: {}".format(label))

    if need_guide:
        decoder = GlobalGuideDecoder(decoder,cfg["node_dim"])
    return decoder

class GraphDecoder(nn.Module):
    def __init__(self,mean=0,std=1):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.need_vector = False

    @abstractmethod
    def forward(self, pos, scalar, vector, batch_index):
        pass

    def standardize(self,v):
        return (v - self.mean) / self.std

    def destandardize(self,v):
        return v * self.std + self.mean

class IntensiveScalarDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None,act_fn=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        if act_fn is None:
            act_fn = cfg["activation"]

        self.decoder = MLP(in_features=in_features,out_features=1,hidden_dim=hidden_dim,act_fn=act_fn)

    def forward(self, pos, scalar, vector, batch_index):
        graph_scalar = scatter(scalar, batch_index, dim=0, reduce="mean")
        out = self.decoder(graph_scalar)
        return out

class ExtensiveScalarDecoder(GraphDecoder):
    def __init__(self, in_features, hidden_dim=None,act_fn=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        if act_fn is None:
            act_fn = cfg["activation"]

        self.decoder = MLP(in_features=in_features,out_features=1,hidden_dim=hidden_dim,act_fn=act_fn)

    def forward(self, pos, scalar, vector, batch_index):
        node_scalar = self.decoder(scalar)
        out = scatter(node_scalar, batch_index, dim=0, reduce="sum")
        return out

class ScalarDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None,act_fn=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        if act_fn is None:
            act_fn = cfg["activation"]

        self.node_decoder = ExtensiveScalarDecoder(in_features=in_features, hidden_dim=hidden_dim,act_fn=act_fn)
        self.graph_decoder = IntensiveScalarDecoder(in_features=in_features, hidden_dim=hidden_dim,act_fn=act_fn)
        #self.w = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, pos, scalar, vector, batch_index):
        node_scalar = self.node_decoder(pos, scalar, vector, batch_index)
        graph_scalar = self.graph_decoder(pos, scalar, vector, batch_index)
        #out = self.w * graph_scalar + (1-self.w) * node_scalar
        out = graph_scalar + node_scalar
        return out

class GlobalGuideDecoder(GraphDecoder):
    def __init__(self,default_decoder,in_features,hidden_dim=None,act_fn=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2

        if act_fn is None:
            act_fn = cfg["activation"]

        self.default_decoder = default_decoder
        if default_decoder.need_vector:
            self.guide_decoder = GateV2(in_features=in_features, hidden_dim=hidden_dim, out_features=1, act_fn=act_fn)
        else:
            self.guide_decoder = IntensiveScalarDecoder(in_features=in_features, hidden_dim=hidden_dim, act_fn=act_fn)

    def forward(self, pos, scalar, vector, batch_index):
        default_out = self.default_decoder(pos, scalar, vector, batch_index)

        if not self.training:
            return default_out

        if self.default_decoder.need_vector:
            s = scatter(scalar, batch_index, dim=0, reduce="mean")
            v = scatter(vector, batch_index, dim=0, reduce="mean")
            s, v = self.guide_decoder(s, v)
            guide_out = s + torch.norm(v, dim=1)
        else:
            guide_out = self.guide_decoder(pos, scalar, vector, batch_index)
        return default_out,guide_out

class DipoleMomentDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None,out_features=1,mean=0,std=1):
        super().__init__(mean, std)
        self.need_vector = True

        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.gate_0 = GateV2(in_features=in_features, out_features=hidden_dim)
        self.act_fn = cfg["activation"]
        self.gate_1 = GateV2(in_features=hidden_dim, out_features=1)
        self.out_features = out_features

        #self.decoder = MLP(in_features=in_features,out_features=1,hidden_dim=hidden_dim)

        #self.graph_decoder = IntensiveScalarDecoder(in_features=in_features, hidden_dim=hidden_dim)
        #self.w = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, pos, scalar, vector, batch_index):
        q_i, mu_i = self.gate_0(scalar,vector)
        q_i = self.act_fn(q_i)
        q_i, mu_i = self.gate_1(q_i, mu_i)
        mu_i = mu_i.squeeze()

        #q_i = self.decoder(scalar)
        q_i = self.destandardize(q_i)

        node_mu = mu_i + q_i * pos
        graph_mu = scatter(node_mu, batch_index, dim=0, reduce="sum")

        out = graph_mu
        if self.out_features == 1:
            out = torch.norm(out, dim=-1, keepdim=True)

        out = self.standardize(out)
        #out = (1-self.w)*out + self.w*self.graph_decoder(pos, scalar, vector, batch_index)
        return out#,self.graph_decoder(pos, scalar, vector, batch_index)

class ElectronicSpatialExtentDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None,mean=0,std=1):
        super().__init__(mean, std)
        if hidden_dim is None:
            hidden_dim = in_features // 2

        #self.decoder_q = MLP(in_features=in_features, out_features=1, hidden_dim=hidden_dim, act_fn=ShiftedSoftplus())
        self.decoder_q = MLP(in_features=in_features, out_features=1, hidden_dim=hidden_dim)

    def forward(self, pos, scalar, vector, batch_index):
        q_i = self.decoder_q(scalar)

        r2_i = torch.norm(pos,dim=1,keepdim=True) ** 2

        out = q_i * r2_i

        out = scatter(out, batch_index, dim=0, reduce="sum")

        out = self.standardize(out)

        return out

class EnergyForceDecoder(GraphDecoder):
    def __init__(self,in_features,hidden_dim=None,mean=0,std=1):
        super().__init__(mean, std)

        if hidden_dim is None:
            hidden_dim = in_features // 2

        self.decoder = ExtensiveScalarDecoder(in_features, hidden_dim)
        #self.decoder = nn.Linear(in_features, 1)

    def forward(self, pos, scalar, vector, batch_index):
        energy_out = self.decoder(pos, scalar, vector, batch_index)
        #energy_out = scatter(self.decoder(scalar), batch_index, dim=0, reduce="sum")

        energy_out = self.destandardize(energy_out)
        energy = energy_out
        #energy_out = self.destandardize(energy_out)

        force_out = -grad(
            outputs=energy,
            inputs=pos,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
            retain_graph=True,
        )[0]

        #force_out = force_out * self.std

        return [energy_out, force_out]
