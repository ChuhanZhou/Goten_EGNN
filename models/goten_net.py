from configs.config import config as cfg
from models.decoder import MLP,get_decoder

import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.utils import scatter,softmax
import math

def init_parameters(layer):
    if isinstance(layer, nn.Linear):
        if cfg["weight_init"]:
            cfg["weight_init"](layer.weight)
        if cfg["bias_init"] and layer.bias is not None:
            cfg["bias_init"](layer.bias)

class GotenNet(nn.Module):
    def __init__(self, out_label=None, mean=0, std=1):
        super().__init__()
        if out_label is None:
            out_label=cfg["predict_label"]
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        self.embedding = Embedding()

        self.gata_list = nn.ModuleList()
        self.eqff_list = nn.ModuleList()
        for i in range(cfg["layer_num"]):
            self.gata_list.append(GATA(init_X=i==0))
            self.eqff_list.append(EQFF())

        self.decoder = get_decoder(out_label,self.standardize,self.destandardize)
        self.apply(init_parameters)

    def forward(self, atoms_pos, mass_center, x_n, x_e, edge_index,batch_index):
        r_ij,h,t_ij,X = self.embedding(x_n,x_e,edge_index)

        for i in range(cfg["layer_num"]):
            h, X, t_ij = self.gata_list[i](h, X, t_ij, r_ij, edge_index)
            h, X = self.eqff_list[i](h, X)

        out = self.decoder(atoms_pos, mass_center, h,X[0],batch_index)

        return out

    def standardize(self,v):
        return (v - self.mean) / self.std

    def destandardize(self,v):
        return v * self.std + self.mean

class GaussianRBFLayer(nn.Module):
    '''
    Learnable Gaussian Smearing Radial Base Function
    '''
    def __init__(self, out_features, start=0.0, end=cfg["cutoff_radius"]):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(start, end, out_features))

        delta = (end - start) / (out_features - 1)
        gamma_init = 0.5 * delta ** -2
        self.gammas = nn.Parameter(torch.ones(self.centers.shape) * gamma_init)

    def forward(self, x):
        centers = self.centers.unsqueeze(0)
        dist = (x - centers) ** 2
        return torch.exp(-self.gammas*dist)

class ExponentialRBFLayer(nn.Module):
    '''
    Exponential Gaussian Smearing Radial Base Function
    '''
    def __init__(self, out_features, start=None, end=1.0, cutoff=cfg["cutoff_radius"]):
        super().__init__()
        self.cutoff = cutoff
        if start is None:
            start = math.exp(-cutoff)

        self.register_buffer("centers", torch.linspace(start, end, out_features))

        delta = (end - start) / (out_features - 1)
        #gamma_init = 0.5 * delta ** -2
        gamma_init = (2 * delta) ** -2
        gammas = torch.tensor(gamma_init, dtype=torch.float32)
        self.register_buffer("gammas", gammas)
        self.alpha = 5.0 / cutoff

    def forward(self, x):
        centers = self.centers.unsqueeze(0)
        x_exp = torch.exp(self.alpha * (-x))
        dist = (x_exp - centers) ** 2
        return torch.exp(-self.gammas*dist) * cos_cutoff(x,self.cutoff)

def cos_cutoff(r,r_cut=cfg["cutoff_radius"]):
    mask = r<r_cut
    cutoff = (torch.cos(torch.pi*r/r_cut)+1)/2
    return cutoff * mask

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.rbf = ExponentialRBFLayer(cfg["rbf_num"])

        self.w_ndp = nn.Linear(cfg["rbf_num"],cfg["node_dim"], bias=True)
        self.a_nbr = nn.Embedding(len(cfg["atom_types"]),cfg["node_dim"])

        self.a_na = nn.Embedding(len(cfg["atom_types"]),cfg["node_dim"])

        self.w_nrd_nru = nn.Sequential(
            nn.Linear(cfg["node_dim"]*2,cfg["node_dim"], bias=True), # w_nrd
            nn.LayerNorm(cfg["node_dim"]),
            cfg["activation"],
            nn.Linear(cfg["node_dim"], cfg["node_dim"], bias=True), # w_nru
        )

        self.w_erp = nn.Linear(cfg["rbf_num"], cfg["edge_dim"], bias=True)

        irreps = o3.Irreps.spherical_harmonics(cfg['degree_max'])
        self.sphere = o3.SphericalHarmonics(irreps, normalize=False, normalization="norm")

    def forward(self, z, p, edge_index):
        '''
        :param z: atomic number
        :param p: relative position of nodes
        :return:
        '''

        #Edge Tensor Representation
        r_0 = torch.norm(p, dim=1).unsqueeze(1)
        r_ij = [r_0] + list(torch.split(self.sphere(p/r_0)[:, 1:], cfg["high_degree_sizes"], dim=1))

        n_j,n_i = edge_index
        rbf_0 = self.rbf(r_0)

        # Node Scalar Feature Initialization
        m_i = scatter(self.a_nbr(z).squeeze(1)[n_j] * (self.w_ndp(rbf_0) * cos_cutoff(r_0)), n_i, dim=0, dim_size=len(z),reduce="sum")
        h = self.w_nrd_nru(torch.cat([self.a_na(z).squeeze(1),m_i],dim=1))

        # Edge Scalar Feature Initialization
        t_ij = (h[n_i]+h[n_j]) * self.w_erp(rbf_0)

        # High-degree Steerable Feature will be initialized in the first GATA module
        X = None

        return r_ij,h,t_ij,X

class SelfAttentionLayer(nn.Module):
    def __init__(self,out_features):
        super().__init__()
        self.head_num = cfg["attention_heads"]
        assert out_features % self.head_num == 0

        self.w_q = nn.Linear(cfg["node_dim"], cfg["node_dim"], bias=True)
        self.w_k = nn.Linear(cfg["node_dim"], cfg["node_dim"], bias=True)

        self.mlp_v = MLP(in_features=cfg["node_dim"],out_features=out_features)

        self.w_re = nn.Linear(cfg["edge_dim"],cfg["node_dim"],bias=True)

        self.act_fn = cfg["activation"]
        self.dropout = nn.Dropout(cfg["dropout"])

        self.combine_heads = None
        if cfg["combine_heads"]:
            self.combine_heads = nn.Linear(in_features=out_features,out_features=out_features,bias=True)

    def forward(self, h, t_ij, edge_index):
        n_j, n_i = edge_index

        q_i = self.w_q(h).reshape(h.shape[0],self.head_num,-1)[n_i]
        k_j = self.w_k(h).reshape(h.shape[0],self.head_num,-1)[n_j]
        v_j = self.mlp_v(h).reshape(h.shape[0],self.head_num,-1)[n_j]

        a_ij = (q_i * k_j * self.act_fn(self.w_re(t_ij)).reshape(t_ij.shape[0],self.head_num,-1)).sum(dim=-1,keepdims=True)
        a_i = softmax(a_ij, n_i, dim=0)

        # not in the paper but in the author's code
        norm = 1.0 / math.sqrt(v_j.shape[2])
        #n_i_edges = scatter(torch.ones([len(n_i),1,1],device=n_i.device),n_i,dim=0,reduce="sum")
        #norm = torch.sqrt(n_i_edges) / math.sqrt(v_j.shape[2])
        #norm = norm[n_i]

        a_i = self.dropout(a_i * norm)
        sea_ij = a_i * v_j
        sea_ij = sea_ij.flatten(1)

        # not in the paper
        if self.combine_heads is not None:
            sea_ij = self.combine_heads(sea_ij)
            sea_ij = self.dropout(sea_ij)
        return sea_ij

class HTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_vq = nn.Linear(cfg["edge_dim"],cfg["edge_ref_dim"],bias=False)
        self.w_vk = nn.ModuleList([nn.Linear(cfg["edge_dim"],cfg["edge_ref_dim"],bias=False) for _ in range(cfg["degree_max"])])

        # pre-norm is not in the paper, but network has high possibility of exploding (numerical overflow) after depth 4
        #self.mlp_w = MLP(in_features=cfg["edge_ref_dim"], out_features=cfg["edge_dim"],pre_norm=True)
        self.mlp_w = nn.Sequential(
            nn.LayerNorm(cfg["edge_ref_dim"]),
            nn.Linear(in_features=cfg["edge_ref_dim"], out_features=cfg["edge_dim"]),
        )
        self.mlp_t = MLP(in_features=cfg["edge_dim"],out_features=cfg["edge_dim"])

        self.act_fn = cfg["activation"]

    def forward(self, X, t_ij, r_ij, edge_index):
        n_j, n_i = edge_index
        X_ls =  torch.cat(X, dim=1)

        eq_i = self.w_vq(X_ls)[n_i]
        ek_j = torch.cat([w_vl_l(X[i]) for i, w_vl_l in enumerate(self.w_vk)], dim=1)[n_j]

        if cfg["vec_rej"]:
            # vector rejection is not mentioned in the paper, but is a part of official implementation
            eq_i = torch.cat([self.vector_rejection(eq_l_i,r_ij[i]) for i,eq_l_i in enumerate(torch.split(eq_i, cfg["high_degree_sizes"], dim=1))], dim=1)
            ek_j = torch.cat([self.vector_rejection(ek_l_j,-r_ij[i]) for i,ek_l_j in enumerate(torch.split(ek_j, cfg["high_degree_sizes"], dim=1))], dim=1)

        w_ij = (eq_i * ek_j).sum(dim=1)
        #w_ij = None
        #l_dim_start = 0
        #for i in range(len(X)):
        #    l_dim = X[i].shape[1]
        #    eq_i_l = eq_i[:,l_dim_start:l_dim_start+l_dim,:]
        #    ek_j_l = ek_j[:,l_dim_start:l_dim_start+l_dim,:]
        #    w_ij_l = (eq_i_l * ek_j_l).sum(dim=1)
        #    if w_ij is None:
        #        w_ij = w_ij_l
        #    else:
        #        w_ij = w_ij + w_ij_l
        #    l_dim_start += l_dim

        dt_ij = self.mlp_w(w_ij) * self.act_fn(self.mlp_t(t_ij))
        return dt_ij

    @staticmethod
    def vector_rejection(x,r_l_ij, eps=1e-8):
        r_l_ij = r_l_ij.unsqueeze(2)
        dot = (x * r_l_ij).sum(dim=1, keepdim=True)
        #denom = (r_l_ij * r_l_ij).sum(dim=1, keepdim=True) + eps
        vec_proj = dot#/denom
        return x - vec_proj * r_l_ij

class GATA(nn.Module):
    def __init__(self,init_X=False):
        super().__init__()

        self.htr = HTR()

        if init_X:
            S = 1 + cfg["degree_max"] # initialize X in first GATA
        else:
            S = 1 + 2*cfg["degree_max"]
        self.sea = SelfAttentionLayer(S * cfg["node_dim"])
        self.w_rs = nn.Linear(cfg["edge_dim"], S * cfg["node_dim"], bias=True)
        self.mlp_s = MLP(in_features=cfg["node_dim"],out_features=S * cfg["node_dim"])

        self.ln = nn.LayerNorm(cfg["node_dim"])

    def forward(self, h, X, t_ij, r_ij, edge_index):
        n_j, n_i = edge_index
        r_0 = r_ij[0]
        h = self.ln(h)

        if X is not None:
            dt_ij = self.htr(X, t_ij, r_ij[1:], edge_index)
            t_ij = t_ij + dt_ij

        sea_ij = self.sea(h, t_ij, edge_index)

        o_ij = sea_ij + self.w_rs(t_ij) * self.mlp_s(h)[n_j] * cos_cutoff(r_0)
        o_ij = torch.split(o_ij.unsqueeze(1), cfg["node_dim"], dim=-1)

        o_ij_s = o_ij[0].squeeze(1)
        dh = scatter(o_ij_s, n_i, dim=0, dim_size=len(h),reduce="sum")
        h = h + dh

        o_ij_d = o_ij[1:1+cfg["degree_max"]]
        o_ij_t = None
        if X is not None:
            o_ij_t = o_ij[1+cfg["degree_max"]:1+cfg["degree_max"]*2]
        else:
            X = []

        for i in range(len(o_ij_d)):
            l = i + 1
            if o_ij_t is not None:
                dX_l = scatter(o_ij_d[i] * r_ij[l].unsqueeze(-1) + o_ij_t[i] * X[i][n_j], n_i, dim=0, dim_size=len(h),reduce="sum")
                X[i] = X[i] + dX_l
            else:
                dX_l = scatter(o_ij_d[i] * r_ij[l].unsqueeze(-1), n_i, dim=0, dim_size=len(h),reduce="sum")
                X.append(dX_l)

        return h, X, t_ij

class EQFF(nn.Module):
    '''
    Equivariant Feed Forward
    '''
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

        self.w_vu = nn.Linear(cfg["node_dim"],cfg["node_dim"],bias=False)
        self.mlp_m = MLP(in_features=2 * cfg["node_dim"],out_features=2 * cfg["node_dim"],hidden_dim=cfg["node_dim"],pre_norm=True)

    def forward(self, h, X):
        X_ls =  torch.cat(X, dim=1)
        X_vu = self.w_vu(X_ls)
        X_vu_l2 = torch.sqrt(torch.sum(X_vu**2, dim=1) + self.eps)

        m1, m2 = torch.split(self.mlp_m(torch.cat([h, X_vu_l2], dim=1)), cfg["node_dim"],dim=-1)

        h = h + m1
        X_ls = X_ls + m2.unsqueeze(1) * X_vu
        X = list(torch.split(X_ls, cfg["high_degree_sizes"], dim=1))
        return h, X

