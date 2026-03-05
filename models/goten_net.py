from configs.config import config as cfg
from models.decoder import MLP,get_decoder

import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter_softmax,scatter_sum

class GotenNet(nn.Module):
    def __init__(self, out_label=None, mean=0, std=1):
        super().__init__()
        if out_label is None:
            out_label=cfg["predict_label"]
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

        self.embedding = Embedding()

        self.gata_list = nn.ModuleList()
        self.eqff_list = nn.ModuleList()
        for i in range(cfg["layer_num"]):
            self.gata_list.append(GATA())
            self.eqff_list.append(EQFF())

        self.decoder = get_decoder(out_label)

    def forward(self, x_n, x_e, edge_index,batch_index):
        r_ij,h,t_ij,X = self.embedding(x_n,x_e,edge_index)

        for i in range(cfg["layer_num"]):
            h, X, t_ij = self.gata_list[i](h, X, t_ij, r_ij, edge_index)
            h, X = self.eqff_list[i](h, X)

        out = self.decoder(h,X,batch_index)

        return out

    def standardize(self,v):
        return (v - self.mean) / self.std

    def destandardize(self,v):
        return v * self.std + self.mean

class RBFLayer(nn.Module):
    '''
    Gaussian Radial Base Function
    '''
    def __init__(self, out_features, start=0.0, end=cfg["cutoff_radius"]):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(start, end, out_features))

        delta = (end - start) / (out_features - 1)
        gamma_init = 1.0 / (2 * delta ** 2)
        self.gamma = nn.Parameter(torch.ones(self.centers.shape) * gamma_init)

    def forward(self, x):
        centers = self.centers.unsqueeze(0)
        dist = (x - centers) ** 2
        return torch.exp(-self.gamma*dist)

def cos_cutoff(d,c=cfg["cutoff_radius"]):
    mask = d<=c
    cutoff = (torch.cos(torch.pi*d/c)+1)/2
    return cutoff * mask

class SelfAttentionLayer(nn.Module):
    def __init__(self,out_features):
        super().__init__()
        self.head_num = cfg["attention_heads"]
        assert out_features % self.head_num == 0

        self.w_q = nn.Linear(cfg["node_dim"], cfg["node_dim"], bias=False)
        self.w_k = nn.Linear(cfg["node_dim"], cfg["node_dim"], bias=False)

        self.mlp_v = MLP(in_features=cfg["node_dim"],out_features=out_features)

        self.w_re = nn.Linear(cfg["edge_dim"],cfg["node_dim"],bias=False)

        self.ln = nn.LayerNorm(cfg["node_dim"])
        self.act = cfg["activation"]
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, h, t_ij, edge_index):
        n_j, n_i = edge_index
        h = self.ln(h)

        q_i = self.w_q(h).reshape(h.shape[0],self.head_num,-1)[n_i]
        k_j = self.w_k(h).reshape(h.shape[0],self.head_num,-1)[n_j]
        v_j = self.mlp_v(h).reshape(h.shape[0],self.head_num,-1)[n_j]

        a_ij = (q_i * k_j * self.act(self.w_re(t_ij).reshape(t_ij.shape[0],self.head_num,-1))).sum(dim=-1,keepdims=True)
        sea_ij = self.dropout(scatter_softmax(a_ij,n_i,dim=0)) * v_j
        sea_ij = sea_ij.reshape(sea_ij.shape[0],-1)
        return sea_ij

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.rbf = RBFLayer(cfg["rbf_num"])

        self.w_ndp = nn.Linear(cfg["rbf_num"],cfg["node_dim"], bias=False)
        self.a_nbr = nn.Linear(len(cfg["atom_types"]),cfg["node_dim"], bias=True)

        self.a_na = nn.Linear(len(cfg["atom_types"]), cfg["node_dim"], bias=True)
        self.w_nrd = nn.Linear(cfg["node_dim"]*2,cfg["node_dim"], bias=False)
        self.w_nru = nn.Linear(cfg["node_dim"], cfg["node_dim"], bias=False)

        self.w_erp = nn.Linear(cfg["rbf_num"], cfg["edge_dim"], bias=False)

        self.sea = SelfAttentionLayer(cfg["degree_max"] * cfg["node_dim"])
        self.w_rs = nn.Linear(cfg["edge_dim"], cfg["degree_max"] * cfg["node_dim"], bias=False)
        self.mlp_s = MLP(in_features=cfg["node_dim"],out_features=cfg["degree_max"] * cfg["node_dim"])

        self.ln = nn.LayerNorm(cfg["node_dim"])
        self.act = cfg["activation"]

    def forward(self, z, p, edge_index):
        '''
        :param z: atomic number
        :param p: relative position of nodes
        :return:
        '''
        z = nn.functional.one_hot(z.type(torch.int64),num_classes=len(cfg["atom_types"])).squeeze(1).float()

        #Edge Tensor Representation
        r_0 = torch.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2).unsqueeze(1)
        r_1 = p/r_0
        r_ij = [r_0,r_1]
        for l in range(len(r_ij),cfg['degree_max']+1):
            r_l = o3.spherical_harmonics(l, r_1, normalize=False, normalization='norm')
            r_ij.append(r_l)

        n_j,n_i = edge_index
        rbf_0 = self.rbf(r_0)

        # Node Scalar Feature Initialization
        m_i = scatter_sum(self.a_nbr(z)[n_j] * (self.w_ndp(rbf_0) * cos_cutoff(r_0)), n_i, dim=0, dim_size=len(z))
        h = self.w_nru(self.act(self.ln(self.w_nrd(torch.cat([self.a_na(z),m_i],dim=1)))))

        # Edge Scalar Feature Initialization
        t_ij = (h[n_i]+h[n_j]) * self.w_erp(rbf_0)

        # High-degree Steerable Feature Initialization
        sea_ij = self.sea(h,t_ij,edge_index)
        #o_ij = torch.split(sea_ij.unsqueeze(1) + self.w_rs(t_ij).unsqueeze(-1) * self.mlp_s(h[n_j]).unsqueeze(1) * cos_cutoff(r_0).unsqueeze(-1), cfg["node_dim"], dim=-1)
        o_ij = torch.split(sea_ij + self.w_rs(t_ij) * self.mlp_s(h)[n_j] * cos_cutoff(r_0),cfg["node_dim"], dim=-1)
        X = []
        for i in range(len(o_ij)):
            l = i + 1
            X_l = scatter_sum(o_ij[i].unsqueeze(1) * r_ij[l].unsqueeze(-1), n_i, dim=0, dim_size=len(z))
            X.append(X_l)

        return r_ij,h,t_ij,X

class HTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_vq = nn.Linear(cfg["edge_dim"],cfg["edge_ref_dim"],bias=False)
        self.w_vk = nn.ModuleList()
        for i in range(cfg["degree_max"]):
            self.w_vk.append(nn.Linear(cfg["edge_dim"],cfg["edge_ref_dim"],bias=False))

        self.mlp_w = MLP(in_features=cfg["edge_ref_dim"], out_features=cfg["edge_dim"])
        self.mlp_t = MLP(in_features=cfg["edge_dim"],out_features=cfg["edge_dim"])

    def forward(self, X, t_ij, edge_index):
        n_j, n_i = edge_index

        eq_i = self.w_vq(torch.cat(X, dim=1))[n_i]
        #eq_i = torch.cat([self.w_vq(X_l) for X_l in X], dim=1)[n_i]
        ek_j = torch.cat([w_vl_l(X[i]) for i, w_vl_l in enumerate(self.w_vk)], dim=1)[n_j]

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

        dt_ij = self.mlp_w(w_ij) * self.mlp_t(t_ij)
        return dt_ij

class GATA(nn.Module):
    def __init__(self):
        super().__init__()

        self.htr = HTR()

        S = 1+2*cfg["degree_max"]
        self.sea = SelfAttentionLayer(S * cfg["node_dim"])
        self.w_rs = nn.Linear(cfg["edge_dim"], S * cfg["node_dim"], bias=False)
        self.mlp_s = MLP(in_features=cfg["node_dim"],out_features=S * cfg["node_dim"])

    def forward(self, h, X, t_ij, r_ij, edge_index):
        n_j, n_i = edge_index
        r_0 = r_ij[0]

        dt_ij = self.htr(X, t_ij, edge_index)
        t_ij = t_ij + dt_ij

        sea_ij = self.sea(h, t_ij, edge_index)
        o_ij = sea_ij + self.w_rs(t_ij) * self.mlp_s(h)[n_j] * cos_cutoff(r_0)
        o_ij = torch.split(o_ij.unsqueeze(1), cfg["node_dim"], dim=-1)

        o_ij_s = o_ij[0].squeeze(1)
        dh = scatter_sum(o_ij_s, n_i, dim=0, dim_size=len(h))
        h = h + dh

        #l_dim_list = [2*l+1 for l in range(1,cfg["degree_max"]+1)]
        #o_ij_d = o_ij[1:1 + cfg["degree_max"]]
        #o_ij_d = [o_ij_d[i].expand(-1, l_dim_list[i], -1) for i in range(len(o_ij_d))]
        #o_ij_d = torch.cat(o_ij_d, dim=1)
        #o_ij_t = o_ij[1 + cfg["degree_max"]:1 + cfg["degree_max"] * 2]
        #o_ij_t = [o_ij_t[i].expand(-1, l_dim_list[i], -1) for i in range(len(o_ij_t))]
        #o_ij_t = torch.cat(o_ij_t, dim=1)
        #r_ij_d = torch.cat(r_ij[1:],dim=1)
        #X = torch.cat(X, dim=1)
        #dX_l = scatter_sum(o_ij_d * r_ij_d.unsqueeze(-1) + o_ij_t * X[n_j], n_i,dim=0, dim_size=len(h))
        #X = X + dX_l
        #X = torch.split(X, l_dim_list, dim=1)

        o_ij_d = o_ij[1:1+cfg["degree_max"]]
        o_ij_t = o_ij[1+cfg["degree_max"]:1+cfg["degree_max"]*2]
        for i in range(len(X)):
           l = i + 1
           dX_l = scatter_sum(o_ij_d[i] * r_ij[l].unsqueeze(-1) + o_ij_t[i] * X[i][n_j], n_i, dim=0, dim_size=len(h))
           X[i] = X[i] + dX_l

        return h, X, t_ij

class EQFF(nn.Module):
    '''
    Equivariant Feed Forward
    '''
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

        self.w_vu = nn.Linear(cfg["node_dim"],cfg["node_dim"],bias=False)
        self.mlp_m = MLP(in_features=2 * cfg["node_dim"],out_features=2 * cfg["node_dim"],hidden_dim=cfg["node_dim"])

    def forward(self, h, X):
        X_ls =  torch.cat(X, dim=1)
        X_vu = self.w_vu(X_ls)

        m1,m2 = torch.split(self.mlp_m(torch.cat([X_vu.norm(dim=1, p=2) + self.epsilon,h],dim=1)),cfg["node_dim"],dim=-1)

        h = h + m1
        X_ls = X_ls + m2.unsqueeze(1) * X_vu
        X = list(torch.split(X_ls, [X_l.shape[1] for X_l in X], dim=1))
        return h, X

