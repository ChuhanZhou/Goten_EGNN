from configs.config import config as cfg
from models.decoder import MLP,get_decoder
from models.goten_net import GotenNet,init_parameters,SelfAttentionLayer,HTR

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_geometric.utils import scatter,softmax
import math

class AblationNet(GotenNet):
    def __init__(self, out_label=None, mean=0, std=1, rbf_type="exp",ablation_type=None):
        super().__init__(out_label,mean,std,rbf_type)

        match ablation_type:
            case "sea_sing":
                for gata in self.gata_list:
                    gata.sea = SingleHeadSEA(gata.S * cfg["node_dim"])
            case "sea_comb":
                for gata in self.gata_list:
                    gata.sea = CombineHeadSEA(gata.S * cfg["node_dim"], cfg["node_dim"])
            case "sea_no":
                for gata in self.gata_list:
                    gata.sea = NoSEAButWeight(gata.S * cfg["node_dim"])
            case "sea_norm":
                for gata in self.gata_list:
                    gata.sea = NoSEAButNorm(gata.S * cfg["node_dim"])
            case "htr_att":
                for gata in self.gata_list:
                    gata.htr = AttHTR() if gata.htr is not None else None
            case "htr_scalar":
                for gata in self.gata_list:
                    gata.htr = ScalarHTR() if gata.htr is not None else None
            case other:
                raise NotImplementedError("Unknown ablation study type: {}".format(other))

        self.apply(init_parameters)
        self.set_decoder(self.out_label, mean, std)

class SingleHeadSEA(SelfAttentionLayer):
    def __init__(self,out_features):
        super().__init__(out_features)

    def forward(self, h, t_ij, edge_index):
        n_j, n_i = edge_index

        q_i = self.w_q(h)[n_i]
        k_j = self.w_k(h)[n_j]
        v_j = self.mlp_v(h)[n_j]

        a_ij = (q_i * k_j * self.act_fn(self.w_re(t_ij))).sum(dim=-1,keepdims=True)
        a_i = softmax(a_ij, n_i, num_nodes=h.shape[0], dim=0)

        # not in the paper but in the author's code
        # norm = 1.0 / math.sqrt(cfg["node_dim"])
        n_i_edges = scatter(torch.ones([len(n_i), 1], device=n_i.device), n_i, dim=0, reduce="sum")
        norm = torch.sqrt(n_i_edges) / math.sqrt(cfg["node_dim"])
        norm = norm[n_i]
        a_i = a_i * norm

        a_i = self.dropout(a_i)
        sea_ij = a_i * v_j  # [E,D*5]

        return sea_ij

class CombineHeadSEA(SelfAttentionLayer):
    def __init__(self, out_features, attn_dim=None):
        attn_dim = out_features if attn_dim is None else attn_dim
        super().__init__(attn_dim)

        self.combine_heads = nn.Linear(in_features=attn_dim, out_features=out_features, bias=True)

class NoSEAButWeight(SelfAttentionLayer):
    def __init__(self,out_features):
        super().__init__(out_features)

        self.w_q = nn.Linear(cfg["node_dim"], cfg["node_dim"]//2)
        self.w_k = nn.Linear(cfg["node_dim"], cfg["node_dim"]//2)
        self.w_re = nn.Linear(cfg["edge_dim"], cfg["node_dim"]//2)

        self.mlp_w = MLP(in_features=cfg["node_dim"]//2*3, out_features=1)

    def forward(self, h, t_ij, edge_index):
        n_j, n_i = edge_index

        q_i = self.w_q(h)[n_i]
        k_j = self.w_k(h)[n_j]
        v_j = self.mlp_v(h)[n_j]

        a_ij = torch.cat([q_i,k_j,self.act_fn(self.w_re(t_ij))], dim=-1)
        a_ij = self.mlp_w(a_ij)
        a_i = softmax(a_ij, n_i, num_nodes=h.shape[0], dim=0)

        n_i_edges = scatter(torch.ones([len(n_i), 1], device=n_i.device), n_i, dim=0, reduce="sum")
        norm = torch.sqrt(n_i_edges) / math.sqrt(cfg["node_dim"])
        norm = norm[n_i]
        a_i = a_i * norm

        a_i = self.dropout(a_i)
        sea_ij = a_i * v_j  # [E,D*5]

        return sea_ij

class NoSEAButNorm(SelfAttentionLayer):
    def __init__(self,out_features):
        super().__init__(out_features)

    def forward(self, h, t_ij, edge_index):
        n_j, n_i = edge_index

        v_j = self.mlp_v(h)[n_j]

        n_i_edges = scatter(torch.ones([len(n_i), 1], device=n_i.device), n_i, dim=0, reduce="sum")
        norm = torch.sqrt(n_i_edges) / math.sqrt(cfg["node_dim"])
        #norm = norm[n_i]

        a = 1/n_i_edges
        a = a * norm
        a_i = a[n_i]
        sea_ij = a_i * v_j

        return sea_ij

class AttHTR(HTR):
    def __init__(self):
        super().__init__()
        self.head_num = cfg["attention_heads"]

        self.mlp_w = None
        self.mlp_t = MLP(in_features=cfg["edge_dim"],out_features=cfg["edge_ref_dim"])

        self.comb = nn.Linear(in_features=cfg["edge_ref_dim"], out_features=cfg["edge_dim"], bias=True)

        self.dropout = nn.Dropout(cfg["dropout"]) if self.head_num > 1 else nn.Identity()

    def forward(self, h, X, t_ij, r_ij, edge_index):
        n_j, n_i = edge_index
        X_ls =  torch.cat(X, dim=1) #[N,8,256]

        eq_i = self.w_vq(X_ls)[n_i] #[E,8,256]
        ek_j = torch.cat([w_vl_l(X[i]) for i, w_vl_l in enumerate(self.w_vk)], dim=1)[n_j] #[E,8,256]
        ev_ij = self.mlp_t(t_ij)

        #muti-head
        eq_i = eq_i.reshape(eq_i.shape[0],eq_i.shape[1],self.head_num,-1)
        ek_j = ek_j.reshape(ek_j.shape[0],ek_j.shape[1],self.head_num,-1)
        ev_ij = ev_ij.reshape(ev_ij.shape[0],self.head_num,-1) #[E,8,32]

        a_ij = (eq_i * ek_j).sum(dim=1) #[E,8]
        a_ij = F.softmax(a_ij,dim=-1)
        a_ij = self.dropout(a_ij)

        dt_ij = a_ij * ev_ij #[E,256]
        dt_ij = dt_ij.flatten(1)

        dt_ij = self.comb(dt_ij)
        return dt_ij

class ScalarHTR(HTR):
    def __init__(self):
        super().__init__()
        self.w_vq = nn.Linear(cfg["edge_dim"],cfg["edge_ref_dim"],bias=True)
        self.w_vk = nn.Linear(cfg["edge_dim"],cfg["edge_ref_dim"],bias=True)

    def forward(self, h, X, t_ij, r_ij, edge_index):
        n_j, n_i = edge_index

        eq_i = self.w_vq(h)[n_i]
        ek_j = self.w_vk(h)[n_j]

        w_ij = eq_i * ek_j

        dt_ij = self.mlp_w(w_ij) * self.mlp_t(t_ij)
        return dt_ij