from configs.config import config as cfg
from models.decoder import MLP,get_decoder
from models.goten_net import GotenNet,SelfAttentionLayer,init_parameters

import torch
import torch.nn as nn
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
                    gata.sea = NoSEA(gata.S * cfg["node_dim"])
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

class NoSEA(SelfAttentionLayer):
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